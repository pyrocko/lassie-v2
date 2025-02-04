from __future__ import annotations

import logging
import re
import struct
import zipfile
from datetime import datetime, timedelta
from functools import cached_property
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from lru import LRU
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    FilePath,
    PrivateAttr,
    ValidationError,
    constr,
    model_validator,
)
from pyrocko import orthodrome as od
from pyrocko import spit
from pyrocko.cake import LayeredModel, PhaseDef, load_model, m2d
from pyrocko.gf import meta
from pyrocko.plot.cake_plot import my_model_plot as earthmodel_plot

from qseek.octree import get_node_coordinates
from qseek.stats import PROGRESS
from qseek.tracers.base import ModelledArrival, RayTracer
from qseek.utils import (
    CACHE_DIR,
    PhaseDescription,
    alog_call,
    datetime_now,
    human_readable_bytes,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from qseek.models.location import Location
    from qseek.models.station import Stations
    from qseek.octree import Node, Octree

logger = logging.getLogger(__name__)

KM = 1e3
GiB = int(1024**3)
MAX_DBS = 16

LRU_CACHE_SIZE = 2000


# TODO: Move to a separate file
DEFAULT_VELOCITY_MODEL = """
-1.00    5.50    3.59    2.7
 0.00    5.50    3.59    2.7
 1.00    5.50    3.59    2.7
 1.00    6.00    3.92    2.7
 4.00    6.00    3.92    2.7
 4.00    6.20    4.05    2.7
 8.00    6.20    4.05    2.7
 8.00    6.30    4.12    2.7
13.00    6.30    4.12    2.7
13.00    6.40    4.18    2.7
17.00    6.40    4.18    2.7
17.00    6.50    4.25    2.7
22.00    6.50    4.25    2.7
22.00    6.60    4.31    2.7
26.00    6.60    4.31    2.7
26.00    6.80    4.44    2.7
30.00    6.80    4.44    2.7
30.00    8.10    5.29    2.7
45.00    8.10    5.29    2.7
"""

DEFAULT_VELOCITY_MODEL_FILE = CACHE_DIR / "velocity_models" / "default.nd"
if not DEFAULT_VELOCITY_MODEL_FILE.exists():
    DEFAULT_VELOCITY_MODEL_FILE.parent.mkdir(exist_ok=True)
    DEFAULT_VELOCITY_MODEL_FILE.write_text(DEFAULT_VELOCITY_MODEL)


class EarthModel(BaseModel):
    filename: FilePath | None = Field(
        default=DEFAULT_VELOCITY_MODEL_FILE,
        description="Path to velocity model.",
    )
    format: Literal["nd", "hyposat"] = Field(
        default="nd",
        description="Format of the velocity model. `nd` or `hyposat` is supported.",
    )
    crust2_profile: constr(to_upper=True) | tuple[float, float] = Field(
        default="",
        description="Crust2 profile name or a tuple of `(lat, lon)` coordinates.",
    )

    raw_file_data: str | None = Field(
        default=None,
        description="Raw `.nd` file data.",
    )
    _layered_model: LayeredModel = PrivateAttr()

    model_config = ConfigDict(ignored_types=(cached_property,))

    @model_validator(mode="after")
    def load_model(self) -> EarthModel:
        if self.filename is not None and self.raw_file_data is None:
            logger.info("loading velocity model from %s", self.filename)
            self.raw_file_data = self.filename.read_text()

        if self.raw_file_data is not None:
            with NamedTemporaryFile("w") as tmpfile:
                tmpfile.write(self.raw_file_data)
                tmpfile.flush()
                self._layered_model = load_model(
                    tmpfile.name,
                    format=self.format,
                    crust2_profile=self.crust2_profile or None,
                )
        elif self.crust2_profile:
            self._layered_model = load_model(crust2_profile=self.crust2_profile)
        else:
            raise AttributeError("No velocity model or crust2 profile defined.")
        return self

    def trim(self, depth_max: float) -> None:
        """Trim the model to a maximum depth.

        Args:
            depth_max (float): Maximum depth in meters.
        """
        logger.debug("trimming earth model to %.1f km depth", depth_max / KM)
        self._layered_model = self.layered_model.extract(depth_max=depth_max)

    @property
    def layered_model(self) -> LayeredModel:
        return self._layered_model

    def get_profile_vp(self) -> np.ndarray:
        return self.layered_model.profile("vp")

    def get_profile_vs(self) -> np.ndarray:
        return self.layered_model.profile("vs")

    def save_plot(self, filename: Path) -> None:
        """Plot the layered model and save the figure to a file.

        Args:
            filename (Path): The path to save the figure.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        earthmodel_plot(self.layered_model, axes=ax)
        fig.savefig(filename, dpi=300)
        if self.filename:
            ax.set_title(f"File: {self.filename}")

        logger.info("saved earth model plot to %s", filename)

    @cached_property
    def hash(self) -> str:
        model_serialised = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            self.layered_model.profile(param).dump(model_serialised)
        return sha1(model_serialised.getvalue()).hexdigest()


class Timing(BaseModel):
    definition: constr(strip_whitespace=True) = "P,p"

    def as_phase_defs(self) -> list[PhaseDef]:
        return [PhaseDef(definition=phase) for phase in self.definition.split(",")]

    def as_pyrocko_timing(self) -> meta.Timing:
        return meta.Timing(f"{{stored:{self.id}}}")

    @property
    def id(self) -> str:
        return re.sub(r"[\,\s\;]", "", self.definition)


def surface_distances(nodes: Sequence[Node], stations: Stations) -> np.ndarray:
    """Returns the surface distance from all nodes to all stations.

    Args:
        nodes (Sequence[Node]): Nodes to calculate distance from.
        stations (Stations): Stations to calculate distance to.

    Returns:
        np.ndarray: Distances in shape (n-nodes, n-stations).
    """
    node_coords = get_node_coordinates(nodes, system="geographic")
    n_nodes = node_coords.shape[0]

    node_coords = np.repeat(node_coords, stations.n_stations, axis=0)
    sta_coords = np.vstack(n_nodes * [stations.get_coordinates(system="geographic")])

    return od.distance_accurate50m_numpy(
        node_coords[:, 0], node_coords[:, 1], sta_coords[:, 0], sta_coords[:, 1]
    ).reshape(-1, stations.n_stations)


class TravelTimeTree(BaseModel):
    earthmodel: EarthModel
    timing: Timing

    distance_bounds: tuple[float, float]
    source_depth_bounds: tuple[float, float]
    receiver_depth_bounds: tuple[float, float]
    time_tolerance: float
    spatial_tolerance: float

    created: datetime = Field(default_factory=datetime_now)

    _sptree: spit.SPTree | None = PrivateAttr(None)
    _file: Path | None = PrivateAttr(None)

    _cached_stations: Stations = PrivateAttr()
    _cached_station_indices: dict[str, int] = PrivateAttr({})
    _node_lut: dict[bytes, np.ndarray] = PrivateAttr(
        default_factory=lambda: LRU(LRU_CACHE_SIZE)
    )

    def calculate_tree(self) -> spit.SPTree:
        layered_model = self.earthmodel.layered_model

        def evaluate(args) -> float | None:
            receiver_depth, source_depth, distances = args
            rays = layered_model.arrivals(
                phases=self.timing.as_phase_defs(),
                distances=[distances * m2d],
                zstart=source_depth,
                zstop=receiver_depth,
            )
            times = np.fromiter((ray.t for ray in rays), float)
            return times.min() if times.size else None

        spatial_bounds = [
            self.receiver_depth_bounds,
            self.source_depth_bounds,
            self.distance_bounds,
        ]
        return spit.SPTree(
            f=evaluate,
            xbounds=np.array(spatial_bounds),
            ftol=np.array(self.time_tolerance),
            xtols=[
                self.spatial_tolerance,
                self.spatial_tolerance,
                self.spatial_tolerance,
            ],
        )

    def is_suited(
        self,
        timing: Timing,
        earthmodel: EarthModel,
        distance_bounds: tuple[float, float],
        source_depth_bounds: tuple[float, float],
        receiver_depth_bounds: tuple[float, float],
        time_tolerance: float,
        spatial_tolerance: float,
    ) -> bool:
        def check_bounds(self, requested: tuple[float, float]) -> bool:
            return self[0] <= requested[0] and self[1] >= requested[1]

        return (
            self.earthmodel.hash == earthmodel.hash
            and self.timing == timing
            and check_bounds(self.distance_bounds, distance_bounds)
            and check_bounds(self.source_depth_bounds, source_depth_bounds)
            and check_bounds(self.receiver_depth_bounds, receiver_depth_bounds)
            and self.time_tolerance <= time_tolerance
            and self.spatial_tolerance <= spatial_tolerance
        )

    @property
    def filename(self) -> Path:
        hash = sha1(self.earthmodel.hash.encode())
        hash.update(
            struct.pack(
                "dddddddd",
                *self.distance_bounds,
                *self.source_depth_bounds,
                *self.receiver_depth_bounds,
                self.time_tolerance,
                self.spatial_tolerance,
            )
        )
        return Path(f"{self.timing.id}-{hash.hexdigest()}.sptree")

    @classmethod
    def new(cls, **data) -> Self:
        """Create new SPTree and calculate traveltime table.

        Takes all input arguments as the __init__.
        """
        model = cls(**data)
        model._sptree = model.calculate_tree()
        return model

    def save(self, path: Path) -> Path:
        """Save the model and traveltimes to an .sptree archive.

        Args:
            path (Path): Folder or file to save tree into. If path is a folder a
                native name from the model's hash is used

        Returns:
            Path: Path to the saved archive.
        """
        file = path / self.filename if path.is_dir() else path
        logger.info("saving traveltimes to %s", file)

        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr(
                "model.json",
                self.model_dump_json(
                    indent=2,
                    exclude={"earthmodel": {"filename"}},
                    # include={"earthmodel": {"raw_file_data"}},
                ),
            )
            with NamedTemporaryFile() as tmpfile:
                self._get_sptree().dump(tmpfile.name)
                archive.write(tmpfile.name, "model.sptree")
        return file

    @classmethod
    def load(cls, file: Path) -> Self:
        """Load model from archive file.

        Args:
            file (Path): Path to archive file.

        Returns:
            Self: Loaded SPTreeModel
        """
        logger.debug("loading cached traveltimes from %s", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model_file = path / "model.json"
            model = cls.model_validate_json(model_file.read_text())
        model._file = file
        return model

    def _load_sptree(self) -> spit.SPTree:
        if not self._file or not self._file.exists():
            raise FileNotFoundError(f"file {self._file} not found")

        with (
            zipfile.ZipFile(self._file, "r") as archive,
            TemporaryDirectory() as temp_dir,
        ):
            archive.extract("model.sptree", path=temp_dir)
            return spit.SPTree(filename=str(Path(temp_dir) / "model.sptree"))

    def _get_sptree(self) -> spit.SPTree:
        if self._sptree is None:
            self._sptree = self._load_sptree()
        return self._sptree

    def _interpolate_traveltimes_sptree(
        self,
        coordinates: np.ndarray | list[float],
    ) -> np.ndarray:
        sptree = self._get_sptree()
        timing = self.timing.as_pyrocko_timing()

        coordinates = np.atleast_2d(np.ascontiguousarray(coordinates))
        return timing.evaluate(
            lambda phase: sptree.interpolate_many,
            coordinates,
        )

    async def init_lut(self, nodes: Sequence[Node], stations: Stations) -> None:
        logger.info(
            "warming up traveltime LUT %s for %d stations and %d nodes",
            self.timing.definition,
            stations.n_stations,
            len(nodes),
        )
        self._cached_stations = stations
        self._cached_station_indices = {
            sta.nsl.pretty: idx for idx, sta in enumerate(stations)
        }
        await self.fill_lut(nodes)

    def lut_fill_level(self) -> float:
        """Return the fill level of the LUT as a float between 0.0 and 1.0."""
        return len(self._node_lut) / self._node_lut.get_size()

    async def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling traveltimes LUT for %d nodes", len(nodes))
        stations = self._cached_stations

        traveltimes = await self._interpolate_travel_times(
            surface_distances(nodes, stations),
            np.array([sta.effective_depth for sta in stations]),
            np.array([node.as_location().effective_depth for node in nodes]),
        )

        node_lut = self._node_lut
        for node, times in zip(nodes, traveltimes, strict=True):
            times = times.astype(np.float32)
            times.setflags(write=False)
            node_lut[node.hash()] = times

    async def get_travel_times(
        self,
        nodes: Sequence[Node],
        stations: Stations,
    ) -> np.ndarray:
        try:
            station_indices = np.fromiter(
                (self._cached_station_indices[sta.nsl.pretty] for sta in stations),
                dtype=int,
            )
        except KeyError as exc:
            raise ValueError(
                "stations not found in cached stations, "
                "was the LUT initialized with `TravelTimeTree.init_lut`?"
            ) from exc

        stations_travel_times = []
        fill_nodes = []
        node_lut = self._node_lut
        for node in nodes:
            try:
                node_travel_times = node_lut[node.hash()][station_indices]
            except KeyError:
                fill_nodes.append(node)
                continue
            stations_travel_times.append(node_travel_times)

        if fill_nodes:
            await self.fill_lut(fill_nodes)

            cache_hits, cache_misses = node_lut.get_stats()
            total_hits = cache_hits + cache_misses
            cache_hit_rate = cache_hits / (total_hits or 1)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                self.lut_fill_level() * 100,
                cache_hit_rate * 100,
            )
            return await self.get_travel_times(nodes, stations)

        return np.asarray(stations_travel_times).astype(float, copy=False)

    async def interpolate_travel_times(
        self,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        receiver_distances = surface_distances(octree.nodes, stations)
        receiver_depths = np.array([sta.effective_depth for sta in stations])
        source_depths = np.array(
            [node.as_location().effective_depth for node in octree]
        )

        return await self._interpolate_travel_times(
            receiver_distances, receiver_depths, source_depths
        )

    async def _interpolate_travel_times(
        self,
        receiver_distances: np.ndarray,
        receiver_depths: np.ndarray,
        source_depths: np.ndarray,
    ) -> np.ndarray:
        coordinates = []
        for distances, source_depth in zip(
            receiver_distances,
            source_depths,
            strict=True,
        ):
            node_receivers_distances = (
                receiver_depths,
                np.full_like(distances, source_depth),
                distances,
            )
            coordinates.append(np.asarray(node_receivers_distances).T)

        n_nodes = len(coordinates)
        status = PROGRESS.add_task(
            f"interpolating {self.timing.definition} travel times "
            f"for {n_nodes} nodes",
            total=len(coordinates),
        )
        travel_times = []
        for coords in coordinates:
            travel_times.append(self._interpolate_traveltimes_sptree(coords))
            PROGRESS.update(status, advance=1)

        PROGRESS.remove_task(status)

        return np.asarray(travel_times).astype(float)

    def get_travel_time(self, source: Location, receiver: Location) -> float:
        coordinates = [
            receiver.effective_depth,
            source.effective_depth,
            receiver.surface_distance_to(source),
        ]
        try:
            travel_time = self._get_sptree().interpolate(coordinates) or np.nan
        except spit.OutOfBounds:
            travel_time = np.nan
        return float(travel_time)


class CakeTracer(RayTracer):
    """Travel time ray tracer for 1D layered earth models."""

    tracer: Literal["CakeTracer"] = "CakeTracer"
    phases: dict[PhaseDescription, Timing] = Field(
        default={
            "cake:P": Timing(definition="P,p"),
            "cake:S": Timing(definition="S,s"),
        },
        description="Dictionary of phases and timings to calculate.",
    )
    earthmodel: EarthModel = Field(
        default_factory=EarthModel,
        description="Earth model to calculate travel times for.",
    )
    trim_earth_model_depth: bool = Field(
        default=True,
        description="Trim earth model to max depth of the octree.",
    )
    lut_cache_size: ByteSize = Field(
        default=2 * GiB,
        description="Size of the LUT cache. Default is `2G`.",
    )

    _travel_time_trees: dict[PhaseDescription, TravelTimeTree] = PrivateAttr({})

    @property
    def cache_dir(self) -> Path:
        path = CACHE_DIR / "cake"
        path.mkdir(exist_ok=True)
        return path

    def clear_cache(self) -> None:
        """Clear cached SPTreeModels from user's cache."""
        logging.info("clearing cached travel time trees in %s", self.cache_dir)
        for file in self.cache_dir.glob("*.sptree"):
            file.unlink()

    def get_available_phases(self) -> tuple[str, ...]:
        return tuple(self.phases.keys())

    def get_vmin(self) -> float:
        earthmodel = self.earthmodel
        vel = np.concatenate((earthmodel.get_profile_vp(), earthmodel.get_profile_vs()))
        return float((vel[vel != 0.0]).min())

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
        rundir: Path | None = None,
    ) -> None:
        global LRU_CACHE_SIZE

        bytes_per_node = stations.n_stations * np.float32().itemsize
        n_trees = len(self.phases)
        LRU_CACHE_SIZE = int(self.lut_cache_size / bytes_per_node / n_trees)

        # TODO: This should be total number nodes. Not only leaf nodes.
        node_cache_fraction = LRU_CACHE_SIZE / octree.total_number_nodes()
        logging.info(
            "limiting traveltime LUT size to %d nodes (%s),"
            " caching %.1f%% of possible octree nodes",
            LRU_CACHE_SIZE,
            human_readable_bytes(self.lut_cache_size),
            node_cache_fraction * 100,
        )

        cached_trees = self._load_cached_trees()
        octree.reset()

        distances = surface_distances(octree.leaf_nodes, stations)
        source_depths = np.asarray(octree.depth_bounds) - octree.location.elevation
        receiver_depths = np.fromiter((sta.effective_depth for sta in stations), float)

        distance_bounds = (distances.min(), distances.max())
        source_depth_bounds = (source_depths.min(), source_depths.max())
        receiver_depths_bounds = (receiver_depths.min(), receiver_depths.max())
        # FIXME: Time tolerance is too hardcoded. Is 5x a good value?
        time_tolerance = octree.smallest_node_size() / (self.get_vmin() * 5.0)
        # if self.trim_earth_model_depth:
        #     self.earthmodel.trim(-source_depth_bounds[1])

        traveltime_tree_args = {
            "earthmodel": self.earthmodel,
            "distance_bounds": distance_bounds,
            "source_depth_bounds": source_depth_bounds,
            "receiver_depth_bounds": receiver_depths_bounds,
            "spatial_tolerance": octree.smallest_node_size() / 2,
            "time_tolerance": time_tolerance,
        }

        for phase_descr, timing in self.phases.items():
            for tree in cached_trees:
                if tree.is_suited(timing=timing, **traveltime_tree_args):
                    logger.info("using cached travel time tree for %s", phase_descr)
                    break
            else:
                logger.info("pre-calculating travel time tree for %s", phase_descr)
                tree = TravelTimeTree.new(timing=timing, **traveltime_tree_args)
                tree.save(self.cache_dir)

            await tree.init_lut(octree.leaf_nodes, stations)
            self._travel_time_trees[phase_descr] = tree

        if rundir:
            cake_plots = rundir / "cake"
            cake_plots.mkdir(exist_ok=True)
            for phase, tree in self._travel_time_trees.items():
                tree.earthmodel.save_plot(
                    cake_plots / f"earthmodel_{phase.replace(':', '_')}.png",
                )

    def _get_sptree_model(self, phase: str) -> TravelTimeTree:
        return self._travel_time_trees[phase]

    def _load_cached_trees(self) -> list[TravelTimeTree]:
        trees = []
        for file in self.cache_dir.glob("*.sptree"):
            try:
                tree = TravelTimeTree.load(file)
            except ValidationError:
                logger.warning("deleting invalid cached travel time tree %s", file)
                file.unlink()
                continue
            trees.append(tree)
        logger.debug("loaded %d cached travel time trees", len(trees))
        return trees

    def get_travel_time_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        try:
            tree = self._get_sptree_model(phase)
        except KeyError as exc:
            raise ValueError(f"Phase {phase} is not defined.") from exc
        return tree.get_travel_time(source, receiver)

    @alog_call
    async def get_travel_times(
        self,
        phase: str,
        nodes: Sequence[Node],
        stations: Stations,
    ) -> np.ndarray:
        try:
            return await self._get_sptree_model(phase).get_travel_times(
                nodes,
                stations,
            )
        except KeyError as exc:
            raise ValueError(f"Phase {phase} is not defined.") from exc

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival | None]:
        traveltimes = self.get_travel_times_locations(
            phase,
            source=source,
            receivers=receivers,
        )
        arrivals = []
        for traveltime, _receiver in zip(traveltimes, receivers, strict=True):
            if np.isnan(traveltime):
                arrivals.append(None)
                continue

            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = ModelledArrival(
                time=arrivaltime,
                phase=phase,
            )
            arrivals.append(arrival)
        return arrivals
