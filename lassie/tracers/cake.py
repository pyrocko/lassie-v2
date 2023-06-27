from __future__ import annotations

import logging
import re
import time
import zipfile
from datetime import datetime, timedelta
from functools import cached_property
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from lru import LRU
from pydantic import BaseModel, Field, PositiveFloat, PrivateAttr, constr
from pyrocko import spit
from pyrocko.cake import LayeredModel, PhaseDef, m2d, read_nd_model_str
from pyrocko.gf import meta
from rich.progress import Progress

from lassie.tracers.base import ModelledArrival, RayTracer
from lassie.utils import (
    CACHE_DIR,
    PhaseDescription,
    datetime_now,
    human_readable_bytes,
    log_call,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from lassie.models.location import Location
    from lassie.models.station import Stations
    from lassie.octree import Octree

logger = logging.getLogger(__name__)

KM = 1e3
MAX_DBS = 16

LRU_CACHE = 2000


class CakeArrival(ModelledArrival):
    tracer: Literal["CakeArrival"] = "CakeArrival"
    phase: str


class EarthModel(BaseModel):
    __root__: list[tuple[float, PositiveFloat, PositiveFloat, PositiveFloat]] = [
        (0.00, 5.50, 3.59, 2.7),
        (1.00, 5.50, 3.59, 2.7),
        (1.00, 6.00, 3.92, 2.7),
        (4.00, 6.00, 3.92, 2.7),
        (4.00, 6.20, 4.05, 2.7),
        (8.00, 6.20, 4.05, 2.7),
        (8.00, 6.30, 4.12, 2.7),
        (13.00, 6.30, 4.12, 2.7),
        (13.00, 6.40, 4.18, 2.7),
        (17.00, 6.40, 4.18, 2.7),
        (17.00, 6.50, 4.25, 2.7),
        (22.00, 6.50, 4.25, 2.7),
        (22.00, 6.60, 4.31, 2.7),
        (26.00, 6.60, 4.31, 2.7),
        (26.00, 6.80, 4.44, 2.7),
        (30.00, 6.80, 4.44, 2.7),
        (30.00, 8.10, 5.29, 2.7),
        (45.00, 8.10, 5.29, 2.7),
    ]

    class Config:
        keep_untouched = (cached_property,)

    def _as_array(self) -> np.ndarray:
        return np.asarray(self.__root__)

    def get_profile_vp(self) -> np.ndarray:
        # TODO: reduce to relevant layers
        return self._as_array()[:, 1] * KM

    def get_profile_vs(self) -> np.ndarray:
        return self._as_array()[:, 2] * KM

    def as_layered_model(self) -> LayeredModel:
        line_tpl = "{} {} {} {}"
        earthmodel = "\n".join(line_tpl.format(*layer) for layer in self.__root__)
        return LayeredModel.from_scanlines(read_nd_model_str(earthmodel))

    @cached_property
    def hash(self) -> str:
        layered_model = self.as_layered_model()
        model_serialised = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            layered_model.profile(param).dump(model_serialised)
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


class TraveltimeTree(BaseModel):
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
    _cache: dict[bytes, np.ndarray] = PrivateAttr(
        default_factory=lambda: LRU(LRU_CACHE)
    )

    def calculate_tree(self) -> spit.SPTree:
        layered_model = self.earthmodel.as_layered_model()

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
        def check_bounds(self, requested) -> bool:
            return self[0] <= requested[0] and self[1] >= requested[1]

        return (
            self.earthmodel == earthmodel
            and self.timing == timing
            and check_bounds(self.distance_bounds, distance_bounds)
            and check_bounds(self.source_depth_bounds, source_depth_bounds)
            and check_bounds(self.receiver_depth_bounds, receiver_depth_bounds)
            and self.time_tolerance <= time_tolerance
            and self.spatial_tolerance <= spatial_tolerance
        )

    @property
    def filename(self) -> Path:
        return Path(f"{self.timing.id}-{self.earthmodel.hash}.sptree")

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
            folder (Path): Folder or file to save tree into. If path is a folder a
                native name from the model's hash is used

        Returns:
            Path: Path to the saved archive.
        """
        file = path / self.filename if path.is_dir() else path
        logger.info("saving traveltimes to %s", file)

        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr("model.json", self.json(indent=2))
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
        logger.debug("loading traveltimes from %s", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model = cls.parse_raw((path / "model.json").read_bytes())
        model._file = file
        return model

    def _load_sptree(self) -> spit.SPTree:
        if not self._file or not self._file.exists():
            raise FileNotFoundError(f"file {self._file} not found")

        with zipfile.ZipFile(
            self._file, "r"
        ) as archive, TemporaryDirectory() as temp_dir:
            archive.extract("model.sptree", path=temp_dir)
            return spit.SPTree(filename=str(Path(temp_dir) / "model.sptree"))

    def _get_sptree(self) -> spit.SPTree:
        if self._sptree is None:
            self._sptree = self._load_sptree()
        return self._sptree

    def _get_traveltime_sptree(
        self,
        coordinates: np.ndarray | list[float],
    ) -> np.ndarray:
        sptree = self._get_sptree()
        timing = self.timing.as_pyrocko_timing()

        coordinates = np.atleast_2d(np.ascontiguousarray(coordinates))
        coord_hash = sha1(coordinates.data).digest()
        if coord_hash not in self._cache:
            traveltimes: np.ndarray = timing.evaluate(
                lambda phase: sptree.interpolate_many,
                coordinates,
            )
            self._cache[coord_hash] = traveltimes.astype(np.float32)
        return self._cache[coord_hash].astype(float)

    def get_traveltime(self, source: Location, receiver: Location) -> float:
        coordinates = [
            receiver.effective_depth,
            source.effective_depth,
            receiver.distance_to(source),
        ]
        traveltime = self._get_traveltime_sptree(coordinates)
        return float(traveltime)

    def get_cache_bytes(self) -> int:
        return sum(traveltimes.nbytes for traveltimes in self._cache.values())

    def get_traveltimes(self, octree: Octree, stations: Stations) -> np.ndarray:
        logger.debug("calculating traveltimes for %d stations", stations.n_stations)
        receiver_depths = np.fromiter((sta.effective_depth for sta in stations), float)
        source_depths = np.fromiter((node.depth for node in octree), float)
        receiver_distances = octree.distances_stations(stations)

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

        with Progress() as progress:
            status = progress.add_task(
                f"interpolating {stations.n_stations*octree.n_nodes} traveltimes",
                total=len(coordinates),
                visible=False,
            )
            start = time.time()

            traveltimes = []
            for coords in coordinates:
                traveltimes.append(self._get_traveltime_sptree(coords))
                progress.update(status, advance=1)
                if time.time() - start > 1.0:
                    progress.update(status, visible=True)

        return np.array(traveltimes)


class CakeTracer(RayTracer):
    tracer: Literal["CakeTracer"] = "CakeTracer"
    timings: dict[PhaseDescription, Timing] = {
        "cake:P": Timing(definition="P,p"),
        "cake:S": Timing(definition="S,s"),
    }
    earthmodel: EarthModel = EarthModel()

    _traveltime_trees: dict[PhaseDescription, TraveltimeTree] = PrivateAttr({})

    @property
    def cache_dir(self) -> Path:
        path = CACHE_DIR / "cake"
        path.mkdir(exist_ok=True)
        return path

    def clear_cache(self) -> None:
        """Clear cached SPTreeModels from user's cache."""
        logging.info("clearing traveltime cache %s", self.cache_dir)
        for file in self.cache_dir.glob("*.sptree"):
            file.unlink()

    def get_available_phases(self) -> tuple[str]:
        return tuple(self.timings.keys())

    def get_vmin(self) -> float:
        earthmodel = self.earthmodel
        vel = np.concatenate((earthmodel.get_profile_vp(), earthmodel.get_profile_vs()))
        return float((vel[vel != 0.0]).min())

    def prepare(self, octree: Octree, stations: Stations) -> None:
        global LRU_CACHE

        LRU_CACHE = octree.n_nodes * 8
        logging.debug("setting LRU cache keys to %d", LRU_CACHE)

        cached_trees = [
            TraveltimeTree.load(file) for file in self.cache_dir.glob("*.sptree")
        ]

        distances = octree.distances_stations(stations)
        source_depths = np.asarray(octree.depth_bounds)
        receiver_depths = np.fromiter((sta.effective_depth for sta in stations), float)

        receiver_depths_bounds = (receiver_depths.min(), receiver_depths.max())
        source_depth_bounds = (source_depths.min(), source_depths.max())
        distance_bounds = (distances.min(), distances.max())
        # TODO: Time tolerance is too hardcoded
        time_tolerance = octree.size_limit / (self.get_vmin() * 3.0)

        traveltime_tree_args = {
            "earthmodel": self.earthmodel,
            "distance_bounds": distance_bounds,
            "source_depth_bounds": source_depth_bounds,
            "receiver_depth_bounds": receiver_depths_bounds,
            "spatial_tolerance": octree.size_limit / 2,
            "time_tolerance": time_tolerance,
        }

        for phase_descr, timing in self.timings.items():
            for tree in cached_trees:
                if tree.is_suited(timing=timing, **traveltime_tree_args):
                    logger.info("using cached traveltime tree for %s", phase_descr)
                    break
            else:
                logger.info("pre-calculating traveltime tree for %s", phase_descr)
                tree = TraveltimeTree.new(timing=timing, **traveltime_tree_args)
                tree.save(self.cache_dir)

            self._traveltime_trees[phase_descr] = tree

    def _get_sptree_model(self, phase: str) -> TraveltimeTree:
        return self._traveltime_trees[phase]

    def get_traveltime_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        if phase not in self.timings:
            raise ValueError(f"Timing {phase} is not defined.")
        tree = self._get_sptree_model(phase)
        return tree.get_traveltime(source, receiver)

    @log_call
    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        if phase not in self.timings:
            raise ValueError(f"Timing {phase} is not defined.")
        tree = self._get_sptree_model(phase)
        logger.debug(
            "%s cache size is %s", phase, human_readable_bytes(tree.get_cache_bytes())
        )
        return tree.get_traveltimes(octree, stations)

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[CakeArrival | None]:
        traveltimes = self.get_traveltimes_locations(
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
            arrival = CakeArrival(time=arrivaltime, phase=phase)
            arrivals.append(arrival)
        return arrivals
