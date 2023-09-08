from __future__ import annotations

import asyncio
import functools
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, Sequence

import numpy as np
from pydantic import BaseModel, ByteSize, Field, PrivateAttr
from pyrocko.modelling import eikonal
from rich.progress import Progress
from scipy.interpolate import RegularGridInterpolator

from lassie.models.location import Location
from lassie.models.station import Station, Stations
from lassie.tracers.base import ModelledArrival, RayTracer
from lassie.tracers.fast_marching.velocity_models import (
    Constant3DVelocityModel,
    VelocityModel3D,
    VelocityModels,
)
from lassie.utils import CACHE_DIR, PhaseDescription, datetime_now, log_call

if TYPE_CHECKING:
    from lassie.octree import Octree


FMM_CACHE_DIR = CACHE_DIR / "fast-marching-cache"

KM = 1e3
GiB = int(1024**3)

logger = logging.getLogger(__name__)


class FastMarchingArrival(ModelledArrival):
    tracer: Literal["FastMarchingArrival"] = "FastMarchingArrival"
    phase: PhaseDescription


class StationTravelTimeVolume(BaseModel):
    center: Location
    station: Station

    velocity_model_hash: str

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]
    grid_spacing: float

    created: datetime = Field(default_factory=datetime_now)

    _travel_times: np.ndarray | None = PrivateAttr(None)

    _north_coords: np.ndarray = PrivateAttr(None)
    _east_coords: np.ndarray = PrivateAttr(None)
    _depth_coords: np.ndarray = PrivateAttr(None)

    # Cached values
    _file: Path | None = PrivateAttr(None)
    _interpolator: RegularGridInterpolator | None = PrivateAttr(None)

    @property
    def travel_times(self) -> np.ndarray:
        if self._travel_times is None:
            self._travel_times = self._load_travel_times()
        return self._travel_times

    def has_travel_times(self) -> bool:
        return self._travel_times is not None or self._file is not None

    def free_cache(self):
        self._interpolator = None
        if self._file is not None:
            logger.warning("cannot free travel time cache, file is not saved")
            self._travel_times = None

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing

        self._east_coords = np.arange(
            self.east_bounds[0],
            self.east_bounds[1],
            grid_spacing,
        )
        self._north_coords = np.arange(
            self.north_bounds[0],
            self.north_bounds[1],
            grid_spacing,
        )
        self._depth_coords = np.arange(
            self.depth_bounds[0],
            self.depth_bounds[1],
            grid_spacing,
        )

    @classmethod
    async def calculate_from_eikonal(
        cls,
        model: VelocityModel3D,
        station: Station,
        save: Path | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> Self:
        arrival_times = model.get_source_arrival_grid(station)

        def eikonal_wrapper(
            velocity_model: VelocityModel3D,
            arrival_times: np.ndarray,
            delta: float,
        ) -> StationTravelTimeVolume:
            logger.debug(
                "calculating travel time volume for %s, grid size %s, spacing %s m...",
                station.pretty_nsl,
                arrival_times.shape,
                velocity_model.grid_spacing,
            )
            eikonal.eikonal_solver_fmm_cartesian(
                velocity_model._velocity_model,
                arrival_times,
                delta=delta,
            )
            station_travel_times = cls(
                center=model.center,
                velocity_model_hash=model.hash(),
                station=station,
                east_bounds=model.east_bounds,
                north_bounds=model.north_bounds,
                depth_bounds=model.depth_bounds,
                grid_spacing=model.grid_spacing,
            )
            station_travel_times._travel_times = arrival_times.astype(np.float32)
            if save:
                station_travel_times.save(save)

            return station_travel_times

        loop = asyncio.get_running_loop()

        work = functools.partial(
            eikonal_wrapper,
            model,
            arrival_times,
            delta=model.grid_spacing,
        )

        return await loop.run_in_executor(executor, work)

    @property
    def filename(self) -> str:
        # TODO: Add origin to hash to avoid collisions
        return f"{self.station.pretty_nsl}-{self.velocity_model_hash}.3dtt"

    def get_traveltime_interpolator(self) -> RegularGridInterpolator:
        if self._interpolator is None:
            self._interpolator = RegularGridInterpolator(
                (self._east_coords, self._north_coords, self._depth_coords),
                self.travel_times,
                bounds_error=False,
                fill_value=np.nan,
            )
        return self._interpolator

    def interpolate_travel_time(
        self,
        location: Location,
        method: Literal["nearest", "linear", "cubic"] = "linear",
    ) -> float:
        interpolator = self.get_traveltime_interpolator()
        offset = location.offset_to(self.center)
        return interpolator([offset], method=method)[0]

    def interpolate_travel_times(
        self,
        octree: Octree,
        method: Literal["nearest", "linear", "cubic"] = "linear",
    ) -> np.ndarray:
        interpolator = self.get_traveltime_interpolator()

        coordinates = []
        for node in octree:
            location = node.as_location()
            coordinates.append(location.offset_to(self.center))

        return interpolator(coordinates, method=method)

    def save(self, path: Path) -> Path:
        """Save travel times to a zip file.

        The zip file contains a model.json file with the model metadata and a
        numpy file with the travel times.

        Args:
            path (Path): path to save the travel times to

        Returns:
            Path: path to the saved travel times
        """
        if not self.has_travel_times():
            raise AttributeError("travel times have not been calculated yet")

        file = path / self.filename if path.is_dir() else path
        logger.debug("saving travel times to %s...", file)

        with zipfile.ZipFile(str(file), "w") as archive:
            archive.writestr("model.json", self.model_dump_json(indent=2))
            travel_times = archive.open("travel_times.npy", "w")
            np.save(travel_times, self.travel_times)
            travel_times.close()

        self._file = file
        return file

    @classmethod
    def load(cls, file: Path) -> Self:
        """Load 3D travel times from a zip file.

        Args:
            file (Path): path to the zip file containing the travel times

        Returns:
            Self: 3D travel times
        """
        logger.debug("loading travel times from %s...", file)
        with zipfile.ZipFile(file, "r") as archive:
            path = zipfile.Path(archive)
            model_file = path / "model.json"
            model = cls.model_validate_json(model_file.read_text())
        model._file = file
        return model

    def _load_travel_times(self) -> np.ndarray:
        if not self._file or not self._file.exists():
            raise FileNotFoundError(f"file {self._file} not found")

        with zipfile.ZipFile(self._file, "r") as archive:
            return np.load(archive.open("travel_times.npy", "r"))


class FastMarchingPhaseTracer(BaseModel):
    velocity_model: VelocityModels = Constant3DVelocityModel()
    interpolation_method: Literal["nearest", "linear", "cubic"] = "nearest"

    _traveltime_models: dict[str, StationTravelTimeVolume] = PrivateAttr({})

    lut_cache_size: ByteSize = Field(
        4 * GiB,
        description="Size of the LUT cache in MB.",
    )

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
        nthreads: int = 0,
    ) -> None:
        velocity_model = self.velocity_model.get_model(octree, stations)
        for station in stations:
            if not velocity_model.is_inside(station):
                stations.blacklist_station(station, reason="outside the velocity model")

        executor = ThreadPoolExecutor(
            max_workers=nthreads or os.cpu_count(), thread_name_prefix="lassie-fmm"
        )

        cache_dir = FMM_CACHE_DIR / f"{velocity_model.hash()}"
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        for file in cache_dir.glob("*.3dtt"):
            try:
                travel_times = StationTravelTimeVolume.load(file)
            except zipfile.BadZipFile:
                logger.warning("removing bad travel time file %s", file)
                file.unlink()
                continue
            self._traveltime_models[travel_times.station.pretty_nsl] = travel_times

        logger.info(
            "loaded %d travel times volumes from cache", len(self._traveltime_models)
        )

        pre_calculate_work = []
        for station in stations:
            if station.pretty_nsl in self._traveltime_models:
                continue

            async def worker_station_travel_time(
                velocity_model: VelocityModel3D,
                station: Station = station,
            ) -> None:
                model = await StationTravelTimeVolume.calculate_from_eikonal(
                    velocity_model,
                    station,
                    save=cache_dir,
                    executor=executor,
                )
                self._traveltime_models[station.pretty_nsl] = model

            pre_calculate_work.append(worker_station_travel_time(velocity_model))

        if not pre_calculate_work:
            return

        logger.info(
            "pre-calculating travel time volumes for %d stations...",
            len(pre_calculate_work),
        )
        start = datetime_now()
        tasks = [asyncio.create_task(work) for work in pre_calculate_work]
        with Progress() as progress:
            status = progress.add_task(
                f"calculating travel time volumes for {len(tasks)} station",
                total=len(tasks),
            )
            for _task in asyncio.as_completed(tasks):
                await _task
                progress.advance(status)

        logger.info("pre-calculated travel time volumes in %s", datetime_now() - start)

    def get_travel_time(self, source: Location, receiver: Location) -> float:
        station_travel_times = self._traveltime_models[hash(receiver)]
        return station_travel_times.interpolate_travel_time(
            source, method=self.interpolation_method
        )

    def get_travel_times(self, octree: Octree, stations: Stations) -> np.ndarray:
        result = []
        for station in stations:
            station_travel_times = self._traveltime_models[station.pretty_nsl]
            result.append(
                station_travel_times.interpolate_travel_times(
                    octree, method=self.interpolation_method
                )
            )
        return np.array(result).T


class FastMarchingRayTracer(RayTracer):
    tracer: Literal["FastMarchingRayTracer"] = "FastMarchingRayTracer"

    tracers: dict[PhaseDescription, FastMarchingPhaseTracer] = {
        "fm:P": FastMarchingPhaseTracer(),
        "fm:S": FastMarchingPhaseTracer(),
    }
    nthreads: int = Field(default_factory=os.cpu_count)

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        for tracer in self.tracers.values():
            await tracer.prepare(octree, stations, nthreads=self.nthreads)

    def get_available_phases(self) -> tuple[str, ...]:
        return tuple(self.tracers.keys())

    def _get_tracer(self, phase: str) -> FastMarchingPhaseTracer:
        return self.tracers[phase]

    def get_travel_time_location(
        self, phase: str, source: Location, receiver: Location
    ) -> float:
        if phase not in self.tracers:
            raise ValueError(f"Phase {phase} is not defined.")
        return self._get_tracer(phase).get_travel_time(source, receiver)

    @log_call
    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        return self._get_tracer(phase).get_travel_times(octree, stations)

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival | None]:
        traveltimes = []
        for receiver in receivers:
            traveltimes.append(self.get_travel_time_location(phase, source, receiver))

        arrivals = []
        for traveltime, _receiver in zip(traveltimes, receivers, strict=True):
            if np.isnan(traveltime):
                arrivals.append(None)
                continue

            arrivaltime = event_time + timedelta(seconds=traveltime)
            arrival = FastMarchingArrival(time=arrivaltime, phase=phase)
            arrivals.append(arrival)
        return arrivals
