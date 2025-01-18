from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Literal, Sequence

import numpy as np
import psutil
from pydantic import (
    AliasChoices,
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    PositiveFloat,
    PrivateAttr,
    computed_field,
    field_validator,
)
from scipy import stats

from qseek.corrections.corrections import StationCorrectionType
from qseek.distance_weights import DistanceWeights
from qseek.features import FeatureExtractorType
from qseek.images.images import ImageFunctions, WaveformImages
from qseek.magnitudes import EventMagnitudeCalculatorType
from qseek.models import Stations
from qseek.models.catalog import EventCatalog
from qseek.models.detection import EventDetection, PhaseDetection
from qseek.models.detection_uncertainty import DetectionUncertainty
from qseek.models.semblance import Semblance
from qseek.octree import Octree
from qseek.pre_processing.frequency_filters import Bandpass
from qseek.pre_processing.module import Downsample, PreProcessing
from qseek.signals import Signal
from qseek.stats import RuntimeStats, Stats
from qseek.tracers.tracers import RayTracer, RayTracers
from qseek.utils import (
    BackgroundTasks,
    PhaseDescription,
    alog_call,
    datetime_now,
    get_cpu_count,
    get_total_memory,
    human_readable_bytes,
    time_to_path,
)
from qseek.waveforms.base import WaveformBatch
from qseek.waveforms.providers import PyrockoSquirrel, WaveformProviderType

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from rich.table import Table
    from typing_extensions import Self

    from qseek.images.base import WaveformImage
    from qseek.octree import Node


logger = logging.getLogger(__name__)

SamplingRate = Literal[10, 20, 25, 50, 100, 200, 400]


class SearchStats(Stats):
    project_name: str = "qseek"
    batch_time: datetime = datetime.min
    batch_count: int = 0
    batch_count_total: int = 0
    processed_time: timedelta = timedelta(seconds=0.0)
    processed_bytes: int = 0
    processing_time: timedelta = timedelta(seconds=0.0)
    latest_processing_rate: float = 0.0
    latest_processing_speed: timedelta = timedelta(seconds=0.0)

    current_stations: int = 0
    total_stations: int = 0

    current_networks: int = 0
    total_networks: int = 0

    memory_total: ByteSize = Field(default_factory=lambda: ByteSize(get_total_memory()))

    _search_start: datetime = PrivateAttr(default_factory=datetime_now)
    _batch_processing_times: Deque[timedelta] = PrivateAttr(
        default_factory=lambda: deque(maxlen=25)
    )
    _position: int = PrivateAttr(0)
    _process: psutil.Process = PrivateAttr(default_factory=psutil.Process)

    @computed_field
    @property
    def time_remaining(self) -> timedelta:
        if not self.batch_count:
            return timedelta()

        remaining_batches = self.batch_count_total - self.batch_count
        if not remaining_batches:
            return timedelta()

        elapsed_time = datetime_now() - self._search_start
        return (elapsed_time / self.batch_count) * remaining_batches

    @computed_field
    @property
    def processing_rate(self) -> float:
        """Calculate the processing rate of the search.

        Returns:
            float: The processing rate in bytes per second.
        """
        if not self.processing_time:
            return 0.0
        return self.processed_bytes / self.processing_time.total_seconds()

    @computed_field
    @property
    def processing_speed(self) -> timedelta:
        if not self.processing_time:
            return timedelta(seconds=0.0)
        return self.processed_time / self.processing_time.total_seconds()

    @computed_field
    @property
    def processed_percent(self) -> float:
        """Calculate the percentage of processed batches.

        Returns:
            float: The percentage of processed batches.
        """
        if not self.batch_count_total:
            return 0.0
        return self.batch_count / self.batch_count_total * 100.0

    @computed_field
    @property
    def memory_used(self) -> int:
        return self._process.memory_info().rss

    @computed_field
    @property
    def cpu_percent(self) -> float:
        return self._process.cpu_percent(interval=None)

    def reset_start_time(self) -> None:
        self._search_start = datetime_now()

    def add_processed_batch(
        self,
        batch: WaveformBatch,
        duration: timedelta,
        show_log: bool = False,
    ) -> None:
        self.batch_count = batch.i_batch + 1
        self.batch_count_total = batch.n_batches
        self.batch_time = batch.end_time
        self.processed_bytes += batch.cumulative_bytes
        self.processed_time += batch.duration
        self.processing_time += duration
        self.latest_processing_rate = batch.cumulative_bytes / duration.total_seconds()
        self.latest_processing_speed = batch.duration / duration.total_seconds()
        self.current_stations = batch.n_stations
        self.current_networks = batch.n_networks

        self._batch_processing_times.append(duration)
        if show_log:
            self.log()

    def log(self) -> None:
        log_str = (
            f"{self.batch_count+1}/{self.batch_count_total or '?'} {self.batch_time}"
        )
        logger.info(
            "%s%% processed - batch %s in %s",
            f"{self.processed_percent:.1f}" if self.processed_percent else "??",
            log_str,
            self._batch_processing_times[-1],
        )
        logger.info(
            "processing rate %s/s", human_readable_bytes(self.latest_processing_rate)
        )

    def _populate_table(self, table: Table) -> None:
        def tts(duration: timedelta) -> str:
            return str(duration).split(".")[0]

        table.add_row(
            "Project",
            f"[bold]{self.project_name}[/bold]",
        )
        table.add_row(
            "Progress ",
            f"[bold]{self.processed_percent:.1f}%[/bold]"
            f" ([bold]{self.batch_count}[/bold]/{self.batch_count_total or '?'},"
            f' {self.batch_time.strftime("%Y-%m-%d %H:%M:%S")})',
        )
        table.add_row(
            "Stations",
            f"{self.current_stations}/{self.total_stations}"
            f" ({self.current_networks}/{self.total_networks} networks)",
        )
        table.add_row(
            "Processing rate",
            f"{human_readable_bytes(self.processing_rate)}/s"
            f" ({tts(self.processing_speed)} tr/s)",
        )
        table.add_row(
            "Resources",
            f"CPU {self.cpu_percent:.1f}%, "
            f"RAM {human_readable_bytes(self.memory_used, decimal=True)}"
            f"/{self.memory_total.human_readable(decimal=True)}",
        )
        table.add_row(
            "Remaining Time",
            f"{tts(self.time_remaining)}, "
            f"finish at {datetime.now() + self.time_remaining:%c}",  # noqa: DTZ005
        )


class SearchProgress(BaseModel):
    time_progress: datetime | None = None


class Search(BaseModel):
    project_dir: Path = Path(".")
    stations: Stations = Field(
        default=Stations(),
        description="Station inventory from StationXML or Pyrocko Station YAML.",
    )
    data_provider: WaveformProviderType = Field(
        default_factory=PyrockoSquirrel.model_construct,
        description="Data provider for waveform data.",
    )
    pre_processing: PreProcessing = Field(
        default=PreProcessing(root=[Downsample(), Bandpass()]),
        description="Pre-processing steps for waveform data.",
    )

    octree: Octree = Field(
        default=Octree(),
        description="Octree volume for the search.",
    )

    image_functions: ImageFunctions = Field(
        default=ImageFunctions(),
        description="Image functions for waveform processing and "
        "phase on-set detection.",
    )
    ray_tracers: RayTracers = Field(
        default=RayTracers(root=[tracer() for tracer in RayTracer.get_subclasses()]),
        description="List of ray tracers for travel time calculation.",
    )
    distance_weights: DistanceWeights | None = Field(
        default=DistanceWeights(),
        validation_alias=AliasChoices("spatial_weights", "distance_weights"),
        description="Spatial weights for distance weighting.",
    )
    station_corrections: StationCorrectionType | None = Field(
        default=None,
        description="Apply station corrections extracted from a previous run.",
    )
    magnitudes: list[EventMagnitudeCalculatorType] = Field(
        default=[],
        description="Magnitude calculators to use.",
    )
    features: list[FeatureExtractorType] = Field(
        default=[],
        description="Event features to extract.",
    )

    semblance_sampling_rate: SamplingRate = Field(
        default=100,
        description="Sampling rate for the semblance image function. "
        "Choose from `10, 20, 25, 50, 100, 200 or 400` Hz.",
    )
    detection_threshold: Literal["MAD"] | PositiveFloat = Field(
        default="MAD",
        description="Detection threshold for semblance.",
    )
    pick_confidence_threshold: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Confidence threshold for picking.",
    )
    min_stations: int = Field(
        default=3,
        ge=0,
        description="Minimum number of stations required for"
        " detection and localization.",
    )
    absorbing_boundary: Literal[False, "with_surface", "without_surface"] = Field(
        default=False,
        description="Ignore events that are inside the first root node layer of"
        " the octree. If `with_surface`, all events inside the boundaries of the volume"
        " are absorbed. If `without_surface`, events at the surface are not absorbed.",
    )
    absorbing_boundary_width: float | Literal["root_node_size"] = Field(
        default="root_node_size",
        description="Width of the absorbing boundary around the octree volume. "
        "If 'octree' the width is set to the root node size of the octree.",
    )
    node_peak_interpolation: bool = Field(
        default=True,
        description="Interpolate intranode locations for detected events using radial"
        " basis functions. If `False`, the node center location is used for "
        "the event hypocentre.",
    )
    detection_blinding: timedelta = Field(
        default=timedelta(seconds=1.0),
        description="Blinding time in seconds before and after the detection peak. "
        "This is used to avoid detecting the same event multiple times. "
        "Default is 2 seconds.",
    )

    power_mean: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Power mean exponent for stacking and combining the image"
        " functions for stacking. A value of 1.0 is the arithmetic mean, 2.0 is the"
        " quadratic mean. A higher value will result in sharper detections"
        " and low values smooth the stacking function.",
    )

    window_length: timedelta = Field(
        default=timedelta(minutes=5),
        description="Window length for processing. Smaller window size will be less RAM"
        " consuming. Default is 5 minutes.",
    )

    n_threads_parstack: int | Literal["auto"] = Field(
        default="auto",
        description="Number of threads for stacking and migration. "
        "`'auto'` will use the maximum number of cores and leaves resources "
        "for I/O and other work. `0` uses all available cores.",
    )
    n_threads_argmax: int | Literal["auto"] = Field(
        default="auto",
        description="Number of threads for argmax. `'auto'` will use the "
        "maximum number of cores and leaves resources for I/O and other work. "
        "`0` uses all available cores.",
    )

    save_images: bool = Field(
        default=False,
        description="Save annotation images to disk for debugging and analysis.",
    )
    created: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _progress: SearchProgress = PrivateAttr(SearchProgress())

    _shift_range: timedelta = PrivateAttr(timedelta(seconds=0.0))
    _window_padding: timedelta = PrivateAttr(timedelta(seconds=0.0))
    _distance_range: tuple[float, float] = PrivateAttr((0.0, 0.0))
    _travel_time_ranges: dict[PhaseDescription, tuple[timedelta, timedelta]] = (
        PrivateAttr({})
    )
    _last_detection_export: int = 0

    _catalog: EventCatalog = PrivateAttr()
    _config_stem: str = PrivateAttr("")
    _rundir: Path = PrivateAttr()

    _compute_semaphore: asyncio.Semaphore = PrivateAttr(
        asyncio.Semaphore(max(1, get_cpu_count() - 4))
    )

    # Signals
    _new_detection: Signal[EventDetection] = PrivateAttr(Signal())

    model_config = ConfigDict(extra="forbid")

    @field_validator("n_threads_parstack", "n_threads_argmax")
    @classmethod
    def check_threads_compute(cls, n_threads: int | Literal["auto"]) -> int:
        # We limit the number of threads to the number of available cores and leave some
        # cores for other processes.
        cpu_count = get_cpu_count()
        max_compute_threads = max(cpu_count - 4, 1)
        if n_threads == "auto":
            n_threads = max_compute_threads
        if n_threads < 0:
            raise ValueError("Number of threads must be greater than 0")
        if n_threads > cpu_count:
            logger.warning(
                "number of compute threads exceeds available cores, reducing to %d",
                max_compute_threads,
            )
            n_threads = cpu_count
        return n_threads

    def init_rundir(self, force: bool = False) -> None:
        rundir = (
            self.project_dir / self._config_stem or f"run-{time_to_path(self.created)}"
        )
        self._rundir = rundir

        if rundir.exists() and not force:
            raise FileExistsError(f"Rundir {rundir} already exists")

        if rundir.exists() and force:
            create_time = time_to_path(
                datetime.fromtimestamp(rundir.stat().st_ctime)  # noqa
            ).split(".")[0]
            rundir_backup = rundir.with_name(f"{rundir.name}.bak-{create_time}")
            rundir.rename(rundir_backup)
            logger.info("created backup of existing rundir at %s", rundir_backup)

        if not rundir.exists():
            rundir.mkdir()

        self._init_logging()

        logger.info("created new rundir %s", rundir)
        self._catalog = EventCatalog(rundir=rundir)

    @property
    def catalog(self) -> EventCatalog:
        return self._catalog

    def _init_logging(self) -> None:
        file_logger = logging.FileHandler(self._rundir / "qseek.log")
        logging.root.addHandler(file_logger)

    def create_folders(self, path: Path | None = None) -> None:
        rundir = path or self._rundir
        csv_dir = rundir / "csv"
        csv_dir.mkdir(exist_ok=True)

    def write_config(self, path: Path | None = None) -> None:
        rundir = path or self._rundir
        path = rundir / "search.json"

        logger.debug("writing search config to %s", path)
        path.write_text(self.model_dump_json(indent=2, exclude_unset=True))

        logger.debug("writing stations")
        self.stations.export_pyrocko_stations(rundir / "pyrocko_stations.yaml")

        csv_dir = rundir / "csv"
        csv_dir.mkdir(exist_ok=True)
        self.stations.export_csv(csv_dir / "stations.csv")

    def set_progress(self, time: datetime) -> None:
        self._progress.time_progress = time
        progress_file = self._rundir / "progress.json"
        progress_file.write_text(self._progress.model_dump_json())

    async def init_boundaries(self) -> None:
        """Initialise search."""
        # Grid/receiver distances
        distances = self.octree.distances_stations(self.stations)
        self._distance_range = (distances.min(), distances.max())

        # Timing ranges
        for phase, tracer in self.ray_tracers.iter_phase_tracer(
            phases=self.image_functions.get_phases()
        ):
            traveltimes = await tracer.get_travel_times(
                phase,
                self.octree,
                self.stations,
            )
            self._travel_time_ranges[phase] = (
                timedelta(seconds=max(0, np.nanmin(traveltimes))),
                timedelta(seconds=np.nanmax(traveltimes)),
            )
            logger.info(
                "time shift ranges: %s / %s - %s",
                phase,
                *self._travel_time_ranges[phase],
            )

        # TODO: minimum shift is calculated on the coarse octree grid, which is
        # not necessarily the same as the fine grid used for semblance calculation
        shift_min = min(chain.from_iterable(self._travel_time_ranges.values()))
        shift_max = max(chain.from_iterable(self._travel_time_ranges.values()))
        self._shift_range = shift_max - shift_min

        self._window_padding = (
            self._shift_range
            + self.detection_blinding
            + self.image_functions.get_blinding(self.semblance_sampling_rate)
        )
        if self.window_length < 2 * self._window_padding + self._shift_range:
            raise ValueError(
                f"window length {self.window_length} is too short for the "
                f"theoretical travel time range {self._shift_range} and "
                f"cummulative window padding of {self._window_padding}."
                " Increase the window_length time to at least "
                f"{self._shift_range +2*self._window_padding }"
            )

        logger.info("using trace window padding: %s", self._window_padding)
        logger.info("time shift range %s", self._shift_range)
        logger.info(
            "source-station distance range: %.1f - %.1f m",
            *self._distance_range,
        )

    async def prepare(self) -> None:
        """Prepares the search by initializing necessary components and data.

        This method prepares the search by performing the following steps:
        1. Prepares the data provider with the given stations.
        2. Prepares the ray tracers with the octree, stations, phases, and rundir.
        3. Prepares each magnitude with the octree and stations.
        4. Prepares the station corrections with the stations, octree, and phases.
        5. Initializes the boundaries.

        Note: This method is asynchronous.

        Returns:
            None
        """
        logger.info("preparing search components")
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: logger.error(context)
        )
        self.stations.prepare(self.octree)
        self.data_provider.prepare(self.stations)
        await self.pre_processing.prepare()
        await self.ray_tracers.prepare(
            self.octree,
            self.stations,
            phases=self.image_functions.get_phases(),
            rundir=self._rundir,
        )

        if self.distance_weights:
            self.distance_weights.prepare(self.stations, self.octree)

        if self.station_corrections:
            await self.station_corrections.prepare(
                self.stations,
                self.octree,
                self.image_functions.get_phases(),
                self._rundir,
            )
        for magnitude in self.magnitudes:
            await magnitude.prepare(self.octree, self.stations)
        await self.init_boundaries()

        self._catalog.prepare()

    async def start(self, force_rundir: bool = False) -> None:
        if not self.has_rundir():
            self.init_rundir(force=force_rundir)

        self.create_folders()
        await self.prepare()
        self.write_config()

        if self._progress.time_progress:
            logger.info("continuing search from %s", self._progress.time_progress)
            await self._catalog.check(repair=True)
            await self._catalog.filter_events_by_time(
                start_time=None,
                end_time=self._progress.time_progress,
            )
        else:
            logger.info("starting search")

        batches = self.data_provider.iter_batches(
            window_increment=self.window_length,
            window_padding=self._window_padding,
            start_time=self._progress.time_progress,
            min_length=2 * self._window_padding,
            min_stations=self.min_stations,
        )
        pre_processed_batches = self.pre_processing.iter_batches(batches)

        stats = SearchStats(
            project_name=self._rundir.name,
            total_stations=self.stations.n_stations,
            total_networks=self.stations.n_networks,
        )
        stats.reset_start_time()

        processing_start = datetime_now()
        console = asyncio.create_task(RuntimeStats.live_view())

        async for images, batch in self.image_functions.iter_images(
            pre_processed_batches
        ):
            batch_processing_start = datetime_now()
            images.set_stations(self.stations)
            images.apply_exponent(self.power_mean)
            search_block = SearchTraces(
                parent=self,
                images=images,
                start_time=batch.start_time,
                end_time=batch.end_time,
            )

            detections, semblance_trace = await search_block.search()

            await self._catalog.save_semblance_trace(semblance_trace)
            if detections:
                BackgroundTasks.create_task(self.new_detections(detections))
            if self.save_images:
                await images.save_mseed(self._rundir / "images")

            stats.add_processed_batch(
                batch,
                duration=datetime_now() - batch_processing_start,
                show_log=True,
            )
            self.set_progress(batch.end_time)

        await BackgroundTasks.wait_all()
        await self._catalog.save()
        await self._catalog.export_detections(
            jitter_location=self.octree.smallest_node_size()
        )
        console.cancel()
        logger.info("finished search in %s", datetime_now() - processing_start)
        logger.info("detected %d events", self._catalog.n_events)

    async def new_detections(self, detections: list[EventDetection]) -> None:
        """Process new detections.

        Args:
            detections (list[EventDetection]): List of new event detections.
        """
        catalog = self.catalog
        await asyncio.gather(
            *(self.add_magnitude_and_features(det) for det in detections)
        )

        for detection in detections:
            await catalog.add(
                detection,
                jitter_location=self.octree.smallest_node_size(),
            )
            await self._new_detection.emit(detection)

        if not catalog.n_events:
            return

    async def add_magnitude_and_features(
        self,
        event: EventDetection,
        recalculate: bool = True,
    ) -> EventDetection:
        """Adds magnitude and features to the given event.

        Args:
            event (EventDetection): The event to add magnitude and features to.
            recalculate (bool, optional): Whether to overwrite existing magnitudes and
                features. Defaults to True.
        """
        if not event.in_bounds:
            return event

        async with self._compute_semaphore:
            for mag_calculator in self.magnitudes:
                if not recalculate and mag_calculator.has_magnitude(event):
                    continue
                logger.debug("adding magnitude from %s", mag_calculator.magnitude)
                await mag_calculator.add_magnitude(self.data_provider, event)

            for feature_calculator in self.features:
                logger.debug("adding features from %s", feature_calculator.feature)
                await feature_calculator.add_features(self.data_provider, event)
        return event

    @classmethod
    def load_rundir(cls, rundir: Path) -> Self:
        search_file = rundir / "search.json"
        search = cls.model_validate_json(search_file.read_bytes())
        search._rundir = rundir
        search._catalog = EventCatalog.load_rundir(rundir)

        progress_file = rundir / "progress.json"
        if progress_file.exists():
            search._progress = SearchProgress.model_validate_json(
                progress_file.read_text()
            )

        search._init_logging()
        return search

    @classmethod
    def from_config(
        cls,
        filename: Path,
    ) -> Self:
        model = super().model_validate_json(filename.read_text())
        # Make relative paths absolute
        filename = Path(filename)
        base_dir = filename.absolute().parent
        for name in model.model_fields_set:
            value = getattr(model, name)
            if isinstance(value, Path) and not value.absolute():
                setattr(model, name, value.relative_to(base_dir))
        model._config_stem = filename.stem
        return model

    def has_rundir(self) -> bool:
        return hasattr(self, "_rundir") and self._rundir.exists()

    # def __del__(self) -> None:
    # FIXME: Replace with signal overserver?
    # if hasattr(self, "_detections"):
    # with contextlib.suppress(Exception):
    #     asyncio.run(
    #         self._detections.export_detections(
    #             jitter_location=self.octree.smallest_node_size()
    #         )
    #     )


class SearchTraces:
    _images: dict[float | None, WaveformImages]

    def __init__(
        self,
        parent: Search,
        images: WaveformImages,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.parent = parent
        self.images = images
        self.start_time = start_time
        self.end_time = end_time

        self._images = {}

    def _n_samples_semblance(self) -> int:
        """Number of samples to use for semblance calculation, includes padding."""
        parent = self.parent
        window_padding = parent._window_padding
        time_span = (self.end_time + window_padding) - (
            self.start_time - window_padding
        )
        return int(round(time_span.total_seconds() * parent.semblance_sampling_rate))

    @alog_call
    async def add_semblance(
        self,
        octree: Octree,
        image: WaveformImage,
        ray_tracer: RayTracer,
        semblance: Semblance,
        nodes: Sequence[Node],
    ) -> None:
        if image.sampling_rate != semblance.sampling_rate:
            raise ValueError(
                f"image sampling rate {image.sampling_rate} does not match "
                f"semblance sampling rate {semblance.sampling_rate}"
            )
        if image.weight == 0.0:
            return

        logger.debug("stacking image %s", image.image_function.name)
        parent = self.parent

        traveltimes = await ray_tracer.get_travel_times(
            image.phase,
            nodes,
            image.stations,
        )

        if parent.station_corrections:
            station_delays = await parent.station_corrections.get_delays(
                image.stations.get_all_nsl(),
                image.phase,
                nodes,
            )
            traveltimes += station_delays

        traveltimes_bad = np.isnan(traveltimes)
        traveltimes[traveltimes_bad] = 0.0
        # station_contribution = (~traveltimes_bad).sum(axis=1, dtype=np.float32)

        shifts = np.round(-traveltimes / image.delta_t).astype(np.int32)
        # weights = np.full_like(shifts, fill_value=image.weight, dtype=np.float32)
        # Normalize by number of station contribution
        # with np.errstate(divide="ignore", invalid="ignore"):
        #     weights /= station_contribution[:, np.newaxis]

        weights = np.full_like(shifts, fill_value=image.weight, dtype=np.float32)
        weights[traveltimes_bad] = 0.0

        if parent.distance_weights:
            weights *= await parent.distance_weights.get_weights(
                nodes,
                image.stations,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            weights /= weights.sum(axis=1, keepdims=True)

        # applying waterlevel
        weights[weights < 1e-3] = 0.0

        def expand_array(array: np.ndarray) -> np.ndarray:
            """Fill arrays with zeros to pad the matrix."""
            array_n_nodes = array.shape[0]
            if array_n_nodes == octree.n_nodes:
                return array

            n_stations = image.stations.n_stations
            filled = np.zeros((octree.n_nodes, n_stations), dtype=array.dtype)
            filled[-array_n_nodes:] = array
            return filled

        await semblance.add_semblance(
            trace_data=image.get_trace_data(),
            offsets=image.get_offsets(self.start_time - parent._window_padding),
            shifts=expand_array(shifts),
            weights=expand_array(weights),
            threads=self.parent.n_threads_parstack,
        )

    async def get_images(self, sampling_rate: float | None = None) -> WaveformImages:
        """Retrieves waveform images for the specified sampling rate.

        Args:
            sampling_rate (float | None, optional): The desired sampling rate in Hz.
                Defaults to None.

        Returns:
            WaveformImages: The waveform images for the specified sampling rate.
        """
        if sampling_rate is None:
            return self.images

        if not isinstance(sampling_rate, float):
            raise TypeError("sampling rate has to be a float or int")

        logger.debug("downsampling images to %g Hz", sampling_rate)
        self.images.resample(sampling_rate, max_normalize=True)

        return self.images

    async def search(
        self,
        octree: Octree | None = None,
        new_nodes: Sequence[Node] | None = None,
        semblance: Semblance | None = None,
    ) -> tuple[list[EventDetection], Trace]:
        """Searches for events in the given traces.

        Args:
            octree (Octree | None, optional): The octree to use for the search.
                Defaults to None.
            new_nodes (Sequence[Node] | None, optional): The new nodes to use for the
                search. If none all nodes from the provided octree are taken.
                Defaults to None.
            semblance (Semblance | None, optional): The semblance to use for the search.
                If None, a new instance is created. Defaults to None.

        Returns:
            tuple[list[EventDetection], Trace]: The event detections and the
                semblance traces used for the search.
        """
        parent = self.parent
        sampling_rate = parent.semblance_sampling_rate

        octree = octree or parent.octree.reset()
        new_nodes = new_nodes or octree.nodes

        images = await self.get_images(sampling_rate=float(sampling_rate))

        padding_samples = int(
            round(parent._window_padding.total_seconds() * sampling_rate)
        )

        if not semblance:
            semblance = Semblance(
                n_samples=self._n_samples_semblance(),
                start_time=self.start_time,
                sampling_rate=sampling_rate,
                padding_samples=padding_samples,
                exponent=1.0 / parent.power_mean,
            )

        for image in images:
            if not image.has_traces():
                continue
            await self.add_semblance(
                octree=octree,
                image=image,
                ray_tracer=parent.ray_tracers.get_phase_tracer(image.phase),
                semblance=semblance,
                nodes=new_nodes,
            )

        # Applying the generalized mean to the semblance
        # semblance.normalize(images.cumulative_weight())

        leaf_node_mask = np.array([node.is_leaf() for node in octree.nodes], dtype=bool)
        semblance.set_leaf_nodes(leaf_node_mask)

        if parent.detection_threshold == "MAD":
            detection_threshold = stats.median_abs_deviation(
                await semblance.maxima_semblance(
                    trim_padding=False,
                    nthreads=parent.n_threads_argmax,
                )
            )
            detection_threshold *= 10.0
            logger.debug("threshold MAD %g", detection_threshold)
        else:
            detection_threshold = parent.detection_threshold

        threshold = detection_threshold ** (1.0 / parent.power_mean)
        detection_idx, detection_semblance = await semblance.find_peaks(
            height=float(threshold),
            prominence=float(threshold),
            distance=round(parent.detection_blinding.total_seconds() * sampling_rate),
            nthreads=parent.n_threads_argmax,
        )

        if detection_idx.size == 0:
            return [], await semblance.get_trace()

        # Split Octree nodes above a semblance threshold. Once octree for all detections
        # in frame
        maxima_node_indices = await semblance.maxima_node_idx()
        refine_nodes: set[Node] = set()

        for time_idx in detection_idx:
            octree.map_semblance(semblance.get_semblance(time_idx))
            node_idx = maxima_node_indices[time_idx]
            source_node = octree.nodes[node_idx]

            if not source_node.is_leaf():
                logger.error("Hit a non-leaf node!")
                continue

            if parent.absorbing_boundary and source_node.is_inside_border(
                with_surface=parent.absorbing_boundary == "with_surface",
                border_width=parent.absorbing_boundary_width,
            ):
                continue
            refine_nodes.update(source_node)
            refine_nodes.update(source_node.get_neighbours())

            densest_node = max(octree.leaf_nodes, key=lambda n: n.semblance_density())
            refine_nodes.add(densest_node)
            refine_nodes.update(densest_node.get_neighbours())

        refine_nodes = {node for node in refine_nodes if node.can_split()}

        # refine_nodes is empty when all sources fall into smallest octree nodes
        new_nodes = []
        if refine_nodes:
            node_size_max = max(node.size for node in refine_nodes)
            new_level = 0
            for node in refine_nodes:
                new_nodes.extend(node.split())
                new_level = max(new_level, node.level + 1)
            logger.info(
                "detected %d energy burst%s - refined %d nodes, level %d (%.1f m)",
                detection_idx.size,
                "s" if detection_idx.size > 1 else "",
                len(refine_nodes),
                new_level,
                node_size_max,
            )
            return await self.search(
                octree,
                semblance=semblance,
                new_nodes=new_nodes,
            )

        detections = []
        for time_idx, semblance_detection in zip(
            detection_idx, detection_semblance, strict=True
        ):
            time = semblance.get_time_from_index(time_idx)
            semblance_event = semblance.get_semblance(time_idx)
            node_idx = maxima_node_indices[time_idx]

            octree.map_semblance(semblance_event)
            source_node = octree.nodes[node_idx]
            if parent.absorbing_boundary and source_node.is_inside_border(
                with_surface=parent.absorbing_boundary == "with_surface",
                border_width=parent.absorbing_boundary_width,
            ):
                continue

            if parent.node_peak_interpolation:
                source_location = await octree.interpolate_max_semblance(source_node)
            else:
                source_location = source_node.as_location()

            detection = EventDetection(
                time=time,
                semblance=float(semblance_detection),
                distance_border=source_node.get_distance_border(),  # rename trough_distance
                n_stations=images.n_stations,
                **source_location.model_dump(),
            )

            # Attach modelled and picked arrivals to receivers
            for image in await self.get_images(sampling_rate=None):
                ray_tracer = parent.ray_tracers.get_phase_tracer(image.phase)
                arrivals_model = ray_tracer.get_arrivals(
                    phase=image.phase,
                    event_time=time,
                    source=source_location,
                    receivers=image.stations,
                )

                arrival_times = [arr.time if arr else None for arr in arrivals_model]
                station_delays = []

                if parent.station_corrections:
                    for nsl in image.stations.get_all_nsl():
                        delay = parent.station_corrections.get_delay(
                            nsl,
                            image.phase,
                            node=source_node,
                        )
                        station_delays.append(timedelta(seconds=delay))

                arrivals_observed = image.search_phase_arrivals(
                    event_time=time,
                    modelled_arrivals=[arr if arr else None for arr in arrival_times],
                    threshold=parent.pick_confidence_threshold,
                )

                phase_detections = [
                    PhaseDetection(
                        phase=image.phase,
                        model=modeled_time,
                        observed=obs,
                    )
                    if modeled_time
                    else None
                    for modeled_time, obs in zip(
                        arrivals_model,
                        arrivals_observed,
                        strict=True,
                    )
                ]
                detection.receivers.add(
                    stations=image.stations,
                    phase_arrivals=phase_detections,
                )
                detection.set_uncertainty(
                    DetectionUncertainty.from_event(
                        source_node=source_node,
                        octree=octree,
                    )
                )

            detections.append(detection)

        return detections, await semblance.get_trace()
