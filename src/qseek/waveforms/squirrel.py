from __future__ import annotations

import asyncio
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, ClassVar, Iterator, Literal

from pydantic import (
    AwareDatetime,
    DirectoryPath,
    Field,
    PositiveInt,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from pyrocko.squirrel import Squirrel
from typing_extensions import Self

from qseek.models.station import Stations
from qseek.stats import Stats
from qseek.utils import QUEUE_SIZE, datetime_now, human_readable_bytes, to_datetime
from qseek.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch
    from rich.table import Table

logger = logging.getLogger(__name__)


class SquirrelPrefetcher:
    queue: asyncio.Queue[Batch | None]
    load_time: timedelta = timedelta(seconds=0.0)

    _load_queue: asyncio.Queue[Batch | None]
    _fetched_batches: int
    _task: asyncio.Task[None]

    def __init__(self, iterator: Iterator[Batch]) -> None:
        self.iterator = iterator
        self.queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._load_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._fetched_batches = 0

        self._task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self) -> None:
        logger.info(
            "start prefetching waveforms, queue size %d",
            self.queue.maxsize,
        )

        async def load_data() -> None | Batch:
            while True:
                start_load = datetime_now()
                logger.debug("loading waveform batch %d", self._fetched_batches)
                batch = await asyncio.to_thread(next, self.iterator, None)
                if batch is None:
                    await self.queue.put(None)
                    return
                logger.debug("read waveform batch in %s", datetime_now() - start_load)
                self._fetched_batches += 1
                self.load_time = datetime_now() - start_load
                await self.queue.put(batch)

        await asyncio.create_task(load_data())
        logger.debug("done loading waveforms")


class SquirrelStats(Stats):
    empty_batches: PositiveInt = 0
    short_batches: PositiveInt = 0
    time_per_batch: timedelta = timedelta(seconds=0.0)
    bytes_per_seconds: float = 0.0

    _queue: asyncio.Queue[Batch | None] | None = PrivateAttr(None)
    _position: int = PrivateAttr(20)

    def set_queue(self, queue: asyncio.Queue[Batch | None]) -> None:
        self._queue = queue

    @computed_field
    @property
    def queue_size(self) -> PositiveInt:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @computed_field
    @property
    def queue_size_max(self) -> PositiveInt:
        if self._queue is None:
            return 0
        return self._queue.maxsize

    def _populate_table(self, table: Table) -> None:
        prefix = "[bold red]" if self.queue_size <= 1 else ""
        table.add_row("Queue", f"{prefix}{self.queue_size} / {self.queue_size_max}")
        table.add_row(
            "Waveform loading",
            f"{human_readable_bytes(self.bytes_per_seconds)}/s",
        )


class PyrockoSquirrel(WaveformProvider):
    """Waveform provider using Pyrocko's Squirrel."""

    provider: Literal["PyrockoSquirrel"] = "PyrockoSquirrel"

    environment: DirectoryPath | None = Field(
        default=None,
        description="Path to a [Pyrocko Squirrel environment](https://pyrocko.org/docs/current/apps/squirrel/reference/squirrel_init.html).",
    )
    persistent: str | None = Field(
        default=None,
        description="Name of the [Squirrel's persistent collection](https://pyrocko.org/docs/current/apps/squirrel/reference/squirrel_persistent.html)"
        " for faster loading of large data sets.",
    )
    waveform_dirs: list[Path] = Field(
        default=[Path("./data")],
        description="List of directories holding the waveform files.",
    )
    start_time: AwareDatetime | None = Field(
        default=None,
        description="Start time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) including timezone. "
        "E.g. `2024-12-30T00:00:00Z`.",
    )
    end_time: AwareDatetime | None = Field(
        default=None,
        description="End time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601) including timezone. "
        "E.g. `2024-12-31T00:00:00Z`.",
    )

    n_threads: PositiveInt = Field(
        default=8,
        description="Number of threads for loading waveforms,"
        " important for large data sets.",
    )
    watch_waveforms: bool | timedelta = Field(
        default=False,
        description="Watch the waveform directories for changes. If `True` it will "
        "check every ten minutes. If a `timedelta` is provided it will check every "
        "specified time. Default is False.",
    )

    _squirrel: Squirrel | None = PrivateAttr(None)
    _stations: Stations = PrivateAttr(None)
    _stats: ClassVar[SquirrelStats] = SquirrelStats()

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        if not self.waveform_dirs and not self.persistent:
            raise ValueError("no waveform directories or persistent selection provided")
        if self.watch_waveforms and not self.waveform_dirs:
            raise ValueError("watch_waveforms requires waveform_dirs")
        if self.watch_waveforms and self.end_time:
            raise ValueError("watch_waveforms does not support end_time")
        return self

    @field_validator("watch_waveforms", mode="after")
    @classmethod
    def _validate_watch(cls, value: bool | timedelta) -> timedelta | bool:
        if value is True:
            return timedelta(minutes=10)
        if isinstance(value, timedelta) and value < timedelta(minutes=10):
            raise ValueError("watch_waveform_dirs must be at least 10 minutes")
        return value

    def get_squirrel(self) -> Squirrel:
        if not self._squirrel:
            logger.info(
                "loading squirrel environment from %s",
                self.environment or "home directory",
            )
            try:
                squirrel = Squirrel(
                    env=str(self.environment.expanduser())
                    if self.environment
                    else None,
                    persistent=self.persistent,
                    n_threads=self.n_threads,
                )
            except TypeError:
                logger.warning("Squirrel does not support n_threads")
                squirrel = Squirrel(
                    env=str(self.environment.expanduser())
                    if self.environment
                    else None,
                    persistent=self.persistent,
                )

            if self.waveform_dirs:
                self.scan_waveform_dirs(squirrel)

            if self._stations:
                for path in self._stations.station_xmls:
                    logger.info("loading StationXML responses from %s", path)
                    squirrel.add(str(path), check=False)
            self._squirrel = squirrel
        return self._squirrel

    def scan_waveform_dirs(self, squirrel: Squirrel) -> None:
        logger.info(
            "scanning waveform directories %s",
            ".".join(map(str, self.waveform_dirs)),
        )
        paths = []
        for path in self.waveform_dirs:
            if "**" in str(path):
                paths.extend(glob.glob(str(path.expanduser()), recursive=True))
            else:
                paths.append(str(path.expanduser()))
        squirrel.add(paths, check=False)

    def prepare(self, stations: Stations) -> None:
        logger.info("preparing squirrel waveform provider")
        self._stations = stations
        squirrel = self.get_squirrel()
        stations.weed_from_squirrel_waveforms(squirrel)

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_length: timedelta | None = None,
        min_stations: int = 0,
    ) -> AsyncIterator[WaveformBatch]:
        if not self._stations:
            raise ValueError("no stations provided. has prepare() been called?")

        squirrel = self.get_squirrel()
        stats = self._stats

        if self.watch_waveforms:
            logger.info("scanning for new waveforms every %s", self.watch_waveforms)

        def init_prefetcher(
            chop_start_time: datetime | None = None,
            chop_end_time: datetime | None = None,
            trim_end: timedelta | None = None,
        ) -> SquirrelPrefetcher:
            sq_tmin, sq_tmax = map(to_datetime, squirrel.get_time_span(["waveform"]))
            chop_start_time = chop_start_time or self.start_time or sq_tmin
            chop_end_time = chop_end_time or self.end_time or sq_tmax
            if trim_end:
                chop_end_time -= trim_end

            logger.info(
                "searching time span from %s to %s (%s)",
                chop_start_time,
                chop_end_time,
                chop_end_time - chop_start_time,
            )

            iterator = squirrel.chopper_waveforms(
                tmin=(chop_start_time + window_padding).timestamp(),
                tmax=(chop_end_time - window_padding).timestamp(),
                tinc=window_increment.total_seconds(),
                tpad=window_padding.total_seconds(),
                want_incomplete=False,
                codes=[(*nsl, "*") for nsl in self._stations.get_all_nsl()],
                channel_priorities=self.channel_selector,
            )
            prefetcher = SquirrelPrefetcher(iterator)
            stats.set_queue(prefetcher.queue)
            return prefetcher

        prefetcher = init_prefetcher(chop_start_time=start_time, chop_end_time=end_time)
        last_batch_end_time = None

        while True:
            start = datetime_now()
            pyrocko_batch = await prefetcher.queue.get()

            if pyrocko_batch is None:
                if isinstance(self.watch_waveforms, timedelta):
                    logger.info(
                        "re-scanning waveform directories in %s", self.watch_waveforms
                    )

                    await asyncio.sleep(self.watch_waveforms.total_seconds())
                    self.scan_waveform_dirs(squirrel)
                    prefetcher = init_prefetcher(
                        chop_start_time=last_batch_end_time,
                        chop_end_time=None,
                        trim_end=timedelta(seconds=30),  # Trim as SeedLink is slow!
                    )
                    continue
                else:
                    logger.debug("no more waveforms to load")
                    prefetcher.queue.task_done()
                    break

            batch = WaveformBatch(
                traces=pyrocko_batch.traces,
                start_time=to_datetime(pyrocko_batch.tmin),
                end_time=to_datetime(pyrocko_batch.tmax),
                i_batch=pyrocko_batch.i,
                n_batches=pyrocko_batch.n,
            )
            batch.clean_traces()

            stats.time_per_batch = datetime_now() - start
            stats.bytes_per_seconds = (
                batch.cumulative_bytes / prefetcher.load_time.total_seconds()
            )

            if not batch.is_healthy(min_stations=min_stations):
                logger.warning(
                    "unhealthy batch %d - %s",
                    batch.i_batch,
                    batch.start_time,
                )
                stats.empty_batches += 1
                continue

            if min_length and batch.duration < min_length:
                logger.warning(
                    "duration of batch %d too short %s",
                    batch.i_batch,
                    batch.duration,
                )
                stats.short_batches += 1
                continue

            last_batch_end_time = batch.end_time
            yield batch

            prefetcher.queue.task_done()
