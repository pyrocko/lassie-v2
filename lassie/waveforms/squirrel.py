from __future__ import annotations

import asyncio
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Literal

from pydantic import (
    AwareDatetime,
    DirectoryPath,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pyrocko.squirrel import Squirrel
from typing_extensions import Self

from lassie.models.station import Stations
from lassie.stats import Stats
from lassie.utils import datetime_now, to_datetime
from lassie.waveforms.base import WaveformBatch, WaveformProvider

if TYPE_CHECKING:
    from pyrocko.squirrel.base import Batch

logger = logging.getLogger(__name__)


class SquirrelPrefetcher:
    queue: asyncio.Queue[Batch | None]
    highpass: float | None
    lowpass: float | None
    _fetched_batches: int
    _task: asyncio.Task[None]

    fetch_time: timedelta = timedelta(seconds=0)

    def __init__(
        self,
        iterator: Iterator[Batch],
        queue_size: int = 4,
        highpass: float | None = None,
        lowpass: float | None = None,
    ) -> None:
        self.iterator = iterator
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.highpass = highpass
        self.lowpass = lowpass
        self._fetched_batches = 0

        self._task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self) -> None:
        logger.info("start prefetching data, queue size %d", self.queue.maxsize)

        def filter_freqs(batch: Batch) -> Batch:
            # Filter traces in-place
            start = None
            if self.highpass:
                start = datetime_now()
                for tr in batch.traces:
                    tr.highpass(4, corner=self.highpass)
            if self.lowpass:
                start = start or datetime_now()
                for tr in batch.traces:
                    tr.lowpass(4, corner=self.lowpass)
            if start:
                logger.debug("filtered traces in %s", datetime_now() - start)
            return batch

        while True:
            start = datetime_now()
            batch = await asyncio.to_thread(lambda: next(self.iterator, None))
            self.fetch_time = datetime_now() - start
            if batch is None:
                logger.debug("squirrel prefetcher finished")
                await self.queue.put(None)
                break

            await asyncio.to_thread(filter_freqs, batch)
            logger.debug("prefetched waveforms in %s", self.fetch_time)
            if self.queue.empty() and self._fetched_batches:
                logger.warning("waveform queue ran empty, prefetching is too slow")

            self._fetched_batches += 1
            await self.queue.put(batch)


class SquirrelStats(Stats):
    queue_size: PositiveInt = 0
    queue_size_max: PositiveInt = 0
    empty_batches: PositiveInt = 0
    short_batches: PositiveInt = 0
    time_per_batch: timedelta = timedelta(seconds=0)


class PyrockoSquirrel(WaveformProvider):
    """Waveform provider using Pyrocko's Squirrel."""

    provider: Literal["PyrockoSquirrel"] = "PyrockoSquirrel"

    environment: DirectoryPath = Field(
        default=Path("."),
        description="Path to a Squirrel environment.",
    )
    waveform_dirs: list[Path] = Field(
        default=[],
        description="List of directories holding the waveform files.",
    )
    start_time: AwareDatetime | None = Field(
        default=None,
        description="Start time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601).",
    )
    end_time: AwareDatetime | None = Field(
        default=None,
        description="End time for the search in "
        "[ISO8601](https://en.wikipedia.org/wiki/ISO_8601).",
    )

    highpass: PositiveFloat | None = Field(
        default=None,
        description="Highpass filter, corner frequency in Hz.",
    )
    lowpass: PositiveFloat | None = Field(
        default=None,
        description="Lowpass filter, corner frequency in Hz.",
    )

    channel_selector: str = Field(
        default="*",
        max_length=3,
        description="Channel selector for waveforms, "
        "use e.g. `EN?` for selection of all accelerometer data.",
    )
    async_prefetch_batches: PositiveInt = Field(
        default=4,
        description="Queue size for asynchronous pre-fetcher.",
    )

    _squirrel: Squirrel | None = PrivateAttr(None)
    _stations: Stations = PrivateAttr()
    _stats: SquirrelStats = PrivateAttr(SquirrelStats())

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        if self.highpass and self.lowpass and self.highpass > self.lowpass:
            raise ValueError("freq_min must be less than freq_max")
        return self

    @field_validator("waveform_dirs")
    def check_dirs(cls, dirs: list[Path]) -> list[Path]:  # noqa: N805
        if not dirs:
            raise ValueError("no waveform directories provided!")
        return dirs

    def get_squirrel(self) -> Squirrel:
        if not self._squirrel:
            logger.debug("initializing squirrel")
            squirrel = Squirrel(str(self.environment.expanduser()))
            paths = []
            for path in self.waveform_dirs:
                if "**" in str(path):
                    paths.extend(glob.glob(str(path.expanduser()), recursive=True))
                else:
                    paths.append(str(path.expanduser()))

            squirrel.add(paths, check=False)
            self._squirrel = squirrel
        return self._squirrel

    def prepare(self, stations: Stations) -> None:
        logger.info("preparing squirrel waveform provider")
        squirrel = self.get_squirrel()
        stations.weed_from_squirrel_waveforms(squirrel)
        self._stats.queue_size_max = self.async_prefetch_batches
        self._stations = stations

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_length: timedelta | None = None,
    ) -> AsyncIterator[WaveformBatch]:
        if not self._stations:
            raise ValueError("no stations provided. has prepare() been called?")

        squirrel = self.get_squirrel()
        stats = self._stats
        sq_tmin, sq_tmax = squirrel.get_time_span(["waveform"])

        start_time = start_time or self.start_time or to_datetime(sq_tmin)
        end_time = end_time or self.end_time or to_datetime(sq_tmax)

        logger.info(
            "searching time span from %s to %s (%s)",
            start_time,
            end_time,
            end_time - start_time,
        )

        iterator = squirrel.chopper_waveforms(
            tmin=start_time.timestamp(),
            tmax=end_time.timestamp(),
            tinc=window_increment.total_seconds(),
            tpad=window_padding.total_seconds(),
            want_incomplete=False,
            codes=[
                (*nsl, self.channel_selector) for nsl in self._stations.get_all_nsl()
            ],
        )
        prefetcher = SquirrelPrefetcher(
            iterator,
            self.async_prefetch_batches,
            self.highpass,
            self.lowpass,
        )

        while True:
            stats.queue_size = prefetcher.queue.qsize()
            pyrocko_batch = await prefetcher.queue.get()
            stats.time_per_batch = prefetcher.fetch_time
            if pyrocko_batch is None:
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

            if batch.is_empty():
                logger.warning("empty batch %d", batch.i_batch)
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

            yield batch

            prefetcher.queue.task_done()
