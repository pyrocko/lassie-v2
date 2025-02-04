from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AsyncIterator,
    ClassVar,
    Iterator,
    Tuple,
    Union,
)

from pydantic import Field, PositiveInt, PrivateAttr, RootModel, computed_field

from qseek.images.base import ImageFunction
from qseek.images.seisbench import SeisBench
from qseek.stats import Stats
from qseek.utils import QUEUE_SIZE, PhaseDescription, datetime_now, human_readable_bytes

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from rich.table import Table

    from qseek.images.base import WaveformImage
    from qseek.models.station import Stations
    from qseek.waveforms.base import WaveformBatch


logger = logging.getLogger(__name__)


ImageFunctionType = Annotated[
    Union[SeisBench, ImageFunction],
    Field(..., discriminator="image"),
]


class ImageFunctionsStats(Stats):
    time_per_batch: timedelta = timedelta()
    bytes_per_second: float = 0.0

    _queue: asyncio.Queue[Tuple[WaveformImages | WaveformBatch] | None] | None = (
        PrivateAttr(None)
    )
    _position: int = PrivateAttr(40)

    def set_queue(
        self,
        queue: asyncio.Queue[Tuple[WaveformImages | WaveformBatch] | None],
    ) -> None:
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
        prefix = "[bold red]" if self.queue_size <= 2 else ""
        table.add_row("Queue", f"{prefix}{self.queue_size} / {self.queue_size_max}")
        table.add_row(
            "Waveform processing",
            f"{human_readable_bytes(self.bytes_per_second)}/s",
        )


class ImageFunctions(RootModel):
    root: list[ImageFunctionType] = [SeisBench()]

    _queue: asyncio.Queue[Tuple[WaveformImages, WaveformBatch] | None] = PrivateAttr(
        asyncio.Queue(maxsize=QUEUE_SIZE)
    )
    _processed_images: int = PrivateAttr(0)
    _stats: ClassVar[ImageFunctionsStats] = ImageFunctionsStats()

    def model_post_init(self, __context: Any) -> None:
        # Check if phases are provided twice
        phases = self.get_phases()
        if len(set(phases)) != len(phases):
            raise ValueError("A phase was provided twice")

    async def process_traces(self, traces: list[Trace]) -> WaveformImages:
        images = []
        for function in self:
            logger.debug("calculating images from %s", function.name)
            images.extend(await function.process_traces(traces))

        return WaveformImages(root=images)

    async def iter_images(
        self,
        batch_iterator: AsyncIterator[WaveformBatch],
    ) -> AsyncIterator[Tuple[WaveformImages, WaveformBatch]]:
        """Iterate over images from batches.

        Args:
            batch_iterator (AsyncIterator[Batch]): Async iterator over batches.
            min_stations (int): Minimum number of stations required in a batch.
                Defaults to 3.

        Yields:
            AsyncIterator[WaveformImages]: Async iterator over images.
        """
        stats = self._stats
        stats.set_queue(self._queue)

        async def worker() -> None:
            logger.info(
                "start pre-processing images, queue size %d", self._queue.maxsize
            )
            async for batch in batch_iterator:
                if not batch.is_healthy():
                    logger.debug("unhealthy batch, skipping")
                    continue

                start_time = datetime_now()
                images = await self.process_traces(batch.traces)
                stats.time_per_batch = datetime_now() - start_time
                stats.bytes_per_second = (
                    batch.cumulative_bytes / stats.time_per_batch.total_seconds()
                )
                self._processed_images += 1
                await self._queue.put((images, batch))

            await self._queue.put(None)

        task = asyncio.create_task(worker())

        while True:
            ret = await self._queue.get()
            if ret is None:
                logger.debug("image function finished")
                break
            yield ret

        logger.debug("waiting for image function to finish")
        await task

    def get_phases(self) -> tuple[PhaseDescription, ...]:
        """Get all phases that are available in the image functions.

        Returns:
            tuple[str, ...]: All available phases.
        """
        return tuple(chain.from_iterable(image.get_provided_phases() for image in self))

    def get_blinding(self, sampling_rate: float) -> timedelta:
        return max(image.get_blinding(sampling_rate) for image in self)

    def __iter__(self) -> Iterator[ImageFunction]:
        return iter(self.root)


@dataclass
class WaveformImages:
    root: list[WaveformImage] = Field([], alias="images")

    @property
    def n_images(self) -> int:
        """Number of image functions."""
        return len(self.root)

    @property
    def n_stations(self) -> int:
        """Number of stations in the images."""
        return max(0, *(image.stations.n_stations for image in self if image.stations))

    def resample(self, sampling_rate: float, max_normalize: bool = False) -> None:
        """Resample traces in-place.

        Args:
            sampling_rate (float): Desired sampling rate in Hz.
            max_normalize (bool): Normalize by maximum value to keep the scale of the
                maximum detection. Defaults to False
        """
        for image in self:
            image.resample(sampling_rate, max_normalize)

    def apply_exponent(self, exponent: float) -> None:
        """Apply exponent to all images.

        Args:
            exponent (float): Exponent to apply.
        """
        for image in self:
            image.apply_exponent(exponent)

    def set_stations(self, stations: Stations) -> None:
        """Set the images stations.

        Args:
            stations (Stations): Stations to set.
        """
        for image in self:
            image.set_stations(stations)

    def cumulative_weight(self) -> float:
        """Get the cumulative weight of all images."""
        return sum(image.weight for image in self)

    def snuffle(self) -> None:
        """Open Pyrocko Snuffler on the image traces."""
        from pyrocko.trace import snuffle

        traces = []
        for img in self:
            traces += img.traces
        snuffle(traces)

    async def save_mseed(self, path: Path) -> None:
        """Save images to disk.

        Args:
            path (Path): Path to save the images.
        """
        logger.debug("saving images to %s", path)
        path.mkdir(exist_ok=True)
        for image in self:
            await image.save_mseed(path)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.root
