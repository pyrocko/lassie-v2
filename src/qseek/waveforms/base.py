from __future__ import annotations

import logging
from asyncio import Queue
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, AsyncIterator, Literal

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, constr
from pyrocko.trace import Trace

from qseek.stats import Stats

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.station import Stations

logger = logging.getLogger(__name__)


@dataclass
class WaveformBatch:
    traces: list[Trace]
    start_time: datetime
    end_time: datetime
    i_batch: int
    n_batches: int = 0

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def cumulative_duration(self) -> timedelta:
        """Cumulative duration of the traces in the batch.

        Returns:
            timedelta: Cumulative duration of the traces in the batch.
        """
        seconds = 0.0
        for tr in self.traces:
            seconds += tr.tmax - tr.tmin
        return timedelta(seconds=seconds)

    @property
    def cumulative_bytes(self) -> int:
        """Cumulative size of the traces in the batch in bytes."""
        return sum(tr.ydata.nbytes for tr in self.traces)

    @property
    def n_stations(self) -> int:
        """Number of unique stations in the batch."""
        return len({(tr.network, tr.station, tr.location) for tr in self.traces})

    @property
    def n_networks(self) -> int:
        """Number of unique networks in the batch."""
        return len({tr.network for tr in self.traces})

    def is_healthy(self, min_stations: int = 0) -> bool:
        """Check if the batch is empty.

        Returns:
            bool: True if the batch is empty, False otherwise.
        """
        if self.n_stations < min_stations:
            logger.warning("batch has only %d stations", self.n_stations)
            return False
        if not self.traces:
            logger.warning("batch is empty")
            return False
        return True

    def clean_traces(self) -> None:
        """Remove empty or bad traces."""
        good_nsl = set()
        for tr in self.traces.copy():
            if not tr.ydata.size or not np.all(np.isfinite(tr.ydata)):
                logger.warning("skipping empty or bad trace: %s", ".".join(tr.nslc_id))
                self.traces.remove(tr)
                continue

            if tr.nslc_id in good_nsl:
                logger.warning("removing duplicate trace: %s", ".".join(tr.nslc_id))
                self.traces.remove(tr)
                continue

            good_nsl.add(tr.nslc_id)


class WaveformProvider(BaseModel):
    provider: Literal["WaveformProvider"] = "WaveformProvider"

    channel_selector: list[constr(to_upper=True, max_length=2, min_length=2)] | None = (
        Field(
            default=None,
            min_length=1,
            description="Channel selector for waveforms, " "e.g. `['HH', 'EN']`.",
        )
    )

    _queue: Queue[WaveformBatch | None] = PrivateAttr(default_factory=lambda: Queue())
    _stats: Stats = PrivateAttr(default_factory=Stats)

    @classmethod
    def get_subclasses(cls) -> tuple[type[WaveformProvider], ...]:
        return tuple(cls.__subclasses__())

    def get_squirrel(self) -> Squirrel:
        raise NotImplementedError

    def prepare(self, stations: Stations) -> None: ...

    async def iter_batches(
        self,
        window_increment: timedelta,
        window_padding: timedelta,
        start_time: datetime | None = None,
        min_length: timedelta | None = None,
        min_stations: int = 0,
    ) -> AsyncIterator[WaveformBatch]:
        yield
        raise NotImplementedError
