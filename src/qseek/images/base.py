from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field
from pyrocko.io import save

from qseek.models.station import Stations
from qseek.utils import SDS_PYROCKO_SCHEME, PhaseDescription, resample

if TYPE_CHECKING:
    from pyrocko.trace import Trace


@dataclass
class ObservedArrival:
    phase: str
    time: datetime
    detection_value: float
    provider: str = ""


class ImageFunction(BaseModel):
    image: Literal["base"] = "base"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def process_traces(self, traces: list[Trace]) -> list[WaveformImage]:
        """Process traces to generate image functions.

        Args:
            traces (list[Trace]): List of traces to process.

        Returns:
            list[WaveformImage]: List of image functions.
        """
        ...

    def get_blinding(self, sampling_rate: float) -> timedelta:
        """Blinding duration for the image function. Added to padded waveforms.

        Args:
            sampling_rate (float): The sampling rate of the waveform.

        Returns:
            timedelta: The blinding duration for the image function.
        """
        raise NotImplementedError("must be implemented by subclass")

    def get_provided_phases(self) -> tuple[PhaseDescription, ...]:
        """Get the phases provided by the image function.

        Returns:
            tuple[PhaseDescription, ...]: The phases provided by the image function.
        """
        raise NotImplementedError("must be implemented by subclass")


@dataclass
class WaveformImage:
    image_function: ImageFunction
    phase: PhaseDescription
    weight: float
    traces: list[Trace]
    detection_half_width: float
    stations: Stations = Field(default_factory=lambda: Stations.model_construct())

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.delta_t

    @property
    def delta_t(self) -> float:
        return self.traces[0].deltat

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    def has_traces(self) -> bool:
        return bool(self.traces)

    def set_stations(self, stations: Stations) -> None:
        """Set stations from the image's available traces."""
        self.stations = stations.select_from_traces(self.traces)

    def resample(self, sampling_rate: float, max_normalize: bool = False) -> None:
        """Resample traces in-place.

        Args:
            sampling_rate (float): Desired sampling rate in Hz.
            max_normalize (bool): Normalize by maximum value to keep the scale of the
                maximum detection. Defaults to False.
        """
        if not self.has_traces():
            return
        if self.sampling_rate == sampling_rate:
            return

        downsample = self.sampling_rate > sampling_rate

        for tr in self.traces:
            resample(tr, sampling_rate)

            if max_normalize and downsample:
                _, max_value = tr.max()
                tr.ydata /= tr.ydata.max()
                tr.ydata *= max_value

    def get_trace_data(self) -> list[np.ndarray]:
        """Get all trace data in a list.

        Returns:
            list[np.ndarray]: List of numpy arrays.
        """
        return [tr.ydata for tr in self.traces if tr.ydata is not None]

    def get_offsets(self, reference: datetime) -> np.ndarray:
        """Get traces timing offsets to a reference time in samples.

        Args:
            reference (datetime): Reference time.

        Returns:
            np.ndarray: Integer offset towards the reference for each trace.
        """
        trace_tmins = np.fromiter((tr.tmin for tr in self.traces), float)
        return np.round((trace_tmins - reference.timestamp()) / self.delta_t).astype(
            np.int32
        )

    def apply_exponent(self, exponent: float) -> None:
        """Apply exponent to all traces.

        Args:
            exponent (float): Exponent to apply.
        """
        if exponent == 1.0:
            return
        for tr in self.traces:
            tr.ydata **= exponent

    def search_phase_arrival(
        self,
        trace_idx: int,
        event_time: datetime,
        modelled_arrival: datetime,
        search_window_seconds: float = 5,
        threshold: float = 0.1,
    ) -> ObservedArrival | None:
        """Search for a peak in all station's image functions.

        Args:
            trace_idx (int): Index of the trace.
            event_time (datetime): Time of the event.
            modelled_arrival (datetime): Time to search around.
            search_window_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.

        Returns:
            datetime | None: Time of arrival, None is none found.
        """
        raise NotImplementedError

    def search_phase_arrivals(
        self,
        event_time: datetime,
        modelled_arrivals: list[datetime | None],
        search_window_seconds: float = 5.0,
        threshold: float = 0.1,
    ) -> list[ObservedArrival | None]:
        """Search for a peak in all station's image functions.

        Args:
            event_time (datetime): Time of the event.
            modelled_arrivals (list[datetime]): Time to search around.
            search_window_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.

        Returns:
            list[datetime | None]: List of arrivals, None is none found.
        """
        return [
            self.search_phase_arrival(
                idx,
                event_time,
                modelled_arrival,
                search_window_seconds=search_window_seconds,
                threshold=threshold,
            )
            if modelled_arrival
            else None
            for idx, modelled_arrival in zip(
                range(self.n_traces),
                modelled_arrivals,
                strict=True,
            )
        ]

    async def save_mseed(self, path: Path) -> None:
        """Save the image traces to disk.

        Args:
            path (Path): Path to save the traces.
        """
        save_traces = [tr.copy() for tr in self.traces]
        for tr in save_traces:
            tr.set_ydata((tr.ydata * 1e3).astype(np.int32))
        await asyncio.to_thread(
            save,
            save_traces,
            f"{path!s}/{SDS_PYROCKO_SCHEME}",
            append=True,
        )

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        snuffle(self.traces)
