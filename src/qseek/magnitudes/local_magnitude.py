from __future__ import annotations

import asyncio
import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, PositiveFloat, PrivateAttr, model_validator
from pyrocko import io
from typing_extensions import Self

from qseek.magnitudes.base import (
    EventMagnitude,
    EventMagnitudeCalculator,
)
from qseek.magnitudes.local_magnitude_model import (
    WOOD_ANDERSON,
    WOOD_ANDERSON_OLD,
    LocalMagnitudeModel,
    ModelName,
    StationLocalMagnitude,
)
from qseek.utils import time_to_path

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver
    from qseek.waveforms.providers import WaveformProvider

logger = logging.getLogger(__name__)

KM = 1e3
MM = 1e3


class LocalMagnitude(EventMagnitude):
    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    model: ModelName = Field(
        default="iaspei-southern-california",
        description="The estimator to use for calculating the local magnitude.",
    )
    station_magnitudes: list[StationLocalMagnitude] = []

    @classmethod
    async def from_model(
        cls,
        model: LocalMagnitudeModel,
        event: EventDetection,
        receivers: list[Receiver],
        grouped_traces: list[list[Trace]],
        min_snr: float = 3.0,
    ) -> Self:
        self = cls(model=model.model_name())

        for traces, receiver in zip(grouped_traces, receivers, strict=True):
            station_magnitude = model.get_station_magnitude(
                event=event,
                traces=traces,
                receiver=receiver,
                min_snr=min_snr,
            )
            if station_magnitude is None:
                continue
            self.station_magnitudes.append(station_magnitude)

        if not self.station_magnitudes:
            logger.warning("No station magnitudes found for event %s", event.time)
            return self

        magnitudes = np.array([sta.magnitude for sta in self.station_magnitudes])

        median = np.median(magnitudes)
        self.average = float(median)
        self.error = float(np.median(np.abs(magnitudes - median)))  # MAD
        return self

    @property
    def n_stations(self) -> int:
        return len(self.station_magnitudes)

    def csv_row(self) -> dict[str, float]:
        return {
            f"ML-{self.model}": self.average,
            f"ML-error-{self.model}": self.error,
        }


class LocalMagnitudeExtractor(EventMagnitudeCalculator):
    """Local magnitude calculator for different regional models."""

    magnitude: Literal["LocalMagnitude"] = "LocalMagnitude"

    noise_window: PositiveFloat = Field(
        default=5.0,
        ge=1.5,
        description="Waveforms to extract before P phase arrival. The noise amplitude "
        "is extracted from before the P phase arrival, with 0.5 s padding.",
    )
    seconds_after: PositiveFloat = Field(
        default=4.0,
        description="Waveforms to extract after S phase arrival.",
    )
    taper_seconds: PositiveFloat = Field(
        default=10.0,
        description="Seconds tapering before and after the extraction window."
        " The taper stabalizes the restitution and is cut off from the traces "
        "before the analysis.",
    )
    min_signal_noise_ratio: float = Field(
        default=1.5,
        ge=1.0,
        description="Minimum signal-to-noise ratio for the local magnitude estimation. "
        "The noise amplitude is extracted from before the P phase arrival,"
        " with 0.5 s padding.",
    )

    model: ModelName = Field(
        default="iaspei-southern-california",
        description="The estimator to use for calculating the local magnitude.",
    )

    export_mseed: Path | None = Field(
        default=None,
        description="Path to export the processed mseed traces to.",
    )

    _model: LocalMagnitudeModel = PrivateAttr()

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        self._model = LocalMagnitudeModel.get_subclass_by_name(self.model)()
        return self

    def has_magnitude(self, event: EventDetection) -> bool:
        for mag in event.magnitudes:
            if type(mag) is LocalMagnitude and mag.model == self.model:
                return True
        return False

    async def add_magnitude(
        self, waveform_provider: WaveformProvider, event: EventDetection
    ) -> None:
        model = self._model

        traces = await event.receivers.get_waveforms_restituted(
            waveform_provider.get_squirrel(),
            seconds_before=self.noise_window,
            seconds_after=self.seconds_after,
            seconds_taper=self.taper_seconds,
            quantity=model.restitution_quantity,
            channels=waveform_provider.channel_selector,
            phase=None,
            cut_off_taper=False,
            filter_clipped=True,
        )
        if not traces:
            logger.warning("No restituted traces found for event %s", event.time)
            return

        if model.max_amplitude == "wood-anderson":
            traces = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        tr.transfer,
                        transfer_function=WOOD_ANDERSON,
                        freqlimits=(0.1, 1.0, 0.40 / tr.deltat, 0.45 / tr.deltat),
                        tfade=self.taper_seconds,
                        cut_off_fading=False,
                        demean=True,
                        invert=False,
                    )
                    for tr in traces
                ]
            )
        elif model.max_amplitude == "wood-anderson-old":
            traces = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        tr.transfer,
                        transfer_function=WOOD_ANDERSON_OLD,
                        freqlimits=(0.1, 1.0, 0.40 / tr.deltat, 0.45 / tr.deltat),
                        tfade=self.taper_seconds,
                        cut_off_fading=False,
                        demean=True,
                        invert=False,
                    )
                    for tr in traces
                ]
            )

        if model.highpass_freq is not None:
            for tr in traces:
                await asyncio.to_thread(
                    tr.highpass, order=4, corner=model.highpass_freq
                )
        if model.lowpass_freq is not None:
            for tr in traces:
                await asyncio.to_thread(tr.lowpass, order=4, corner=model.lowpass_freq)

        for tr in traces:
            await asyncio.to_thread(
                tr.chop,
                tr.tmin + self.taper_seconds,
                tr.tmax - self.taper_seconds,
            )

        if self.export_mseed is not None:
            file_name = self.export_mseed / f"{time_to_path(event.time)}.mseed"
            logger.debug("saving restituted mseed traces to %s", file_name)
            await asyncio.to_thread(io.save, traces, str(file_name))

        grouped_traces = []
        receivers = []
        for nsl, grp_traces in itertools.groupby(traces, key=lambda tr: tr.nslc_id[:3]):
            grouped_traces.append(list(grp_traces))
            receivers.append(event.receivers.get_receiver(nsl))

        local_magnitude = await LocalMagnitude.from_model(
            model=model,
            grouped_traces=grouped_traces,
            receivers=receivers,
            event=event,
            min_snr=self.min_signal_noise_ratio,
        )

        event.add_magnitude(local_magnitude)
