from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from lassie.features_receiver.base import ReceiverFeature, ReceiverFeatureExtractor
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from lassie.models.detection import PhaseDetection


class AmplitudeMeasurement(ReceiverFeature):
    feature: Literal["AmplitudeMeasurement"] = "AmplitudeMeasurement"
    seconds_before: float
    seconds_after: float
    vertical_counts: int
    horizontal_counts: int


@dataclass
class Selector:
    channels: str
    number_channels: int


class Selectors:
    Horizontal = Selector("EN23", 2)
    Vertical = Selector("Z0", 1)


class WaveformAmplitudes(ReceiverFeatureExtractor):
    feature: Literal["WaveformAmplitudes"] = "WaveformAmplitudes"
    phase: PhaseDescription = "cake:S"
    seconds_before: float = 3.0
    seconds_after: float = 3.0

    def _get_maximum(self, traces: list[Trace], selector: Selector) -> float:
        horizontal_traces = [tr for tr in traces if tr.channel[-1] in selector.channels]
        if len(horizontal_traces) != selector.number_channels:
            raise KeyError("cannot get two horizontal channels")

        data = np.array([tr.ydata for tr in horizontal_traces])
        norm_traces = np.linalg.norm(data, axis=0)
        return float(np.abs(norm_traces).max())

    async def get_features(
        self,
        squirrel: Squirrel,
        phase_detection: PhaseDetection,
    ) -> list[AmplitudeMeasurement | None]:
        if not phase_detection.phase == self.phase:
            raise KeyError(f"Bad phase {phase_detection.phase}")

        features = []
        for receiver in phase_detection.receivers:
            traces = receiver.get_waveforms(
                squirrel,
                seconds_after=self.seconds_after,
                seconds_before=self.seconds_before,
            )
            try:
                horizontal_counts = self._get_maximum(traces, Selectors.Horizontal)
                vertical_counts = self._get_maximum(traces, Selectors.Vertical)
                amplitudes = AmplitudeMeasurement(
                    seconds_before=self.seconds_before,
                    seconds_after=self.seconds_after,
                    horizontal_counts=int(round(horizontal_counts)),
                    vertical_counts=int(round(vertical_counts)),
                )
            except KeyError:
                amplitudes = None
            features.append(amplitudes)
        return features
