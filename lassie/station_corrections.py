from __future__ import annotations

import logging
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, cast, get_args
from uuid import UUID, uuid4

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Extra, Field

from lassie.models.detection import EventDetection, PhaseDetection, Receiver
from lassie.models.location import Location
from lassie.models.station import Station
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from typing_extensions import Self

    from lassie.models.detection import Detections

logger = logging.getLogger(__name__)
KM = 1e3

ArrivalWeighting = Literal[
    "none",
    "PhaseNet",
    "semblance",
    "add-PhaseNet-semblance",
    "mul-PhaseNet-semblance",
]


class EventCorrection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    semblance: float
    distance_border: float

    phase_arrivals: dict[PhaseDescription, PhaseDetection] = {}

    def phases(self) -> tuple[PhaseDescription, ...]:
        return tuple(self.phase_arrivals.keys())


def weighted_median(data: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Calculate the weighted median of an array/list using numpy."""
    if not weights:
        return float(np.median(data))

    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(
        np.array, zip(*sorted(zip(data, weights, strict=True)), strict=True)
    )
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return float(w_median)


class StationCorrection(BaseModel):
    station: Station
    events: list[EventCorrection] = []

    class Config:
        extra = Extra.ignore

    def add_event(self, event: EventCorrection) -> None:
        self.events.append(event)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def phases(self) -> set[PhaseDescription]:
        return set(chain.from_iterable(event.phases() for event in self.events))

    def get_num_picks(self, phase: PhaseDescription) -> int:
        return sum(1 for _ in self.iter_phase_arrivals(phase=phase))

    def get_traveltime_delays(self, phase: PhaseDescription) -> np.ndarray:
        return np.fromiter(
            (
                phase.traveltime_delay.total_seconds()
                for phase in self.iter_phase_arrivals(phase=phase)
                if phase.traveltime_delay is not None
            ),
            dtype=float,
        )

    def get_event_semblances(self, phase: PhaseDescription) -> np.ndarray:
        return np.fromiter(
            (event.semblance for event in self.iter_events(phase=phase)),
            dtype=float,
        )

    def get_detection_values(self, phase: PhaseDescription) -> np.ndarray:
        return np.fromiter(
            (
                phase.observed.detection_value
                for phase in self.iter_phase_arrivals(phase=phase)
            ),
            dtype=float,
        )

    def get_arrival_weights(
        self, phase: PhaseDescription, weight: ArrivalWeighting = "none"
    ) -> np.ndarray | None:
        weights = None
        if weight == "PhaseNet":
            weights = self.get_detection_values(phase)
        elif weight == "semblance":
            weights = self.get_event_semblances(phase)
        elif weight == "add-PhaseNet-semblance":
            detections = self.get_detection_values(phase)
            semblances = self.get_event_semblances(phase)
            weights = detections / detections.max() + semblances / semblances.max()
        elif weight == "mul-PhaseNet-semblance":
            weights = self.get_detection_values(phase) * self.get_event_semblances(
                phase
            )
        return weights

    def get_average_delay(
        self,
        phase: PhaseDescription,
        weight: ArrivalWeighting = "none",
    ) -> float:
        """Average delay times. Weighted and unweighted.

        Args:
            phase (PhaseDescription): Name of the phase
            weighted (ArrivalWeighting, optional): Weights, choose from "none",
                "PhaseNet" and "PhaseNet+EventSemblance". Defaults to "none".

        Returns:
            float: Delay time.
        """
        traveltime_delays = self.get_traveltime_delays(phase)
        if not traveltime_delays.size:
            return 0.0
        return float(
            np.average(
                traveltime_delays,
                weights=self.get_arrival_weights(phase, weight),
            )
        )

    def get_median_delay(
        self,
        phase: PhaseDescription,
        weight: ArrivalWeighting = "none",
    ) -> float:
        """Average delay times. Weighted and unweighted.

        Args:
            phase (PhaseDescription): Name of the phase
            weighted (ArrivalWeighting, optional): Weights, choose from "none",
                "PhaseNet" and "PhaseNet+EventSemblance". Defaults to "none".

        Returns:
            float: Delay time.
        """
        traveltime_delays = self.get_traveltime_delays(phase)
        if not traveltime_delays.size:
            return 0.0
        return weighted_median(
            traveltime_delays,
            weights=self.get_arrival_weights(phase, weight),
        )

    def get_delays_std(self, phase: PhaseDescription) -> float:
        return float(np.std(self.get_traveltime_delays(phase)))

    def iter_events(
        self, phase: PhaseDescription, observed_only: bool = True
    ) -> Iterable[EventCorrection]:
        for event in self.events:
            phase_detection = event.phase_arrivals.get(phase)
            if not phase_detection:
                continue
            if observed_only and not phase_detection.observed:
                continue
            yield event

    def iter_phase_arrivals(
        self, phase: PhaseDescription, observed_only: bool = True
    ) -> Iterable[PhaseDetection]:
        for event in self.iter_events(phase, observed_only=observed_only):
            yield event.phase_arrivals[phase]

    def get_csv_data(self) -> dict[str, float]:
        station = self.station
        data = {"lat": station.effective_lat, "lon": station.effective_lon}
        average = self.get_average_delay
        median = self.get_median_delay
        for phase in self.phases:
            for aggregator, weight_name in zip(
                (average, median), ("avg", "median"), strict=True
            ):
                for weight in get_args(ArrivalWeighting):
                    data[f"{phase}-{weight_name}-{weight}"] = aggregator(phase, weight)
            data[f"{phase}-num-picks"] = self.get_num_picks(phase)
        return data

    def plot(self, filename: Path) -> None:
        phases = self.phases
        n_phases = len(phases)
        arrival_weights = get_args(ArrivalWeighting)

        fig, axes = plt.subplots(len(arrival_weights), n_phases)
        fig.set_size_inches(8, 12)
        axes = axes.T

        def plot_histogram(
            ax: plt.Axes, phase: PhaseDescription, weight: ArrivalWeighting
        ) -> None:
            delays = self.get_traveltime_delays(phase=phase)
            n_delays = len(delays)
            if not n_delays:
                return
            weights = self.get_arrival_weights(phase, weight=weight)
            ax.hist(delays, weights=weights)
            ax.axvline(
                self.get_average_delay(phase, weight=weight),
                ls="--",
                c="k",
                label="mean",
            )
            ax.axvline(
                self.get_median_delay(phase, weight=weight),
                ls=":",
                c="k",
                label="median",
            )
            ax.text(
                0.05,
                0.95,
                f"{weight}\n{phase} ({n_delays} picks)",
                fontsize="small",
                transform=ax.transAxes,
                va="top",
            )

        for ax_col, phase in zip(axes, phases, strict=True):
            for ax, weight in zip(ax_col, arrival_weights, strict=True):
                plot_histogram(ax, phase, weight)

            for ax in ax_col:
                ax = cast(plt.Axes, ax)
                ax.axvline(0.0, c="k", alpha=0.5)
                xlim = max(np.abs(ax.get_xlim()))
                ax.set_xlim(-xlim, xlim)
                ax.grid(alpha=0.4)

            ax_col[0].legend(fontsize="small")
            ax_col[-1].set_xlabel("Time Residual [s]")

        logger.info("saving residual plot to %s", filename)
        fig.tight_layout()
        fig.savefig(str(filename))
        plt.close()

    @classmethod
    def from_receiver(cls, receiver: Receiver) -> Self:
        return cls(station=Station.parse_obj(receiver))


class StationCorrections(BaseModel):
    station_corrections: dict[tuple[str, str, str], StationCorrection] = {}

    def load_detections(self, detections: Detections) -> None:
        for event in detections:
            self.add_event(event)

    def add_event(self, event: EventDetection) -> None:
        # FIXME: use event.in_bounds
        if event.distance_border < KM:
            return

        logger.info("loading event %s", event.time)
        for receiver in event.receivers:
            try:
                sta_correction = self.get_station_correction(receiver.nsl)
            except KeyError:
                sta_correction = StationCorrection.from_receiver(receiver)
                self.station_corrections[receiver.nsl] = sta_correction

            sta_correction.add_event(
                EventCorrection.construct(
                    phase_arrivals=receiver.phase_arrivals,
                    **event.dict(),
                )
            )

    def get_station_correction(self, nsl: tuple[str, str, str]) -> StationCorrection:
        return self.station_corrections[nsl]

    def get_correction(
        self, phase: PhaseDescription, station_nsl: tuple[str, str, str]
    ) -> float:
        ...

    def save_plots(self, folder: Path) -> None:
        folder.mkdir(exist_ok=True)
        for correction in self.station_corrections.values():
            correction.plot(
                filename=folder / f"corrections-{correction.station.pretty_nsl}.png"
            )

    def save_csv(self, filename: Path) -> None:
        logger.info("writing corrections to %s", filename)
        csv_data = [correction.get_csv_data() for correction in self]
        columns = set(chain.from_iterable(data.keys() for data in csv_data))
        with filename.open("w") as file:
            file.write(f"{', '.join(columns)}\n")
            for data in csv_data:
                file.write(
                    f"{', '.join(str(data.get(key, -9999.9)) for key in columns)}\n"
                )

    def __iter__(self) -> Iterator[StationCorrection]:
        return iter(self.station_corrections.values())

    @classmethod
    def from_detections(cls, detections: Detections) -> Self:
        logger.info("loading detections")
        station_corrections = cls()
        station_corrections.load_detections(detections=detections)
        return station_corrections