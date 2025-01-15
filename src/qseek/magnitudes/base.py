from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field
from pyrocko.trace import Trace
from typing_extensions import Self

from qseek.utils import NSL

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection, Receiver
    from qseek.models.station import Stations
    from qseek.octree import Octree

KM = 1e3

PeakMeasurement = Literal["peak-to-peak", "max-amplitude"]


class EventMagnitude(BaseModel):
    magnitude: Literal["EventMagnitude"] = "EventMagnitude"

    average: float = Field(
        default=0.0,
        description="Average local magnitude.",
    )
    median: float = Field(
        default=0.0,
        description="Average local magnitude.",
    )
    error: float = Field(
        default=0.0,
        description="Average error of local magnitude.",
    )

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitude], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def csv_row(self) -> dict[str, float]:
        return {
            "magnitude": self.average,
            "error": self.error,
        }

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        station_distances_hypo = np.array(
            [sta.distance_hypo for sta in self.station_magnitudes]
        )

        fig = plt.figure()
        ax = fig.gca()
        ax.errorbar(
            station_distances_hypo,
            self.magnitudes,
            yerr=[sta.magnitude_error for sta in self.station_magnitudes],
            marker="o",
            mec="k",
            mfc="k",
            ms=2,
            ecolor=(0.0, 0.0, 0.0, 0.1),
            capsize=1,
            ls="none",
        )
        ax.axhline(
            self.average,
            color="k",
            linestyle="dotted",
            alpha=0.5,
            label=rf"Median $M_L$ {self.average:.2f} $\pm${self.error:.2f}",
        )
        ax.set_xlabel("Distance to Hypocenter [km]")
        ax.set_ylabel("$M_L$")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x / KM))
        ax.grid(alpha=0.3)
        ax.legend(title=f"Estimator: {self.model}", loc="lower right")
        ax.text(
            0.05,
            0.05,
            f"{self.n_observations} Stations",
            transform=ax.transAxes,
            alpha=0.5,
        )
        plt.show()


class EventMagnitudeCalculator(BaseModel):
    magnitude: Literal["MagnitudeCalculator"] = "MagnitudeCalculator"

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitudeCalculator], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    def has_magnitude(self, event: EventDetection) -> bool:
        """Check if the given event has a magnitude.

        Args:
            event (EventDetection): The event to check.

        Returns:
            bool: True if the event has a magnitude, False otherwise.
        """
        raise NotImplementedError

    async def add_magnitude(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        """Adds a magnitude to the squirrel for the given event.

        Args:
            squirrel (Squirrel): The squirrel object to add the magnitude to.
            event (EventDetection): The event detection object.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
    ) -> None:
        """Prepare the magnitudes calculation by initializing necessary data structures.

        Args:
            octree (Octree): The octree containing seismic event data.
            stations (Stations): The stations containing seismic station data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        ...


class StationAmplitudes(NamedTuple):
    station_nsl: NSL
    peak: float
    noise: float
    std_noise: float

    distance_epi: float
    distance_hypo: float

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio."""
        if self.noise == 0.0:
            return 0.0
        return self.peak / self.noise

    @classmethod
    def create(
        cls,
        receiver: Receiver,
        traces: list[Trace],
        event: EventDetection,
        noise_padding: float = 0.5,
        measurement: PeakMeasurement = "max-amplitude",
    ) -> Self:
        time_arrival = min(receiver.get_arrivals_time_window()).timestamp()

        noise_traces = [
            tr.chop(tmin=tr.tmin, tmax=time_arrival - noise_padding, inplace=False)
            for tr in traces
        ]

        if measurement == "peak-to-peak":
            peak_amp = max(np.ptp(tr.ydata) / 2 for tr in traces)
            noise_amp = max(np.ptp(tr.ydata) / 2 for tr in noise_traces)
        elif measurement == "max-amplitude":
            peak_amp = max(np.max(np.abs(tr.ydata)) for tr in traces)
            noise_amp = max(np.max(np.abs(tr.ydata)) for tr in noise_traces)
        else:
            raise ValueError(f"Invalid measurement: {measurement}")
        std_noise = max(np.std(tr.ydata) for tr in noise_traces)

        return cls(
            station_nsl=receiver.nsl,
            peak=peak_amp,
            noise=noise_amp,
            std_noise=std_noise,
            distance_hypo=receiver.distance_to(event),
            distance_epi=receiver.surface_distance_to(event),
        )
