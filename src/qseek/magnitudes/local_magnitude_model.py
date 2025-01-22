from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Type

import numpy as np
from pyrocko.trace import PoleZeroResponse

from qseek.magnitudes.base import PeakMeasurement, StationAmplitudes
from qseek.utils import NSL, ChannelSelector, ChannelSelectors, MeasurementUnit

if TYPE_CHECKING:
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver

logger = logging.getLogger(__name__)

KM = 1e3
MM = 1e3
UM = 1e6
M2NM = 1e9

Component = Literal[
    "all",
    "all-abs",
    "horizontal-abs",
    "horizontal-avg",
    "vertical",
    "north-east-separate",
]

MaxAmplitudeQuantity = Literal[
    "acceleration",
    "velocity",
    "displacement",
    "wood-anderson",
    "wood-anderson-old",
]
# All models from Bormann (2012) https://doi.org/10.2312/GFZ.NMSOP-2_DS_3.1
# Page 5

_COMPONENT_MAP: dict[Component, ChannelSelector] = {
    "all": ChannelSelectors.All,
    "all-abs": ChannelSelectors.AllAbsolute,
    "horizontal-abs": ChannelSelectors.HorizontalAbs,
    "horizontal-avg": ChannelSelectors.HorizontalAvg,
    "vertical": ChannelSelectors.Vertical,
    "north-east-separate": ChannelSelectors.NorthEast,
}

_QUANTITY_MAP: dict[MaxAmplitudeQuantity, MeasurementUnit] = {
    "acceleration": "acceleration",
    "velocity": "velocity",
    "displacement": "displacement",
    "wood-anderson": "displacement",
    "wood-anderson-old": "displacement",
}

# New corrections from NMSOP 3.3
# 10.2312/GFZ.NMSOP-2_IS_3.3, Page 5
WOOD_ANDERSON = PoleZeroResponse(
    poles=[
        -5.49779 + 5.60886j,
        -5.49779 - 5.60886j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

WOOD_ANDERSON_OLD = PoleZeroResponse(
    poles=[
        -6.283 + 4.7124j,
        -6.283 - 4.7124j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2080.0,
)

WOOD_ANDERSON_PIDSA = PoleZeroResponse(
    poles=[
        -6.283 + 4.7124j,
        -6.283 - 4.7124j,
    ],
    zeros=[0.0 + 0.0j, 0.0 + 0.0j],
    constant=2800.0,
)


class Range(NamedTuple):
    min: float
    max: float

    def inside(self, value: float) -> bool:
        return self.min <= value <= self.max


class StationLocalMagnitude(NamedTuple):
    station: NSL
    magnitude: float
    magnitude_error: float
    peak_amp: float
    distance_epi: float
    distance_hypo: float
    snr: float = 0.0


class LocalMagnitudeModel:
    epicentral_range: ClassVar[Range | None] = None
    hypocentral_range: ClassVar[Range | None] = None

    max_amplitude: ClassVar[MaxAmplitudeQuantity] = "wood-anderson"

    author: ClassVar[str] = "Unknown"
    doi: ClassVar[str] = ""
    component: ClassVar[Component] = "vertical"
    peak_measurement: ClassVar[PeakMeasurement] = "peak-to-peak"

    # Used only for non-Wood-Anderson models
    highpass_freq: ClassVar[float | None] = None
    lowpass_freq: ClassVar[float | None] = None

    @classmethod
    def get_subclass_by_name(cls, name: str) -> Type[LocalMagnitudeModel]:
        subclasses = {sub.model_name(): sub for sub in cls.__subclasses__()}
        if name in subclasses:
            return subclasses[name]
        raise ValueError(f"Model {name} not found. Choose from {', '.join(subclasses)}")

    @classmethod
    def model_name(cls) -> str:
        """Get the name of the model in kebab case."""
        return re.sub(r"(?<!^)(?=[A-Z])", "-", cls.__name__).lower()

    @classmethod
    def model_names(cls) -> tuple[str, ...]:
        return tuple(sub.model_name() for sub in cls.__subclasses__())

    @property
    def restitution_quantity(self) -> MeasurementUnit:
        return _QUANTITY_MAP[self.max_amplitude]

    def get_magnitude(
        self, amplitude: float, distance_hypo: float, distance_epi: float
    ) -> float:
        raise NotImplementedError

    def get_station_magnitude(
        self,
        event: EventDetection,
        receiver: Receiver,
        traces: list[Trace],
        min_snr: float = 3.0,
    ) -> StationLocalMagnitude | None:
        """Calculate the local magnitude for a given event and receiver.

        Args:
            event (EventDetection): The event to calculate the magnitude for.
            receiver (Receiver): The seismic station to calculate the magnitude for.
            traces (list[Trace]): The traces to calculate the magnitude for.
            min_snr (float, optional): The minimum amplitude to noise ratio.
                Defaults to 3.0.

        Returns:
            StationMagnitude | None: The calculated magnitude or None if the magnitude.
        """
        if self.epicentral_range and not self.epicentral_range.inside(
            receiver.surface_distance_to(event)
        ):
            return None
        if self.hypocentral_range and not self.hypocentral_range.inside(
            receiver.distance_to(event)
        ):
            return None

        try:
            traces = _COMPONENT_MAP[self.component](traces)
        except (KeyError, AttributeError) as exc:
            logger.warning(
                "Could get all required channels for %s: %s",
                receiver.nsl.pretty,
                str(exc),
            )
            return None
        if not traces:
            return None

        sta = StationAmplitudes.create(
            receiver=receiver,
            traces=traces,
            event=event,
            measurement=self.peak_measurement,
        )
        if sta.snr < min_snr:
            return None

        with np.errstate(divide="ignore"):
            magnitude = self.get_magnitude(
                sta.peak, sta.distance_hypo, sta.distance_epi
            )
            magnitude_error_upper = self.get_magnitude(
                sta.peak + sta.noise, sta.distance_hypo, sta.distance_epi
            )
            magnitude_error_lower = self.get_magnitude(
                sta.peak - sta.noise, sta.distance_hypo, sta.distance_epi
            )

        if not np.isfinite(magnitude):
            return None

        if not np.isfinite(magnitude_error_lower):
            magnitude_error_lower = magnitude - (magnitude_error_upper - magnitude)

        return StationLocalMagnitude(
            station=sta.station_nsl,
            magnitude=magnitude,
            magnitude_error=(magnitude_error_upper + abs(magnitude_error_lower)) / 2,
            peak_amp=sta.peak,
            snr=sta.snr,
            distance_epi=sta.distance_epi,
            distance_hypo=sta.distance_hypo,
        )


class WebnetWesternBohemia(LocalMagnitudeModel):
    author = "Horálek et al. (2000)"
    doi = "10.1023/A:1022198406514"

    hypocentral_range = Range(0.0 * KM, 100.0 * KM)
    component = "all-abs"
    max_amplitude = "velocity"

    peak_measurement = "max-amplitude"
    highpass_freq = 1.0

    def get_magnitude(
        self,
        amplitude: float,
        distance_hypo: float,
        distance_epi: float,
    ) -> float:
        return (
            np.log10(amplitude * UM)
            - np.log10(np.pi)
            + 2.1 * np.log10(distance_hypo / KM)
            - 1.7
        )


class WoodAnderson:
    max_amplitude: ClassVar[MaxAmplitudeQuantity] = "wood-anderson"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        """Get the amplitude attenuation for a given distance, the `-log10(A_0)`term."""
        raise NotImplementedError

    def get_magnitude(
        self,
        amplitude: float,
        distance_hypo: float,
        distance_epi: float,
    ) -> float:
        """Get the magnitude from a given amplitude."""
        amplitude = amplitude * MM  # m to mm
        return np.log10(amplitude) + self.get_amp_attenuation(
            distance_hypo / KM, distance_epi / KM
        )


class SouthernCalifornia(WoodAnderson, LocalMagnitudeModel):
    author = "Hutton and Boore (1987)"

    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "north-east-separate"
    max_amplitude = "wood-anderson-old"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km / 100) + 0.00189 * (dist_hypo_km - 100) + 3


class CentralCalifornia(WoodAnderson, LocalMagnitudeModel):
    author = "Bakun and Joyner (1984)"
    doi = "10.1785/BSSA0740051827"

    epicentral_range = Range(0.0 * KM, 400.0 * KM)
    component = "horizontal-avg"
    max_amplitude = "wood-anderson-old"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return np.log10(dist_hypo_km) + 0.00301 * dist_hypo_km + 0.7


class IaspeiSouthernCalifornia(WoodAnderson, LocalMagnitudeModel):
    author = "Hutton and Boore (1987)"

    hypocentral_range = Range(10.0 * KM, 700.0 * KM)
    component = "north-east-separate"
    max_amplitude = "wood-anderson-old"

    def get_magnitude(
        self, amplitude: float, distance_hypo: float, distance_epi: float
    ) -> float:
        amp = amplitude * M2NM  # m to nm
        return (
            np.log10(amp / WOOD_ANDERSON.constant)
            + 1.11 * np.log10(distance_hypo / KM)
            + 0.00189 * (distance_hypo / KM)
            - 2.09
        )


class EasternNorthAmerica(WoodAnderson, LocalMagnitudeModel):
    author = "Kim (1998)"

    epicentral_range = Range(100.0 * KM, 800.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.55 * np.log10(dist_epi_km) - 0.22


class Albania(WoodAnderson, LocalMagnitudeModel):
    author = "Muco and Minga (1991)"

    epicentral_range = Range(10.0 * KM, 600.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.6627 * np.log10(dist_epi_km) + 0.0008 * dist_epi_km - 0.433


class SouthWestGermany(WoodAnderson, LocalMagnitudeModel):
    author = "Stange (2006)"

    hypocentral_range = Range(10.0 * KM, 1000.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.11 * np.log10(dist_hypo_km) + 0.95 * dist_hypo_km * 1e-3 + 0.69


class SouthAustralia(WoodAnderson, LocalMagnitudeModel):
    author = "Greenhalgh and Singh (1986)"

    epicentral_range = Range(40.0 * KM, 700.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1 * np.log10(dist_epi_km) + 0.0013 * dist_epi_km + 0.7


class NorwayFennoscandia(WoodAnderson, LocalMagnitudeModel):
    author = "Alsaker et al. (1991)"

    hypocentral_range = Range(0.0 * KM, 1500.0 * KM)
    component = "vertical"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.91 * np.log10(dist_hypo_km) + 0.00087 * dist_hypo_km + 1.01


class IcelandAskja(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.4406 * np.log10(dist_hypo_km / 17) + 0.003 * (dist_hypo_km - 17) + 2


class IcelandBardabunga(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.2534 * np.log10(dist_hypo_km / 17) - 0.0032 * (dist_hypo_km - 17) + 2


class IcelandAskjaBardabungaCombined(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2020)"

    hypocentral_range = Range(0.0 * KM, 150.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.1999 * np.log10(dist_hypo_km / 17) - 0.0016 * (dist_hypo_km - 17) + 2


class IcelandReykjanes(WoodAnderson, LocalMagnitudeModel):
    author = "Greenfield et al. (2022)"

    hypocentral_range = Range(0.0 * KM, 40.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.6902 * np.log10(dist_hypo_km / 17) + 0.0318 * (dist_hypo_km - 17) + 2


class Azores(WoodAnderson, LocalMagnitudeModel):
    author = "Gongora et al. (2004)"

    epicentral_range = Range(10.0 * KM, 800.0 * KM)
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 0.89 * np.log10(dist_epi_km / 100) + 0.00256 * (dist_epi_km - 100) + 3


class ArgentinaVolcanoes(WoodAnderson, LocalMagnitudeModel):
    author = "Montenegro et al. (2021)"

    epicentral_range = Range(0.0 * KM, 100.0 * KM)  # Bounds are not clear
    component = "north-east-separate"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 2.76 * np.log10(dist_epi_km) - 2.48


class NetherlandsGroningen(WoodAnderson, LocalMagnitudeModel):
    author = "Dost et al. (2018)"

    epicentral_range = Range(0.0 * KM, 80.0 * KM)
    component = "horizontal-avg"

    @staticmethod
    def get_amp_attenuation(dist_hypo_km: float, dist_epi_km: float) -> float:
        return 1.33 * np.log10(dist_hypo_km) + 0.00139 * dist_hypo_km + 0.424


ModelName = Literal[LocalMagnitudeModel.model_names()]
