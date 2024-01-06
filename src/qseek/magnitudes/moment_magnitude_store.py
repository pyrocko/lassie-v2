from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import struct
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, NamedTuple, get_args
from uuid import UUID, uuid4

import numpy as np
import pyrocko.moment_tensor as pmt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PrivateAttr,
    ValidationError,
)
from pyrocko import gf
from rich.progress import track
from typing_extensions import Self

from qseek.utils import (
    ChannelSelector,
    ChannelSelectors,
    MeasurementUnit,
    Range,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pyrocko.cake import LayeredModel
    from pyrocko.trace import Trace

KM = 1e3
NM = 1e-9

logger = logging.getLogger(__name__)

PeakAmplitude = Literal["horizontal", "vertical", "absolute"]
Interpolation = Literal["nearest_neighbor", "multilinear"]

Components = Literal["Z", "R", "T"]
_DIPS: dict[Components, float] = {"Z": -90.0, "R": 0.0, "T": 0.0}
_AZIMUTHS: dict[Components, float] = {"Z": 0.0, "R": 0.0, "T": 0.0}


def _get_dip(component: Components) -> float:
    return _DIPS[component]


def _get_azimuth(component: Components) -> float:
    azi = _AZIMUTHS[component]
    if component == "T":
        azi += 90.0
    return azi


def _get_target(targets: list[gf.Target], nsl: tuple[str, str, str]) -> gf.Target:
    for target in targets:
        if nsl == target.codes[:3]:
            return target
    raise KeyError(f"No target for {nsl}.")


def trace_amplitude(traces: list[Trace], channel_selector: ChannelSelector) -> float:
    """
    Normalize traces channels.

    Args:
        traces (list[Trace]): A list of traces to normalize.
        components (str): The components to normalize.

    Returns:
        Trace: The normalized trace.

    Raises:
        KeyError: If there are no traces to normalize.
    """
    trace_selection = channel_selector(traces)
    if not trace_selection:
        raise KeyError("No traces to normalize.")

    data = np.array([tr.ydata for tr in trace_selection])
    data = np.linalg.norm(data, axis=0)
    return float(data.max())


class PeakAmplitudesBase(BaseModel):
    gf_store_id: str = Field(
        default="moment_magnitude",
        description="Pyrocko Store ID for peak amplitude models.",
    )
    quantity: MeasurementUnit = Field(
        default="displacement",
        description="Quantity for the peak amplitude.",
    )
    frequency_range: Range | None = Field(
        default=None,
        description="Frequency range for the peak amplitude.",
    )
    max_distance: PositiveFloat = Field(
        default=100.0 * KM,
        description="Maximum surface distances to the source for the receivers.",
    )
    reference_magnitude: float = Field(
        default=1.0,
        ge=-1.0,
        le=8.0,
        description="Reference magnitude in Mw.",
    )
    rupture_velocities: Range = Field(
        default=(0.9, 1.0),
        description="Rupture velocity range as fraction of the shear wave velocity.",
    )
    stress_drop: Range = Field(
        default=(1.0e6, 10.0e6),
        description="Stress drop range in MPa.",
    )
    gf_interpolation: Interpolation = Field(
        default="nearest_neighbor",
        description="Interpolation method for the Pyrocko GF Store",
    )


class SiteAmplitude(NamedTuple):
    distance: float
    peak_horizontal: float
    peak_vertical: float
    peak_absolute: float

    @classmethod
    def from_traces(cls, receiver: gf.Receiver, traces: list[Trace]) -> Self:
        surface_distance = np.sqrt(receiver.north_shift**2 + receiver.east_shift**2)
        return cls(
            distance=surface_distance,
            peak_horizontal=trace_amplitude(
                traces,
                ChannelSelectors.Horizontal,
            ),
            peak_vertical=trace_amplitude(
                traces,
                ChannelSelectors.Vertical,
            ),
            peak_absolute=trace_amplitude(
                traces,
                ChannelSelectors.All,
            ),
        )


class ModelledAmplitude(NamedTuple):
    distance: float
    peak_amplitude: PeakAmplitude
    amplitude_avg: float
    amplitude_median: float
    std: float

    def combine(
        self,
        amplitude: ModelledAmplitude,
        weight: float = 1.0,
    ) -> ModelledAmplitude:
        """
        Combines with another ModelledAmplitude using a weighted average.

        Args:
            amplitude (ModelledAmplitude): The ModelledAmplitude to be combined with.
            weight (float, optional): The weight of the amplitude being combined.
                Defaults to 1.0.

        Returns:
            Self: A new instance of the ModelledAmplitude class with the combined values.

        Raises:
            ValueError: If the weight is not between 0.0 and 1.0 (inclusive).
            ValueError: If the distances of the amplitudes are different.
            ValueError: If the peak amplitudes of the amplitudes are different.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Invalid weight {weight}.")
        if self.distance != amplitude.distance:
            raise ValueError(
                f"Cannot add amplitudes with different distances "
                f"{self.distance} and {amplitude.distance}."
            )
        if self.peak_amplitude != amplitude.peak_amplitude:
            raise ValueError(
                f"Cannot add amplitudes with different peak amplitudes "
                f"{self.peak_amplitude} and {amplitude.peak_amplitude}."
            )
        inv_weight = 1.0 - weight
        return ModelledAmplitude(
            distance=self.distance,
            peak_amplitude=self.peak_amplitude,
            amplitude_avg=self.amplitude_avg * inv_weight
            + amplitude.amplitude_avg * weight,
            amplitude_median=self.amplitude_median * inv_weight
            + amplitude.amplitude_median * weight,
            std=self.std * inv_weight + amplitude.std * weight,
        )


class SiteAmplitudesCollection(BaseModel):
    source_depth: float
    quantity: MeasurementUnit
    reference_magnitude: float
    rupture_velocities: Range
    stress_drop: Range
    gf_store_id: str
    frequency_range: Range

    site_amplitudes: list[SiteAmplitude] = Field(default_factory=list)

    @staticmethod
    def _get_numpy_attribute(attribute: str) -> Callable:
        def wrapped(self) -> np.ndarray:
            return np.array([getattr(sa, attribute) for sa in self.site_amplitudes])

        return wrapped

    _distances = cached_property[np.ndarray](_get_numpy_attribute("distance"))
    _vertical = cached_property[np.ndarray](_get_numpy_attribute("peak_vertical"))
    _absolute = cached_property[np.ndarray](_get_numpy_attribute("peak_absolute"))
    _horizontal = cached_property[np.ndarray](_get_numpy_attribute("peak_horizontal"))

    def get_amplitude(
        self,
        distance: float,
        n_amplitudes: int,
        max_distance: float = 0.0,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> ModelledAmplitude:
        """
        Get the amplitudes for a given distance.

        Args:
            distance (float): The distance at which to retrieve the amplitudes.
            n_amplitudes (int): The number of amplitudes to retrieve.
            max_distance (float): The maximum distance allowed for
                the retrieved amplitudes. If 0.0, no maximum distance is applied and the
                number of amplitudes will be exactly n_amplitudes. Defaults to 0.0.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".

        Returns:
            ModelledAmplitude: The modelled amplitudes.

        Raises:
            ValueError: If there are not enough amplitudes in the specified range.
            ValueError: If the peak amplitude type is unknown.
        """
        site_distances = np.abs(self._distances - distance)
        distance_idx = np.argsort(site_distances)
        idx = distance_idx[:n_amplitudes]
        distances = site_distances[idx]
        if max_distance and distances.max() > max_distance:
            raise ValueError(f"Not enough amplitudes in range {max_distance}.")

        match peak_amplitude:
            case "horizontal":
                amplitudes = self._horizontal[idx]
            case "vertical":
                amplitudes = self._vertical[idx]
            case "absolute":
                amplitudes = self._absolute[idx]
            case _:
                raise ValueError(f"Unknown peak amplitude type {peak_amplitude}.")

        return ModelledAmplitude(
            distance=distance,
            peak_amplitude=peak_amplitude,
            amplitude_avg=amplitudes.mean(),
            amplitude_median=amplitudes.mean(),
            std=amplitudes.std(),
        )

    def fill(self, receivers: list[gf.Receiver], traces: list[list[Trace]]) -> None:
        for receiver, rcv_traces in zip(receivers, traces, strict=True):
            self.site_amplitudes.append(SiteAmplitude.from_traces(receiver, rcv_traces))
        self._clear_cache()

    def distance_range(self) -> Range:
        """
        Get the distance range of the site amplitudes.

        Returns:
            Range: The distance range.
        """
        return Range(self._distances.min(), self._distances.max())

    @property
    def n_amplitudes(self) -> int:
        """
        Get the number of amplitudes in the collection.

        Returns:
            int: The number of amplitudes.
        """
        return len(self.site_amplitudes)

    def plot(
        self,
        axes: Axes | None = None,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> None:
        from matplotlib.ticker import FuncFormatter

        if axes is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        else:
            ax = axes

        labels: dict[MeasurementUnit, str] = {
            "displacement": "u [nm]",
            "velocity": "v [nm/s]",
            "acceleration": "a [nm/s²]",
        }
        interp_amplitudes: list[ModelledAmplitude] = []
        for distance in np.arange(*self.distance_range(), 1 * KM):
            interp_amplitudes.append(
                self.get_amplitude(
                    distance=distance,
                    n_amplitudes=25,
                    peak_amplitude=peak_amplitude,
                )
            )

        interp_dists = np.array([amp.distance for amp in interp_amplitudes])
        interp_amps = np.array([amp.amplitude_median for amp in interp_amplitudes])
        interp_std = np.array([amp.std for amp in interp_amplitudes])

        site_amplitudes = getattr(self, f"_{peak_amplitude.replace('peak_', '')}")
        dynamic = Range.from_list(site_amplitudes)

        ax.scatter(
            self._distances,
            site_amplitudes / NM,
            marker="o",
            c="k",
            s=2.0,
            alpha=0.05,
        )
        ax.scatter(
            interp_dists,
            interp_amps / NM,
            marker="o",
            c="forestgreen",
            s=6.0,
            alpha=1.0,
        )
        ax.fill_between(
            interp_dists,
            (interp_amps - interp_std) / NM,
            (interp_amps + interp_std) / NM,
            alpha=0.1,
            color="forestgreen",
        )

        ax.set_xlabel("Distance [km]")
        ax.set_ylabel(labels[self.quantity])
        ax.set_yscale("log")
        ax.text(
            0.025,
            0.025,
            rf"n={self.n_amplitudes}\n"
            rf"$M_w^r$={self.reference_magnitude}\n"
            rf"$z$={self.source_depth / KM} km\n"
            rf"$v_r$=[{self.rupture_velocities.min}, {self.rupture_velocities.max}]"
            r" $\cdot v_s$\n"
            rf"$\Delta\sigma$=[{self.stress_drop.min / 1e6},"
            rf" {self.stress_drop.max / 1e6}] MPa\n"
            rf"$f$=[{self.frequency_range.min}, {self.frequency_range.max}] Hz",
            alpha=0.5,
            transform=ax.transAxes,
            va="bottom",
            fontsize="small",
        )
        ax.text(
            0.95,
            0.95,
            f"Measure: {peak_amplitude}\n"
            f"Dynamic: {(dynamic.max - dynamic.min) / NM:g}",
            alpha=0.5,
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.grid(alpha=0.1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: x / KM))
        if axes is None:
            plt.show()

    def _clear_cache(self) -> None:
        self.__dict__.pop("_distances", None)
        self.__dict__.pop("_horizontal", None)
        self.__dict__.pop("_vertical", None)
        self.__dict__.pop("_absolute", None)


class PeakAmplitudesStore(PeakAmplitudesBase):
    uuid: UUID = Field(
        default_factory=uuid4,
        description="Unique ID of the amplitude store.",
    )
    site_amplitudes: list[SiteAmplitudesCollection] = Field(
        default_factory=list,
        description="Site amplitudes per source depth.",
    )
    frequency_range: Range = Field(
        ...,
        description="Frequency range for the peak amplitude.",
    )

    _rng: np.random.Generator = PrivateAttr(default_factory=np.random.default_rng)
    _engine: ClassVar[gf.LocalEngine | None] = None
    _cache_dir: ClassVar[Path | None] = None

    model_config: ConfigDict = {"extra": "ignore"}

    @classmethod
    def set_engine(cls, engine: gf.LocalEngine) -> None:
        """
        Set the GF engine for the store.

        Args:
            engine (gf.LocalEngine): The engine to use.
        """
        cls._engine = engine

    @classmethod
    def from_selector(cls, selector: PeakAmplitudesBase) -> Self:
        """
        Create a new PeakAmplitudesStore from the given selector.

        Args:
            selector (PeakAmplitudesSelector): The selector to use.

        Returns:
            PeakAmplitudesStore: The newly created store.
        """

        if cls._engine is None:
            raise EnvironmentError(
                "No GF engine available to determine frequency range."
            )
        config = cls._engine.get_store(selector.gf_store_id).config
        if not isinstance(config, gf.ConfigTypeA):
            raise EnvironmentError("GF store is not of type ConfigTypeA.")
        store_frequency_range = Range(0.0, 1.0 / config.deltat)

        if (
            selector.frequency_range
            and selector.frequency_range.max > store_frequency_range.max
        ):
            raise ValueError(
                f"Selector frequency range {selector.frequency_range} "
                f"exceeds store frequency range {store_frequency_range}."
            )

        kwargs = selector.model_dump()
        kwargs["frequency_range"] = selector.frequency_range or store_frequency_range
        return cls(**kwargs)

    @property
    def source_depth_range(self) -> Range:
        return Range.from_list([sa.source_depth for sa in self.site_amplitudes])

    def has_depth(self, depth: float) -> bool:
        """
        Check if the moment magnitude store has a given depth.

        Args:
            depth (float): The depth to check.

        Returns:
            bool: True if the store has the given depth, False otherwise.
        """
        depths = [sa.source_depth for sa in self.site_amplitudes]
        return depth in depths

    def get_store(self) -> gf.Store:
        """
        Load the GF store for the given store ID.
        """
        if self._engine is None:
            raise EnvironmentError("No GF engine available.")

        try:
            store = self._engine.get_store(self.gf_store_id)
        except Exception as exc:
            raise EnvironmentError(
                f"Failed to load GF store {self.gf_store_id}."
            ) from exc

        config = store.config
        if not isinstance(config, gf.ConfigTypeA):
            raise EnvironmentError("GF store is not of type ConfigTypeA.")
        if 1.0 / config.deltat < self.frequency_range.max:
            raise ValueError(
                f"Pyrocko GF store frequency {1.0 / config.deltat} too low."
            )
        return store

    def is_suited(self, selector: PeakAmplitudesBase) -> bool:
        """
        Check if the given selector is suited for this store.

        Args:
            selector (PeakAmpliutdesSelector): The selector to check.

        Returns:
            bool: True if the selector is suited for this store.
        """
        result = (
            self.gf_store_id == selector.gf_store_id
            and self.max_distance >= selector.max_distance
            and self.gf_interpolation == selector.gf_interpolation
            and self.quantity == selector.quantity
            and self.reference_magnitude == selector.reference_magnitude
            and self.rupture_velocities.min == selector.rupture_velocities.min
            and self.rupture_velocities.max == selector.rupture_velocities.max
            and self.stress_drop.min == selector.stress_drop.min
            and self.stress_drop.max == selector.stress_drop.max
        )
        if selector.frequency_range:
            result = (
                result
                and self.frequency_range.min == selector.frequency_range.min
                and self.frequency_range.max == selector.frequency_range.max
            )
        return result

    def _get_random_source(self, depth: float) -> gf.MTSource:
        """
        Generates a random seismic source with the given depth.

        Args:
            depth (float): The depth of the seismic source.

        Returns:
            gf.MTSource: A random moment tensor source.
        """
        rng = self._rng
        store = self.get_store()
        velocity_model: LayeredModel = store.config.earthmodel_1d
        vs = np.interp(depth, velocity_model.profile("z"), velocity_model.profile("vs"))

        stress_drop = rng.uniform(*self.stress_drop)
        rupture_velocity = rng.uniform(*self.rupture_velocities) * vs

        radius = (
            pmt.magnitude_to_moment(self.reference_magnitude) * 7.0 / 16.0 / stress_drop
        ) ** (1.0 / 3.0)
        duration = 1.5 * radius / rupture_velocity
        moment_tensor = pmt.MomentTensor.random_dc(magnitude=self.reference_magnitude)
        return gf.MTSource(
            m6=moment_tensor.m6(),
            depth=depth,
            stf=gf.HalfSinusoidSTF(effective_duration=duration),
        )

    def _get_random_targets(
        self,
        distance_range: Range,
        n_receivers: int,
    ) -> list[gf.Target]:
        """
        Generate a list of receivers with random angles and distances.

        Args:
            n_receivers (int): The number of receivers to generate.

        Returns:
            list[gf.Receiver]: A list of receivers with random angles and distances.
        """
        rng = self._rng
        angles = rng.uniform(0.0, 360.0, size=n_receivers)
        distances = np.exp(rng.uniform(*np.log(distance_range), size=n_receivers))
        targets: list[gf.Receiver] = []

        for i_receiver, (angle, distance) in enumerate(
            zip(angles, distances, strict=True)
        ):
            for component in get_args(Components):
                target = gf.Target(
                    quantity=self.quantity,
                    store_id=self.gf_store_id,
                    interpolation=self.gf_interpolation,
                    depth=0.0,
                    dip=_get_dip(component),
                    azimuth=_get_azimuth(component),
                    north_shift=distance * np.cos(np.radians(angle)),
                    east_shift=distance * np.sin(np.radians(angle)),
                    codes=("PA", f"{i_receiver:05d}", "", component),
                )
                targets.append(target)
        return targets  # type: ignore

    async def fill_source_depth(
        self,
        source_depth: float,
        n_targets: int = 20,
        n_sources: int = 100,
    ) -> SiteAmplitudesCollection:
        """
        Fills the moment magnitude store with amplitudes calculated
        for a specific source depth.

        Args:
            source_depth (float): The depth of the seismic source.
            n_targets (int, optional): The number of target locations to calculate
                amplitudes for. Defaults to 20.
            n_sources (int, optional): The number of source locations to generate
                random sources from. Defaults to 100.
        """
        if self._engine is None:
            raise EnvironmentError("No GF engine available.")

        store = self.get_store()

        engine = self._engine
        target_distances = Range(
            min=store.config.distance_min,
            max=min(store.config.distance_max, self.max_distance),
        )

        async def get_modelled_waveforms() -> tuple[gf.Response, list[gf.Target]]:
            targets = self._get_random_targets(target_distances, n_targets)
            source = self._get_random_source(source_depth)
            return await asyncio.to_thread(engine.process, source, targets), targets

        receivers = []
        receiver_traces = []
        logger.info(
            "calculating %d amplitudes for depth %f",
            n_sources * n_targets,
            source_depth,
        )
        try:
            collection = self.get_collection(source_depth)
        except KeyError:
            collection = self.new_collection(source_depth)

        for _ in track(
            range(n_sources),
            total=n_sources,
            description="Calculating amplitudes",
        ):
            response, targets = await get_modelled_waveforms()

            traces: list[Trace] = response.pyrocko_traces()
            for tr in traces:
                if self.frequency_range:
                    if self.frequency_range.min > 0.0:
                        tr.highpass(4, self.frequency_range.min, demean=False)
                    if self.frequency_range.max < 1.0 / tr.deltat:
                        tr.lowpass(4, self.frequency_range.max, demean=False)

            for nsl, grp_traces in itertools.groupby(
                traces, key=lambda tr: tr.nslc_id[:3]
            ):
                receivers.append(_get_target(targets, nsl))
                receiver_traces.append(list(grp_traces))

        collection.fill(receivers, receiver_traces)
        self.save()
        return collection

    def get_collection(self, source_depth: float) -> SiteAmplitudesCollection:
        """
        Get the site amplitudes collection for the given source depth.

        Args:
            depth (float): The source depth.

        Returns:
            SiteAmplitudesCollection: The site amplitudes collection.
        """
        for site_amplitudes in self.site_amplitudes:
            if site_amplitudes.source_depth == source_depth:
                return site_amplitudes
        raise KeyError(f"No site amplitudes for depth {source_depth}.")

    def new_collection(self, depth: float) -> SiteAmplitudesCollection:
        """
        Creates a new SiteAmplitudesCollection object for the given depth and
        adds it to the list of site amplitudes.

        Args:
            depth (float): The depth for which the site amplitudes collection is
                created.

        Returns:
            SiteAmplitudesCollection: The newly created SiteAmplitudesCollection object.
        """
        try:
            collection = self.get_collection(depth)
            self.site_amplitudes.remove(collection)
        except KeyError:
            pass
        collection = SiteAmplitudesCollection(source_depth=depth, **self.model_dump())
        self.site_amplitudes.append(collection)
        return collection

    async def get_amplitude(
        self,
        source_depth: float,
        distance: float,
        n_amplitudes: int = 25,
        max_distance: float = 1.0 * KM,
        peak_amplitude: PeakAmplitude = "absolute",
        auto_fill: bool = True,
        interpolation: Literal["nearest", "linear"] = "linear",
    ) -> ModelledAmplitude:
        """
        Retrieves the amplitude for a given depth and distance.

        Args:
            depth (float): The depth of the event.
            distance (float): The surface distance from the event.
            n_amplitudes (int, optional): The number of amplitudes to retrieve.
                Defaults to 10.
            max_distance (float, optional): The maximum distance to consider in [m].
                Defaults to 1000.0.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".
            auto_fill (bool, optional): If True, the site amplitudes are calculated
                if they are not available. Defaults to True.

        Returns:
            ModelledAmplitude: The modelled amplitude for the given depth and distance.
        """
        if not self.source_depth_range.inside(source_depth):
            raise ValueError(f"Source depth {source_depth} outside range.")

        source_depths = np.array([sa.source_depth for sa in self.site_amplitudes])
        match interpolation:
            case "nearest":
                idx = [np.abs(source_depths - source_depth).argmin()]
            case "linear":
                idx = np.argsort(np.abs(source_depths - source_depth))[:2]
            case _:
                raise ValueError(f"Unknown interpolation method {interpolation}.")

        collections = [self.site_amplitudes[i] for i in idx]

        amplitudes: list[ModelledAmplitude] = []
        for collection in collections:
            try:
                amplitude = collection.get_amplitude(
                    distance=distance,
                    n_amplitudes=n_amplitudes,
                    max_distance=max_distance,
                    peak_amplitude=peak_amplitude,
                )
                amplitudes.append(amplitude)
            except ValueError:
                if auto_fill:
                    await self.fill_source_depth(source_depth)
                    logger.info("auto-filling amplitudes for depth %f", source_depth)
                    return await self.get_amplitude(
                        source_depth=source_depth,
                        distance=distance,
                        n_amplitudes=n_amplitudes,
                        max_distance=max_distance,
                        peak_amplitude=peak_amplitude,
                        interpolation=interpolation,
                        auto_fill=True,
                    )
                raise

        if not amplitudes:
            raise ValueError(f"No site amplitudes for depth {source_depth}.")

        if interpolation == "nearest":
            return amplitudes[0]

        if interpolation == "linear":
            if len(amplitudes) != 2:
                raise ValueError(
                    f"Cannot interpolate amplitudes with {len(amplitudes)} "
                    f"source depths."
                )
            depths = source_depths[idx]
            weight = abs((source_depth - depths[0]) / abs(depths[1] - depths[0]))
            return amplitudes[0].combine(amplitudes[1], weight=weight)

        raise ValueError(f"Unknown interpolation method {interpolation}.")

    def hash(self) -> str:
        """
        Calculate the hash of the store from store parameters.

        Returns:
            str: The hash of the store.
        """
        data = struct.pack(
            "ddddddddss",
            self.frequency_range.min,
            self.frequency_range.max,
            self.reference_magnitude,
            self.max_distance,
            self.rupture_velocities.min,
            self.rupture_velocities.max,
            self.stress_drop.min,
            self.stress_drop.max,
            self.gf_store_id,
            self.gf_interpolation,
        )
        return hashlib.sha1(data).hexdigest()

    def __hash__(self) -> int:
        return hash(self.hash())

    def save(self, path: Path | None = None) -> None:
        """
        Save the site amplitudes to a JSON file.

        The site amplitudes are saved in a directory called 'site_amplitudes'
        within the cache directory. The file name is generated based on the store ID and
        a hash of the store parameters.
        """
        if not path:
            if not self._cache_dir:
                return
            path = self._cache_dir

        file = path / f"{self.gf_store_id}-{self.quantity}-{self.hash()}.json"
        logger.info("saving site amplitudes to %s", file)
        file.write_text(self.model_dump_json())


class CacheStats(NamedTuple):
    path: Path
    n_stores: int
    bytes: int


class PeakAmplitudeStoreCache:
    cache_dir: Path
    engine: gf.LocalEngine

    def __init__(self, cache_dir: Path, engine: gf.LocalEngine | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("cache stats: %s", self.cache_stats())
        self.clean_cache()

        self.engine = engine or gf.LocalEngine(store_superdirs=["."])
        PeakAmplitudesStore.set_engine(engine)

    def clear_cache(self):
        """
        Clear the cache directory.

        This method deletes all files in the cache directory.
        """
        logger.info("clearing cache directory %s", self.cache_dir)
        for file in self.cache_dir.glob("*"):
            file.unlink()

    def clean_cache(self, keep_files: int = 100) -> None:
        """
        Clean the cache directory.

        Args:
            keep_files (int, optional): The number of most recent files to keep in the
                cache directory. Defaults to 100.
        """
        files = sorted(self.cache_dir.glob("*"), key=lambda f: f.stat().st_mtime)
        if len(files) <= keep_files:
            return
        logger.info("cleaning cache directory %s", self.cache_dir)
        for file in files[keep_files:]:
            file.unlink()

    def cache_stats(self) -> CacheStats:
        """
        Get the cache statistics.

        Returns:
            CacheStats: The cache statistics.
        """
        n_stores = 0
        nbytes = 0
        for file in self.cache_dir.glob("*.json"):
            n_stores += 1
            nbytes += file.stat().st_size
        return CacheStats(path=self.cache_dir, n_stores=n_stores, bytes=nbytes)

    def get_cached_stores(
        self, store_id: str, quantity: MeasurementUnit
    ) -> list[PeakAmplitudesStore]:
        """
        Get the cached peak amplitude stores for the given store ID and quantity.

        Args:
            store_id (str): The store ID.
            quantity (MeasurementUnit): The quantity.

        Returns:
            list[PeakAmplitudesStore]: A list of peak amplitude stores.
        """
        cache_dir = self.cache_dir / "site_amplitudes"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stores = []
        for file in cache_dir.glob("*.json"):
            try:
                store_id, quantity, _ = file.stem.split("-")  # type: ignore
            except ValueError:
                logger.warning("Invalid file name %s, deleting file", file)
                file.unlink()

            if store_id == store_id and quantity == quantity:
                try:
                    store = PeakAmplitudesStore.model_validate_json(file.read_text())
                except ValidationError:
                    logger.warning("Invalid store %s, deleting file", file)
                    file.unlink()
                    continue
                stores.append(store)
        return stores

    def get_store(self, selector: PeakAmplitudesBase) -> PeakAmplitudesStore:
        """
        Get a peak amplitude store for the given selector, either from the cache
        or by creating a new store.

        Args:
            selector (PeakAmplitudesSelector): The selector to use.

        Returns:
            PeakAmplitudesStore: The peak amplitude store.
        """
        for store in self.get_cached_stores(selector.gf_store_id, selector.quantity):
            if store.is_suited(selector):
                return store
        return PeakAmplitudesStore.from_selector(selector)
