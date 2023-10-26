from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Annotated, Any, Iterator, Union

from pydantic import Field, RootModel

from lassie.images.base import ImageFunction, PickedArrival
from lassie.images.phase_net import PhaseNet, PhaseNetPick
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from datetime import timedelta

    from pyrocko.trace import Trace

    from lassie.images.base import WaveformImage
    from lassie.models.station import Stations


logger = logging.getLogger(__name__)


ImageFunctionType = Annotated[
    Union[PhaseNet, ImageFunction],
    Field(..., discriminator="image"),
]

# Make this a Union when more picks are implemented
ImageFunctionPick = Annotated[
    Union[PhaseNetPick, PickedArrival],
    Field(..., discriminator="provider"),
]


class ImageFunctions(RootModel):
    root: list[ImageFunctionType] = [PhaseNet()]

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

    def get_phases(self) -> tuple[PhaseDescription, ...]:
        """Get all phases that are available in the image functions.

        Returns:
            tuple[str, ...]: All available phases.
        """
        return tuple(chain.from_iterable(image.get_provided_phases() for image in self))

    def get_blinding(self) -> timedelta:
        return max(image.blinding for image in self)

    def __iter__(self) -> Iterator[ImageFunction]:
        return iter(self.root)


@dataclass
class WaveformImages:
    root: list[WaveformImage] = Field([], alias="images")

    @property
    def n_images(self) -> int:
        return len(self.root)

    def downsample(self, sampling_rate: float, max_normalize: bool = False) -> None:
        """Downsample traces in-place.

        Args:
            sampling_rate (float): Desired sampling rate in Hz.
            max_normalize (bool): Normalize by maximum value to keep the scale of the
                maximum detection. Defaults to False
        """
        for image in self:
            image.downsample(sampling_rate, max_normalize)

    def apply_exponent(self, exponent: float) -> None:
        """Apply exponent to all images.

        Args:
            exponent (float): Exponent to apply.
        """
        for image in self:
            image.apply_exponent(exponent)

    def set_stations(self, stations: Stations) -> None:
        """Set the images stations."""
        for image in self:
            image.set_stations(stations)

    def cumulative_weight(self) -> float:
        return sum(image.weight for image in self)

    def snuffle(self) -> None:
        from pyrocko.trace import snuffle

        traces = []
        for img in self:
            traces += img.traces
        snuffle(traces)

    def __iter__(self) -> Iterator[WaveformImage]:
        yield from self.root
