from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from lassie.models.location import Location
    from lassie.models.station import Stations
    from lassie.octree import Octree


class RayTracer(BaseModel):
    tracer: Literal["RayTracer"] = "RayTracer"

    def prepare(self, octree: Octree, stations: Stations):
        ...

    def get_available_phases(self) -> tuple[str]:
        ...

    def get_traveltime_location(
        self,
        phase: str,
        source: Location,
        receiver: Location,
    ) -> float:
        ...

    def get_traveltimes_locations(
        self,
        phase: str,
        source: Location,
        receivers: Sequence[Location],
    ) -> np.ndarray:
        return np.array(
            [self.get_traveltime_location(phase, source, recv) for recv in receivers]
        )

    def get_traveltimes(
        self,
        phase: str,
        octree: Octree,
        stations: Stations,
    ) -> np.ndarray:
        ...
