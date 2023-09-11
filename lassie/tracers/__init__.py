from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Iterator, Union

from pydantic import Field, RootModel

from lassie.tracers.cake import CakeArrival, CakeTracer
from lassie.tracers.constant_velocity import (
    ConstantVelocityArrival,
    ConstantVelocityTracer,
)
from lassie.tracers.fast_marching import FastMarchingArrival, FastMarchingTracer

if TYPE_CHECKING:
    from lassie.models.station import Stations
    from lassie.octree import Octree
    from lassie.tracers.base import RayTracer
    from lassie.utils import PhaseDescription

logger = logging.getLogger(__name__)


RayTracerType = Annotated[
    Union[ConstantVelocityTracer, CakeTracer, FastMarchingTracer],
    Field(..., discriminator="tracer"),
]

RayTracerArrival = Annotated[
    Union[ConstantVelocityArrival, CakeArrival, FastMarchingArrival],
    Field(..., discriminator="tracer"),
]


class RayTracers(RootModel):
    root: list[RayTracerType] = []

    async def prepare(
        self,
        octree: Octree,
        stations: Stations,
        phases: tuple[PhaseDescription, ...],
    ) -> None:
        logger.info("preparing ray tracers")
        for tracer in self:
            tracer_phases = tracer.get_available_phases()
            for phase in phases:  # Only prepare tracers for requested phases
                if phase in tracer_phases:
                    break
            else:
                continue
            await tracer.prepare(octree, stations)

    def get_available_phases(self) -> tuple[str]:
        phases = []
        for tracer in self:
            phases.extend([*tracer.get_available_phases()])
        if len(set(phases)) != len(phases):
            raise ValueError("A phase was provided twice")
        return tuple(phases)

    def get_phase_tracer(self, phase: str) -> RayTracer:
        for tracer in self:
            if phase in tracer.get_available_phases():
                return tracer
        raise ValueError(
            f"No tracer found for phase {phase}."
            f" Available phases: {', '.join(self.get_available_phases())}"
        )

    def __iter__(self) -> Iterator[RayTracer]:
        yield from self.root

    def iter_phase_tracer(self) -> Iterator[tuple[PhaseDescription, RayTracer]]:
        for tracer in self:
            for phase in tracer.get_available_phases():
                yield (phase, tracer)
