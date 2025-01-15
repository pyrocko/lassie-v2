from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, computed_field
from typing_extensions import Self

if TYPE_CHECKING:
    from qseek.octree import Node, Octree


# Equivalent to one standard deviation
PERCENTILE = 0.02


class DetectionUncertainty(BaseModel):
    east: tuple[float, float] = Field(
        ...,
        description="Uncertainty in east direction in [m].",
    )
    north: tuple[float, float] = Field(
        ...,
        description="Uncertainty in north direction in [m].",
    )
    depth: tuple[float, float] = Field(
        ...,
        description="Uncertainty in depth in [m].",
    )

    @classmethod
    def from_event(
        cls, source_node: Node, octree: Octree, percentile: float = PERCENTILE
    ) -> Self:
        """Calculate the uncertainty of an event detection.

        Args:
            source_node (Node): The source node of the event.
            octree (Octree): The octree to use for the calculation.
            percentile (float): The percentile to use for the calculation.
                Defaults to 0.02 (2%).

        Returns:
            The calculated uncertainty.
        """
        if not source_node.semblance:
            raise ValueError("Source node must have semblance value.")

        nodes = octree.get_nodes_by_threshold(
            semblance_threshold=source_node.semblance * (1.0 - percentile)
        )
        vicinity_coords = np.array(
            [(node.east, node.north, node.depth) for node in nodes]
        )
        relative_node_offsets = vicinity_coords - np.array(
            [source_node.east, source_node.north, source_node.depth]
        )
        min_offsets = np.min(relative_node_offsets, axis=0)
        max_offsets = np.max(relative_node_offsets, axis=0)

        return cls(
            east=(float(min_offsets[0]), float(max_offsets[0])),
            north=(float(min_offsets[1]), float(max_offsets[1])),
            depth=(float(min_offsets[2]), float(max_offsets[2])),
        )

    @computed_field
    def total(self) -> float:
        """Calculate the total uncertainty in [m]."""
        return float(
            np.sqrt(sum(self.east) ** 2 + sum(self.north) ** 2 + sum(self.depth) ** 2)
        )

    @computed_field
    def horizontal(self) -> float:
        """Calculate the horizontal uncertainty in [m]."""
        return float(np.sqrt(sum(self.east) ** 2 + sum(self.north) ** 2))

    @computed_field
    def vertical(self) -> float:
        """Calculate the vertical uncertainty in [m]."""
        return float(self.depth[1] - self.depth[0])
