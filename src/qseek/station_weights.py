from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pyrocko.orthodrome as od
from lru import LRU
from pydantic import BaseModel, ByteSize, Field, PositiveFloat, PrivateAttr

from qseek.octree import get_node_coordinates

if TYPE_CHECKING:
    from qseek.models.station import Station, Stations
    from qseek.octree import Node, Octree

MB = 1024**2

logger = logging.getLogger(__name__)


class StationWeights(BaseModel):
    exponent: float = Field(
        default=0.5,
        description="Exponent of the exponential decay function. Default is 1.5.",
        ge=0.0,
        le=3.0,
    )
    radius_meters: PositiveFloat = Field(
        default=8000.0,
        description="Radius in meters for the exponential decay function. "
        "Default is 8000.",
    )
    lut_cache_size: ByteSize = Field(
        default=200 * MB,
        description="Size of the LRU cache in bytes. Default is 1e9.",
    )

    _node_lut: dict[bytes, np.ndarray] = PrivateAttr()
    _cached_stations_indices: dict[str, int] = PrivateAttr()
    _station_coords_ecef: np.ndarray = PrivateAttr()

    def get_distances(self, nodes: Iterable[Node]) -> np.ndarray:
        node_coords = get_node_coordinates(nodes, system="geographic")
        node_coords = np.array(od.geodetic_to_ecef(*node_coords.T)).T
        return np.linalg.norm(
            self._station_coords_ecef - node_coords[:, np.newaxis], axis=2
        )

    def calc_weights(self, distances: np.ndarray) -> np.ndarray:
        exp = self.exponent
        # radius = distances.min(axis=1)[:, np.newaxis]
        radius = self.radius_meters
        return np.exp(-(distances**exp) / (radius**exp))

    def prepare(self, stations: Stations, octree: Octree) -> None:
        logger.info("preparing station weights")

        bytes_per_node = stations.n_stations * np.float32().itemsize
        lru_cache_size = int(self.lut_cache_size / bytes_per_node)
        self._node_lut = LRU(size=lru_cache_size)

        sta_coords = stations.get_coordinates(system="geographic")
        self._station_coords_ecef = np.array(od.geodetic_to_ecef(*sta_coords.T)).T
        self._cached_stations_indices = {
            sta.nsl.pretty: idx for idx, sta in enumerate(stations)
        }
        self.fill_lut(nodes=list(octree))

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling weight lut for %d nodes", len(nodes))
        distances = self.get_distances(nodes)
        for node, sta_distances in zip(nodes, distances, strict=True):
            sta_distances = sta_distances.astype(np.float32)
            sta_distances.setflags(write=False)
            self._node_lut[node.hash()] = sta_distances

    def get_node_weights(self, node: Node, stations: list[Station]) -> np.ndarray:
        try:
            distances = self._node_lut[node.hash()]
        except KeyError:
            self.fill_lut([node])
            return self.get_node_weights(node, stations)
        return self.calc_weights(distances)

    def lut_fill_level(self) -> float:
        """Return the fill level of the LUT as a float between 0.0 and 1.0."""
        return len(self._node_lut) / self._node_lut.get_size()

    async def get_weights(self, octree: Octree, stations: Stations) -> np.ndarray:
        station_indices = np.fromiter(
            (self._cached_stations_indices[sta.nsl.pretty] for sta in stations),
            dtype=int,
        )
        distances = np.zeros(
            shape=(octree.n_nodes, stations.n_stations), dtype=np.float32
        )

        fill_nodes = []
        for idx, node in enumerate(octree):
            try:
                distances[idx] = self._node_lut[node.hash()][station_indices]
            except KeyError:
                cache_hits, cache_misses = self._node_lut.get_stats()
                total_hits = cache_hits + cache_misses
                cache_hit_rate = cache_hits / (total_hits or 1)
                logger.debug(
                    "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                    self.lut_fill_level() * 100,
                    cache_hit_rate * 100,
                )
                fill_nodes.append(node)
                continue

        if fill_nodes:
            self.fill_lut(fill_nodes)
            return await self.get_weights(octree, stations)

        return self.calc_weights(distances)
