from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy as np
import pyrocko.orthodrome as od
from lru import LRU
from pydantic import BaseModel, ByteSize, Field, PositiveFloat, PrivateAttr

from qseek.octree import get_node_coordinates
from qseek.utils import alog_call

if TYPE_CHECKING:
    from qseek.models.station import Station, Stations
    from qseek.octree import Node, Octree

MB = 1024**2

logger = logging.getLogger(__name__)


class DistanceWeights(BaseModel):
    exponent: float = Field(
        default=3.0,
        description="Exponent of the spatial decay function. Default is 3.",
        ge=0.0,
    )
    radius_meters: PositiveFloat | Literal["mean_interstation"] = Field(
        default="mean_interstation",
        description="Cutoff distance for the spatial decay function in meters."
        " Default is 8000.",
    )
    waterlevel: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Waterlevel for the exponential decay function. Default is 0.0.",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize the weights to the range [0, 1]. Default is True.",
    )
    lut_cache_size: ByteSize = Field(
        default=200 * MB,
        description="Size of the LRU cache in bytes. Default is 200 MB.",
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

    def calc_weights_exp(self, distances: np.ndarray) -> np.ndarray:
        exp = self.exponent
        # radius = distances.min(axis=1)[:, np.newaxis]
        radius = self.radius_meters
        return np.exp(-(distances**exp) / (radius**exp))

    def calc_weights(self, distances: np.ndarray) -> np.ndarray:
        exp = self.exponent
        radius = self.radius_meters
        weights = (1 - self.waterlevel) / (
            1 + (distances / radius) ** exp
        ) + self.waterlevel
        if self.normalize:
            weights /= weights.max()
        return weights

    def prepare(self, stations: Stations, octree: Octree) -> None:
        logger.info("preparing distance weights")

        if self.radius_meters == "mean_interstation":
            self.radius_meters = stations.mean_interstation_distance()
            logger.info(
                "using mean interstation distance as radius: %g m",
                self.radius_meters,
            )

        bytes_per_node = stations.n_stations * np.float32().itemsize
        lru_cache_size = int(self.lut_cache_size / bytes_per_node)
        self._node_lut = LRU(size=lru_cache_size)

        sta_coords = stations.get_coordinates(system="geographic")
        self._station_coords_ecef = np.array(od.geodetic_to_ecef(*sta_coords.T)).T
        self._cached_stations_indices = {
            sta.nsl.pretty: idx for idx, sta in enumerate(stations)
        }
        self.fill_lut(nodes=octree.nodes)

    def fill_lut(self, nodes: Sequence[Node]) -> None:
        logger.debug("filling distance weight LUT for %d nodes", len(nodes))
        distances = self.get_distances(nodes)
        node_lut = self._node_lut
        for node, sta_distances in zip(nodes, distances, strict=True):
            sta_distances = sta_distances.astype(np.float32)
            sta_distances.setflags(write=False)
            node_lut[node.hash()] = sta_distances

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

    @alog_call
    async def get_weights(
        self, nodes: Sequence[Node], stations: Stations
    ) -> np.ndarray:
        n_nodes = len(nodes)
        station_indices = np.fromiter(
            (self._cached_stations_indices[sta.nsl.pretty] for sta in stations),
            dtype=int,
        )
        distances = np.zeros(shape=(n_nodes, stations.n_stations), dtype=np.float32)

        fill_nodes = []
        node_lut = self._node_lut
        for idx, node in enumerate(nodes):
            try:
                distances[idx] = node_lut[node.hash()][station_indices]
            except KeyError:
                fill_nodes.append(node)
                continue

        if fill_nodes:
            self.fill_lut(fill_nodes)
            cache_hits, cache_misses = node_lut.get_stats()
            total_hits = cache_hits + cache_misses
            cache_hit_rate = cache_hits / (total_hits or 1)
            logger.debug(
                "node LUT cache fill level %.1f%%, cache hit rate %.1f%%",
                self.lut_fill_level() * 100,
                cache_hit_rate * 100,
            )
            return await self.get_weights(nodes, stations)

        return self.calc_weights(distances)
