from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from lassie.models.location import Location
from lassie.tracers.cake import TraveltimeTree

if TYPE_CHECKING:
    from lassie.models.station import Stations
    from lassie.octree import Octree

KM = 1e3


def test_sptree_model(traveltime_tree: TraveltimeTree):
    model = traveltime_tree

    with TemporaryDirectory() as d:
        tmp = Path(d)
        file = model.save(tmp)

        model2 = TraveltimeTree.load(file)
        model2._load_sptree()

    source = Location(
        lat=0.0,
        lon=0.0,
        north_shift=1 * KM,
        east_shift=1 * KM,
        depth=5.0 * KM,
    )
    receiver = Location(
        lat=0.0,
        lon=0.0,
        north_shift=0 * KM,
        east_shift=0 * KM,
        depth=0,
    )

    model.get_traveltime(source, receiver)


def test_lut(
    traveltime_tree: TraveltimeTree, octree: Octree, stations: Stations
) -> None:
    model = traveltime_tree
    model.init_lut(octree, stations)

    traveltimes_tree = model.interpolate_traveltimes(octree, stations)
    traveltimes_lut = model.get_traveltimes(octree, stations)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)

    # Test refilling the LUT
    model._node_lut.clear()
    traveltimes_tree = model.interpolate_traveltimes(octree, stations)
    traveltimes_lut = model.get_traveltimes(octree, stations)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)
    assert len(model._node_lut) > 0, "did not refill lut"

    stations_selection = stations.copy()
    stations_selection.stations = stations_selection.stations[:5]
    traveltimes_tree = model.interpolate_traveltimes(octree, stations_selection)
    traveltimes_lut = model.get_traveltimes(octree, stations_selection)
    np.testing.assert_equal(traveltimes_tree, traveltimes_lut)
