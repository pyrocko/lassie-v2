from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
from pydantic import BaseModel, Field, FilePath, constr
from pyrocko.io.stationxml import load_xml
from pyrocko.model import Station as PyrockoStation
from pyrocko.model import dump_stations_yaml, load_stations

from qseek.utils import NSL

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

from qseek.models.location import CoordSystem, Location

NSL_RE = r"^[a-zA-Z0-9]{0,2}\.[a-zA-Z0-9]{0,5}\.[a-zA-Z0-9]{0,3}$"

logger = logging.getLogger(__name__)


class Station(Location):
    network: str = Field(..., max_length=2)
    station: str = Field(..., max_length=5)
    location: str = Field(default="", max_length=2)

    @classmethod
    def from_pyrocko_station(cls, station: PyrockoStation) -> Station:
        return cls(
            network=station.network,
            station=station.station,
            location=station.location,
            lat=station.lat,
            lon=station.lon,
            east_shift=station.east_shift,
            north_shift=station.north_shift,
            elevation=station.elevation,
            depth=station.depth,
        )

    def as_pyrocko_station(self) -> PyrockoStation:
        return PyrockoStation(**self.model_dump(exclude={"effective_lat_lon"}))

    @property
    def nsl(self) -> NSL:
        """Network Station Location code as tuple.

        Returns:
            tuple[str, str, str]: Network, Station, Location
        """
        return NSL(self.network, self.station, self.location)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.nsl))


class Stations(BaseModel):
    pyrocko_station_yamls: list[FilePath] = Field(
        default=[],
        description="List of [Pyrocko station YAML]"
        "(https://pyrocko.org/docs/current/formats/yaml.html) files.",
    )
    station_xmls: list[FilePath] = Field(
        default=[],
        description="List of StationXML files.",
    )

    blacklist: set[constr(pattern=NSL_RE)] = Field(
        default=set(),
        description="Blacklist stations and exclude from detecion. "
        "Format is `['NET.STA.LOC', ...]`.",
    )
    stations: list[Station] = []

    def model_post_init(self, __context: Any) -> None:
        loaded_stations = []
        for file in self.pyrocko_station_yamls:
            loaded_stations += load_stations(filename=str(file.expanduser()))

        for file in self.station_xmls:
            station_xml = load_xml(filename=str(file.expanduser()))
            loaded_stations += station_xml.get_pyrocko_stations()

        for sta in loaded_stations:
            sta = Station.from_pyrocko_station(sta)
            if sta not in self.stations:
                self.stations.append(sta)

        self.weed_stations()

    def weed_stations(self) -> None:
        """Remove stations with bad coordinates or duplicates."""
        logger.debug("weeding bad stations")

        seen_nsls = set()
        for sta in self.stations.copy():
            if sta.lat == 0.0 or sta.lon == 0.0:
                logger.warning("removing station %s: bad coordinates", sta.nsl.pretty)
                self.stations.remove(sta)
                continue

            if sta.nsl.pretty in seen_nsls:
                logger.warning("removing duplicate station: %s", sta.nsl.pretty)
                self.stations.remove(sta)
                continue
            seen_nsls.add(sta.nsl.pretty)

        # if not self.stations:
        #     logger.warning("no stations available, add stations to start detection")

    def blacklist_station(self, station: Station, reason: str) -> None:
        logger.warning("blacklisting station %s: %s", station.nsl.pretty, reason)
        self.blacklist.add(station.nsl.pretty)
        if self.n_stations == 0:
            raise ValueError("no stations available, all stations blacklisted")

    def weed_from_squirrel_waveforms(self, squirrel: Squirrel) -> None:
        """Remove stations without waveforms from squirrel instances.

        Args:
            squirrel (Squirrel): Squirrel instance
        """
        available_squirrel_codes = squirrel.get_codes(kind="waveform")
        available_squirrel_nsls = {
            ".".join(code[0:3]) for code in available_squirrel_codes
        }

        n_removed_stations = 0
        for sta in self.stations.copy():
            if sta.nsl.pretty not in available_squirrel_nsls:
                logger.warning(
                    "removing station %s: no waveforms available in squirrel",
                    sta.nsl.pretty,
                )
                self.stations.remove(sta)
                n_removed_stations += 1

        if n_removed_stations:
            logger.warning("removed %d stations without waveforms", n_removed_stations)
        if not self.stations:
            raise ValueError("no stations available, add waveforms to start detection")

    def __iter__(self) -> Iterator[Station]:
        return (sta for sta in self.stations if sta.nsl.pretty not in self.blacklist)

    @property
    def n_stations(self) -> int:
        """Number of stations in the stations object."""
        return sum(1 for _ in self)

    def get_all_nsl(self) -> list[tuple[str, str, str]]:
        """Get all NSL codes from all stations."""
        return [sta.nsl for sta in self]

    def select_from_traces(self, traces: Iterable[Trace]) -> Stations:
        """Select stations by NSL code.

        Args:
            selection (Iterable[Trace]): Iterable of Pyrocko Traces

        Returns:
            Stations: Containing only selected stations.
        """
        selected_stations = []
        for nsl in ((tr.network, tr.station, tr.location) for tr in traces):
            for sta in self:
                if sta.nsl == nsl:
                    selected_stations.append(sta)
                    break
            else:
                raise ValueError(f"could not find a station for {'.'.join(nsl)} ")
        return Stations.model_construct(stations=selected_stations)

    def get_centroid(self) -> Location:
        """Get centroid location from all stations.

        Returns:
            Location: Centroid Location.
        """
        centroid_lat, centroid_lon, centroid_elevation = np.mean(
            [(*sta.effective_lat_lon, sta.elevation) for sta in self],
            axis=0,
        )
        return Location(
            lat=centroid_lat,
            lon=centroid_lon,
            elevation=centroid_elevation,
        )

    def get_coordinates(self, system: CoordSystem = "geographic") -> np.ndarray:
        return np.array(
            [(*sta.effective_lat_lon, sta.effective_elevation) for sta in self]
        )

    def export_pyrocko_stations(self, filename: Path) -> None:
        """Dump stations to pyrocko station yaml file.

        Args:
            filename (Path): Path to yaml file.
        """
        dump_stations_yaml(
            [sta.as_pyrocko_station() for sta in self],
            filename=str(filename.expanduser()),
        )

    def export_csv(self, filename: Path) -> None:
        """Dump stations to CSV file.

        Args:
            filename (Path): Path to CSV file.
        """
        with filename.open("w") as f:
            f.write("network,station,location,latitude,longitude,elevation,depth\n")
            for sta in self:
                f.write(
                    f"{sta.network},{sta.station},{sta.location},"
                    f"{sta.lat},{sta.lon},{sta.elevation},{sta.depth}\n"
                )

    def export_vtk(self, reference: Location | None = None) -> None:
        ...

    def __hash__(self) -> int:
        return hash(sta for sta in self)
