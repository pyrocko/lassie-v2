from __future__ import annotations

import logging
import re
from hashlib import sha1
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PositiveFloat,
    PrivateAttr,
    model_validator,
)
from pydantic.dataclasses import dataclass
from pyevtk.hl import gridToVTK
from pyrocko.cake import LayeredModel, load_model
from scipy.interpolate import RegularGridInterpolator
from typing_extensions import Self

from lassie.models.location import Location

if TYPE_CHECKING:
    from lassie.octree import Octree


KM = 1e3
logger = logging.getLogger(__name__)


class VelocityModel3D(BaseModel):
    center: Location

    grid_spacing: float

    east_bounds: tuple[float, float]
    north_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]

    _east_coords: np.ndarray = PrivateAttr()
    _north_coords: np.ndarray = PrivateAttr()
    _depth_coords: np.ndarray = PrivateAttr()

    _velocity_model: np.ndarray = PrivateAttr()

    _hash: str | None = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        grid_spacing = self.grid_spacing

        self._east_coords = np.arange(
            self.east_bounds[0],
            self.east_bounds[1],
            grid_spacing,
        )
        self._north_coords = np.arange(
            self.north_bounds[0],
            self.north_bounds[1],
            grid_spacing,
        )
        self._depth_coords = np.arange(
            self.depth_bounds[0],
            self.depth_bounds[1],
            grid_spacing,
        )

        self._velocity_model = np.zeros(
            (
                self._east_coords.size,
                self._north_coords.size,
                self._depth_coords.size,
            )
        )

    def set_velocity_model(self, velocity_model: np.ndarray) -> None:
        if velocity_model.shape != self._velocity_model.shape:
            raise ValueError(
                f"Velocity model shape {velocity_model.shape} does not match"
                f" expected shape {self._velocity_model.shape}"
            )
        self._velocity_model = velocity_model.astype(float, copy=False)

    @property
    def velocity_model(self) -> np.ndarray:
        if self._velocity_model is None:
            raise ValueError("Velocity model not set.")
        return self._velocity_model

    @property
    def east_coords(self) -> np.ndarray:
        return self._east_coords

    @property
    def north_coords(self) -> np.ndarray:
        return self._north_coords

    @property
    def depth_coords(self) -> np.ndarray:
        return self._depth_coords

    @property
    def east_size(self) -> float:
        return self._east_coords.size * self.grid_spacing

    @property
    def north_size(self) -> float:
        return self._north_coords.size * self.grid_spacing

    @property
    def depth_size(self) -> float:
        return self._depth_coords.size * self.grid_spacing

    def hash(self) -> str:
        """Return hash of velocity model.

        Returns:
            str: The hash.
        """
        if self._hash is None:
            self._hash = sha1(self._velocity_model.tobytes()).hexdigest()
        return self._hash

    def _get_location_indices(self, location: Location) -> tuple[int, int, int]:
        """Return indices of location in velocity model, by nearest neighbor.

        Args:
            location (Location): The location.

        Returns:
            tuple[int, int, int]: The indices as (east, north, depth).
        """
        if not self.is_inside(location):
            raise ValueError("Location is outside of velocity model.")
        station_offset = location.offset_from(self.center)
        east_idx = np.argmin(np.abs(self._east_coords - station_offset[0]))
        north_idx = np.argmin(np.abs(self._north_coords - station_offset[1]))
        depth_idx = np.argmin(np.abs(self._depth_coords - station_offset[2]))
        return int(round(east_idx)), int(round(north_idx)), int(round(depth_idx))

    def get_velocity(self, location: Location) -> float:
        """Return velocity at location in [m/s], nearest neighbor.

        Args:
            location (Location): The location.

        Returns:
            float: The velocity in m/s.
        """
        east_idx, north_idx, depth_idx = self._get_location_indices(location)
        return self.velocity_model[east_idx, north_idx, depth_idx]

    def get_source_arrival_grid(self, location: Location) -> np.ndarray:
        """Return travel times grid for Eikonal for specific.

        The initial travel time grid is filled with -1.0, except for the source
        location, which is set to 0.0 s.

        Args:
            location (Location): The location.

        Returns:
            np.ndarray: The initial travel times grid.
        """
        times = np.full_like(self.velocity_model, fill_value=-1.0)
        east_idx, north_idx, depth_idx = self._get_location_indices(location)
        times[east_idx, north_idx, depth_idx] = 0.0
        return times

    def is_inside(self, location: Location) -> bool:
        """Return True if location is inside velocity model.

        Args:
            location (Location): The location.

        Returns:
            bool: True if location is inside velocity model.
        """
        offset_from_center = location.offset_from(self.center)
        return (
            self.east_bounds[0] <= offset_from_center[0] <= self.east_bounds[1]
            and self.north_bounds[0] <= offset_from_center[1] <= self.north_bounds[1]
            and self.depth_bounds[0] <= offset_from_center[2] <= self.depth_bounds[1]
        )

    def get_meshgrid(self) -> list[np.ndarray]:
        """Return meshgrid of velocity model coordinates.

        Returns:
            list[np.ndarray]: The meshgrid as list of numpy arrays for east, north,
                depth.
        """
        return np.meshgrid(
            self._east_coords,
            self._north_coords,
            self._depth_coords,
            indexing="ij",
        )

    def resample(
        self,
        grid_spacing: float,
        method: Literal["nearest", "linear", "cubic"] = "linear",
    ) -> Self:
        """Resample velocity model to new grid spacing.

        Args:
            grid_spacing (float): The new grid spacing in [m].
            method (Literal['nearest', 'linear', 'cubic'], optional): Interpolation
                method. Defaults to "linear".

        Returns:
            Self: A new, resampled velocity model.
        """
        if grid_spacing == self.grid_spacing:
            return self

        logger.info("resampling velocity model to grid spacing %s m", grid_spacing)
        interpolator = RegularGridInterpolator(
            (self._east_coords, self._north_coords, self._depth_coords),
            self._velocity_model,
            method=method,
            bounds_error=False,
        )
        resampled_model = VelocityModel3D(
            center=self.center,
            grid_spacing=grid_spacing,
            east_bounds=self.east_bounds,
            north_bounds=self.north_bounds,
            depth_bounds=self.depth_bounds,
        )
        coordinates = np.array(
            [coords.ravel() for coords in resampled_model.get_meshgrid()]
        ).T
        resampled_model._velocity_model = interpolator(coordinates).reshape(
            resampled_model._velocity_model.shape
        )
        return resampled_model

    def export_vtk(self, filename: Path, reference: Location | None = None) -> None:
        offset = reference.offset_from(self.center) if reference else np.zeros(3)

        out_file = gridToVTK(
            str(filename),
            self._east_coords + offset[0],
            self._north_coords + offset[1],
            np.array((-self._depth_coords + offset[2])[::-1]),
            pointData={"velocity": np.array(self._velocity_model[:, :, ::-1])},
        )
        logger.info("vtk: exported velocity model to %s", out_file)


class VelocityModelFactory(BaseModel):
    model: Literal["VelocityModelFactory"] = "VelocityModelFactory"

    grid_spacing: PositiveFloat | Literal["octree"] = Field(
        default="octree",
        description="Grid spacing in meters."
        " If 'octree' defaults to smallest octreee node size.",
    )

    def get_model(self, octree: Octree) -> VelocityModel3D:
        raise NotImplementedError


class Constant3DVelocityModel(VelocityModelFactory):
    """This model is for mere testing of the method."""

    model: Literal["Constant3DVelocityModel"] = "Constant3DVelocityModel"

    velocity: PositiveFloat = 5000.0

    def get_model(self, octree: Octree) -> VelocityModel3D:
        if self.grid_spacing == "octree":
            grid_spacing = octree.smallest_node_size()
        else:
            grid_spacing = self.grid_spacing

        model = VelocityModel3D(
            center=octree.location,
            grid_spacing=grid_spacing,
            east_bounds=octree.east_bounds,
            north_bounds=octree.north_bounds,
            depth_bounds=octree.depth_bounds,
        )
        model._velocity_model.fill(self.velocity)

        return model


NonLinLocGridType = Literal["VELOCITY", "VELOCITY_METERS", "SLOW_LEN"]
GridDtype = Literal["FLOAT", "DOUBLE"]
DTYPE_MAP = {"FLOAT": np.float32, "DOUBLE": float}


@dataclass
class NonLinLocHeader:
    """Helper class representing a NonLinLoc header file."""

    origin: Location
    nx: int
    ny: int
    nz: int
    delta_x: float
    delta_y: float
    delta_z: float
    grid_dtype: GridDtype
    grid_type: NonLinLocGridType

    @classmethod
    def from_header_file(
        cls,
        file: Path,
        reference_location: Location | None = None,
    ) -> Self:
        """Load NonLinLoc velocity model header file.

        Args:
            file (Path): Path to NonLinLoc model header file.
            reference_location (Location | None, optional): relative location of
                NonLinLoc model, used for models with relative coordinates.
                Defaults to None.

        Raises:
            ValueError: If grid spacing is not equal in all dimensions.

        Returns:
            Self: The header.
        """
        logger.info("loading NonLinLoc velocity model header file %s", file)
        header_text = file.read_text().split("\n")[0]
        header_text = re.sub(r"\s+", " ", header_text)  # remove excessive spaces
        (
            nx,
            ny,
            nz,
            orig_x,
            orig_y,
            orig_z,
            delta_x,
            delta_y,
            delta_z,
            grid_type,
            grid_dtype,
        ) = header_text.split()

        if not delta_x == delta_y == delta_z:
            raise ValueError("NonLinLoc velocity model must have equal spacing.")

        if reference_location:
            origin = reference_location.model_copy()
            origin.east_shift += float(orig_x) * KM
            origin.north_shift += float(orig_y) * KM
            origin.elevation -= float(orig_z) * KM
        else:
            origin = Location(
                lon=float(orig_x),
                lat=float(orig_y),
                elevation=-float(orig_z) * KM,
            )

        return cls(
            origin=origin,
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            delta_x=float(delta_x) * KM,
            delta_y=float(delta_y) * KM,
            delta_z=float(delta_z) * KM,
            grid_dtype=grid_dtype,
            grid_type=grid_type,
        )

    @property
    def dtype(self) -> np.dtype:
        """dtype of the grid."""
        return DTYPE_MAP[self.grid_dtype]

    @property
    def grid_spacing(self) -> float:
        """grid spacing, homogeneous in three directions."""
        return self.delta_x

    @property
    def east_bounds(self) -> tuple[float, float]:
        """Relative to center location."""
        return -self.delta_x * self.nx / 2, self.delta_x * self.nx / 2

    @property
    def north_bounds(self) -> tuple[float, float]:
        """Relative to center location."""
        return -self.delta_y * self.ny / 2, self.delta_y * self.ny / 2

    @property
    def depth_bounds(self) -> tuple[float, float]:
        """Relative to center location."""
        return 0, self.delta_z * self.nz

    @property
    def center(self) -> Location:
        """Return center location of velocity model.

        Returns:
            Location: The center location of the grid.
        """
        center = self.origin.model_copy(deep=True)
        center.east_shift += self.delta_x * self.nx / 2
        center.north_shift += self.delta_y * self.ny / 2
        return center


class NonLinLocVelocityModel(VelocityModelFactory):
    model: Literal["NonLinLocVelocityModel"] = "NonLinLocVelocityModel"

    header_file: FilePath = Field(
        ...,
        description="Path to NonLinLoc model header file file. "
        "The file should be in the format of a NonLinLoc velocity model header file. "
        "Binary data has to have the same name and `.buf` suffix.",
    )

    grid_spacing: PositiveFloat | Literal["octree", "input"] = Field(
        default="input",
        description="Grid spacing in meters. "
        "If 'octree' defaults to smallest octreee node size. If 'input' uses the"
        " grid spacing from the NonLinLoc header file.",
    )
    interpolation: Literal["nearest", "linear", "cubic"] = Field(
        default="linear",
        description="Interpolation method for resampling the grid"
        " for the fast-marching method.",
    )

    reference_location: Location | None = Field(
        default=None,
        description="relative location of NonLinLoc model,"
        " used for models with relative coordinates.",
    )

    _header: NonLinLocHeader = PrivateAttr()
    _velocity_model: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def load_model(self) -> Self:
        self._header = NonLinLocHeader.from_header_file(
            self.header_file,
            reference_location=self.reference_location,
        )
        buffer_file = self.header_file.with_suffix(".buf")
        if not buffer_file.exists():
            raise FileNotFoundError(f"Buffer file {buffer_file} not found.")

        logger.debug("loading NonLinLoc velocity model buffer file %s", buffer_file)
        self._velocity_model = np.fromfile(
            buffer_file, dtype=self._header.dtype
        ).reshape((self._header.nx, self._header.ny, self._header.nz))

        if self._header.grid_type == "SLOW_LEN":
            logger.debug("converting NonLinLoc SLOW_LEN model to velocity")
            self._velocity_model = 1.0 / (
                self._velocity_model / self._header.grid_spacing
            )
        elif self._header.grid_type == "VELOCITY":
            self._velocity_model *= KM

        logging.info(
            "NonLinLoc velocity model: %s"
            " east_bounds: %s, north_bounds %s, depth_bounds %s",
            self._header.center,
            self._header.east_bounds,
            self._header.north_bounds,
            self._header.depth_bounds,
        )
        return self

    def get_model(self, octree: Octree) -> VelocityModel3D:
        if self.grid_spacing == "octree":
            grid_spacing = octree.smallest_node_size()
        elif self.grid_spacing == "input":
            grid_spacing = self._header.grid_spacing
        elif isinstance(self.grid_spacing, float):
            grid_spacing = self.grid_spacing
        else:
            raise ValueError(f"Invalid grid_spacing {self.grid_spacing}")

        header = self._header

        velocity_model = VelocityModel3D(
            center=header.center,
            grid_spacing=header.grid_spacing,
            east_bounds=header.east_bounds,
            north_bounds=header.north_bounds,
            depth_bounds=header.depth_bounds,
        )
        velocity_model.set_velocity_model(self._velocity_model)
        return velocity_model.resample(grid_spacing, self.interpolation)


class VelocityModelLayered(VelocityModelFactory):
    # For mere testing purposes of the 3D tracer against Pyrocko cake 2D travel times
    model: Literal["VelocityModel2D"] = "VelocityModel2D"
    velocity: Literal["vp", "vs"] = Field(
        default="vp",
        description="velocity to extract from the 2D model, choose from 'vp' or 'vs'.",
    )
    format: Literal["nd", "hyposat"] = Field(
        default="nd",
        description="Format of the velocity model. nd or hyposat is supported.",
    )
    filename: FilePath = Field(
        ...,
        description="Path to `.nd` file holding the 2D velocity model information.",
    )
    raw_file_data: str | None = Field(
        default=None,
        description="Raw `.nd` file data.",
    )

    _layered_model: LayeredModel = PrivateAttr()

    @model_validator(mode="after")
    def load_model(self) -> VelocityModelLayered:
        if self.filename is not None:
            logger.info("loading velocity model from %s", self.filename)
            self.raw_file_data = self.filename.read_text()

        if self.raw_file_data is not None:
            with NamedTemporaryFile("w") as tmpfile:
                tmpfile.write(self.raw_file_data)
                tmpfile.flush()
                self._layered_model = load_model(
                    tmpfile.name,
                    format=self.format,
                )
        else:
            raise AttributeError("No velocity model or crust2 profile defined.")
        return self

    def get_model(self, octree: Octree) -> VelocityModel3D:
        if self.grid_spacing == "octree":
            grid_spacing = octree.smallest_node_size()
        else:
            grid_spacing = self.grid_spacing

        model = VelocityModel3D(
            center=octree.location,
            grid_spacing=grid_spacing,
            east_bounds=octree.east_bounds,
            north_bounds=octree.north_bounds,
            depth_bounds=octree.depth_bounds,
        )

        velocities = []
        for depth in model.depth_coords:
            material = self._layered_model.material(z=depth)
            if self.velocity == "vp":
                velocities.append(material.vp)
            elif self.velocity == "vs":
                velocities.append(material.vs)
            else:
                raise ValueError(f"Invalid velocity {self.velocity}")

        velocities = np.array(velocities)

        model.velocity_model[:, :, :] = velocities[np.newaxis, np.newaxis, :]
        return model


VelocityModels = Annotated[
    Union[Constant3DVelocityModel, NonLinLocVelocityModel, VelocityModelLayered],
    Field(..., discriminator="model"),
]
