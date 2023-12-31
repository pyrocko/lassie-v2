from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection


class EventMagnitude(BaseModel):
    magnitude: Literal["EventMagnitude"] = "EventMagnitude"

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitude], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def average(self) -> float:
        raise NotImplementedError

    @property
    def error(self) -> float:
        raise NotImplementedError

    def csv_row(self) -> dict[str, float]:
        return {
            "magnitude": self.average,
            "error": self.error,
        }


class EventMagnitudeCalculator(BaseModel):
    magnitude: Literal["MagnitudeCalculator"] = "MagnitudeCalculator"

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventMagnitudeCalculator], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    async def add_magnitude(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        raise NotImplementedError
