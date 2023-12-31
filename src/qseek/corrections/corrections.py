from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from qseek.corrections.base import StationCorrections

# Has to be imported to register as subclass
from qseek.corrections.simple import SimpleCorrections  # noqa: F401

StationCorrectionType = Annotated[
    Union[(StationCorrections, *StationCorrections.get_subclasses())],
    Field(..., discriminator="corrections"),
]