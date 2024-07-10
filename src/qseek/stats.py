from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, NoReturn, Type
from weakref import WeakValueDictionary

from pydantic import BaseModel, PrivateAttr, create_model
from pydantic.fields import ComputedFieldInfo, FieldInfo
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from typing_extensions import Self

logger = logging.getLogger(__name__)

STATS_INSTANCES: WeakValueDictionary[str, Stats] = WeakValueDictionary()


PROGRESS = Progress()


def titelify(name: str) -> str:
    return " ".join(word for word in name.split("_")).capitalize()


class RuntimeStats(BaseModel):
    @classmethod
    def model(cls) -> Type[Self]:
        return create_model(
            "RuntimeStats",
            **{stats.__name__: (stats, None) for stats in Stats.get_subclasses()},
            __base__=cls,
        )

    @classmethod
    def current(cls) -> Self:
        """Get the current runtime stats instance."""
        return cls.model()(**STATS_INSTANCES)

    @classmethod
    async def live_view(cls) -> NoReturn:
        def generate_grid() -> Table:
            """Make a new table."""
            table = Table(show_header=False, box=None)
            stats_instaces = sorted(
                STATS_INSTANCES.values(),
                key=lambda s: s._position,
            )
            for stats in stats_instaces:
                table.add_row(
                    f"{stats.__class__.__name__.removesuffix('Stats')}", style="bold"
                )
                table.add_section()
                stats._populate_table(table)
            grid = table.grid(expand=True)
            grid.add_row(PROGRESS)
            grid.add_row(Panel(table, title="QSeek"))
            return grid

        with Live(
            generate_grid(),
            refresh_per_second=4,
            # screen=True,
        ) as live:
            while True:
                live.update(generate_grid())
                try:
                    await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    break


class Stats(BaseModel):
    _position: int = PrivateAttr(10)

    @classmethod
    def get_subclasses(cls) -> set[type[Stats]]:
        return set(cls.__subclasses__())

    def model_post_init(self, __context: Any) -> None:
        STATS_INSTANCES[self.__class__.__name__] = self

    def _populate_table(self, table: Table) -> None:
        for name, field in self.iter_fields():
            title = field.title or titelify(name)
            table.add_row(
                title,
                str(getattr(self, name)),
                style="dim",
            )

    def iter_fields(self) -> Iterator[tuple[str, FieldInfo | ComputedFieldInfo]]:
        yield from self.model_fields.items()
        yield from self.model_computed_fields.items()

    def __rich__(self) -> Panel:
        table = Table(box=None, row_styles=["", "dim"])
        self._populate_table(table)
        return Panel(table, title=self.__class__.__name__)
