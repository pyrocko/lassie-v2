from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Iterable,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
import psutil
from pydantic import AfterValidator, BaseModel, BeforeValidator, ByteSize, constr
from pyrocko.util import UnavailableDecimation
from rich.logging import RichHandler

if TYPE_CHECKING:
    from contextvars import Context

    from pyrocko.trace import Trace

PYTHON_VERSION = (sys.version_info.major, sys.version_info.minor)

logger = logging.getLogger(__name__)
FORMAT = "%(message)s"

SDS_PYROCKO_SCHEME = (
    "%(tmin_year)s/%(network)s/%(station)s/%(channel)s.D"
    "/%(network)s.%(station)s.%(location)s.%(channel)s.D"
    ".%(tmin_year)s.%(julianday)s"
)

PhaseDescription = Annotated[str, constr(pattern=r"[a-zA-Z]*:[a-zA-Z]*")]

QUEUE_SIZE = 16
CACHE_DIR = Path.home() / ".cache" / "qseek"
if not CACHE_DIR.exists():
    logger.info("creating cache dir %s", CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)


class Symbols:
    Target = "🞋"
    Check = "✓"
    CheckerBoard = "🙾"
    Cross = "✗"


def setup_rich_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )


class BackgroundTasks:
    tasks: ClassVar[set[asyncio.Task]] = set()

    @classmethod
    def create_task(
        cls,
        coro: Coroutine,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Task:
        if PYTHON_VERSION >= (3, 11):
            task = asyncio.create_task(coro, name=name, context=context)
        else:
            task = asyncio.create_task(coro, name=name)
        cls.tasks.add(task)
        task.add_done_callback(cls.tasks.remove)
        return task

    @classmethod
    def cancel_all(cls) -> None:
        for task in cls.tasks:
            task.cancel()

    @classmethod
    async def wait_all(cls) -> None:
        if not cls.tasks:
            return
        logger.debug("waiting for %d tasks to finish", len(cls.tasks))
        await asyncio.gather(*cls.tasks)


class _NSL(NamedTuple):
    network: str
    station: str
    location: str

    @property
    def pretty(self) -> str:
        return ".".join(self)

    def match(self, other: NSL) -> bool:
        """Check if the current NSL object matches another NSL object.

        Args:
            other (NSL): The NSL object to compare with.

        Returns:
            bool: True if the objects match, False otherwise.
        """
        if self.location:
            return self == other
        if self.station:
            return self.network == other.network and self.station == other.station
        return self.network == other.network

    @classmethod
    def parse(cls, nsl: str | NSL | list[str] | tuple[str, str, str]) -> NSL:
        """Parse the given NSL string and return an NSL object.

        Args:
            nsl (str): The NSL string to parse.

        Returns:
            NSL: The parsed NSL object.

        Raises:
            ValueError: If the NSL string is empty or invalid.
        """
        if not nsl:
            raise ValueError(f"invalid empty NSL: {nsl}")
        if type(nsl) is _NSL:
            return nsl
        if isinstance(nsl, (list, tuple)):
            return cls(*nsl)
        if not isinstance(nsl, str):
            raise ValueError(f"invalid NSL {nsl}")

        parts = nsl.split(".")
        n_parts = len(parts)
        if n_parts >= 3:
            return cls(*parts[:3])
        if n_parts == 2:
            return cls(parts[0], parts[1], "")
        if n_parts == 1:
            return cls(parts[0], "", "")
        raise ValueError(
            f"invalid NSL `{nsl}`, expecting `<net>.<sta>.<loc>`, "
            "e.g. `6A.STA130.00`, `6A.`, `6A.STA130` or `.STA130`"
        )

    def _check(self) -> NSL:
        """Check if the current NSL object matches another NSL object.

        Args:
            nsl (NSL): The NSL object to compare with.

        Returns:
            bool: True if the objects match, False otherwise.
        """
        if len(self.network) > 2:
            raise ValueError(
                f"invalid network {self.network} for {self.pretty},"
                " expected 0-2 characters for network code"
            )
        if len(self.station) > 5:
            raise ValueError(
                f"invalid station {self.station} for {self.pretty},"
                " expected 0-5 characters for station code"
            )
        if len(self.location) > 2:
            raise ValueError(
                f"invalid location {self.location} for {self.pretty},"
                " expected 0-2 characters for location code"
            )
        return self


NSL = Annotated[_NSL, BeforeValidator(_NSL.parse), AfterValidator(_NSL._check)]


class _Range(NamedTuple):
    min: float
    max: float

    def inside(self, value: float) -> bool:
        """Check if a value is inside the range.

        Args:
            value (float): The value to check.

        Returns:
            bool: True if the value is inside the range, False otherwise.
        """
        return self.min <= value <= self.max

    @classmethod
    def from_list(cls, array: np.ndarray | list[float]) -> _Range:
        """Create a Range object from a numpy array.

        Parameters:
        - array: numpy.ndarray
            The array from which to create the Range object.

        Returns:
        - _Range: The created Range object.
        """
        return cls(min=np.min(array), max=np.max(array))


def _range_validator(v: _Range) -> _Range:
    if v.min > v.max:
        raise ValueError(f"Bad range {v}, must be (min, max)")
    return v


Range = Annotated[_Range, AfterValidator(_range_validator)]


def time_to_path(datetime: datetime) -> str:
    """Converts a datetime object to a string representation of a file path.

    Args:
        datetime (datetime): The datetime object to convert.

    Returns:
        str: The string representation of the file path.
    """
    return datetime.isoformat(sep="T", timespec="milliseconds").replace(":", "")


def as_array(
    iterable: Iterable[float | Iterable[float]], dtype: np.dtype = float
) -> np.ndarray:
    """Convert an iterable of floats into a NumPy array.

    Parameters:
        iterable (Iterable[float, ...]): An iterable containing float values.

    Returns:
        np.ndarray: A NumPy array containing the float values from the iterable.
    """
    return np.fromiter(iterable, dtype=dtype)


def weighted_median(data: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Calculate the weighted median of an array/list using numpy.

    Parameters:
        data (np.ndarray): The input array/list.
        weights (np.ndarray | None): The weights corresponding to each
            element in the data array/list.
            If None, the function calculates the regular median.

    Returns:
        float: The weighted median.

    Raises:
        TypeError: If the data and weights arrays/lists cannot be sorted together.
    """
    if weights is None:
        return float(np.median(data))

    if weights.sum() == 0.0:
        raise ValueError("weights sum to zero, can't normalize")

    data = np.atleast_1d(data.squeeze())
    weights = np.atleast_1d(weights.squeeze())

    sorted_indices = np.argsort(data)
    s_data = data[sorted_indices]
    s_weights = weights[sorted_indices]

    midpoint = 0.5 * s_weights.sum()
    if np.any(weights > midpoint):
        w_median = (data[weights == weights.max()])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return float(w_median)


async def async_weighted_median(
    data: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Asynchronously calculate the weighted median of an array/list using numpy.

    Parameters:
        data (np.ndarray): The input array/list.
        weights (np.ndarray | None): The weights corresponding to each
            element in the data array/list.
            If None, the function calculates the regular median.

    Returns:
        float: The weighted median.

    Raises:
        TypeError: If the data and weights arrays/lists cannot be sorted together.
    """
    if weights is None:
        return float(await asyncio.to_thread(np.median, data))

    if weights.sum() == 0.0:
        raise ValueError("weights sum to zero, can't normalize")

    data = np.atleast_1d(data.squeeze())
    weights = np.atleast_1d(weights.squeeze())

    sorted_indices = await asyncio.to_thread(np.argsort, data)
    s_data = data[sorted_indices]
    s_weights = weights[sorted_indices]

    midpoint = 0.5 * s_weights.sum()
    if np.any(weights > midpoint):
        w_median = (data[weights == weights.max()])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return float(w_median)


def to_datetime(time: float) -> datetime:
    """Convert a UNIX timestamp to a datetime object in UTC timezone.

    Args:
        time (float): The UNIX timestamp to convert.

    Returns:
        datetime: The corresponding datetime object in UTC timezone.
    """
    return datetime.fromtimestamp(time, tz=timezone.utc)


def resample(trace: Trace, sampling_rate: float) -> None:
    """Downsamples the given trace to the specified sampling rate in-place.

    Args:
        trace (Trace): The trace to be downsampled.
        sampling_rate (float): The desired sampling rate.
    """
    deltat = 1.0 / sampling_rate
    trace_sampling_rate = 1.0 / trace.deltat

    if trace.deltat == deltat:
        return

    if trace_sampling_rate < sampling_rate:
        trace.resample(deltat)
    else:
        try:
            trace.downsample_to(deltat, demean=False, snap=False, allow_upsample_max=4)
        except UnavailableDecimation:
            logger.warning("using resample instead of decimation")
            trace.resample(deltat)


T = TypeVar("T")
P = ParamSpec("P")


def log_call(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = func(*args, **kwargs)
        logger.debug(
            "executed %s in %s",
            func.__qualname__,
            timedelta(seconds=time.time() - start),
        )
        return ret

    return wrapper


def alog_call(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        ret = await func(*args, **kwargs)
        logger.debug(
            "executed %s in %s",
            func.__qualname__,
            timedelta(seconds=time.time() - start),
        )
        return ret

    return wrapper


def human_readable_bytes(size: int | float, decimal: bool = False) -> str:
    """Convert a size in bytes to a human-readable string representation.

    Args:
        size (int | float): The size in bytes.
        decimal: If True, use decimal units (e.g. 1000 bytes per KB).
            If False, use binary units (e.g. 1024 bytes per KiB).

    Returns:
        str: The human-readable string representation of the size.

    """
    return ByteSize(size).human_readable(decimal=decimal)


def datetime_now() -> datetime:
    """Get the current datetime in UTC timezone.

    Returns:
        datetime: The current datetime in UTC timezone.
    """
    return datetime.now(tz=timezone.utc)


def get_cpu_count() -> int:
    """Get the number of CPUs available for the current job/task.

    The function first checks if the environment variable SLURM_CPUS_PER_TASK is set.
    If it is set, the value is returned as the number of CPUs.

    If SLURM_CPUS_PER_TASK is not set, the function then checks if the environment
    variable PBS_NUM_PPN is set. If it is set, the value is returned as the number of
    CPUs.

    If neither SLURM_CPUS_PER_TASK nor PBS_NUM_PPN is set, the function returns the
    number of CPUs available on the system using os.cpu_count().

    Returns:
        int: The number of CPUs available for the current job/task.
    """
    return int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            os.environ.get(
                "PBS_NUM_PPN",
                os.cpu_count() or 0,
            ),
        )
    )


def get_total_memory() -> int:
    """Get the total memory in bytes for the current job/task.

    The function first checks if the environment variable SLURM_MEM_PER_CPU is set.
    If it is set, the value is returned as the available memory.

    If SLURM_MEM_PER_CPU is not set, the function then checks if the environment
    variable PBS_VMEM is set. If it is set, the value is returned as the available
    memory.

    If neither SLURM_MEM_PER_CPU nor PBS_VMEM is set, the function returns from psutil.

    Returns:
        int: The available memory in bytes for the current job/task.
    """
    return int(
        os.environ.get(
            "SLURM_MEM_PER_NODE",
            os.environ.get(
                "PBS_VMEM",
                psutil.virtual_memory().total,
            ),
        )
    )


def filter_clipped_traces(
    traces: list[Trace],
    counts_threshold: int = 20,
    max_bits: tuple[int, ...] = (24, 32),
) -> list[Trace]:
    """Filters out clipped traces from the given list of traces.

    Args:
        traces (list[Trace]): The list of traces to filter.
        counts_threshold (int, optional): The threshold for the distance between the
            maximum value of a trace and the clip value. Defaults to 20.
        max_bits (tuple[int, ...], optional): The clip bits to check for.
            Defaults to (24, 32).

    Raises:
        TypeError: If a trace is not of type int, np.int32 or np.int64.

    Returns:
        list[Trace]: The filtered list of traces.
    """
    for tr in traces.copy():
        if tr.ydata is None:
            continue
        if tr.ydata.dtype not in (int, np.int32, np.int64):
            raise TypeError(f"trace {tr.nslc_id} has invalid dtype {tr.ydata.dtype}")

        max_val = np.abs(tr.ydata).max()
        for bits in max_bits:
            clip_value = 2 ** (bits - 1) - 1
            distance_counts = abs(max_val - clip_value)
            if distance_counts < counts_threshold:
                logger.warning(
                    "trace %s likely clipped, distance to %d bits clip are %d counts",
                    ".".join(tr.nslc_id),
                    bits,
                    distance_counts,
                )
                traces.remove(tr)
    return traces


def camel_case_to_snake_case(name: str) -> str:
    """Converts a camel case string to snake case.

    Args:
        name (str): The camel case string to be converted.

    Returns:
        str: The snake case string.

    Example:
        >>> camel_case_to_snake_case("camelCaseString")
        'camel_case_string'
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def load_insights() -> None:
    """Imports the qseek.insights package if available.

    This function attempts to import the qseek.insights package and logs a debug message
    indicating whether the package was successfully imported or not.

    Raises:
        ImportError: If the qseek.insights package is not installed.
    """
    try:
        import qseek.insights  # noqa: F401

        logger.info("loaded qseek.insights package")
    except ImportError:
        logger.debug("package qseek.insights not installed")


MeasurementUnit = Literal[
    "displacement",
    "velocity",
    "acceleration",
]


@dataclass
class ChannelSelector:
    channels: str  # Channel selectors
    number_channels: int  # Number of required channels for the selector
    absolute: bool = False
    average: bool = False

    def get_traces(self, traces_flt: list[Trace]) -> list[Trace]:
        """Filter and normalize a list of traces based on the specified channels.

        Args:
            traces_flt (list[Trace]): The list of traces to filter.

        Returns:
            list[Trace]: The filtered and normalized list of traces.

        Raises:
            KeyError: If the number of channels in the filtered list does not match
                the expected number of channels.
        """
        nsls = {tr.nslc_id[:3] for tr in traces_flt}
        if len(nsls) != 1:
            raise AttributeError(
                f"cannot get traces for selector {self.channels}"
                f" available: {', '.join('.'.join(tr.nslc_id) for tr in traces_flt)}"
            )

        traces_flt = [tr for tr in traces_flt if tr.channel[-1] in self.channels]

        tmins = {tr.tmin for tr in traces_flt}
        tmaxs = {tr.tmax for tr in traces_flt}
        if len(tmins) != 1 or len(tmaxs) != 1:
            raise AttributeError(
                f"unhealthy timing on channels {self.channels}",
                f" for: {', '.join('.'.join(tr.nslc_id) for tr in traces_flt)}",
            )

        if len(traces_flt) != self.number_channels:
            raise KeyError(
                f"cannot get {self.number_channels} channels"
                f" for selector {self.channels}"
                f" available: {', '.join('.'.join(tr.nslc_id) for tr in traces_flt)}"
            )
        if self.absolute:
            traces_norm = traces_flt[0].copy()
            data = np.atleast_2d(np.array([tr.ydata for tr in traces_flt]))
            traces_norm.ydata = np.linalg.norm(data, axis=0)
            return [traces_norm]

        if self.average:
            traces_avg = traces_flt[0].copy()
            data = np.atleast_2d(np.array([tr.ydata for tr in traces_flt]))
            traces_avg.ydata = np.mean(data, axis=0)
            return [traces_avg]

        return traces_flt

    __call__ = get_traces


class ChannelSelectors:
    All = ChannelSelector("ENZ0123RT", number_channels=3)
    AllAbsolute = ChannelSelector("ENZ0123RT", number_channels=3, absolute=True)
    HorizontalAbs = ChannelSelector("EN123RT", number_channels=2, absolute=True)
    HorizontalAvg = ChannelSelector("EN123RT", number_channels=2, average=True)
    Horizontal = ChannelSelector("EN123RT", number_channels=2)
    Vertical = ChannelSelector("Z0", number_channels=1)
    NorthEast = ChannelSelector("NE", number_channels=2)


def generate_docs(model: BaseModel, exclude: dict | set | None = None) -> str:
    """Takes model and dumps markdown for documentation."""

    def generate_submodel(model: BaseModel) -> list[str]:
        lines = []
        for name, field in model.model_fields.items():
            if field.description is None:
                continue
            lines += [
                f"        - **`{name}`** *`{field.annotation}`*\n",
                f"            {field.description}",
            ]
        return lines

    model_name = model.__class__.__name__
    lines = [f"### {model_name} Module"]
    if model.__class__.__doc__ is not None:
        lines += [f"{model.__class__.__doc__}\n"]
    lines += [f'=== "Config {model_name}"']
    for name, field in model.model_fields.items():
        annotation = ""

        if field.annotation in (int, float, bool, dict, str):
            annotation = f"{field.default}"
        elif field.annotation in (list, set):
            annotation = f"{field.annotation}"
        elif get_origin(field.annotation) is Literal:
            annotation = f"{' | '.join(map(str, get_args(field.annotation)))}"
        else:
            ...

        if annotation:
            annotation = f": `{annotation}`"

        if field.description is None:
            continue
        lines += [
            f"    **`{name}`{annotation}**\n",
            f"    :   {field.description}\n",
        ]

    def dump_json() -> list[str]:
        dump = model.model_dump_json(by_alias=False, indent=2, exclude=exclude)
        lines = dump.split("\n")
        return [f"    {line}" for line in lines]

    lines += ['=== "JSON :material-code-braces:"']
    lines += [f"    ```json title='JSON for {model_name}'"]
    lines.extend(dump_json())
    lines += ["    ```"]
    return "\n".join(lines)
