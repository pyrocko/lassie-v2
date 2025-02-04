from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np
from obspy import Stream
from pydantic import Field, PositiveFloat, PositiveInt, PrivateAttr, confloat
from pyrocko import obspy_compat
from pyrocko.trace import NoData
from scipy import signal
from seisbench import logger as seisbench_logger

from qseek.images.base import ImageFunction, ObservedArrival, WaveformImage
from qseek.utils import alog_call, to_datetime

obspy_compat.plant()

seisbench_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyrocko.trace import Trace
    from seisbench.models import WaveformModel


ModelName = Literal[
    "PhaseNet",
    "EQTransformer",
    "OBSTransformer",
    "LFEDetect",
    "GPD",
]


PreTrainedName = Literal[
    "cascadia",
    "cms",
    "diting",
    "dummy",
    "ethz",
    "geofon",
    "instance",
    "iquique",
    "jcms",
    "jcs",
    "jms",
    "lendb",
    "mexico",
    "nankai",
    "neic",
    "obs",
    "obst2024",
    "original",
    "original_nonconservative",
    "san_andreas",
    "scedc",
    "stead",
    "volpick",
]

PhaseName = Literal["P", "S"]
StackMethod = Literal["avg", "max"]


class PhaseNetImage(WaveformImage):
    def search_phase_arrival(
        self,
        trace_idx: int,
        event_time: datetime,
        modelled_arrival: datetime,
        search_window_seconds: float = 5.0,
        threshold: float = 0.1,
        detection_blinding_seconds: float = 1.0,
    ) -> ObservedArrival | None:
        """Search for the closest peak (pick) in the station's image functions.

        Args:
            trace_idx (int): Index of the trace.
            event_time (datetime): Time of the event.
            modelled_arrival (datetime): Time to search around.
            search_window_seconds (float, optional): Total search length in seconds
                around modelled arrival time. Defaults to 5.
            threshold (float, optional): Threshold for detection. Defaults to 0.1.
            detection_blinding_seconds (float, optional): Blinding time in seconds for
                the peak detection. Defaults to 1 second.

        Returns:
            datetime | None: Time of arrival, None is none found.
        """
        # TODO adapt threshold to the seisbench model
        trace = self.traces[trace_idx]
        window_length = timedelta(seconds=search_window_seconds)
        try:
            search_trace = trace.chop(
                tmin=(modelled_arrival - window_length / 2).timestamp(),
                tmax=(modelled_arrival + window_length / 2).timestamp(),
                inplace=False,
            )
        except NoData:
            logger.warning("No data to pick phase arrival %s.", ".".join(trace.nslc_id))
            return None

        peak_idx, _ = signal.find_peaks(
            search_trace.ydata,
            height=threshold,
            prominence=threshold,
            distance=int(detection_blinding_seconds * 1 / search_trace.deltat),
        )
        if False:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            time = search_trace.get_xdata()
            std = np.std(search_trace.get_ydata())

            ax.plot(time, search_trace.get_ydata())
            ax.grid(alpha=0.3)
            ax.axhline(threshold, color="r", linestyle="--", label="threshold")
            ax.axhline(std, color="g", linestyle="--", label="std")
            ax.axhline(3 * std, color="b", linestyle="dotted", label="3*std")
            ax.axvline(
                modelled_arrival.timestamp(),
                color="k",
                alpha=0.3,
                label="modelled arrival",
            )
            if peak_idx.size:
                ax.axvline(time[peak_idx], color="m", linestyle="--", label="peaks")
            plt.show()

        times = search_trace.get_xdata()
        peak_times = times[peak_idx]
        peak_delays = peak_times - event_time.timestamp()

        # Limit to post-event peaks
        post_event_peaks = peak_delays > 0.0
        peak_idx = peak_idx[post_event_peaks]
        peak_times = peak_times[post_event_peaks]
        peak_residuals = peak_times - modelled_arrival.timestamp()

        if not peak_idx.size:
            return None

        peak_values = search_trace.get_ydata()[peak_idx]
        closest_peak_idx = np.argmin(np.abs(peak_residuals))

        return ObservedArrival(
            time=to_datetime(peak_times[closest_peak_idx]),
            detection_value=peak_values[closest_peak_idx],
            phase=self.phase,
        )


class SeisBench(ImageFunction):
    """PhaseNet image function. For more details see SeisBench documentation."""

    image: Literal["SeisBench"] = "SeisBench"

    model: ModelName = Field(
        default="PhaseNet",
        description="The model to use for the image function. Currently supported "
        "models are `PhaseNet`, `EQTransformer`, `GPD`, `OBSTransformer`, `LFEDetect`.",
    )

    pretrained: PreTrainedName = Field(
        default="original",
        description="SeisBench pre-trained model to use. "
        "Choose from `ethz`, `geofon`, `instance`, `iquique`, `lendb`, `neic`, `obs`,"
        " `original`, `scedc`, `stead`."
        " For more details see SeisBench documentation",
    )
    window_overlap_samples: int = Field(
        default=1500,
        ge=1000,
        le=3000,
        description="Window overlap in samples.",
    )
    torch_use_cuda: bool | int = Field(
        default=True,
        description="Use CUDA for inference. If `True` use default device, if `int` use"
        " the specified device.",
    )
    torch_cpu_threads: PositiveInt = Field(
        default=4,
        description="Number of CPU threads to use if only CPU is used.",
    )
    batch_size: int = Field(
        default=128,
        ge=64,
        description="Batch size for inference, larger values can improve performance.",
    )
    stack_method: StackMethod = Field(
        default="avg",
        description="Method to stack the overlaping blocks internally. "
        "Choose from `avg` and `max`.",
    )
    rescale_input: PositiveFloat = Field(
        default=1.0,
        description="Upscale input by factor. "
        "This augments the input data from e.g. 100 Hz to 50 Hz (factor: `2`). Can be"
        " useful for high-frequency microseismic events.",
    )
    phase_map: dict[PhaseName, str] = Field(
        default={
            "P": "cake:P",
            "S": "cake:S",
        },
        description="Phase mapping from SeisBench PhaseNet to Lassie phases.",
    )
    weights: dict[PhaseName, confloat(ge=0.0)] = Field(
        default={
            "P": 1.0,
            "S": 1.0,
        },
        description="Weights for each phase.",
    )

    _seisbench_model: WaveformModel = PrivateAttr(None)

    @property
    def seisbench_model(self) -> WaveformModel:
        if self._seisbench_model is None:
            self._prepare()
        return self._seisbench_model

    def _prepare(self) -> None:
        import seisbench.models as sbm
        import torch

        torch.set_num_threads(self.torch_cpu_threads)

        match self.model:
            case "PhaseNet":
                model = sbm.PhaseNet
            case "EQTransformer":
                model = sbm.EQTransformer
            case "GPD":
                model = sbm.GPD
            case "OBSTransformer":
                model = sbm.OBSTransformer
            case "LFEDetect":
                model = sbm.LFEDetect
            case _:
                raise ValueError(f"Model `{self.model}` not available.")

        self._seisbench_model = model.from_pretrained(self.pretrained)
        if self.torch_use_cuda:
            try:
                if isinstance(self.torch_use_cuda, bool):
                    self._seisbench_model.cuda()
                else:
                    self._seisbench_model.cuda(self.torch_use_cuda)
            except (RuntimeError, AssertionError) as exc:
                logger.warning(
                    "failed to use CUDA for SeisBench model, using CPU.",
                    exc_info=exc,
                )
        self._seisbench_model.eval()
        try:
            logger.info("compiling SeisBench model...")
            self._seisbench_model = torch.compile(
                self._seisbench_model,
                mode="max-autotune",
            )
        except RuntimeError as exc:
            logger.warning(
                "failed to compile SeisBench model, using uncompiled model.",
                exc_info=exc,
            )

    def get_blinding_samples(self) -> tuple[int, int]:
        if self.model == "GPD":
            return (0, 0)
        try:
            return self.seisbench_model.default_args["blinding"]
        except KeyError:
            return self.seisbench_model._annotate_args["blinding"][1]

    def get_blinding(self, sampling_rate: float) -> timedelta:
        scaled_blinding_samples = max(self.get_blinding_samples()) / self.rescale_input
        return timedelta(seconds=scaled_blinding_samples / sampling_rate)

    def _detection_half_width(self) -> float:
        """Half width of the detection window in seconds."""
        # The 0.2 seconds is the default value from SeisBench training
        return 0.2 / self.rescale_input

    @alog_call
    async def process_traces(self, traces: list[Trace]) -> list[PhaseNetImage]:
        stream = Stream(tr.to_obspy_trace() for tr in traces)
        if self.rescale_input != 1.0:
            scale = self.rescale_input
            for tr in stream:
                tr.stats.sampling_rate /= scale

        annotations: Stream = await asyncio.to_thread(
            self.seisbench_model.annotate,
            stream,
            overlap=self.window_overlap_samples,
            batch_size=self.batch_size,
            copy=False,
        )

        if self.rescale_input != 1.0:
            scale = self.rescale_input
            for tr in annotations:
                tr.stats.sampling_rate *= scale
                blinding_samples = max(self.get_blinding_samples())
                # 100 Hz is the native sampling rate of PhaseNet
                blinding_seconds = (blinding_samples / 100.0) * (1.0 - 1 / scale)
                tr.stats.starttime -= blinding_seconds

        annotated_traces: list[Trace] = [
            tr.to_pyrocko_trace()
            for tr in annotations
            if tr.stats.channel.endswith("P") or tr.stats.channel.endswith("S")
        ]

        annotation_p = PhaseNetImage(
            image_function=self,
            weight=self.weights["P"],
            phase=self.phase_map["P"],
            detection_half_width=self._detection_half_width(),
            traces=[tr for tr in annotated_traces if tr.channel.endswith("P")],
        )
        annotation_s = PhaseNetImage(
            image_function=self,
            weight=self.weights["S"],
            phase=self.phase_map["S"],
            detection_half_width=self._detection_half_width(),
            traces=[tr for tr in annotated_traces if tr.channel.endswith("S")],
        )

        for tr in annotation_s.traces + annotation_p.traces:
            tr.set_channel(tr.channel[-1])

        return [annotation_s, annotation_p]

    def get_provided_phases(self) -> tuple[str, ...]:
        return tuple(self.phase_map.values())
