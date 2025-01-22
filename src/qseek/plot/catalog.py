from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np

from qseek.magnitudes.local_magnitude import LocalMagnitude
from qseek.magnitudes.moment_magnitude import MomentMagnitude
from qseek.models.catalog import EventCatalog
from qseek.plot.base import BasePlot, LassieFigure

HOUR = 3600
DAY = 24 * HOUR
KM = 1e3


DetectionAttribute = Literal["semblance", "magnitude"]


class DetectionsDistribution(BasePlot):
    attribute: ClassVar[DetectionAttribute] = "semblance"

    def get_figure(self) -> LassieFigure:
        return self.create_figure(attribute=self.attribute)

    def create_figure(
        self,
        attribute: DetectionAttribute = "semblance",
    ) -> LassieFigure:
        figure = self.new_figure("event-distribution-{attribute}.png")
        axes = figure.get_axes()

        detections = self.detections

        values = [getattr(detection, attribute) for detection in detections]
        times = [
            detection.time.replace(tzinfo=None)  # Stupid fix for matplotlib bug
            for detection in detections
        ]

        axes.scatter(times, values, cmap="viridis_r", c=values, s=3, alpha=0.5)
        axes.set_ylabel(attribute.capitalize())
        axes.grid(axis="x", alpha=0.3)
        # axes.figure.autofmt_xdate()

        cum_axes = axes.twinx()

        cummulative_detections = np.cumsum(np.ones(detections.n_detections))
        cum_axes.plot(
            times,
            cummulative_detections,
            color="black",
            alpha=0.8,
            label="Cumulative Detections",
        )
        cum_axes.set_ylabel("# Detections")

        to_timestamps = np.vectorize(lambda d: d.timestamp())
        from_timestamps = np.vectorize(
            lambda t: datetime.fromtimestamp(t, tz=timezone.utc)
        )
        detection_time_span = times[-1] - times[0]
        daily_rate, edges = np.histogram(
            to_timestamps(times),
            bins=detection_time_span.days,
        )

        cum_axes.stairs(
            daily_rate,
            from_timestamps(edges),
            color="gray",
            fill=True,
            alpha=0.5,
            label="Daily Detections",
        )
        cum_axes.legend(loc="upper left", fontsize="small")
        return figure


def plot_magnitudes(catalog: EventCatalog):
    local_mags = []
    mw_mags = []
    depths = []
    for ev in catalog:
        local_mag = None
        mw_mag = None
        for mag in ev.magnitudes[::-1]:
            if local_mag and mw_mag:
                break
            if isinstance(mag, LocalMagnitude) and not local_mag:
                local_mag = mag
            elif isinstance(mag, MomentMagnitude) and not mw_mag:
                mw_mag = mag
        if not local_mag or not mw_mag:
            continue
        depths.append(ev.effective_depth)
        local_mags.append(local_mag.average)
        mw_mags.append(mw_mag.magnitude)

    depth_sorted = np.argsort(depths)
    depths = np.array(depths)[depth_sorted] / KM
    local_mags = np.array(local_mags)[depth_sorted]
    mw_mags = np.array(mw_mags)[depth_sorted]

    fig, ax = plt.subplots()
    c = ax.scatter(local_mags, mw_mags, s=3, c=depths, alpha=0.2, cmap="magma")
    fig.colorbar(c, ax=ax, label="Depth [km]")
    ax.grid(alpha=0.3)
    ax.set_xlabel("$M_L$")
    ax.set_ylabel("$M_W$")
    ax.set_aspect("equal")
    ax.set_xlim(-0.0, 5.5)
    ax.set_ylim(-0.0, 5.5)
    ax.plot(
        [-1.0, 6.0],
        [-1.0, 6.0],
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    ax.text(
        0.99,
        0.01,
        f"$N_{{Ev}}={len(local_mags)}$",
        transform=ax.transAxes,
        ha="right",
        alpha=0.5,
    )
    plt.show()
