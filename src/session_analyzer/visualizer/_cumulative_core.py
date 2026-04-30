"""Internal helpers shared by ``cumulative_record`` and ``cumulative_diff``.

Not part of the public API. Only the two sibling modules in this package
should import from here.

Reference
---------
Lattal, K. A. (2004). Steps and pips in the history of the cumulative
recorder. Journal of the Experimental Analysis of Behavior, 82(3), 329-355.
https://doi.org/10.1901/jeab.2004.82-329
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

CANONICAL_PIP_ANGLE_DEG: float = -30.0
"""Canonical Ferster & Skinner reinforcement-pip angle, measured in display
coordinates as degrees from horizontal. Negative = tick slopes downward to
the right (Lattal, 2004)."""

CANONICAL_PIP_LENGTH_PT: float = 7.0
"""Canonical pip length in typographic points (~2 mm on Gerbrands paper)."""

CANONICAL_RESET_RESPONSES: int = 550
"""Canonical Gerbrands C-3/C-4 reset threshold (Lattal, 2004)."""


EventTime = Union[float, tuple[float, str]]
"""Either a bare time (rendered as a point marker) or ``(time, "on"|"off")``
for binary event-pen transitions."""


@dataclass(frozen=True)
class EventMarkerSpec:
    """Style for a named event channel drawn on the event pen."""

    label: str
    symbol: str = "|"
    color: str | None = None


def wrap_cumulative(
    response_times: Sequence[float], reset_responses: int | None
) -> tuple[list[float], list[float], list[tuple[float, int]]]:
    """Compute the sawtooth (pen-reset) cumulative trace.

    When ``reset_responses`` is set, at each multiple of ``reset_responses``
    the pen drops from the threshold back to 0 at the same x coordinate and
    the next response starts at y=1.

    Pip anchor semantics at the reset boundary: the returned ``mark_index``
    records each response's *post-reset* displayed y. A reinforcer delivered
    exactly on the reset response is therefore placed on the freshly-reset
    lower segment (y=1), not at the pre-reset peak. This matches canonical
    pen behaviour: the pen is already at baseline when the tick fires.

    Returns
    -------
    xs, ys:
        Step arrays suitable for ``ax.step(xs, ys, where="post")``.
    mark_index:
        For each input response (in order), ``(x, displayed_y)``.
    """
    xs: list[float] = []
    ys: list[float] = []
    mark_index: list[tuple[float, int]] = []

    current = 0
    for t in response_times:
        current += 1
        if reset_responses is not None and current > reset_responses:
            xs.append(t)
            ys.append(0)
            current = 1
        xs.append(t)
        ys.append(current)
        mark_index.append((t, current))
    return xs, ys, mark_index


def find_response_y(
    response_marks: Sequence[tuple[float, int]], t_event: float
) -> int:
    """Return displayed cumulative y at ``t_event`` via binary search.

    Precondition: ``response_marks`` must be sorted ascending by x. This
    invariant holds for any output of :func:`wrap_cumulative` because the
    function consumes ``response_times`` in order; violating it silently
    yields wrong values.
    """
    if not response_marks:
        return 0
    xs = [rm[0] for rm in response_marks]
    idx = bisect.bisect_right(xs, t_event) - 1
    if idx < 0:
        return 0
    return response_marks[idx][1]


def _data_point_plus_offset(
    ax: Axes,
    fig: Figure,
    x: float,
    y: float,
    dx_pt: float,
    dy_pt: float,
) -> tuple[float, float]:
    """Convert data point + point-offset to a data-space endpoint.

    Ensures the pip has a fixed visual length/angle independent of axis
    scale. Must be called after ``tight_layout`` so the axes transform is
    finalised.
    """
    px_per_pt = fig.dpi / 72.0
    start_disp = ax.transData.transform((x, y))
    end_disp = start_disp + np.array([dx_pt * px_per_pt, dy_pt * px_per_pt])
    end_data = ax.transData.inverted().transform(end_disp)
    return float(end_data[0]), float(end_data[1])


def draw_pip(
    ax: Axes,
    fig: Figure,
    x: float,
    y: float,
    *,
    angle_deg: float,
    length_pt: float,
    color: str,
    linewidth: float,
) -> None:
    """Draw a single reinforcement pip as a Line2D labeled ``_pip``."""
    rad = math.radians(angle_deg)
    dx_pt = length_pt * math.cos(rad)
    dy_pt = length_pt * math.sin(rad)
    x_end, y_end = _data_point_plus_offset(ax, fig, x, y, dx_pt, dy_pt)
    line = Line2D(
        [x, x_end],
        [y, y_end],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        label="_pip",
    )
    ax.add_line(line)


def draw_event_pen(
    ax: Axes,
    event_markers: Mapping[str, Sequence[EventTime]],
    event_marker_styles: Mapping[str, EventMarkerSpec] | None,
    default_color: str,
    linewidth: float,
    x_lo: float,
    x_hi: float,
) -> None:
    """Draw event channels on the event-pen axis."""
    styles = dict(event_marker_styles or {})
    ax.set_ylim(-0.5, max(1.0, len(event_markers)) - 0.5)
    ax.set_yticks(list(range(len(event_markers))))
    ax.set_yticklabels(list(event_markers.keys()))
    ax.set_xlim(x_lo, x_hi)
    ax.grid(False)

    for lane_idx, (channel, entries) in enumerate(event_markers.items()):
        spec = styles.get(channel, EventMarkerSpec(label=channel))
        color = spec.color or default_color

        on_off: list[tuple[float, str]] = []
        point_times: list[float] = []
        for item in entries:
            if isinstance(item, tuple) and len(item) == 2 and item[1] in ("on", "off"):
                on_off.append((float(item[0]), str(item[1])))
            else:
                point_times.append(float(item))  # type: ignore[arg-type]

        if on_off:
            on_off_sorted = sorted(on_off, key=lambda p: p[0])
            xs: list[float] = [x_lo]
            ys: list[float] = [lane_idx]
            for t, state in on_off_sorted:
                xs.append(t)
                ys.append(lane_idx + (0.4 if state == "on" else 0.0))
            xs.append(x_hi)
            ys.append(ys[-1])
            step_line = Line2D(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                label=channel,
                drawstyle="steps-post",
            )
            ax.add_line(step_line)

        if point_times:
            ax.scatter(
                point_times,
                [lane_idx] * len(point_times),
                marker=spec.symbol,
                color=color,
                s=40,
                linewidths=linewidth,
                label=channel if not on_off else f"_{channel}_pts",
            )
