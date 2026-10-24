"""Cumulative-record plotting in the canonical Ferster & Skinner style.

Canonical defaults reflect the Gerbrands C-3/C-4 cumulative recorder as
documented by Lattal (2004):

- Reinforcement mark: a short diagonal *pip* (~30 deg below horizontal)
  drawn as a pen deflection off the response-stepping trace.
- Response pen reset: every 550 responses the pen returns to baseline.
- Event pen: a separate on/off trace below the cumulative record records
  stimulus-event (e.g., SD) state in binary step form.

Reference
---------
Lattal, K. A. (2004). Steps and pips in the history of the cumulative
recorder. Journal of the Experimental Analysis of Behavior, 82(3), 329-355.
https://doi.org/10.1901/jeab.2004.82-329
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Union

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from session_recorder import (
    EVENT_TYPE_REINFORCER_START,
    EVENT_TYPE_RESPONSE,
    iter_records,
)

from ._cumulative_core import (
    CANONICAL_PIP_ANGLE_DEG,
    CANONICAL_PIP_LENGTH_PT,
    CANONICAL_RESET_RESPONSES,
    EventMarkerSpec,
    EventTime,
    draw_event_pen,
    draw_pip,
    find_response_y,
    wrap_cumulative,
)
from .themes import ThemeSpec, get_theme
from .themes.mpl_adapter import apply_theme_to_axes, theme_to_rcparams

__all__ = [
    "CANONICAL_PIP_ANGLE_DEG",
    "CANONICAL_PIP_LENGTH_PT",
    "CANONICAL_RESET_RESPONSES",
    "EventMarkerSpec",
    "EventTime",
    "build_cumulative_record_figure",
    "render_cumulative_record",
    "plot_cumulative_record",
]

logger = logging.getLogger(__name__)

ThemeArg = Union[str, ThemeSpec]
OutputTarget = Union[Path, IO[bytes]]

_ALLOWED_FORMATS = frozenset({"png", "svg", "pdf"})


def _resolve_theme(theme: ThemeArg) -> ThemeSpec:
    if isinstance(theme, ThemeSpec):
        return theme
    return get_theme(theme)


def _infer_format(output: OutputTarget, fmt: str | None) -> str:
    if fmt is not None:
        fmt_norm = fmt.lower().lstrip(".")
        if fmt_norm not in _ALLOWED_FORMATS:
            raise ValueError(
                f"Unsupported format {fmt!r}; expected one of {sorted(_ALLOWED_FORMATS)}."
            )
        return fmt_norm
    if isinstance(output, Path):
        ext = output.suffix.lower().lstrip(".")
        if ext in _ALLOWED_FORMATS:
            return ext
    return "png"


def build_cumulative_record_figure(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    *,
    theme: ThemeArg = "readable",
    pip_angle_deg: float = CANONICAL_PIP_ANGLE_DEG,
    pip_length_pt: float = CANONICAL_PIP_LENGTH_PT,
    reset_responses: int | None = CANONICAL_RESET_RESPONSES,
    show_event_pen: bool = True,
    event_markers: Mapping[str, Sequence[EventTime]] | None = None,
    event_marker_styles: Mapping[str, EventMarkerSpec] | None = None,
    time_range: tuple[float, float] | None = None,
) -> tuple[Figure, Axes, Axes | None]:
    """Build a canonical F&S cumulative-record figure.

    Returns the figure and its axes (main cumulative axis, optional
    event-pen axis). Callers are responsible for saving/closing the figure;
    this is the seam tests hook into so they can inspect artists directly.

    ``time_range`` sets the time-axis limits ``(start, end)`` in seconds;
    when omitted the axis spans the first to the last response.
    """
    spec = _resolve_theme(theme)

    with matplotlib.rc_context(theme_to_rcparams(spec)):
        fig = Figure(figsize=(spec.figure_width_in, spec.figure_height_in))

        event_ax: Axes | None
        if show_event_pen:
            axes = fig.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [4, 1]},
            )
            main_ax, event_ax = axes[0], axes[1]
        else:
            main_ax = fig.subplots()
            event_ax = None

        apply_theme_to_axes(main_ax, spec)
        if event_ax is not None:
            apply_theme_to_axes(event_ax, spec)

        xs_wrap, ys_wrap, response_marks = wrap_cumulative(
            list(response_times), reset_responses
        )

        if xs_wrap:
            main_line = Line2D(
                xs_wrap,
                ys_wrap,
                color=spec.foreground,
                linewidth=spec.line_width_pt,
                drawstyle="steps-post",
                label="_cumulative",
            )
            main_ax.add_line(main_line)
        if time_range is not None:
            x_lo, x_hi = float(time_range[0]), float(time_range[1])
        elif xs_wrap:
            x_lo, x_hi = min(xs_wrap), max(xs_wrap)
        else:
            x_lo, x_hi = 0.0, 1.0
        if xs_wrap:
            main_ax.set_xlim(x_lo, max(x_hi, x_lo + 1e-6))
            y_top = (
                reset_responses
                if reset_responses is not None
                else (max(ys_wrap) if ys_wrap else 1)
            )
            main_ax.set_ylim(0, max(y_top, max(ys_wrap) if ys_wrap else 1))

        main_ax.set_xlabel("Time (seconds)")
        main_ax.set_ylabel("Cumulative Responses")
        main_ax.set_title("Cumulative Record")

        if event_ax is not None and event_markers:
            draw_event_pen(
                event_ax,
                event_markers,
                event_marker_styles,
                default_color=spec.foreground,
                linewidth=spec.line_width_pt,
                x_lo=x_lo,
                x_hi=max(x_hi, x_lo + 1e-6),
            )
            event_ax.set_xlabel("Time (seconds)")
            main_ax.set_xlabel("")
        elif event_ax is not None:
            event_ax.set_yticks([])
            event_ax.set_xlabel("Time (seconds)")
            main_ax.set_xlabel("")

        # Legend proxies: only emit when the corresponding data is present.
        if xs_wrap:
            main_ax.add_line(
                Line2D(
                    [],
                    [],
                    color=spec.foreground,
                    linewidth=spec.line_width_pt,
                    label="Responses",
                )
            )
        if reinforcement_times:
            main_ax.add_line(
                Line2D(
                    [],
                    [],
                    color=spec.foreground,
                    linewidth=spec.line_width_pt,
                    label="Reinforcement",
                )
            )
        main_ax.legend(loc="best")

        fig.tight_layout()

        # Draw pips AFTER tight_layout so the data-space offset is computed
        # against the final axes transform; otherwise later layout shifts
        # subtly skew the display-space angle.
        for rt in reinforcement_times:
            y_at_r = find_response_y(response_marks, float(rt))
            draw_pip(
                main_ax,
                fig,
                float(rt),
                float(y_at_r),
                angle_deg=pip_angle_deg,
                length_pt=pip_length_pt,
                color=spec.foreground,
                linewidth=spec.line_width_pt,
            )
    return fig, main_ax, event_ax


def render_cumulative_record(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    output: OutputTarget,
    *,
    theme: ThemeArg = "readable",
    fmt: str | None = None,
    pip_angle_deg: float = CANONICAL_PIP_ANGLE_DEG,
    pip_length_pt: float = CANONICAL_PIP_LENGTH_PT,
    reset_responses: int | None = CANONICAL_RESET_RESPONSES,
    show_event_pen: bool = True,
    event_markers: Mapping[str, Sequence[EventTime]] | None = None,
    event_marker_styles: Mapping[str, EventMarkerSpec] | None = None,
    time_range: tuple[float, float] | None = None,
) -> bool:
    """Render a canonical Ferster & Skinner cumulative record to ``output``.

    Returns ``True`` on success, ``False`` if ``response_times`` is empty
    (no figure is written).
    """
    output_format = _infer_format(output, fmt)

    if not response_times:
        logger.info("No responses to plot; skipping render.")
        return False

    fig, _, _ = build_cumulative_record_figure(
        response_times,
        reinforcement_times,
        theme=theme,
        pip_angle_deg=pip_angle_deg,
        pip_length_pt=pip_length_pt,
        reset_responses=reset_responses,
        show_event_pen=show_event_pen,
        event_markers=event_markers,
        event_marker_styles=event_marker_styles,
        time_range=time_range,
    )
    try:
        fig.savefig(output, format=output_format)
    finally:
        fig.clear()
    return True


def plot_cumulative_record(
    log_path: Path,
    output_path: Path,
    *,
    theme: ThemeArg = "readable",
    pip_angle_deg: float = CANONICAL_PIP_ANGLE_DEG,
    pip_length_pt: float = CANONICAL_PIP_LENGTH_PT,
    reset_responses: int | None = CANONICAL_RESET_RESPONSES,
    show_event_pen: bool = True,
    event_markers: Mapping[str, Sequence[EventTime]] | None = None,
    event_marker_styles: Mapping[str, EventMarkerSpec] | None = None,
) -> None:
    """Plot a cumulative response record from an OKL v1 session log.

    Reads ``log_path`` via :func:`session_recorder.iter_records` so the
    OKL v1 stream parsing is owned by ``session-recorder`` (the format
    authority). Type discrimination uses
    :data:`session_recorder.EVENT_TYPE_RESPONSE` /
    :data:`session_recorder.EVENT_TYPE_REINFORCER_START` rather than
    string literals; unknown / future record types are ignored so that
    additive format evolution does not break this view.

    The time axis spans the whole log, from its first record (0 s) to its
    last record (for example the end of the final reinforcer), not only
    the span between the first and last responses.
    """
    records = list(iter_records(log_path))
    if not records:
        logger.info("No events found in log: %s", log_path)
        return

    # Anchor at the first record's timestamp so plots are always
    # zero-anchored regardless of clock origin (real-time / virtual /
    # resumed sessions).
    start_time = float(records[0].timestamp)

    response_times: list[float] = []
    reinforcement_times: list[float] = []
    for record in records:
        rel_time = float(record.timestamp) - start_time
        rtype = record.type
        if rtype == EVENT_TYPE_RESPONSE:
            response_times.append(rel_time)
        elif rtype == EVENT_TYPE_REINFORCER_START:
            reinforcement_times.append(rel_time)

    if not response_times:
        logger.info("No responses recorded in log: %s", log_path)
        return

    session_end = max(float(record.timestamp) for record in records) - start_time

    rendered = render_cumulative_record(
        response_times,
        reinforcement_times,
        output_path,
        theme=theme,
        pip_angle_deg=pip_angle_deg,
        pip_length_pt=pip_length_pt,
        reset_responses=reset_responses,
        show_event_pen=show_event_pen,
        event_markers=event_markers,
        event_marker_styles=event_marker_styles,
        time_range=(0.0, session_end),
    )
    if rendered:
        logger.info("Graph saved to %s", output_path)
