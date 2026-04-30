"""IRT-coded cumulative record, rendered in canonical F&S style.

Extends the basic cumulative record by color-coding each response
according to its preceding inter-response time (IRT). Useful for
visually assessing differential-reinforcement-of-low-rate (DRL) and
differential-reinforcement-of-other-behavior (DRO) performance, where
the contingency depends on whether an IRT meets or exceeds a threshold.

Semantics
---------
For DRL(t): responses whose preceding IRT is >= t satisfy the
contingency and are typically reinforced; shorter IRTs reset the
timer and are not reinforced. The two colors distinguish these
populations at a glance.

For DRO(t): the contingency reinforces the *absence* of responding
for at least t seconds, so any response resets the interval. In a
DRO context this plot highlights "failures" (IRTs < t) in one color
and "successes" (a reinforcer delivery without an intervening
response) as diagonal pips on the cumulative trace.

Reinforcement marks, pen reset, and the event pen follow the
canonical Gerbrands C-3/C-4 conventions documented in
cumulative_record.py.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Union

import matplotlib
from matplotlib.axes import Axes
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

logger = logging.getLogger(__name__)

ThemeArg = Union[str, ThemeSpec]
OutputTarget = Union[Path, IO[bytes]]

_ALLOWED_FORMATS = frozenset({"png", "svg", "pdf"})


def _infer_format_target(output: OutputTarget, fmt: str | None) -> str:
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


def _resolve_theme(theme: ThemeArg) -> ThemeSpec:
    return theme if isinstance(theme, ThemeSpec) else get_theme(theme)


def _infer_format(output_path: Path) -> str:
    ext = output_path.suffix.lower().lstrip(".")
    return ext if ext in _ALLOWED_FORMATS else "png"


def build_irt_coded_figure(
    response_points: Sequence[tuple[float, int, bool]],
    reinforcement_points: Sequence[tuple[float, int]],
    *,
    irt_threshold_sec: float,
    theme: ThemeArg = "readable",
    short_irt_color: str | None = None,
    long_irt_color: str | None = None,
    reinforcement_color: str | None = None,
    title: str | None = None,
    pip_angle_deg: float = CANONICAL_PIP_ANGLE_DEG,
    pip_length_pt: float = CANONICAL_PIP_LENGTH_PT,
    reset_responses: int | None = CANONICAL_RESET_RESPONSES,
    show_event_pen: bool = True,
    event_markers: Mapping[str, Sequence[EventTime]] | None = None,
    event_marker_styles: Mapping[str, EventMarkerSpec] | None = None,
) -> tuple[Figure, Axes, Axes | None]:
    """Build the IRT-coded cumulative-record figure.

    ``response_points`` is a sequence of ``(rel_time, cum_count, is_long_irt)``
    tuples. ``reinforcement_points`` is ``(rel_time, cum_count_at_event)``.
    ``cum_count`` values are the *unwrapped* cumulative counts; wrap logic
    is applied internally.
    """
    spec = _resolve_theme(theme)

    # Resolve colors.
    palette = list(spec.palette)
    _short = short_irt_color if short_irt_color is not None else palette[-1]
    _long = (
        long_irt_color
        if long_irt_color is not None
        else (palette[0] if len(palette) == 1 else palette[-2])
    )
    _reinf = reinforcement_color if reinforcement_color is not None else spec.foreground

    response_times = [p[0] for p in response_points]
    reinforcement_times = [p[0] for p in reinforcement_points]

    with matplotlib.rc_context(theme_to_rcparams(spec)):
        fig = Figure(figsize=(spec.figure_width_in, spec.figure_height_in))
        event_ax: Axes | None
        if show_event_pen:
            axes = fig.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
            )
            main_ax, event_ax = axes[0], axes[1]
        else:
            main_ax = fig.subplots()
            event_ax = None

        apply_theme_to_axes(main_ax, spec)
        if event_ax is not None:
            apply_theme_to_axes(event_ax, spec)

        # Build the wrapped backdrop cumulative trace (grey, thin).
        xs_wrap, ys_wrap, response_marks = wrap_cumulative(
            response_times, reset_responses
        )

        if xs_wrap:
            backdrop = Line2D(
                xs_wrap,
                ys_wrap,
                color=spec.foreground,
                linewidth=spec.line_width_pt * 0.8,
                alpha=0.5,
                drawstyle="steps-post",
                label="_cumulative",
            )
            main_ax.add_line(backdrop)
            main_ax.set_xlim(min(xs_wrap), max(xs_wrap) if max(xs_wrap) > min(xs_wrap) else min(xs_wrap) + 1e-6)
            y_top = (
                reset_responses
                if reset_responses is not None
                else (max(ys_wrap) if ys_wrap else 1)
            )
            main_ax.set_ylim(0, max(y_top, max(ys_wrap) if ys_wrap else 1))

        # Scatter IRT-coded response points at their wrapped y positions.
        short_xy: list[tuple[float, int]] = []
        long_xy: list[tuple[float, int]] = []
        for (t, _unwrapped, is_long), (_, disp_y) in zip(
            response_points, response_marks, strict=True
        ):
            if is_long:
                long_xy.append((t, disp_y))
            else:
                short_xy.append((t, disp_y))

        if short_xy:
            sx, sy = zip(*short_xy, strict=True)
            main_ax.scatter(
                list(sx),
                list(sy),
                marker="o",
                color=_short,
                s=20,
                label=f"IRT < {irt_threshold_sec:g}s",
            )
        if long_xy:
            lx, ly = zip(*long_xy, strict=True)
            main_ax.scatter(
                list(lx),
                list(ly),
                marker="s" if spec.color_blind_safe else "o",
                color=_long,
                s=20,
                label=f"IRT >= {irt_threshold_sec:g}s",
            )

        main_ax.set_xlabel("Time (seconds)")
        main_ax.set_ylabel("Cumulative Responses")
        main_ax.set_title(
            title
            if title is not None
            else f"IRT-coded Cumulative Record (threshold={irt_threshold_sec:g}s)"
        )

        # Proxy legend for reinforcement pip so it appears with the rest.
        if reinforcement_times:
            main_ax.add_line(
                Line2D(
                    [], [],
                    color=_reinf,
                    linewidth=spec.line_width_pt,
                    label="Reinforcement",
                )
            )
        main_ax.legend(loc="lower right")

        if event_ax is not None and event_markers:
            lo = min(xs_wrap) if xs_wrap else 0.0
            hi = max(xs_wrap) if xs_wrap else 1.0
            draw_event_pen(
                event_ax,
                event_markers,
                event_marker_styles,
                default_color=spec.foreground,
                linewidth=spec.line_width_pt,
                x_lo=lo,
                x_hi=hi,
            )
            event_ax.set_xlabel("Time (seconds)")
            main_ax.set_xlabel("")
        elif event_ax is not None:
            event_ax.set_yticks([])
            event_ax.set_xlabel("Time (seconds)")
            main_ax.set_xlabel("")

        fig.tight_layout()

        # Draw pips after layout for stable display-space angles.
        for rt in reinforcement_times:
            y_at_r = find_response_y(response_marks, float(rt))
            draw_pip(
                main_ax,
                fig,
                float(rt),
                float(y_at_r),
                angle_deg=pip_angle_deg,
                length_pt=pip_length_pt,
                color=_reinf,
                linewidth=spec.line_width_pt,
            )

    return fig, main_ax, event_ax


def plot_irt_coded_cumulative_record(
    log_path: Path,
    output_path: Path,
    *,
    irt_threshold_sec: float,
    short_irt_color: str | None = None,
    long_irt_color: str | None = None,
    reinforcement_color: str | None = None,
    title: str | None = None,
    theme: ThemeArg = "readable",
    pip_angle_deg: float = CANONICAL_PIP_ANGLE_DEG,
    pip_length_pt: float = CANONICAL_PIP_LENGTH_PT,
    reset_responses: int | None = CANONICAL_RESET_RESPONSES,
    show_event_pen: bool = True,
    event_markers: Mapping[str, Sequence[EventTime]] | None = None,
    event_marker_styles: Mapping[str, EventMarkerSpec] | None = None,
) -> None:
    """Plot an IRT-coded cumulative record with canonical F&S styling."""
    if irt_threshold_sec <= 0:
        raise ValueError(
            f"irt_threshold_sec must be positive, got {irt_threshold_sec!r}"
        )

    output_format = _infer_format(output_path)

    records = list(iter_records(log_path))
    if not records:
        logger.info("No events found in log: %s", log_path)
        return

    start_time = float(records[0].timestamp)

    response_points: list[tuple[float, int, bool]] = []
    reinforcement_points: list[tuple[float, int]] = []
    current_count = 0
    last_response_time: float | None = None

    for record in records:
        rel_t = float(record.timestamp) - start_time
        rtype = record.type
        if rtype == EVENT_TYPE_RESPONSE:
            current_count += 1
            is_long = (
                True
                if last_response_time is None
                else (rel_t - last_response_time) >= irt_threshold_sec
            )
            response_points.append((rel_t, current_count, is_long))
            last_response_time = rel_t
        elif rtype == EVENT_TYPE_REINFORCER_START:
            reinforcement_points.append((rel_t, current_count))

    if not response_points:
        logger.info("No responses recorded in log: %s", log_path)
        return

    fig, _, _ = build_irt_coded_figure(
        response_points,
        reinforcement_points,
        irt_threshold_sec=irt_threshold_sec,
        theme=theme,
        short_irt_color=short_irt_color,
        long_irt_color=long_irt_color,
        reinforcement_color=reinforcement_color,
        title=title,
        pip_angle_deg=pip_angle_deg,
        pip_length_pt=pip_length_pt,
        reset_responses=reset_responses,
        show_event_pen=show_event_pen,
        event_markers=event_markers,
        event_marker_styles=event_marker_styles,
    )
    try:
        fig.savefig(output_path, format=output_format)
    finally:
        fig.clear()
    logger.info("Graph saved to %s", output_path)


def _classify_irts(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    irt_threshold_sec: float,
) -> tuple[list[tuple[float, int, bool]], list[tuple[float, int]]]:
    """Classify responses into long/short IRT populations.

    Returns ``(response_points, reinforcement_points)`` as consumed by
    :func:`build_irt_coded_figure`. ``response_times`` is assumed to be
    monotonically non-decreasing (the LiveAggregator invariant for
    response_times); under that assumption the reinforcement-time
    classification is O(M log N) via bisect.
    """
    response_points: list[tuple[float, int, bool]] = []
    last_t: float | None = None
    rt_floats: list[float] = []
    for i, t in enumerate(response_times, start=1):
        tf = float(t)
        is_long = True if last_t is None else (tf - last_t) >= irt_threshold_sec
        response_points.append((tf, i, is_long))
        rt_floats.append(tf)
        last_t = tf

    import bisect

    reinforcement_points: list[tuple[float, int]] = []
    for r_t in reinforcement_times:
        cum_at_r = bisect.bisect_right(rt_floats, float(r_t))
        reinforcement_points.append((float(r_t), cum_at_r))

    return response_points, reinforcement_points


def render_irt_coded_cumulative_record(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    output: OutputTarget,
    *,
    irt_threshold_sec: float,
    theme: ThemeArg = "readable",
    fmt: str | None = None,
    short_irt_color: str | None = None,
    long_irt_color: str | None = None,
    reinforcement_color: str | None = None,
    title: str | None = None,
    pip_angle_deg: float = CANONICAL_PIP_ANGLE_DEG,
    pip_length_pt: float = CANONICAL_PIP_LENGTH_PT,
    reset_responses: int | None = CANONICAL_RESET_RESPONSES,
    show_event_pen: bool = True,
    event_markers: Mapping[str, Sequence[EventTime]] | None = None,
    event_marker_styles: Mapping[str, EventMarkerSpec] | None = None,
) -> bool:
    """Render a canonical F&S IRT-coded cumulative record to ``output``.

    In-memory counterpart to :func:`plot_irt_coded_cumulative_record`. Used
    by HTTP endpoint handlers that already have live snapshot arrays.

    Returns ``True`` on success, ``False`` when ``response_times`` is empty
    (no figure is written).
    """
    if irt_threshold_sec <= 0:
        raise ValueError(
            f"irt_threshold_sec must be positive, got {irt_threshold_sec!r}"
        )

    output_format = _infer_format_target(output, fmt)

    if not response_times:
        logger.info("No responses to plot; skipping render.")
        return False

    response_points, reinforcement_points = _classify_irts(
        response_times, reinforcement_times, irt_threshold_sec
    )

    fig, _, _ = build_irt_coded_figure(
        response_points,
        reinforcement_points,
        irt_threshold_sec=irt_threshold_sec,
        theme=theme,
        short_irt_color=short_irt_color,
        long_irt_color=long_irt_color,
        reinforcement_color=reinforcement_color,
        title=title,
        pip_angle_deg=pip_angle_deg,
        pip_length_pt=pip_length_pt,
        reset_responses=reset_responses,
        show_event_pen=show_event_pen,
        event_markers=event_markers,
        event_marker_styles=event_marker_styles,
    )
    try:
        fig.savefig(output, format=output_format)
    finally:
        fig.clear()
    return True
