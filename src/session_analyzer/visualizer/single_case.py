"""Single-case design graph with phase-change lines, level and trend lines.

Draws the conventional applied-behavior-analysis line graph: one point
per session, the data path broken at each phase change, a dashed vertical
phase-change line and the phase label above each phase. Each phase also
gets a horizontal line at its mean (level) and its least-squares line
(trend), from :func:`session_analyzer.analytics.visual_analysis.describe_phases`.

Several series (a multiple-baseline design) are stacked as panels that
share the session axis. With ``by="label"`` (alternating treatments) the
points of each label are joined into one path and no phase-change lines
are drawn.

Line ``gid`` values identify the elements (``"data"``, ``"phase-change"``,
``"level"``, ``"trend"``) for callers that post-process the figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import IO

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..analytics.single_case import SingleCaseSeries
from ..analytics.visual_analysis import GroupBy, describe_phases
from .themes import ThemeSpec, get_theme
from .themes.mpl_adapter import apply_theme_to_axes, theme_to_rcparams

__all__ = [
    "build_single_case_figure",
    "render_single_case_plot",
]

ThemeArg = str | ThemeSpec
OutputTarget = Path | IO[bytes]

_ALLOWED_FORMATS = frozenset({"png", "svg", "pdf"})


def build_single_case_figure(
    series: SingleCaseSeries | Sequence[SingleCaseSeries],
    *,
    by: GroupBy = "run",
    theme: ThemeArg = "readable",
    show_level: bool = True,
    show_trend: bool = True,
    ylabel: str = "Value",
) -> Figure:
    """Build the figure; one panel per series, sharing the session axis."""
    cases = [series] if isinstance(series, SingleCaseSeries) else list(series)
    if not cases:
        raise ValueError("at least one series is required")
    spec = theme if isinstance(theme, ThemeSpec) else get_theme(theme)
    with matplotlib.rc_context(theme_to_rcparams(spec)):
        fig = Figure(figsize=(spec.figure_width_in, spec.figure_height_in * len(cases)))
        axes = fig.subplots(len(cases), 1, sharex=True, squeeze=False)[:, 0]
        for ax, case in zip(axes, cases, strict=True):
            apply_theme_to_axes(ax, spec)
            _draw_case(ax, case, by, spec, show_level, show_trend)
            ax.set_ylabel(ylabel)
        axes[-1].set_xlabel("Session")
        fig.tight_layout()
    return fig


def render_single_case_plot(
    series: SingleCaseSeries | Sequence[SingleCaseSeries],
    output: OutputTarget,
    *,
    by: GroupBy = "run",
    theme: ThemeArg = "readable",
    fmt: str | None = None,
    show_level: bool = True,
    show_trend: bool = True,
    ylabel: str = "Value",
) -> None:
    """Write the single-case graph to ``output`` (PNG, SVG or PDF)."""
    fig = build_single_case_figure(
        series, by=by, theme=theme, show_level=show_level, show_trend=show_trend, ylabel=ylabel
    )
    try:
        fig.savefig(output, format=_infer_format(output, fmt))
    finally:
        fig.clear()


def _draw_case(
    ax: Axes,
    case: SingleCaseSeries,
    by: GroupBy,
    spec: ThemeSpec,
    show_level: bool,
    show_trend: bool,
) -> None:
    sessions = list(range(1, len(case.values) + 1))
    phases = describe_phases(case, by)
    markers = list(spec.marker_cycle) or ["o"]
    lw = spec.line_width_pt
    for index, phase in enumerate(phases):
        if by == "run":
            xs = sessions[phase.start : phase.stop]
            marker = markers[0]
        else:
            xs = [s for s, p in zip(sessions, case.phases, strict=True) if p == phase.label]
            marker = markers[index % len(markers)]
        ys = [case.values[x - 1] for x in xs]
        ax.plot(
            xs, ys, color=spec.foreground, linewidth=lw, linestyle="-", marker=marker,
            markersize=4, gid="data", label=phase.label if by == "label" else None,
        )
        # Session numbers in the descriptives start at 0 in each phase.
        x0, x1 = xs[0], xs[-1]
        if show_level:
            ax.plot([x0, x1], [phase.mean, phase.mean], color=spec.foreground,
                    linewidth=lw * 0.6, linestyle=":", gid="level")
        if show_trend and phase.n >= 2:
            ax.plot([x0, x1], [phase.intercept, phase.intercept + phase.slope * (x1 - x0)],
                    color=spec.foreground, linewidth=lw * 0.6, linestyle="--", gid="trend")
        if by == "run":
            ax.text((x0 + x1) / 2, 1.0, phase.label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom")
            if index > 0:
                ax.axvline(x0 - 0.5, color=spec.foreground, linewidth=lw * 0.8,
                           linestyle="--", gid="phase-change")
    if by == "label":
        ax.legend(loc="best")
    if case.name is not None:
        ax.text(0.99, 0.04, case.name, transform=ax.transAxes, ha="right", va="bottom")
    ax.set_xlim(0.5, len(sessions) + 0.5)


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
