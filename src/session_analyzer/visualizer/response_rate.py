"""Windowed response-rate timeline visualizer.

Computes a moving-window rate (responses per minute) over a session and
renders it as a themed timeline plot. Complements the cumulative record
by exposing rate dynamics directly.

Window semantics
----------------
For each time point ``t`` in a regular grid (``step_sec`` spacing), the
reported rate is the number of responses in ``(t - window_sec, t]``
divided by ``window_sec``, scaled to responses per minute. The first
bin is at ``t = response_times[0] + window_sec`` so every reported
bin sees a full window.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Union

import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .themes import ThemeSpec, get_theme
from .themes.mpl_adapter import apply_theme_to_axes, theme_to_rcparams

__all__ = [
    "compute_windowed_rate",
    "render_response_rate_timeline",
]

logger = logging.getLogger(__name__)

ThemeArg = Union[str, ThemeSpec]
OutputTarget = Union[Path, IO[bytes]]

_ALLOWED_FORMATS = frozenset({"png", "svg", "pdf"})


def _resolve_theme(theme: ThemeArg) -> ThemeSpec:
    return theme if isinstance(theme, ThemeSpec) else get_theme(theme)


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


def compute_windowed_rate(
    response_times: Sequence[float],
    *,
    window_sec: float,
    step_sec: float,
) -> tuple[list[float], list[float]]:
    """Return ``(times, rates_rpm)`` for a moving-window rate.

    ``response_times`` must be monotonically non-decreasing (the
    LiveAggregator invariant). Rates are scaled to responses per minute.

    Raises
    ------
    ValueError:
        If ``window_sec <= 0`` or ``step_sec <= 0``.
    """
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec!r}")
    if step_sec <= 0:
        raise ValueError(f"step_sec must be positive, got {step_sec!r}")

    if not response_times:
        return [], []

    rt = [float(x) for x in response_times]
    start = rt[0] + window_sec
    end = rt[-1]
    if end < start:
        # Data span shorter than a single window; still emit one bin,
        # but pin it to the last observed response so the x coordinate
        # stays within actual session time (avoiding a point floating
        # beyond the observation interval).
        times = [rt[-1]]
        lo = rt[-1] - window_sec
        count = bisect.bisect_right(rt, rt[-1]) - bisect.bisect_right(rt, lo)
        return times, [count / window_sec * 60.0]

    times: list[float] = []
    rates: list[float] = []
    t = start
    while t <= end + 1e-9:
        lo = t - window_sec
        count = bisect.bisect_right(rt, t) - bisect.bisect_right(rt, lo)
        times.append(t)
        rates.append(count / window_sec * 60.0)
        t += step_sec
    return times, rates


def render_response_rate_timeline(
    response_times: Sequence[float],
    output: OutputTarget,
    *,
    window_sec: float,
    step_sec: float,
    theme: ThemeArg = "readable",
    fmt: str | None = None,
    reinforcement_times: Sequence[float] | None = None,
    title: str | None = None,
) -> bool:
    """Render a themed response-rate timeline.

    Returns ``True`` on success, ``False`` when ``response_times`` is empty
    (no figure is written).
    """
    spec = _resolve_theme(theme)
    output_format = _infer_format(output, fmt)

    if not response_times:
        logger.info("No responses to plot; skipping render.")
        return False

    times, rates = compute_windowed_rate(
        response_times, window_sec=window_sec, step_sec=step_sec
    )

    with matplotlib.rc_context(theme_to_rcparams(spec)):
        fig = Figure(figsize=(spec.figure_width_in, spec.figure_height_in))
        ax = fig.subplots()
        apply_theme_to_axes(ax, spec)

        if times:
            ax.add_line(
                Line2D(
                    times,
                    rates,
                    color=spec.foreground,
                    linewidth=spec.line_width_pt,
                    label="Rate",
                )
            )
            ax.set_xlim(times[0], times[-1] if len(times) > 1 else times[0] + 1.0)
            max_rate = max(rates) if rates else 1.0
            ax.set_ylim(0.0, max_rate * 1.1 if max_rate > 0 else 1.0)

        if reinforcement_times:
            palette = list(spec.palette)
            reinf_color = palette[-1] if len(palette) > 1 else spec.foreground
            for r_t in reinforcement_times:
                ax.axvline(
                    float(r_t),
                    color=reinf_color,
                    linewidth=spec.line_width_pt * 0.5,
                    alpha=0.5,
                )
            # Legend proxy
            ax.add_line(
                Line2D(
                    [],
                    [],
                    color=reinf_color,
                    linewidth=spec.line_width_pt * 0.5,
                    alpha=0.5,
                    label="Reinforcement",
                )
            )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"Responses / min (window = {window_sec:g}s)")
        ax.set_title(title if title is not None else "Response Rate Timeline")
        ax.legend(loc="best")

        fig.tight_layout()
        try:
            fig.savefig(output, format=output_format)
        finally:
            fig.clear()
    return True
