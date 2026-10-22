"""Visual-analysis aids for single-case designs: level, trend, variability,
immediacy and overlap.

Visual analysis of single-case data examines, within and between phases,
"(1.) level, (2.) trend, (3.) variability, (4.) immediacy of the effect,
(5.) overlap, and (6.) consistency of data patterns across similar
phases" (Kratochwill et al., 2010, p. 18). This module computes the
quantities those judgements are made on; it does not decide whether an
effect was demonstrated.

Definitions (Kratochwill et al., 2010, pp. 17-18, and how they are
computed here):

Level
    "the mean score for the data within a phase" — ``mean`` (the median
    is reported too).
Trend
    "the slope of the best-fitting straight line for the data within a
    phase" — ordinary least squares of the values on session number.
    Session numbers start at 0 in each phase, so ``intercept`` is the
    fitted value at the first session of the phase (as in scan
    ``trend()``).
Variability
    "the range or standard deviation of data about the best-fitting
    straight line" — ``residual_sd`` is ``sqrt(SS_residual / (n - 2))``;
    ``sd`` (about the mean) and ``minimum`` / ``maximum`` are reported
    too.
Immediacy of the effect
    "the change in level between the last three data points in one phase
    and the first three data points of the next" — mean of the first
    ``k`` points of the later phase minus mean of the last ``k`` points of
    the earlier phase, with ``k = min(3, n_earlier, n_later)``.
Overlap
    "the proportion of data from one phase that overlaps with data from
    the previous phase" — the proportion of later-phase points inside the
    earlier phase's ``[minimum, maximum]``. NAP and PND of the later phase
    against the earlier one (in the series' ``improvement`` direction, see
    :mod:`session_analyzer.analytics.single_case`) are reported alongside.

Consistency across similar phases is left to the reader: compare the
:class:`PhaseDescriptives` of phases with the same label.

Phases are contiguous runs of the same label (``by="run"``), so a
reversal design labelled ``A, A, B, B, A, A, B, B`` has four phases and
three contrasts. For alternating treatments, where the label changes
every session, use ``by="label"``: each label is pooled (trend in session
order) and no contrasts are formed.

References
----------
Kratochwill, T. R., Hitchcock, J. H., Horner, R. H., Levin, J. R., Odom,
S. L., Rindskopf, D. M., & Shadish, W. R. (2010). *Single-case designs
technical documentation*. What Works Clearinghouse.
https://ies.ed.gov/ncee/wwc/Docs/ReferenceResources/wwc_scd.pdf
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Literal

from .single_case import SingleCaseSeries, nap, pnd

GroupBy = Literal["run", "label"]

_GROUP_BY: tuple[str, ...] = ("run", "label")
IMMEDIACY_POINTS = 3


@dataclass(frozen=True)
class PhaseDescriptives:
    """Level, trend and variability of one phase.

    ``start`` / ``stop`` are the slice of the series the phase occupies
    (``stop`` exclusive). With ``by="label"`` they span from the first to
    the last session of the label, which may contain other labels.
    ``sd``, ``slope`` / ``intercept`` and ``residual_sd`` are NaN when the
    phase has fewer than 2, 2 and 3 observations respectively.
    """

    label: str
    n: int
    start: int
    stop: int
    mean: float
    median: float
    sd: float
    minimum: float
    maximum: float
    slope: float
    intercept: float
    residual_sd: float


@dataclass(frozen=True)
class PhaseContrast:
    """Change from one phase to the next (``to`` minus ``from``)."""

    from_label: str
    to_label: str
    level_change: float
    trend_change: float
    immediacy: float
    immediacy_points: int
    overlap: float
    nap: float
    pnd: float


@dataclass(frozen=True)
class VisualAnalysis:
    """Descriptives of every phase and contrasts of adjacent phases."""

    name: str | None
    by: GroupBy
    improvement: str
    phases: tuple[PhaseDescriptives, ...]
    contrasts: tuple[PhaseContrast, ...]


def describe_phases(series: SingleCaseSeries, by: GroupBy = "run") -> tuple[PhaseDescriptives, ...]:
    """Level, trend and variability of each phase, in order of appearance."""
    return tuple(
        _describe(label, positions, [series.values[i] for i in positions])
        for label, positions in _groups(series, by)
    )


def contrast_phases(series: SingleCaseSeries) -> tuple[PhaseContrast, ...]:
    """Contrasts of each pair of adjacent phases (contiguous runs)."""
    groups = _groups(series, "run")
    described = describe_phases(series, "run")
    contrasts = []
    for i in range(len(groups) - 1):
        before = [series.values[j] for j in groups[i][1]]
        after = [series.values[j] for j in groups[i + 1][1]]
        contrasts.append(
            _contrast(described[i], described[i + 1], before, after, series.improvement)
        )
    return tuple(contrasts)


def visual_analysis(series: SingleCaseSeries, by: GroupBy = "run") -> VisualAnalysis:
    """Descriptives per phase, plus adjacent-phase contrasts when ``by="run"``."""
    return VisualAnalysis(
        name=series.name,
        by=by,
        improvement=series.improvement,
        phases=describe_phases(series, by),
        contrasts=contrast_phases(series) if by == "run" else (),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _groups(series: SingleCaseSeries, by: str) -> list[tuple[str, list[int]]]:
    if by not in _GROUP_BY:
        raise ValueError(f"by must be one of {_GROUP_BY!r}, got {by!r}")
    indexed = list(enumerate(series.phases))
    if by == "run":
        return [
            (label, [i for i, _ in run])
            for label, run in groupby(indexed, key=lambda item: item[1])
        ]
    labels = dict.fromkeys(series.phases)
    return [(label, [i for i, p in indexed if p == label]) for label in labels]


def _describe(label: str, positions: Sequence[int], values: Sequence[float]) -> PhaseDescriptives:
    n = len(values)
    first = positions[0]
    xs = [p - first for p in positions]
    slope = intercept = residual_sd = math.nan
    if n >= 2:
        slope, intercept = statistics.linear_regression(xs, values)
    if n >= 3:
        ss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, values, strict=True))
        residual_sd = math.sqrt(ss / (n - 2))
    return PhaseDescriptives(
        label=label,
        n=n,
        start=first,
        stop=positions[-1] + 1,
        mean=statistics.fmean(values),
        median=float(statistics.median(values)),
        sd=statistics.stdev(values) if n >= 2 else math.nan,
        minimum=min(values),
        maximum=max(values),
        slope=slope,
        intercept=intercept,
        residual_sd=residual_sd,
    )


def _contrast(
    before: PhaseDescriptives,
    after: PhaseDescriptives,
    before_values: Sequence[float],
    after_values: Sequence[float],
    improvement: str,
) -> PhaseContrast:
    k = min(IMMEDIACY_POINTS, len(before_values), len(after_values))
    immediacy = statistics.fmean(after_values[:k]) - statistics.fmean(before_values[-k:])
    inside = sum(1 for y in after_values if before.minimum <= y <= before.maximum)
    return PhaseContrast(
        from_label=before.label,
        to_label=after.label,
        level_change=after.mean - before.mean,
        trend_change=after.slope - before.slope,
        immediacy=immediacy,
        immediacy_points=k,
        overlap=inside / len(after_values),
        nap=nap(before_values, after_values, improvement),  # type: ignore[arg-type]
        pnd=pnd(before_values, after_values, improvement),  # type: ignore[arg-type]
    )
