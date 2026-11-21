"""Change points in a learning curve (Gallistel, Fairhurst & Balsam, 2004).

The analysis works on the cumulative record of a measure taken once per unit
(trial, time bin, episode) or on the times of events, and finds the points
where the rate changes. It follows the four stages of the original:

1. For each point ``N``, the putative change point ``R(N)`` is the earlier
   point where the cumulative record is farthest from the straight line that
   joins the origin and ``N``.
2. The evidence for a change at ``R(N)`` is the common logarithm of the odds
   against the no-change hypothesis.
3. The first ``N`` whose absolute logit exceeds the decision criterion makes
   ``R(N)`` a change point, and the record is cut there.
4. The analysis restarts with the change point as the origin, and ends when no
   point exceeds the criterion.

Only the binomial ("random rate") test is implemented. If the rate is
constant, the number of responses that fall after ``R`` among the ``Cum[N]``
responses up to ``N`` is binomial with ``p = (N - R) / N``; for event times the
number of events after ``R`` among ``N`` events is binomial with
``p = (T_N - T_R) / T_N``. With ``n_a`` that observed number, the logit is
``log10(P(X <= n_a) / (1 - P(X <= n_a) + P(X = n_a)))``. Both the numerator and
the denominator contain ``P(X = n_a)``, so this is a pseudo-logit; that reading
of the original follows the R port cpdetectoR. The chi-square, Kolmogorov-
Smirnov and t-test forms of the original are not implemented.

The original does not fix a default decision criterion; it varies the
criterion (logit 1.3 to 6, that is p 0.05 to 0.000001) to see its effect. The
default 4.0 is this module's choice. Because the test is repeated at every
point, stationary records also show change points: among 200 stationary 0/1
records of length 300 (p = 0.3), about 77 do at criterion 2 and 0 to 1 at
criterion 4.

``acquisition_summary`` gives the per-subject measures of the original: onset
latency (units before the first upward change), the mean of the second half of
the units (the original's estimate of the asymptote, which is not always
stable), the dynamic interval (units from the first upward change point to the
change point after which the level is at least 80% of that mean) and the first
fraction (the level between the first and second change points divided by that
mean).

Reference
---------
Gallistel, C. R., Fairhurst, S., & Balsam, P. (2004). The learning curve:
    Implications of a quantitative analysis. *Proceedings of the National
    Academy of Sciences*, *101*(36), 13124-13131.
    https://doi.org/10.1073/pnas.0404965101
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DEFAULT_CRITERION = 4.0
_DYNAMIC_FRACTION = 0.8


@dataclass(frozen=True)
class ChangePointResult:
    """Change points of one record.

    ``change_points`` are the numbers of units (or events) before each change,
    in increasing order. ``levels`` has one more entry: the rate between
    successive change points (mean per unit, or events per unit time).
    """

    change_points: tuple[int, ...]
    levels: tuple[float, ...]
    criterion: float
    n: int
    method: str


@dataclass(frozen=True)
class AcquisitionSummary:
    """Per-subject acquisition measures (``None`` where undefined)."""

    onset_latency: int | None
    asymptote: float
    dynamic_interval: int | None
    first_fraction: float | None
    levels: tuple[float, ...]


def _check_criterion(criterion: float) -> float:
    value = float(criterion)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"criterion must be a positive finite number, got {criterion!r}")
    return value


def _check_counts(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("values must be non-negative finite numbers")
    if not np.all(arr == np.floor(arr)):
        raise ValueError("values must be integer counts (0/1 or numbers of responses)")
    return arr.astype(np.int64)


def _putative(cum: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """For each N (1-based), the r (1-based, 0 if none) farthest from the chord.

    The deviation at r is ``|cum[r] - axis[r] * cum[N] / axis[N]|``; ties go to
    the smallest r.
    """
    n = cum.size
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dev = np.abs(cum[: i + 1] - axis[: i + 1] * (cum[i] / axis[i]))
        top = dev.max()
        if top > 0:
            out[i] = int(np.argmax(np.isclose(dev, top, rtol=0.0, atol=1e-9))) + 1
    return out


def _logit(n_after: float, trials: float, p: float) -> float:
    le = binom.cdf(n_after, trials, p)
    ge = 1.0 - le + binom.pmf(n_after, trials, p)
    if le <= 0:
        return -math.inf
    if ge <= 0:
        return math.inf
    return math.log10(le / ge)


def _first_change(cum: np.ndarray, axis: np.ndarray, timed: bool, criterion: float) -> int:
    """The first change point of a record (1-based), or 0 if none."""
    # The timed form measures events against time, so the roles swap.
    r_of = _putative(axis, cum) if timed else _putative(cum, axis)
    for i in range(1, cum.size):
        r = int(r_of[i])
        if r == 0:
            continue
        if timed:
            p = (cum[i] - cum[r - 1]) / cum[i]
            stat = _logit(i + 1 - r, i + 1, p)
        else:
            p = (i + 1 - r) / (i + 1)
            stat = _logit(cum[i] - cum[r - 1], cum[i], p)
        if abs(stat) > criterion:
            return r
    return 0


def _segment(x: np.ndarray, timed: bool, criterion: float) -> tuple[int, ...]:
    found: list[int] = []
    offset = 0
    rest = x
    while rest.size > 1:
        cum = np.cumsum(rest).astype(float)
        axis = np.arange(1, rest.size + 1, dtype=float)
        r = _first_change(cum, axis, timed, criterion)
        if r == 0:
            break
        offset += r
        found.append(offset)
        rest = rest[r:]
    return tuple(found)


def _edges(n: int, cps: Sequence[int]) -> list[tuple[int, int]]:
    bounds = [0, *cps, n]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def change_points(
    values: Sequence[float], *, criterion: float = DEFAULT_CRITERION
) -> ChangePointResult:
    """Change points of a record of non-negative integer counts per unit.

    ``values`` holds one count per unit (0/1 per trial, or the number of
    responses per time bin or episode). ``criterion`` is the logit the evidence
    must exceed (the original uses 1.3 to 6).
    """
    crit = _check_criterion(criterion)
    x = _check_counts(values)
    cps = _segment(x, timed=False, criterion=crit)
    levels = tuple(float(x[a:b].mean()) for a, b in _edges(x.size, cps))
    return ChangePointResult(cps, levels, crit, int(x.size), "counts")


def change_points_from_times(
    times: Sequence[float], *, criterion: float = DEFAULT_CRITERION
) -> ChangePointResult:
    """Change points of a record of event times (strictly increasing, > 0).

    Change points are numbers of events; levels are events per unit time.
    """
    crit = _check_criterion(criterion)
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(t)) or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite, positive and strictly increasing")
    intervals = np.diff(np.concatenate([[0.0], t]))
    cps = _segment(intervals, timed=True, criterion=crit)
    start = np.concatenate([[0.0], t])
    levels = tuple(float((b - a) / (start[b] - start[a])) for a, b in _edges(t.size, cps))
    return ChangePointResult(cps, levels, crit, int(t.size), "times")


def acquisition_summary(
    values: Sequence[float],
    change_points: ChangePointResult | Sequence[int],
    *,
    asymptote: float | None = None,
) -> AcquisitionSummary:
    """Onset latency, dynamic interval and first fraction of one subject.

    ``values`` is the per-unit record and ``change_points`` its change points
    (a ``ChangePointResult`` or the positions). ``asymptote`` defaults to the
    mean of the second half of the units.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("values must be a non-empty sequence of non-negative finite numbers")
    given = change_points
    cps = tuple(given.change_points if isinstance(given, ChangePointResult) else given)
    inside = all(int(c) == c and 0 < c < x.size for c in cps)
    if not inside or any(b <= a for a, b in zip(cps, cps[1:], strict=False)):
        raise ValueError("change points must be strictly increasing integers inside the record")
    cps = tuple(int(c) for c in cps)
    levels = tuple(float(x[a:b].mean()) for a, b in _edges(x.size, cps))
    level = float(x[x.size // 2 :].mean()) if asymptote is None else float(asymptote)
    if not math.isfinite(level) or level < 0:
        raise ValueError("asymptote must be a non-negative finite number")

    first_fraction = levels[1] / level if cps and level > 0 else None
    up = next((k for k in range(len(cps)) if levels[k + 1] > levels[k]), None)
    if up is None:
        return AcquisitionSummary(None, level, None, first_fraction, levels)
    reached = next(
        (j for j in range(up, len(cps)) if levels[j + 1] >= _DYNAMIC_FRACTION * level), None
    )
    dynamic = None if reached is None else cps[reached] - cps[up]
    return AcquisitionSummary(cps[up], level, dynamic, first_fraction, levels)
