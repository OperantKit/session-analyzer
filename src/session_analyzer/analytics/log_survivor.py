"""Log-survivor analysis of inter-response times (IRTs).

The empirical survival function ``S(t) = P(IRT > t)`` plotted on a log
scale is a classical diagnostic for the temporal structure of responding
(Shull, 1991; Shull, Gaynor, & Grimes, 2001):

- A single Poisson process produces a *linear* log-survivor with slope
  equal to the negative response rate.
- A bout-structured response stream produces a *bilinear* (broken-line)
  log-survivor: a steep initial limb (within-bout, fast IRTs) and a
  shallow tail (inter-bout, slow IRTs).

This module provides both the raw curve (for plotting) and a simple
broken-line least-squares fit that returns the knot time and the two
slopes.

References
----------
Shull, R. L. (1991). Mathematical description of operant behavior: An
introduction. In I. H. Iversen & K. A. Lattal (Eds.),
*Experimental analysis of behavior* (Vol. 2, pp. 243-282). Elsevier.

Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate
viewed as engagement bouts. *Journal of the Experimental Analysis of
Behavior*, 75(3), 247-274. https://doi.org/10.1901/jeab.2001.75-247
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_BILINEAR_IRTS: int = 20


@dataclass(frozen=True)
class LogSurvivorCurve:
    """Empirical log-survivor curve of IRTs.

    Attributes
    ----------
    times : numpy.ndarray
        Sorted ascending IRTs (seconds).
    survival : numpy.ndarray
        Empirical survival function ``S(t_i) = (n - i) / n + 1/n`` for the
        ``i``-th smallest IRT; bounded in ``[1/n, 1]``.
    log_survival : numpy.ndarray
        ``log(survival)``.
    n_irts : int
        Number of valid (positive) IRTs.
    """

    times: np.ndarray
    survival: np.ndarray
    log_survival: np.ndarray
    n_irts: int


@dataclass(frozen=True)
class LogSurvivorBilinearFit:
    """Broken-line least-squares fit on the log-survivor curve.

    Attributes
    ----------
    knot_time_sec : float
        Time (seconds) at which the two linear pieces join.
    slope_fast : float
        Slope of the steep first segment (within-bout, < slope_slow).
    slope_slow : float
        Slope of the shallow second segment (inter-bout).
    intercept_fast : float
        log-survival intercept of the first segment at ``t=0``.
    intercept_slow : float
        log-survival intercept of the second segment.
    sse : float
        Sum of squared residuals at the optimal knot.
    n_irts : int
        Number of IRTs used in the fit.
    """

    knot_time_sec: float
    slope_fast: float
    slope_slow: float
    intercept_fast: float
    intercept_slow: float
    sse: float
    n_irts: int


def compute_log_survivor(irts: list[float]) -> LogSurvivorCurve | None:
    """Compute the empirical log-survivor curve.

    Parameters
    ----------
    irts : list of float
        IRTs in seconds. Non-positive values are dropped.

    Returns
    -------
    LogSurvivorCurve or None
        ``None`` if no positive IRTs are available.
    """
    valid = [x for x in irts if x > 0.0]
    if not valid:
        return None

    times = np.sort(np.asarray(valid, dtype=np.float64))
    n = len(times)
    # S(t_i) at the i-th sorted IRT: starts at 1, decreases by 1/n at each
    # successive IRT, ends at 1/n.
    ranks = np.arange(n, dtype=np.float64)
    survival = (n - ranks) / n
    log_survival = np.log(survival)

    return LogSurvivorCurve(
        times=times,
        survival=survival,
        log_survival=log_survival,
        n_irts=n,
    )


def _segment_sse(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit; return (slope, intercept, sse)."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    sse = float(np.sum((y - pred) ** 2))
    return float(a), float(b), sse


def fit_bilinear_log_survivor(
    irts: list[float],
) -> LogSurvivorBilinearFit | None:
    """Fit a broken-line model to the log-survivor curve.

    Searches the interior of the sorted IRT array for the knot index
    that minimises the sum of two-segment OLS residuals.

    Parameters
    ----------
    irts : list of float
        IRTs in seconds.

    Returns
    -------
    LogSurvivorBilinearFit or None
        ``None`` when fewer than 20 valid IRTs are available.
    """
    curve = compute_log_survivor(irts)
    if curve is None or curve.n_irts < _MIN_BILINEAR_IRTS:
        return None

    t = curve.times
    y = curve.log_survival
    n = curve.n_irts

    # Need at least 3 points on each side
    best_sse = np.inf
    best_k = -1
    best_left = (0.0, 0.0, 0.0)
    best_right = (0.0, 0.0, 0.0)
    for k in range(3, n - 3):
        left = _segment_sse(t[: k + 1], y[: k + 1])
        right = _segment_sse(t[k:], y[k:])
        sse = left[2] + right[2]
        if sse < best_sse:
            best_sse = sse
            best_k = k
            best_left = left
            best_right = right

    if best_k < 0:
        return None

    slope_left, intercept_left, _ = best_left
    slope_right, intercept_right, _ = best_right

    # Order so that "fast" = steeper (more negative) slope.
    if slope_left <= slope_right:
        slope_fast, intercept_fast = slope_left, intercept_left
        slope_slow, intercept_slow = slope_right, intercept_right
    else:
        slope_fast, intercept_fast = slope_right, intercept_right
        slope_slow, intercept_slow = slope_left, intercept_left

    return LogSurvivorBilinearFit(
        knot_time_sec=round(float(t[best_k]), 6),
        slope_fast=round(slope_fast, 6),
        slope_slow=round(slope_slow, 6),
        intercept_fast=round(intercept_fast, 6),
        intercept_slow=round(intercept_slow, 6),
        sse=round(float(best_sse), 6),
        n_irts=n,
    )
