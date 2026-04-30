"""Survival / extinction curve estimation.

The Kaplan-Meier (1958) estimator produces a non-parametric estimate
of the survival function ``S(t) = P(T > t)`` from time-to-event data,
accommodating right-censored observations. In behavior analysis it is
applied to time-to-extinction, time-to-resurgence, latency-to-first-
response, and related failure-time outcomes.

Reference
---------
Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
    incomplete observations. *Journal of the American Statistical
    Association*, *53*(282), 457-481.
    https://doi.org/10.1080/01621459.1958.10501452
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SurvivalCurve:
    """Kaplan-Meier survival function estimate.

    Attributes
    ----------
    times : tuple of float
        Distinct event times at which the curve is updated, ascending.
    survival : tuple of float
        ``S(t)`` immediately after each event time.
    at_risk : tuple of int
        Number of subjects at risk just prior to each event time.
    events : tuple of int
        Number of uncensored events occurring at each time.
    n : int
        Total number of observations supplied.
    n_events : int
        Total number of uncensored events.
    median_survival_time : float or None
        First time at which ``S(t) <= 0.5``; ``None`` if the curve
        never reaches 0.5 (e.g. many censored observations).
    """

    times: tuple[float, ...]
    survival: tuple[float, ...]
    at_risk: tuple[int, ...]
    events: tuple[int, ...]
    n: int
    n_events: int
    median_survival_time: float | None


def kaplan_meier(
    times: Sequence[float],
    censored: Sequence[bool] | None = None,
) -> SurvivalCurve:
    """Estimate the Kaplan-Meier survival function.

    Parameters
    ----------
    times : sequence of float
        Observed times (event or censoring), strictly positive.
    censored : sequence of bool, optional
        ``True`` if the observation is right-censored (subject did not
        experience the event by its observation time). Default is
        ``False`` for all observations (all are events).

    Returns
    -------
    SurvivalCurve

    Raises
    ------
    ValueError
        If ``times`` is empty, contains non-positive values, or has
        a different length from ``censored``.
    """
    if len(times) == 0:
        raise ValueError("times must be non-empty")
    if censored is None:
        censored_arr = np.zeros(len(times), dtype=bool)
    else:
        if len(censored) != len(times):
            raise ValueError(
                "censored and times must have the same length "
                f"(got {len(censored)} and {len(times)})"
            )
        censored_arr = np.array(censored, dtype=bool)

    t_arr = np.array(times, dtype=np.float64)
    if np.any(t_arr <= 0.0):
        raise ValueError("times must be strictly positive")

    order = np.argsort(t_arr)
    t_sorted = t_arr[order]
    c_sorted = censored_arr[order]

    # Aggregate events and censorings by unique time.
    unique_times = np.unique(t_sorted)
    out_times: list[float] = []
    out_surv: list[float] = []
    out_at_risk: list[int] = []
    out_events: list[int] = []

    survival = 1.0
    at_risk = len(t_sorted)
    cursor = 0  # index into t_sorted

    for t in unique_times:
        # Count events / censorings at this time
        d_i = 0
        c_i = 0
        while cursor < len(t_sorted) and t_sorted[cursor] == t:
            if c_sorted[cursor]:
                c_i += 1
            else:
                d_i += 1
            cursor += 1
        n_i = at_risk
        if d_i > 0:
            survival *= 1.0 - d_i / n_i
            out_times.append(float(t))
            out_surv.append(float(survival))
            out_at_risk.append(int(n_i))
            out_events.append(int(d_i))
        # Subjects at risk drop by events and censorings observed at t.
        at_risk -= d_i + c_i

    # Median survival: first time S(t) <= 0.5.
    median: float | None = None
    for t, s in zip(out_times, out_surv, strict=False):
        if s <= 0.5:
            median = t
            break

    return SurvivalCurve(
        times=tuple(out_times),
        survival=tuple(out_surv),
        at_risk=tuple(out_at_risk),
        events=tuple(out_events),
        n=len(t_sorted),
        n_events=int((~c_sorted).sum()),
        median_survival_time=median,
    )
