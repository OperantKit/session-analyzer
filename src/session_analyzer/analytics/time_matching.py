"""Time-allocation matching law (Baum & Rachlin, 1969).

The matching law applies to the *time* spent on each alternative as
well as to response counts:

    log(T1 / T2) = a * log(R1 / R2) + log(b)

Time allocation is sometimes a more sensitive measure of preference
than response allocation, especially when response topographies differ
across alternatives or when responses are not discrete.

Mathematically identical to :func:`session_analyzer.analytics.matching.
estimate_matching_parameters`, but typed and named for the
time-allocation interpretation. Logarithms are natural, so the bias is
``ln b`` (``log10 b = bias / ln 10``). Baum & Rachlin (1969) report a
constant of proportionality below 1 for every pigeon (a right-side bias,
i.e. ``bias < 0`` with T1 = left).

References
----------
Baum, W. M., & Rachlin, H. C. (1969). Choice as time allocation.
*Journal of the Experimental Analysis of Behavior*, 12(6), 861-874.
https://doi.org/10.1901/jeab.1969.12-861
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class TimeMatchingParameters:
    """Time-allocation generalized matching law fit.

    Attributes
    ----------
    sensitivity : float
        Slope ``a``. ``a = 1`` is strict matching of time to
        reinforcement; ``a < 1`` is undermatching (more equal time
        than predicted), ``a > 1`` is overmatching.
    bias : float
        Intercept ``ln b`` (natural log). ``ln b > 0`` indicates a fixed preference
        for alternative 1 unrelated to reinforcement.
    r_squared : float
        Coefficient of determination of the log-log fit.
    n : int
        Number of valid (positive) data pairs used for the fit.
    """

    sensitivity: float
    bias: float
    r_squared: float
    n: int


def _line(x: np.ndarray, a: float, log_b: float) -> np.ndarray:
    return a * x + log_b


def estimate_time_matching_parameters(
    time_ratios: list[float],
    reinforcement_ratios: list[float],
) -> TimeMatchingParameters:
    """Fit the time-allocation generalized matching law.

    Parameters
    ----------
    time_ratios : list of float
        Per-condition ratios T1 / T2 of time allocated to alternatives.
    reinforcement_ratios : list of float
        Per-condition ratios R1 / R2 of reinforcement obtained.

    Raises
    ------
    ValueError
        If lengths differ or fewer than 2 strictly-positive paired
        ratios remain after filtering.
    """
    if len(time_ratios) != len(reinforcement_ratios):
        raise ValueError(
            "time_ratios and reinforcement_ratios must have the same length "
            f"(got {len(time_ratios)} and {len(reinforcement_ratios)})"
        )

    valid = [
        (t, r)
        for t, r in zip(time_ratios, reinforcement_ratios, strict=False)
        if t > 0.0 and r > 0.0
    ]
    if len(valid) < 2:
        raise ValueError(
            "time-allocation matching fit requires at least 2 positive ratio pairs; "
            f"got {len(valid)} after filtering"
        )

    y = np.log(np.asarray([t for t, _ in valid], dtype=np.float64))
    x = np.log(np.asarray([r for _, r in valid], dtype=np.float64))

    popt, _ = curve_fit(_line, x, y, p0=[1.0, 0.0])
    a, log_b = float(popt[0]), float(popt[1])

    pred = _line(x, a, log_b)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    return TimeMatchingParameters(
        sensitivity=round(a, 6),
        bias=round(log_b, 6),
        r_squared=round(float(r2), 6),
        n=len(valid),
    )
