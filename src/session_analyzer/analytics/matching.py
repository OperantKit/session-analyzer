"""Generalized Matching Law (GML) parameter estimation.

The Generalized Matching Law (Baum, 1974) expresses choice as

    log(B1 / B2) = a * log(R1 / R2) + log(b)

where ``a`` is sensitivity (slope) and ``log b`` is the bias (intercept).
Perfect matching corresponds to ``a = 1`` and ``log b = 0``.

References
----------
Baum, W. M. (1974). On two types of deviation from the matching law: bias
and undermatching. Journal of the Experimental Analysis of Behavior, 22(1),
231-242. https://doi.org/10.1901/jeab.1974.22-231
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class MatchingParameters:
    """Result of fitting the Generalized Matching Law.

    Attributes
    ----------
    sensitivity : float
        Slope ``a`` of the log-log fit. ``a = 1`` is strict matching;
        ``a < 1`` is undermatching, ``a > 1`` is overmatching.
    bias : float
        Intercept ``log b`` of the log-log fit. ``log b = 0`` is unbiased.
    r_squared : float
        Coefficient of determination of the linear fit on log-log axes.
    n : int
        Number of valid (positive) data points used for the fit.
    """

    sensitivity: float
    bias: float
    r_squared: float
    n: int


def generalized_matching_law(
    log_reinforcement_ratio: np.ndarray,
    a: float,
    log_b: float,
) -> np.ndarray:
    """GML equation: ``log(B1/B2) = a * log(R1/R2) + log b``."""
    return a * log_reinforcement_ratio + log_b


def estimate_matching_parameters(
    responses_ratios: list[float],
    reinforcement_ratios: list[float],
) -> MatchingParameters:
    """Estimate GML parameters from response and reinforcement ratios.

    Parameters
    ----------
    responses_ratios : list of float
        Per-condition ratios B1 / B2 of responses on alternative 1 to
        alternative 2.
    reinforcement_ratios : list of float
        Per-condition ratios R1 / R2 of reinforcement on alternative 1 to
        alternative 2.

    Returns
    -------
    MatchingParameters
        Fitted slope ``a``, intercept ``log b``, R^2, and effective ``n``.

    Raises
    ------
    ValueError
        If the two input lists have different lengths, or if fewer than
        two valid (positive, paired) data points remain after filtering
        non-positive ratios.
    """
    if len(responses_ratios) != len(reinforcement_ratios):
        raise ValueError(
            "responses_ratios and reinforcement_ratios must have the same length "
            f"(got {len(responses_ratios)} and {len(reinforcement_ratios)})"
        )

    valid_indices = [
        i
        for i, (b, r) in enumerate(zip(responses_ratios, reinforcement_ratios, strict=False))
        if b > 0 and r > 0
    ]

    if len(valid_indices) < 2:
        raise ValueError(
            "GML fitting requires at least 2 valid (positive) ratio pairs; "
            f"got {len(valid_indices)} after filtering"
        )

    y = np.log(np.array([responses_ratios[i] for i in valid_indices], dtype=np.float64))
    x = np.log(np.array([reinforcement_ratios[i] for i in valid_indices], dtype=np.float64))

    popt, _pcov = curve_fit(generalized_matching_law, x, y, p0=[1.0, 0.0])
    a, log_b = float(popt[0]), float(popt[1])

    residuals = y - generalized_matching_law(x, a, log_b)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0

    return MatchingParameters(
        sensitivity=a,
        bias=log_b,
        r_squared=float(r_squared),
        n=len(valid_indices),
    )
