"""Generalized Matching Law (GML) and single-alternative matching fits.

**Generalized Matching Law** (Baum, 1974):

    log(B1 / B2) = a * log(R1 / R2) + log(b)

where ``a`` is sensitivity (slope) and ``log b`` is the bias (intercept).
Perfect matching corresponds to ``a = 1`` and ``log b = 0``.

Two estimators are provided:

- :func:`estimate_matching_parameters` — multi-condition log-log curve
  fit using ``scipy.optimize.curve_fit``. Reports ``a``, ``log b``,
  ``r_squared``, and ``n``. Use this for cross-condition designs where
  you have several (B1/B2, R1/R2) pairs.
- :func:`estimate_single_observation_gml` — closed-form solver from a
  single (B1, B2, R1, R2) tuple. Pure Python, no numpy/scipy. Use this
  on a live dashboard where only running totals are available; the
  returned bias is the multiplicative ``b`` (not ``log b``).

**Single-alternative hyperbola** (Herrnstein, 1970):

    B = k * R / (R + R_e)

where ``B`` is response rate, ``R`` is reinforcement rate, ``k`` is the
asymptotic response rate, and ``R_e`` is the rate of extraneous
reinforcement (all other competing reinforcement sources combined).

References
----------
Baum, W. M. (1974). On two types of deviation from the matching law: bias
    and undermatching. *Journal of the Experimental Analysis of Behavior*,
    22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231

Herrnstein, R. J. (1970). On the law of effect. *Journal of the
    Experimental Analysis of Behavior*, 13(2), 243-266.
    https://doi.org/10.1901/jeab.1970.13-243
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class SingleAlternativeMatchingParameters:
    """Herrnstein (1970) single-alternative hyperbola fit result.

    Attributes
    ----------
    k : float
        Asymptotic response rate, in the same units as the input
        ``response_rates``. Represents the maximum response rate the
        subject can emit.
    r_e : float
        Rate of extraneous (non-programmed) reinforcement, in the same
        units as the input ``reinforcement_rates``.
    r_squared : float
        Coefficient of determination of the non-linear fit.
    n : int
        Number of (R, B) data pairs used for the fit.
    """

    k: float
    r_e: float
    r_squared: float
    n: int


def _herrnstein_hyperbola(r: np.ndarray, k: float, r_e: float) -> np.ndarray:
    return k * r / (r + r_e)


def fit_single_alternative_matching(
    reinforcement_rates: list[float],
    response_rates: list[float],
) -> SingleAlternativeMatchingParameters:
    """Fit Herrnstein's (1970) single-alternative hyperbola ``B = k R / (R + R_e)``.

    Parameters
    ----------
    reinforcement_rates : list of float
        Per-condition reinforcement rates ``R`` (non-negative).
    response_rates : list of float
        Per-condition response rates ``B`` (non-negative).

    Returns
    -------
    SingleAlternativeMatchingParameters

    Raises
    ------
    ValueError
        If input lengths differ, or if fewer than 2 pairs remain after
        filtering out pairs with negative values.
    """
    if len(reinforcement_rates) != len(response_rates):
        raise ValueError(
            "reinforcement_rates and response_rates must have the same length "
            f"(got {len(reinforcement_rates)} and {len(response_rates)})"
        )

    pairs = [
        (r, b)
        for r, b in zip(reinforcement_rates, response_rates, strict=False)
        if r >= 0.0 and b >= 0.0
    ]
    if len(pairs) < 2:
        raise ValueError(
            "single-alternative matching fit requires at least 2 non-negative pairs; "
            f"got {len(pairs)} after filtering"
        )

    r_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    b_arr = np.array([p[1] for p in pairs], dtype=np.float64)

    k_init = float(b_arr.max()) * 1.2 if b_arr.max() > 0.0 else 1.0
    r_e_init = float(np.median(r_arr)) if r_arr.size else 1.0
    r_e_init = max(r_e_init, 1e-3)

    popt, _ = curve_fit(
        _herrnstein_hyperbola,
        r_arr,
        b_arr,
        p0=[k_init, r_e_init],
        bounds=([1e-9, 1e-12], [np.inf, np.inf]),
        maxfev=20_000,
    )
    k_fit, r_e_fit = float(popt[0]), float(popt[1])

    predicted = _herrnstein_hyperbola(r_arr, k_fit, r_e_fit)
    ss_res = float(np.sum((b_arr - predicted) ** 2))
    ss_tot = float(np.sum((b_arr - np.mean(b_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    return SingleAlternativeMatchingParameters(
        k=k_fit,
        r_e=r_e_fit,
        r_squared=float(r_squared),
        n=len(pairs),
    )


def estimate_single_observation_gml(
    responses_a: int,
    responses_b: int,
    reinforcements_a: int,
    reinforcements_b: int,
) -> tuple[float, float] | None:
    """Closed-form Generalized Matching Law estimate from a single observation.

    Solves the GML

        log(B1 / B2) = a * log(R1 / R2) + log b

    for ``(a, b)`` from one (B1, B2, R1, R2) tuple. With a single
    observation there are two unknowns and effectively one usable
    equation, so the system is solved by setting ``log b = 0`` whenever
    the reinforcement ratio is nontrivial — i.e. attribute all of the
    log response ratio to sensitivity. This closed form is appropriate
    on a live dashboard where only running totals from a single
    condition are available.

    For proper multi-condition fitting (with non-trivial bias and a
    reported R^2), use :func:`estimate_matching_parameters`.

    Parameters
    ----------
    responses_a, responses_b : int
        Total responses on alternative A / B during the observation
        window.
    reinforcements_a, reinforcements_b : int
        Total reinforcements obtained on alternative A / B during the
        same window.

    Returns
    -------
    (sensitivity, bias) : tuple of float
        ``sensitivity`` is the slope ``a``; ``bias`` is the
        multiplicative constant ``b`` (not ``log b``). Both are floats.
        Strict matching corresponds to ``a = 1.0, b = 1.0``.
    None
        Returned when the inputs cannot support an estimate: any
        negative count, both responses zero, both reinforcements zero,
        or B's denominator zero (cannot form a finite ratio).
    """
    if (
        responses_a < 0
        or responses_b < 0
        or reinforcements_a < 0
        or reinforcements_b < 0
    ):
        return None
    if responses_a == 0 and responses_b == 0:
        return None
    if reinforcements_a == 0 and reinforcements_b == 0:
        return None
    if responses_b == 0 or reinforcements_b == 0:
        return None

    resp_ratio = responses_a / responses_b
    rf_ratio = reinforcements_a / reinforcements_b

    if resp_ratio <= 0 or rf_ratio <= 0:
        return None

    log_resp_ratio = math.log(resp_ratio)
    log_rf_ratio = math.log(rf_ratio)

    if log_rf_ratio == 0:
        # r_A == r_B; sensitivity is undetermined from a single point.
        if log_resp_ratio == 0:
            # Equal responses + equal reinforcers: convention a = 1.
            return 1.0, 1.0
        # Unequal responses with equal reinforcers implies infinite
        # sensitivity or extreme bias; either way not estimable.
        return None

    a = log_resp_ratio / log_rf_ratio
    log_b = log_resp_ratio - a * log_rf_ratio
    b = math.exp(log_b)
    return float(a), float(b)
