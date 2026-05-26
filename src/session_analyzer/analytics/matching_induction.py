"""Matching-with-induction fit (Baum & Grace, 2020).

The Generalized Matching Law (Baum, 1974) — see
:mod:`session_analyzer.analytics.matching` — describes choice
allocation as

.. math::

    \\log(B_1 / B_2) = a \\, \\log(R_1 / R_2) + \\log b

with sensitivity ``a`` and bias ``log b``. In concurrent-ratio and other
arrangements, observed allocation often departs systematically from this
law: behavior overshoots strict matching toward the richer alternative.
Baum & Grace (2020) argue that matching and induction (Pavlovian
elicitation by the locally-richer reinforcer context) jointly determine
operant performance.

This module implements a phenomenological three-parameter extension that
adds an induction term proportional to the *relative* reinforcer-rate
difference:

.. math::

    \\log(B_1 / B_2) = a \\, \\log(R_1 / R_2) + \\log b
        + w \\, \\frac{R_1 - R_2}{R_1 + R_2}

where ``w`` is the induction weight. The term :math:`(R_1 - R_2) /
(R_1 + R_2)` is bounded in ``[-1, 1]`` and is positive when alternative
1 is the richer source. ``w > 0`` therefore amplifies the predicted
log-ratio beyond strict matching toward the richer alternative
(induction-driven overmatching); ``w < 0`` attenuates it (induction
acting to equalize allocation). The strict GML is recovered at ``w = 0``.

This is one parameterization consistent with the matching + induction
framework; the framework itself is broader than any specific equation
form.

References
----------
Baum, W. M. (1974). On two types of deviation from the matching law:
    Bias and undermatching. *Journal of the Experimental Analysis of
    Behavior*, *22*(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231

Baum, W. M., & Grace, R. C. (2020). Matching theory and induction
    explain operant performance. *Journal of the Experimental Analysis
    of Behavior*, *113*(1), 1-22. https://doi.org/10.1002/jeab.583

Baum, W. M., Aparicio, C. F., & Alonso-Alvarez, B. (2022). Rate
    matching, probability matching, and optimization in concurrent
    ratio schedules. *Journal of the Experimental Analysis of
    Behavior*, *117*(2), 234-252. https://doi.org/10.1002/jeab.771
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class MatchingInductionParameters:
    """Result of fitting the matching-with-induction extension.

    Attributes
    ----------
    sensitivity : float
        Generalized-matching slope ``a``. Strict matching corresponds
        to ``a = 1`` once induction is partialled out.
    bias : float
        Generalized-matching intercept ``log b``. Unbiased allocation
        corresponds to ``log b = 0``.
    induction_weight : float
        Induction weight ``w``. ``w = 0`` recovers strict GML;
        ``w > 0`` indicates induction-amplified preference toward the
        richer alternative; ``w < 0`` indicates induction-attenuated
        allocation.
    r_squared : float
        Coefficient of determination of the fit on the predicted
        ``log(B1 / B2)`` axis.
    n : int
        Number of valid (positive, paired) data points used for the
        fit.
    """

    sensitivity: float
    bias: float
    induction_weight: float
    r_squared: float
    n: int


def predict_log_ratio_with_induction(
    log_reinforcement_ratio: np.ndarray,
    reinforcement_rates_1: np.ndarray,
    reinforcement_rates_2: np.ndarray,
    a: float,
    log_b: float,
    w: float,
) -> np.ndarray:
    """Evaluate the matching-with-induction prediction equation."""
    r1 = np.asarray(reinforcement_rates_1, dtype=np.float64)
    r2 = np.asarray(reinforcement_rates_2, dtype=np.float64)
    induction_term = (r1 - r2) / (r1 + r2)
    return a * log_reinforcement_ratio + log_b + w * induction_term


def fit_matching_with_induction(
    *,
    responses_ratios: Sequence[float],
    reinforcement_ratios: Sequence[float],
    reinforcement_rates_1: Sequence[float],
    reinforcement_rates_2: Sequence[float],
) -> MatchingInductionParameters:
    """Fit the matching-with-induction model.

    Parameters
    ----------
    responses_ratios :
        Per-condition response ratios ``B1 / B2``.
    reinforcement_ratios :
        Per-condition reinforcement ratios ``R1 / R2``.
    reinforcement_rates_1, reinforcement_rates_2 :
        Per-condition absolute reinforcement rates on alternatives 1
        and 2. Required (alongside ``reinforcement_ratios``) because
        the induction term depends on the absolute rate difference, not
        just the ratio.

    Returns
    -------
    MatchingInductionParameters

    Raises
    ------
    ValueError
        If the input sequences have inconsistent lengths or fewer than
        three valid (positive, paired) observations remain after
        filtering.
    """
    inputs = (
        responses_ratios,
        reinforcement_ratios,
        reinforcement_rates_1,
        reinforcement_rates_2,
    )
    n = len(responses_ratios)
    if not all(len(seq) == n for seq in inputs):
        raise ValueError(
            "responses_ratios, reinforcement_ratios, reinforcement_rates_1, "
            "and reinforcement_rates_2 must all have the same length"
        )

    valid: list[int] = []
    for i in range(n):
        b = responses_ratios[i]
        r = reinforcement_ratios[i]
        r1_i = reinforcement_rates_1[i]
        r2_i = reinforcement_rates_2[i]
        if b > 0 and r > 0 and r1_i > 0 and r2_i > 0:
            valid.append(i)

    if len(valid) < 3:
        raise ValueError(
            "fitting matching-with-induction requires at least 3 valid "
            f"(positive, paired) observations; got {len(valid)}"
        )

    y = np.log(np.array([responses_ratios[i] for i in valid], dtype=np.float64))
    log_r = np.log(np.array([reinforcement_ratios[i] for i in valid], dtype=np.float64))
    r1 = np.array([reinforcement_rates_1[i] for i in valid], dtype=np.float64)
    r2 = np.array([reinforcement_rates_2[i] for i in valid], dtype=np.float64)

    def _model(x_dummy: np.ndarray, a: float, log_b: float, w: float) -> np.ndarray:
        del x_dummy  # log_r / r1 / r2 are captured from the enclosing scope.
        return predict_log_ratio_with_induction(log_r, r1, r2, a=a, log_b=log_b, w=w)

    popt, _pcov = curve_fit(
        _model,
        log_r,  # dummy independent axis; real inputs are closed-over
        y,
        p0=[1.0, 0.0, 0.0],
    )
    a_hat, log_b_hat, w_hat = (float(popt[0]), float(popt[1]), float(popt[2]))

    fitted = predict_log_ratio_with_induction(
        log_r, r1, r2, a=a_hat, log_b=log_b_hat, w=w_hat,
    )
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 1.0

    return MatchingInductionParameters(
        sensitivity=a_hat,
        bias=log_b_hat,
        induction_weight=w_hat,
        r_squared=float(r_squared),
        n=len(valid),
    )
