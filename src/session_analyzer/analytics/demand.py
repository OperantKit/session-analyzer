"""Exponential demand-curve fitting (Hursh & Silberberg, 2008).

The exponential demand model expresses consumption ``Q`` as a function
of unit price ``C``:

    log10(Q) = log10(Q0) + k * (exp(-alpha * Q0 * C) - 1)

where:

- ``Q0`` is consumption at minimal price (demand intensity),
- ``alpha`` is the rate of change in elasticity (the *essential value*
  parameter; smaller ``alpha`` means higher essential value),
- ``k`` is the span of consumption in log10 units (observed range).

From the fit, two derived indices are commonly reported:

- ``P_max`` — the price at which **expenditure** (``P * Q``) is maximal.
  Point of unit elasticity; demand transitions from inelastic (slope > -1)
  below ``P_max`` to elastic (slope < -1) above.
- ``O_max`` — maximal expenditure, equal to ``P_max * Q(P_max)``.

Closed-form ``P_max`` (Hursh & Roma, 2013):

    P_max = W0(1 / (alpha * Q0 * k * ln(10))) / (alpha * Q0)   (approx.)

Here we compute ``P_max`` numerically as the interior maximum of
``P -> P * Q(P)`` to avoid branch-choice ambiguities of the Lambert-W
approximation.

References
----------
Hursh, S. R., & Silberberg, A. (2008). Economic demand and essential
    value. *Psychological Review*, *115*(1), 186-198.
    https://doi.org/10.1037/0033-295X.115.1.186

Hursh, S. R., & Roma, P. G. (2013). Behavioral economics and empirical
    public policy. *Journal of the Experimental Analysis of Behavior*,
    *99*(1), 98-124. https://doi.org/10.1002/jeab.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


@dataclass(frozen=True)
class DemandParameters:
    """Exponential-demand fit result.

    Attributes
    ----------
    q0 : float
        Demand intensity (consumption at minimal price).
    alpha : float
        Rate of change in elasticity. Inversely related to essential
        value: smaller ``alpha`` => higher essential value.
    k : float
        Span of consumption in log10 units.
    r_squared : float
        Coefficient of determination of the non-linear fit on linear Q.
    n : int
        Number of (price, consumption) pairs used for the fit.
    p_max : float
        Price at which expenditure is maximized (unit elasticity point).
    o_max : float
        Maximal expenditure (= p_max * Q(p_max)).
    """

    q0: float
    alpha: float
    k: float
    r_squared: float
    n: int
    p_max: float
    o_max: float


def _demand(price: np.ndarray, q0: float, alpha: float, k: float) -> np.ndarray:
    """Hursh & Silberberg (2008) exponential demand Q(P)."""
    return q0 * np.power(10.0, k * (np.exp(-alpha * q0 * price) - 1.0))


def _compute_pmax(q0: float, alpha: float, k: float) -> tuple[float, float]:
    """Return (p_max, o_max) at the first unit-elasticity point.

    Expenditure ``P * Q(P)`` under the exponential demand model is
    unimodal only on ``(0, 1/(alpha*q0)]`` — the interval bounded above
    by the price of maximum elasticity (u=1 in the reparameterization
    ``u = alpha * q0 * P``). Beyond that price, expenditure rises again
    asymptotically because ``Q(P)`` plateaus at ``q0 * 10^(-k)``. The
    conventional ``P_max`` (Hursh & Silberberg, 2008) refers to the
    local expenditure maximum at unit elasticity (``epsilon = -1``),
    which lies strictly below the elasticity-minimum price.
    """

    def neg_expenditure(price: float) -> float:
        q = q0 * math.pow(10.0, k * (math.exp(-alpha * q0 * price) - 1.0))
        return -price * q

    upper = 1.0 / (alpha * q0)  # price at maximum elasticity (u=1).
    result = minimize_scalar(
        neg_expenditure,
        bounds=(1e-12 * upper, upper),
        method="bounded",
        options={"xatol": 1e-10 * upper},
    )
    p_max = float(result.x)
    o_max = float(-result.fun)
    return p_max, o_max


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_exponential_demand(
    prices: Sequence[float],
    consumption: Sequence[float],
    *,
    k: float | None = None,
) -> DemandParameters:
    """Fit the Hursh & Silberberg (2008) exponential demand model.

    Parameters
    ----------
    prices : sequence of float
        Unit prices (strictly positive).
    consumption : sequence of float
        Observed consumption at each price (strictly positive).
    k : float or None
        Span parameter in log10 units. If ``None``, estimated jointly
        with ``q0`` and ``alpha``. If the data span a small log-consumption
        range, fixing ``k`` (commonly to ``log10(Qmax) - log10(Qmin) + 0.5``
        or a value used across conditions) improves identifiability.

    Returns
    -------
    DemandParameters

    Raises
    ------
    ValueError
        On length mismatch, non-positive inputs, or fewer than 3 data
        points (the full model has 3 free parameters; fixing ``k``
        lowers the requirement to 2 free parameters, but we still
        require 3 data points for a meaningful residual).
    """
    if len(prices) != len(consumption):
        raise ValueError(
            "prices and consumption must have the same length "
            f"(got {len(prices)} and {len(consumption)})"
        )
    if len(prices) < 3:
        raise ValueError(
            f"exponential demand fit requires at least 3 data points, got {len(prices)}"
        )
    p = np.array(prices, dtype=np.float64)
    q = np.array(consumption, dtype=np.float64)
    if np.any(p <= 0.0) or np.any(q <= 0.0):
        raise ValueError("prices and consumption must be strictly positive")

    q0_init = float(np.max(q))
    alpha_init = 1.0 / (q0_init * float(np.median(p)))
    k_init = max(float(np.log10(q.max() / q.min()) + 0.5), 0.5)

    if k is None:
        popt, _ = curve_fit(
            _demand,
            p,
            q,
            p0=[q0_init, alpha_init, k_init],
            bounds=([1e-9, 1e-12, 1e-3], [np.inf, np.inf, 20.0]),
            maxfev=20_000,
        )
        q0_fit, alpha_fit, k_fit = float(popt[0]), float(popt[1]), float(popt[2])
    else:
        if k <= 0.0:
            raise ValueError(f"k must be positive when fixed, got {k}")
        popt, _ = curve_fit(
            lambda price, q0, alpha: _demand(price, q0, alpha, k),
            p,
            q,
            p0=[q0_init, alpha_init],
            bounds=([1e-9, 1e-12], [np.inf, np.inf]),
            maxfev=20_000,
        )
        q0_fit, alpha_fit, k_fit = float(popt[0]), float(popt[1]), float(k)

    predicted = _demand(p, q0_fit, alpha_fit, k_fit)
    r2 = _r_squared(q, predicted)
    p_max, o_max = _compute_pmax(q0_fit, alpha_fit, k_fit)

    return DemandParameters(
        q0=q0_fit,
        alpha=alpha_fit,
        k=k_fit,
        r_squared=r2,
        n=len(p),
        p_max=p_max,
        o_max=o_max,
    )
