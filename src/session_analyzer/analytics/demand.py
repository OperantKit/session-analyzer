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

Two definitions of ``P_max`` are available (``pmax_method``):

- ``"exact"`` (default) — the true local maximum of expenditure. Setting
  ``d(P * Q)/dP = 0`` with ``u = alpha * Q0 * P`` gives
  ``u * k * ln(10) * exp(-u) = 1``, whose solution on the principal
  Lambert-W branch is ``P_max = -W0(-1 / (k ln 10)) / (alpha * Q0)``
  (real only for ``k * ln(10) >= e``). For smaller ``k`` there is no
  unit-elasticity point; the numerical search over ``(0, 1/(alpha Q0)]``
  then returns the upper end of that interval.
- ``"approximate"`` — the derived approximation reported by the beezdemand
  R package as ``Pmaxd`` (attributed by Kaplan et al., 2019 to Hursh, 2014
  and Hursh & Roma, 2013): ``P_max ~= (0.083 k + 0.65) / (Q0 * alpha * k^1.5)``.
  ``O_max`` is then ``P_max * Q(P_max)`` at that price.

Two fitting spaces are available (``fit_space``):

- ``"linear"`` (default) — least squares on ``Q``. Requires strictly
  positive prices.
- ``"log10"`` — least squares on ``log10(Q)``, the error structure of the
  original model (Equation 2 of Kaplan et al., 2019) and of beezdemand's
  ``equation = "hs"``. Accepts the free price ``C = 0``.

References
----------
Hursh, S. R., & Silberberg, A. (2008). Economic demand and essential
    value. *Psychological Review*, *115*(1), 186-198.
    https://doi.org/10.1037/0033-295X.115.1.186

Kaplan, B. A., Gilroy, S. P., Reed, D. D., Koffarnus, M. N., & Hursh,
    S. R. (2019). The R package beezdemand: Behavioral economic easy demand.
    *Perspectives on Behavior Science*, *42*(1), 163-180.
    https://doi.org/10.1007/s40614-018-00187-7

Hursh, S. R., & Roma, P. G. (2013). Behavioral economics and empirical
    public policy. *Journal of the Experimental Analysis of Behavior*,
    *99*(1), 98-124. https://doi.org/10.1002/jeab.7
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import lambertw


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


FitSpace = Literal["linear", "log10"]
PmaxMethod = Literal["exact", "approximate"]


def _log10_demand(price: np.ndarray, q0: float, alpha: float, k: float) -> np.ndarray:
    """Hursh & Silberberg (2008) exponential demand in log10 units."""
    result: np.ndarray = math.log10(q0) + k * (np.exp(-alpha * q0 * price) - 1.0)
    return result


def _exact_pmax(q0: float, alpha: float, k: float) -> float:
    """Price of the first unit-elasticity point (local expenditure maximum).

    Expenditure ``P * Q(P)`` under the exponential demand model is
    unimodal only on ``(0, 1/(alpha*q0)]`` — the interval bounded above
    by the price of maximum elasticity (u=1 in the reparameterization
    ``u = alpha * q0 * P``). Beyond that price, expenditure rises again
    asymptotically because ``Q(P)`` plateaus at ``q0 * 10^(-k)``. The
    conventional ``P_max`` (Hursh & Silberberg, 2008) refers to the
    local expenditure maximum at unit elasticity (``epsilon = -1``),
    which lies strictly below the elasticity-minimum price.
    """

    branch_arg = -1.0 / (k * math.log(10.0))
    if branch_arg >= -1.0 / math.e:
        # Closed form on the principal branch (real when k ln 10 >= e).
        return float(-lambertw(branch_arg, 0).real / (alpha * q0))

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
    return float(result.x)


def compute_pmax(
    q0: float, alpha: float, k: float, *, method: PmaxMethod = "exact"
) -> tuple[float, float]:
    """Return ``(p_max, o_max)`` for given demand parameters.

    ``method="exact"`` gives the true local maximum of expenditure;
    ``method="approximate"`` gives ``(0.083 k + 0.65) / (q0 alpha k^1.5)``
    (beezdemand ``Pmaxd``). In both cases ``o_max = p_max * Q(p_max)``.
    """
    if method == "exact":
        p_max = _exact_pmax(q0, alpha, k)
    elif method == "approximate":
        p_max = (0.083 * k + 0.65) / (q0 * alpha * k**1.5)
    else:
        raise ValueError(f"method must be 'exact' or 'approximate', got {method!r}")
    q_at = q0 * math.pow(10.0, k * (math.exp(-alpha * q0 * p_max) - 1.0))
    return p_max, p_max * q_at


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
    fit_space: FitSpace = "linear",
    pmax_method: PmaxMethod = "exact",
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
    fit_space : {"linear", "log10"}
        Space in which squared residuals are minimised. ``"log10"`` is the
        Hursh & Silberberg (2008) error structure (beezdemand ``"hs"``)
        and admits a zero price; ``"linear"`` requires positive prices.
    pmax_method : {"exact", "approximate"}
        Definition of ``p_max`` (see the module docstring).

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
    if fit_space not in ("linear", "log10"):
        raise ValueError(f"fit_space must be 'linear' or 'log10', got {fit_space!r}")
    p = np.array(prices, dtype=np.float64)
    q = np.array(consumption, dtype=np.float64)
    if np.any(q <= 0.0):
        raise ValueError("consumption must be strictly positive")
    if fit_space == "log10":
        if np.any(p < 0.0):
            raise ValueError("prices must be non-negative")
    elif np.any(p <= 0.0):
        raise ValueError(
            "prices must be strictly positive (use fit_space='log10' for a free price)"
        )

    q0_init = float(np.max(q))
    positive_p = p[p > 0.0]
    alpha_init = 1.0 / (q0_init * float(np.median(positive_p if positive_p.size else p)))
    model = _log10_demand if fit_space == "log10" else _demand
    target = np.log10(q) if fit_space == "log10" else q
    k_init = max(float(np.log10(q.max() / q.min()) + 0.5), 0.5)

    if k is None:
        popt, _ = curve_fit(
            model,
            p,
            target,
            p0=[q0_init, alpha_init, k_init],
            bounds=([1e-9, 1e-12, 1e-3], [np.inf, np.inf, 20.0]),
            maxfev=20_000,
        )
        q0_fit, alpha_fit, k_fit = float(popt[0]), float(popt[1]), float(popt[2])
    else:
        if k <= 0.0:
            raise ValueError(f"k must be positive when fixed, got {k}")
        fixed_k = k
        popt, _ = curve_fit(
            lambda price, q0, alpha: model(price, q0, alpha, fixed_k),
            p,
            target,
            p0=[q0_init, alpha_init],
            bounds=([1e-9, 1e-12], [np.inf, np.inf]),
            maxfev=20_000,
        )
        q0_fit, alpha_fit, k_fit = float(popt[0]), float(popt[1]), float(k)

    predicted = _demand(p, q0_fit, alpha_fit, k_fit)
    r2 = _r_squared(q, predicted)
    p_max, o_max = compute_pmax(q0_fit, alpha_fit, k_fit, method=pmax_method)

    return DemandParameters(
        q0=q0_fit,
        alpha=alpha_fit,
        k=k_fit,
        r_squared=r2,
        n=len(p),
        p_max=p_max,
        o_max=o_max,
    )
