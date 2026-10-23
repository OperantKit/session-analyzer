"""Delay-discounting model fitting.

Two standard models are provided:

- **Hyperbolic** (Mazur, 1987): ``V = A / (1 + k * D)``
- **Hyperboloid** (Myerson & Green, 1995; Green & Myerson, 2004):
  ``V = A / (1 + k * D)^s``

The hyperboloid here raises the whole denominator to ``s``. Rachlin's
(2006) two-parameter form raises delay itself to a power,
``V = A / (1 + k * D^s)``; it is a different function and is not the
model fitted by :func:`fit_hyperboloid_discounting`.

where ``A`` is the undiscounted amount, ``D`` is delay, ``V`` is the
subjective (indifference-point) value, ``k`` is the discounting rate,
and ``s`` is the non-linearity exponent.

The hyperboloid reduces to the hyperbolic model when ``s = 1``. Fitting
is performed on the raw (V, D) pairs via non-linear least squares
(``scipy.optimize.curve_fit``) rather than a log-transform, so that
``A`` need not equal ``V(0)`` exactly in the data (useful when the
zero-delay value is itself a psychometric estimate).

References
----------
Mazur, J. E. (1987). An adjusting procedure for studying delayed
    reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, &
    H. Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5:
    The effect of delay and of intervening events on reinforcement
    value* (pp. 55-73). Lawrence Erlbaum Associates.

Myerson, J., & Green, L. (1995). Discounting of delayed rewards: Models
    of individual choice. *Journal of the Experimental Analysis of
    Behavior*, *64*(3), 263-276. https://doi.org/10.1901/jeab.1995.64-263

Green, L., & Myerson, J. (2004). A discounting framework for choice
    with delayed and probabilistic rewards. *Psychological Bulletin*,
    *130*(5), 769-792. https://doi.org/10.1037/0033-2909.130.5.769

Rachlin, H. (2006). Notes on discounting. *Journal of the Experimental
    Analysis of Behavior*, *85*(3), 425-435.
    https://doi.org/10.1901/jeab.2006.85-05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class HyperbolicDiscountingParameters:
    """Mazur (1987) hyperbolic discounting fit result.

    Attributes
    ----------
    k : float
        Discounting rate. Larger ``k`` means steeper discounting.
    r_squared : float
        Coefficient of determination of the non-linear fit.
    n : int
        Number of (delay, value) pairs used for the fit.
    """

    k: float
    r_squared: float
    n: int


@dataclass(frozen=True)
class HyperboloidDiscountingParameters:
    """Myerson & Green (1995) / Green & Myerson (2004) hyperboloid fit result.

    Attributes
    ----------
    k : float
        Discounting rate.
    s : float
        Non-linearity exponent. ``s = 1`` reduces to Mazur's hyperbolic.
    r_squared : float
        Coefficient of determination of the non-linear fit.
    n : int
        Number of (delay, value) pairs used for the fit.
    """

    k: float
    s: float
    r_squared: float
    n: int


def _hyperbolic(delay: np.ndarray, k: float, amount: float) -> np.ndarray:
    return amount / (1.0 + k * delay)


def _hyperboloid(delay: np.ndarray, k: float, s: float, amount: float) -> np.ndarray:
    return amount / np.power(1.0 + k * delay, s)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _validate_inputs(
    delays: Sequence[float],
    values: Sequence[float],
    amount: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(delays) != len(values):
        raise ValueError(
            "delays and values must have the same length "
            f"(got {len(delays)} and {len(values)})"
        )
    if amount <= 0.0:
        raise ValueError(f"amount must be positive, got {amount}")
    if len(delays) < 2:
        raise ValueError(
            f"discounting fits require at least 2 data points, got {len(delays)}"
        )
    d = np.array(delays, dtype=np.float64)
    v = np.array(values, dtype=np.float64)
    if np.any(d < 0.0):
        raise ValueError("delays must be non-negative")
    return d, v


def fit_hyperbolic_discounting(
    delays: Sequence[float],
    values: Sequence[float],
    *,
    amount: float,
) -> HyperbolicDiscountingParameters:
    """Fit Mazur's (1987) hyperbolic discounting model ``V = A / (1 + k D)``.

    Parameters
    ----------
    delays : sequence of float
        Delays (non-negative, same units as ``k``'s inverse).
    values : sequence of float
        Subjective (indifference-point) values for each delay.
    amount : float
        Undiscounted amount ``A`` (strictly positive).

    Returns
    -------
    HyperbolicDiscountingParameters
    """
    d, v = _validate_inputs(delays, values, amount)

    popt, _ = curve_fit(
        lambda delay, k: _hyperbolic(delay, k, amount),
        d,
        v,
        p0=[0.01],
        maxfev=10_000,
    )
    k = float(popt[0])
    predicted = _hyperbolic(d, k, amount)
    return HyperbolicDiscountingParameters(
        k=k,
        r_squared=_r_squared(v, predicted),
        n=len(d),
    )


def fit_hyperboloid_discounting(
    delays: Sequence[float],
    values: Sequence[float],
    *,
    amount: float,
) -> HyperboloidDiscountingParameters:
    """Fit the Myerson & Green (1995) hyperboloid ``V = A / (1 + k D)^s``.

    Parameters
    ----------
    delays : sequence of float
        Delays (non-negative).
    values : sequence of float
        Subjective values for each delay.
    amount : float
        Undiscounted amount ``A`` (strictly positive).

    Returns
    -------
    HyperboloidDiscountingParameters
    """
    d, v = _validate_inputs(delays, values, amount)

    popt, _ = curve_fit(
        lambda delay, k, s: _hyperboloid(delay, k, s, amount),
        d,
        v,
        p0=[0.01, 1.0],
        maxfev=20_000,
    )
    k, s = float(popt[0]), float(popt[1])
    predicted = _hyperboloid(d, k, s, amount)
    return HyperboloidDiscountingParameters(
        k=k,
        s=s,
        r_squared=_r_squared(v, predicted),
        n=len(d),
    )
