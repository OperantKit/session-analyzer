"""Stimulus generalization gradient fitting.

After discrimination training with a training stimulus ``S+``, response
rate (or proportion) typically falls off as a function of distance
from ``S+`` along a physical stimulus dimension. Two canonical forms
are supported:

- **Gaussian** (Guttman & Kalish, 1956 descriptive fits):

      R(d) = R0 * exp(-d^2 / (2 * sigma^2))

  with a symmetric bell-shaped gradient around the peak.

- **Exponential** (Shepard, 1987 universal law of generalization):

      R(d) = R0 * exp(-d / lambda)

  predicted from a Bayesian consequential-region argument; slope and
  intercept reveal generalization decay rate independent of Gaussian
  assumptions.

Both fits estimate a peak-response parameter ``R0`` jointly with the
width (``sigma`` or ``lambda``).

References
----------
Guttman, N., & Kalish, H. I. (1956). Discriminability and stimulus
    generalization. *Journal of Experimental Psychology*, *51*(1),
    79-88. https://doi.org/10.1037/h0046219

Shepard, R. N. (1987). Toward a universal law of generalization for
    psychological science. *Science*, *237*(4820), 1317-1323.
    https://doi.org/10.1126/science.3629243
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class GaussianGradientParameters:
    """Gaussian generalization-gradient fit result.

    Attributes
    ----------
    peak_response : float
        Response level at zero distance from the training stimulus.
    sigma : float
        Standard-deviation-like width of the gradient, in the same
        units as the stimulus distance axis.
    r_squared : float
        Coefficient of determination of the non-linear fit.
    n : int
        Number of (distance, response) pairs used.
    """

    peak_response: float
    sigma: float
    r_squared: float
    n: int


@dataclass(frozen=True)
class ExponentialGradientParameters:
    """Exponential (Shepard, 1987) generalization-gradient fit result.

    Attributes
    ----------
    peak_response : float
        Response level at zero distance from the training stimulus.
    length_scale : float
        Decay constant ``lambda`` in ``R0 * exp(-d / lambda)``; larger
        means broader generalization.
    r_squared : float
        Coefficient of determination of the non-linear fit.
    n : int
        Number of (distance, response) pairs used.
    """

    peak_response: float
    length_scale: float
    r_squared: float
    n: int


def _gaussian(d: np.ndarray, r0: float, sigma: float) -> np.ndarray:
    return r0 * np.exp(-(d**2) / (2.0 * sigma**2))


def _exponential(d: np.ndarray, r0: float, lam: float) -> np.ndarray:
    return r0 * np.exp(-d / lam)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _validate(
    distances: Sequence[float],
    responses: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(distances) != len(responses):
        raise ValueError(
            "distances and responses must have the same length "
            f"(got {len(distances)} and {len(responses)})"
        )
    if len(distances) < 3:
        raise ValueError(
            f"gradient fits require at least 3 data points, got {len(distances)}"
        )
    d = np.array(distances, dtype=np.float64)
    r = np.array(responses, dtype=np.float64)
    if np.any(d < 0.0):
        raise ValueError("distances must be non-negative")
    if np.any(r < 0.0):
        raise ValueError("responses must be non-negative")
    return d, r


def fit_gaussian_gradient(
    distances: Sequence[float],
    responses: Sequence[float],
) -> GaussianGradientParameters:
    """Fit ``R(d) = R0 * exp(-d^2 / (2*sigma^2))``.

    Parameters
    ----------
    distances : sequence of float
        Non-negative distances from the training stimulus.
    responses : sequence of float
        Non-negative response levels at each distance.

    Returns
    -------
    GaussianGradientParameters
    """
    d, r = _validate(distances, responses)
    r0_init = float(r.max()) if r.max() > 0.0 else 1.0
    sigma_init = float(np.std(d)) if np.std(d) > 0.0 else 1.0

    popt, _ = curve_fit(
        _gaussian,
        d,
        r,
        p0=[r0_init, sigma_init],
        bounds=([1e-9, 1e-9], [np.inf, np.inf]),
        maxfev=20_000,
    )
    r0, sigma = float(popt[0]), float(popt[1])
    predicted = _gaussian(d, r0, sigma)
    return GaussianGradientParameters(
        peak_response=r0,
        sigma=sigma,
        r_squared=_r_squared(r, predicted),
        n=len(d),
    )


def fit_exponential_gradient(
    distances: Sequence[float],
    responses: Sequence[float],
) -> ExponentialGradientParameters:
    """Fit Shepard's (1987) universal law ``R(d) = R0 * exp(-d / lambda)``.

    Parameters
    ----------
    distances : sequence of float
        Non-negative distances from the training stimulus.
    responses : sequence of float
        Non-negative response levels at each distance.

    Returns
    -------
    ExponentialGradientParameters
    """
    d, r = _validate(distances, responses)
    r0_init = float(r.max()) if r.max() > 0.0 else 1.0
    lam_init = float(np.mean(d[d > 0.0])) if np.any(d > 0.0) else 1.0

    popt, _ = curve_fit(
        _exponential,
        d,
        r,
        p0=[r0_init, lam_init],
        bounds=([1e-9, 1e-9], [np.inf, np.inf]),
        maxfev=20_000,
    )
    r0, lam = float(popt[0]), float(popt[1])
    predicted = _exponential(d, r0, lam)
    return ExponentialGradientParameters(
        peak_response=r0,
        length_scale=lam,
        r_squared=_r_squared(r, predicted),
        n=len(d),
    )
