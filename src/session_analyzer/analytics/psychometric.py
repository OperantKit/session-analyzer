"""Psychometric-function fitting (probit and logit).

A psychometric function expresses the probability of a "yes" / correct
response as a function of stimulus magnitude ``x``. Two link functions
are supported:

- **Logistic (logit)**:

      P(yes | x) = gamma + (1 - gamma - lapse) / (1 + exp(-(x - mu) / sigma))

- **Cumulative-Gaussian (probit)**:

      P(yes | x) = gamma + (1 - gamma - lapse) * Phi((x - mu) / sigma)

``mu`` is the threshold (50 % point on the linear inner range), ``sigma``
is the slope width, ``gamma`` is the lower asymptote (guess rate), and
``lapse`` is the upper-asymptote shortfall.

References
----------
Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I.
    Fitting, sampling, and goodness of fit. *Perception &
    Psychophysics*, *63*(8), 1293-1313.
    https://doi.org/10.3758/BF03194544
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


LinkType = Literal["logit", "probit"]


@dataclass(frozen=True)
class PsychometricFit:
    """Psychometric-function fit result.

    Attributes
    ----------
    link : str
        Either ``"logit"`` or ``"probit"``.
    mu : float
        Threshold (50 % point on the inner range).
    sigma : float
        Slope width (positive).
    gamma : float
        Lower asymptote (guess rate).
    lapse : float
        Upper-asymptote shortfall.
    log_likelihood : float
        Maximised log-likelihood under a binomial observation model.
    n : int
        Number of stimulus levels.
    """

    link: str
    mu: float
    sigma: float
    gamma: float
    lapse: float
    log_likelihood: float
    n: int


def _sigmoid_inner(z: np.ndarray, link: LinkType) -> np.ndarray:
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-z))
    return norm.cdf(z)


def _predict(
    x: np.ndarray,
    mu: float,
    sigma: float,
    gamma: float,
    lapse: float,
    link: LinkType,
) -> np.ndarray:
    z = (x - mu) / sigma
    inner = _sigmoid_inner(z, link)
    return gamma + (1.0 - gamma - lapse) * inner


def fit_psychometric(
    stimuli: Sequence[float],
    n_correct: Sequence[int],
    n_total: Sequence[int],
    *,
    link: LinkType = "logit",
    fit_lapse: bool = False,
    gamma: float = 0.0,
) -> PsychometricFit:
    """Fit a psychometric function via maximum likelihood.

    Parameters
    ----------
    stimuli : sequence of float
        Stimulus magnitude per level.
    n_correct : sequence of int
        Correct (or "yes") responses per level.
    n_total : sequence of int
        Total trials per level (>= ``n_correct[i]``).
    link : "logit" or "probit"
        Link function.
    fit_lapse : bool
        If True, fit the lapse rate jointly. Default False (lapse fixed
        at 0).
    gamma : float
        Lower asymptote (guess rate). Default 0; set to 0.5 for 2AFC
        designs.

    Returns
    -------
    PsychometricFit
    """
    if link not in ("logit", "probit"):
        raise ValueError(f"link must be 'logit' or 'probit', got {link!r}")
    if not (len(stimuli) == len(n_correct) == len(n_total)):
        raise ValueError(
            "stimuli, n_correct, n_total must have the same length"
        )
    if len(stimuli) < 3:
        raise ValueError(
            f"psychometric fit requires at least 3 stimulus levels, got {len(stimuli)}"
        )
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must lie in [0, 1), got {gamma}")

    x = np.array(stimuli, dtype=np.float64)
    nc = np.array(n_correct, dtype=np.int64)
    nt = np.array(n_total, dtype=np.int64)
    if np.any(nc < 0) or np.any(nt <= 0) or np.any(nc > nt):
        raise ValueError(
            "require 0 <= n_correct[i] <= n_total[i] and n_total[i] > 0"
        )

    mu_init = float(np.median(x))
    sigma_init = max(float(np.std(x)), 1e-6)

    def neg_log_lik(params: np.ndarray) -> float:
        mu = float(params[0])
        sigma = float(params[1])
        lapse = float(params[2]) if fit_lapse else 0.0
        if sigma <= 0.0 or lapse < 0.0 or lapse > 0.5:
            return 1e12
        if 1.0 - gamma - lapse <= 0.0:
            return 1e12
        p = _predict(x, mu, sigma, gamma, lapse, link)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        ll = float(np.sum(nc * np.log(p) + (nt - nc) * np.log(1.0 - p)))
        return -ll

    init = np.array(
        [mu_init, sigma_init, 0.02 if fit_lapse else 0.0],
        dtype=np.float64,
    )
    result = minimize(
        neg_log_lik,
        x0=init,
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000},
    )
    mu_fit = float(result.x[0])
    sigma_fit = float(result.x[1])
    lapse_fit = float(result.x[2]) if fit_lapse else 0.0
    return PsychometricFit(
        link=link,
        mu=mu_fit,
        sigma=sigma_fit,
        gamma=float(gamma),
        lapse=lapse_fit,
        log_likelihood=-float(result.fun),
        n=len(x),
    )
