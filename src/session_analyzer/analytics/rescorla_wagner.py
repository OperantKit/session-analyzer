"""Rescorla-Wagner (1972) associative-strength simulation and fitting.

Per-trial update for a single CS:

    Delta V_t = alpha * beta * (lambda_t - V_t)

where ``alpha`` is CS salience, ``beta`` is US salience (combined into a
single learning-rate-like product when fit jointly), ``lambda_t`` is the
asymptotic strength supported by the US on trial ``t`` (positive on
reinforced trials, zero on extinction trials), and ``V_t`` is associative
strength entering the trial.

Reference
---------
Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian
    conditioning: Variations in the effectiveness of reinforcement and
    nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.),
    *Classical conditioning II: Current research and theory*
    (pp. 64-99). Appleton-Century-Crofts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RescorlaWagnerParameters:
    """Fit result for the single-CS Rescorla-Wagner model.

    Attributes
    ----------
    alpha_beta : float
        Combined learning rate ``alpha * beta`` (these enter the update
        only as a product, so they are not separately identifiable from
        a single CS without independent manipulation).
    lambda_ : float
        Asymptotic associative strength supported by the US on
        reinforced trials.
    initial_v : float
        Associative strength assumed at the start of the simulated
        trace.
    sse : float
        Sum of squared errors between simulated and observed trace.
    r_squared : float
        Coefficient of determination.
    n_trials : int
        Number of trials used.
    """

    alpha_beta: float
    lambda_: float
    initial_v: float
    sse: float
    r_squared: float
    n_trials: int


def simulate_rescorla_wagner(
    us_present: Sequence[bool],
    *,
    alpha_beta: float,
    lambda_: float,
    initial_v: float = 0.0,
) -> list[float]:
    """Simulate associative strength over trials under a single CS.

    Returns the post-trial ``V`` value for each trial. ``V`` is updated
    by ``alpha_beta * (lambda - V)`` on reinforced trials and by
    ``alpha_beta * (0 - V)`` on extinction trials.
    """
    if not 0.0 < alpha_beta <= 1.0:
        raise ValueError(
            f"alpha_beta must lie in (0, 1], got {alpha_beta}"
        )
    v = float(initial_v)
    trace: list[float] = []
    for present in us_present:
        target = lambda_ if present else 0.0
        v = v + alpha_beta * (target - v)
        trace.append(v)
    return trace


def _simulate_array(
    us_present: np.ndarray, alpha_beta: float, lambda_: float, initial_v: float
) -> np.ndarray:
    v = initial_v
    out = np.empty(us_present.size, dtype=np.float64)
    for i in range(us_present.size):
        target = lambda_ if us_present[i] else 0.0
        v = v + alpha_beta * (target - v)
        out[i] = v
    return out


def fit_rescorla_wagner(
    observed: Sequence[float],
    us_present: Sequence[bool],
    *,
    initial_v: float = 0.0,
) -> RescorlaWagnerParameters:
    """Fit ``alpha_beta`` and ``lambda_`` to an observed trace.

    Parameters
    ----------
    observed : sequence of float
        Observed associative strength (or any monotone proxy) at the
        end of each trial.
    us_present : sequence of bool
        Whether the US occurred on each trial.
    initial_v : float
        Initial associative strength prior to trial 0.

    Returns
    -------
    RescorlaWagnerParameters

    Raises
    ------
    ValueError
        On length mismatch or fewer than 2 trials.
    """
    if len(observed) != len(us_present):
        raise ValueError(
            "observed and us_present must have the same length "
            f"(got {len(observed)} and {len(us_present)})"
        )
    if len(observed) < 2:
        raise ValueError(
            f"Rescorla-Wagner fitting requires at least 2 trials, got {len(observed)}"
        )

    obs = np.array(observed, dtype=np.float64)
    us = np.array(us_present, dtype=bool)

    def negative_log_loss(params: np.ndarray) -> float:
        ab, lam = params
        if not (0.0 < ab <= 1.0):
            return 1e12
        if lam <= 0.0:
            return 1e12
        predicted = _simulate_array(us, float(ab), float(lam), float(initial_v))
        return float(np.sum((predicted - obs) ** 2))

    init_lam = float(np.max(obs)) if obs.size else 1.0
    init_lam = max(init_lam, 1e-3)
    result = minimize(
        negative_log_loss,
        x0=np.array([0.3, init_lam]),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000},
    )
    ab_fit, lam_fit = float(result.x[0]), float(result.x[1])
    predicted = _simulate_array(us, ab_fit, lam_fit, float(initial_v))
    sse = float(np.sum((predicted - obs) ** 2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 0.0
    return RescorlaWagnerParameters(
        alpha_beta=ab_fit,
        lambda_=lam_fit,
        initial_v=float(initial_v),
        sse=sse,
        r_squared=float(r2),
        n_trials=len(obs),
    )
