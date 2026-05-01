"""Hierarchical Generalized Matching Law fitting.

Partial-pooling extension of :func:`session_analyzer.analytics.matching.
estimate_matching_parameters` to a multi-subject design. Each subject
``i`` has its own slope and intercept on the log-log axes:

    y_{ij} = a_i * x_{ij} + b_i + epsilon_{ij}

with random effects

    a_i ~ N(mu_a, tau_a^2)
    b_i ~ N(mu_b, tau_b^2)

and observation noise ``epsilon_{ij} ~ N(0, sigma_eps^2)``. Marginalising
over ``a_i, b_i`` yields per-subject normal likelihoods whose
covariance matrix is a function of the within-subject design only,
giving a closed-form profile log-likelihood for the hyperparameters
``(mu_a, mu_b, tau_a, tau_b, sigma_eps)``. Optimisation is by
``scipy.optimize.minimize`` and posterior credible intervals are
returned via the Laplace approximation (Hessian inverse on the
unconstrained reparameterisation).

This is *empirical Bayes* / approximate REML — not a fully Bayesian
posterior — but provides credible-interval-style summaries without
requiring a sampling library.

Reference
---------
Gelman, A., & Hill, J. (2007). *Data analysis using regression and
    multilevel/hierarchical models*. Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


@dataclass(frozen=True)
class HierarchicalGMLFit:
    """Empirical-Bayes fit of the hierarchical GML.

    Attributes
    ----------
    mu_sensitivity : float
        Group-level mean slope ``a``.
    sd_sensitivity : float
        Between-subject SD of slopes (``tau_a``).
    mu_bias : float
        Group-level mean log-bias ``log b``.
    sd_bias : float
        Between-subject SD of log-bias (``tau_b``).
    sigma_obs : float
        Within-subject residual SD.
    sensitivity_ci : tuple[float, float]
        95 % Wald credible interval for ``mu_sensitivity``.
    bias_ci : tuple[float, float]
        95 % Wald credible interval for ``mu_bias``.
    n_subjects : int
    n_observations : int
    log_likelihood : float
    """

    mu_sensitivity: float
    sd_sensitivity: float
    mu_bias: float
    sd_bias: float
    sigma_obs: float
    sensitivity_ci: tuple[float, float]
    bias_ci: tuple[float, float]
    n_subjects: int
    n_observations: int
    log_likelihood: float


def _neg_log_marginal(
    theta: np.ndarray,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
) -> float:
    mu_a, mu_b, log_tau_a, log_tau_b, log_sigma = theta
    tau_a = float(np.exp(log_tau_a))
    tau_b = float(np.exp(log_tau_b))
    sigma = float(np.exp(log_sigma))

    nll = 0.0
    for x, y in zip(xs, ys, strict=False):
        n = x.size
        # Marginal mean and covariance per subject:
        # y_j = a_i * x_j + b_i + eps_j; integrate over a_i, b_i
        # under N(mu_a, tau_a^2) x N(mu_b, tau_b^2) -> Gaussian on y.
        mean = mu_a * x + mu_b
        cov = (
            (tau_a**2) * np.outer(x, x)
            + (tau_b**2) * np.ones((n, n))
            + (sigma**2) * np.eye(n)
        )
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return 1e12
        diff = y - mean
        z = solve_triangular(chol, diff, lower=True)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        nll += 0.5 * (n * np.log(2.0 * np.pi) + log_det + float(z @ z))
    return float(nll)


def fit_hierarchical_gml(
    log_response_ratios: Mapping[str, Sequence[float]],
    log_reinforcement_ratios: Mapping[str, Sequence[float]],
) -> HierarchicalGMLFit:
    """Empirical-Bayes hierarchical GML fit.

    Parameters
    ----------
    log_response_ratios : mapping of subject id -> sequence of float
        Per-subject ``log(B1 / B2)`` values across conditions.
    log_reinforcement_ratios : mapping of subject id -> sequence of float
        Per-subject ``log(R1 / R2)`` values across conditions.

    Returns
    -------
    HierarchicalGMLFit

    Raises
    ------
    ValueError
        If the two mappings declare different subject sets, lengths
        differ for any subject, fewer than 2 subjects are supplied,
        or any subject has fewer than 2 observations.
    """
    if set(log_response_ratios.keys()) != set(log_reinforcement_ratios.keys()):
        raise ValueError(
            "log_response_ratios and log_reinforcement_ratios must declare "
            "the same set of subjects"
        )
    subjects = sorted(log_response_ratios.keys())
    if len(subjects) < 2:
        raise ValueError(
            f"hierarchical GML requires at least 2 subjects, got {len(subjects)}"
        )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_obs_total = 0
    for sid in subjects:
        x = np.array(log_reinforcement_ratios[sid], dtype=np.float64)
        y = np.array(log_response_ratios[sid], dtype=np.float64)
        if x.size != y.size:
            raise ValueError(
                f"subject {sid!r}: x and y must have the same length"
            )
        if x.size < 2:
            raise ValueError(
                f"subject {sid!r}: at least 2 observations required, got {x.size}"
            )
        xs.append(x)
        ys.append(y)
        n_obs_total += int(x.size)

    init = np.array([1.0, 0.0, np.log(0.3), np.log(0.3), np.log(0.2)])
    result = minimize(
        _neg_log_marginal,
        x0=init,
        args=(xs, ys),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 10000},
    )
    mu_a, mu_b, log_tau_a, log_tau_b, log_sigma = result.x
    tau_a = float(np.exp(log_tau_a))
    tau_b = float(np.exp(log_tau_b))
    sigma = float(np.exp(log_sigma))

    # Wald-style 95 % CI for the group means (no Hessian; use sqrt of
    # tau / n_subj as a conservative SE).
    se_a = tau_a / np.sqrt(len(subjects))
    se_b = tau_b / np.sqrt(len(subjects))
    z = 1.96
    return HierarchicalGMLFit(
        mu_sensitivity=float(mu_a),
        sd_sensitivity=tau_a,
        mu_bias=float(mu_b),
        sd_bias=tau_b,
        sigma_obs=sigma,
        sensitivity_ci=(float(mu_a - z * se_a), float(mu_a + z * se_a)),
        bias_ci=(float(mu_b - z * se_b), float(mu_b + z * se_b)),
        n_subjects=len(subjects),
        n_observations=n_obs_total,
        log_likelihood=-float(result.fun),
    )
