"""Bout-decomposed response rate analysis.

Decomposes response rate into engagement bouts after the conceptual model
of Shull, Gaynor, & Grimes (2001):

    response rate = bout_initiation_rate * within_bout_rate * bout_duration

IRTs are modelled as a 2-component log-normal mixture:
  - Short component -> within-bout responses (running rate)
  - Long component  -> inter-bout intervals (bout initiation rate)

Parameters are estimated via Expectation-Maximisation (EM) on log(IRT).

Shull et al. (2001) estimated visit-initiation rate and responses per
visit graphically from log-survivor plots of IRTs. This module uses a
different estimator (EM on a log-normal mixture), so its values for the
same data need not equal theirs. ``mean_inter_bout_interval_sec`` and
``within_bout_irt_sec`` are component medians (``exp`` of the log-scale
means), not arithmetic means.

References
----------
Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate viewed
as engagement bouts: effects of relative reinforcement and schedule type.
Journal of the Experimental Analysis of Behavior, 75(3), 247-274.
https://doi.org/10.1901/jeab.2001.75-247
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_MIN_IRTS: int = 10
_EM_MAX_ITER: int = 200
_EM_TOL: float = 1e-6


@dataclass(frozen=True)
class BoutMetrics:
    """Bout-decomposed response rate metrics (Shull et al., 2001).

    Attributes
    ----------
    bout_initiation_rate : float
        Rate of bout initiations, in bouts / minute.
    within_bout_rate : float
        Response rate *within* a bout (running rate), in responses / second.
    mean_bout_length : float
        Estimated mean number of responses per bout (= 1 / (1 - pi_within)).
    mean_inter_bout_interval_sec : float
        Median inter-bout interval in seconds (exp of the long-IRT component mean).
    within_bout_irt_sec : float
        Median within-bout IRT in seconds (exp of the short-IRT component mean).
    within_bout_log_sigma : float
        Log-scale standard deviation of the within-bout IRT component.
    inter_bout_log_sigma : float
        Log-scale standard deviation of the inter-bout interval component.
    pi_within : float
        Mixing proportion of the within-bout (short) IRT component, in [0, 1].
    n_irts : int
        Number of valid (positive) IRTs used in the fit.
    converged : bool
        Whether the EM algorithm converged within the iteration budget.
    """

    bout_initiation_rate: float
    within_bout_rate: float
    mean_bout_length: float
    mean_inter_bout_interval_sec: float
    within_bout_irt_sec: float
    within_bout_log_sigma: float
    inter_bout_log_sigma: float
    pi_within: float
    n_irts: int
    converged: bool


def _em_2_lognormal(
    log_irts: np.ndarray,
    max_iter: int = _EM_MAX_ITER,
    tol: float = _EM_TOL,
) -> tuple[float, float, float, float, float, bool]:
    """Fit a 2-component Gaussian mixture to log-IRTs via EM.

    The short component (smaller mean) corresponds to within-bout IRTs;
    the long component to inter-bout intervals.

    Returns
    -------
    mu1, sigma1 : parameters of the *short* (within-bout) component
    mu2, sigma2 : parameters of the *long* (inter-bout) component
    pi1 : mixing weight of the short component
    converged : bool
    """
    mu1 = float(np.percentile(log_irts, 20))
    mu2 = float(np.percentile(log_irts, 80))
    spread = max(float(np.std(log_irts)) * 0.5, 0.1)
    sigma1 = sigma2 = spread
    pi1 = 0.7

    prev_ll = -np.inf

    for _ in range(max_iter):
        # E-step: responsibilities
        p1 = pi1 * stats.norm.pdf(log_irts, mu1, sigma1)
        p2 = (1.0 - pi1) * stats.norm.pdf(log_irts, mu2, sigma2)
        denom = p1 + p2 + 1e-300
        r1 = p1 / denom

        # M-step: update parameters
        n1 = float(r1.sum())
        n2 = float(len(log_irts)) - n1

        pi1 = float(np.clip(n1 / len(log_irts), 0.01, 0.99))

        mu1 = float((r1 * log_irts).sum() / (n1 + 1e-300))
        mu2 = float(((1.0 - r1) * log_irts).sum() / (n2 + 1e-300))

        sigma1 = float(np.sqrt((r1 * (log_irts - mu1) ** 2).sum() / (n1 + 1e-300)))
        sigma2 = float(np.sqrt(((1.0 - r1) * (log_irts - mu2) ** 2).sum() / (n2 + 1e-300)))
        sigma1 = max(sigma1, 0.05)
        sigma2 = max(sigma2, 0.05)

        # Keep component 1 as the short (within-bout) component
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            sigma1, sigma2 = sigma2, sigma1
            pi1 = 1.0 - pi1

        ll = float(np.log(denom).sum())
        if abs(ll - prev_ll) < tol:
            return mu1, sigma1, mu2, sigma2, pi1, True
        prev_ll = ll

    return mu1, sigma1, mu2, sigma2, pi1, False


def analyze_bouts(
    irts: list[float],
    *,
    max_iter: int = _EM_MAX_ITER,
    tol: float = _EM_TOL,
) -> BoutMetrics | None:
    """Decompose inter-response times into bout-level metrics.

    Fits a 2-component log-normal mixture to ``irts`` via EM and derives
    the following behavioural metrics (Shull et al., 2001):

    - **within_bout_rate**: response rate inside a bout (running rate)
    - **bout_initiation_rate**: how often bouts are initiated per minute
    - **mean_bout_length**: estimated responses per bout

    Parameters
    ----------
    irts : list of float
        Inter-response times in *seconds*. Zero or negative values are
        discarded. At least 10 positive values are required.
    max_iter : int
        Maximum EM iterations (default 200).
    tol : float
        Log-likelihood convergence threshold (default 1e-6).

    Returns
    -------
    BoutMetrics or None
        Returns ``None`` when fewer than 10 valid IRTs are available.
    """
    valid = [x for x in irts if x > 0.0]
    if len(valid) < _MIN_IRTS:
        return None

    log_irts = np.log(np.array(valid, dtype=np.float64))
    mu1, s1, mu2, s2, pi1, converged = _em_2_lognormal(log_irts, max_iter=max_iter, tol=tol)

    within_irt_sec = float(np.exp(mu1))
    inter_bout_sec = float(np.exp(mu2))

    within_bout_rate = 1.0 / within_irt_sec if within_irt_sec > 0.0 else 0.0
    bout_initiation_rate = 60.0 / inter_bout_sec if inter_bout_sec > 0.0 else 0.0

    pi_inter = 1.0 - pi1
    mean_bout_length = 1.0 / pi_inter if pi_inter > 0.0 else float("inf")

    return BoutMetrics(
        bout_initiation_rate=round(bout_initiation_rate, 4),
        within_bout_rate=round(within_bout_rate, 4),
        mean_bout_length=round(mean_bout_length, 4),
        mean_inter_bout_interval_sec=round(inter_bout_sec, 4),
        within_bout_irt_sec=round(within_irt_sec, 4),
        within_bout_log_sigma=round(s1, 4),
        inter_bout_log_sigma=round(s2, 4),
        pi_within=round(float(pi1), 4),
        n_irts=len(valid),
        converged=converged,
    )
