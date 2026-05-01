"""Hidden Markov Model segmentation of behavioral time series.

Generalises Schneider's (1969) two-state break-and-run analysis to a
``K``-state stationary Hidden Markov Model with Gaussian emissions.
The procedure:

1. **Fit** an HMM to the observation series via the Baum-Welch
   (EM) algorithm, returning maximum-likelihood estimates of the
   initial-state distribution ``pi``, transition matrix ``A``, and
   per-state emission ``(mu_k, sigma_k)``.
2. **Decode** the most likely state sequence with the Viterbi
   algorithm.
3. **Segment** runs of the same state into ``(start_index, end_index,
   state, mean_observation)`` tuples.

Reference
---------
Rabiner, L. R. (1989). A tutorial on hidden Markov models and
    selected applications in speech recognition. *Proceedings of the
    IEEE*, *77*(2), 257-286. https://doi.org/10.1109/5.18626
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HMMSegment:
    """Run-length segment of a decoded state sequence.

    Attributes
    ----------
    start_index, end_index : int
        Inclusive start, exclusive end into the original observation
        array.
    state : int
        State id (0..K-1).
    mean_observation : float
    """

    start_index: int
    end_index: int
    state: int
    mean_observation: float


@dataclass(frozen=True)
class GaussianHMMFit:
    """Maximum-likelihood Gaussian HMM fit + decoded segmentation.

    Attributes
    ----------
    pi : tuple of float
        Initial-state distribution (length K).
    transition_matrix : tuple[tuple[float, ...], ...]
        ``A[i][j]`` = P(state_t+1 = j | state_t = i).
    means : tuple of float
        Per-state emission mean.
    sigmas : tuple of float
        Per-state emission SD.
    log_likelihood : float
    state_sequence : tuple of int
        Viterbi-decoded state for each observation.
    segments : tuple of HMMSegment
    n : int
    n_iterations : int
    converged : bool
    """

    pi: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]
    log_likelihood: float
    state_sequence: tuple[int, ...]
    segments: tuple[HMMSegment, ...]
    n: int
    n_iterations: int
    converged: bool


def _log_gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return -0.5 * np.log(2.0 * np.pi * sigma * sigma) - 0.5 * ((x - mu) ** 2) / (sigma * sigma)


def _forward_backward(
    obs: np.ndarray, pi: np.ndarray, A: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    n = obs.size
    k = pi.size
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_emit = np.empty((n, k))
    for s in range(k):
        log_emit[:, s] = _log_gauss(obs, float(mus[s]), float(sigmas[s]))

    log_alpha = np.empty((n, k))
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, n):
        for j in range(k):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_A[:, j]) + log_emit[t, j]
            )

    log_beta = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        for i in range(k):
            log_beta[t, i] = np.logaddexp.reduce(
                log_A[i, :] + log_emit[t + 1] + log_beta[t + 1]
            )

    log_likelihood = float(np.logaddexp.reduce(log_alpha[-1]))
    return log_alpha, log_beta, log_likelihood


def _viterbi(
    obs: np.ndarray, pi: np.ndarray, A: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
) -> np.ndarray:
    n = obs.size
    k = pi.size
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_emit = np.empty((n, k))
    for s in range(k):
        log_emit[:, s] = _log_gauss(obs, float(mus[s]), float(sigmas[s]))

    delta = np.empty((n, k))
    psi = np.zeros((n, k), dtype=np.int64)
    delta[0] = log_pi + log_emit[0]
    for t in range(1, n):
        for j in range(k):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = float(scores[psi[t, j]] + log_emit[t, j])
    states = np.empty(n, dtype=np.int64)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def fit_gaussian_hmm(
    observations: Sequence[float],
    *,
    n_states: int,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> GaussianHMMFit:
    """Fit a Gaussian HMM via Baum-Welch and decode it with Viterbi.

    Parameters
    ----------
    observations : sequence of float
        Time series to segment.
    n_states : int
        Number of hidden states (>= 2).
    max_iter : int
    tol : float
        Convergence threshold on log-likelihood improvement.
    seed : int, optional
        Seed for k-means-style state initialisation. Pass ``None`` for
        non-deterministic init.

    Returns
    -------
    GaussianHMMFit
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if len(observations) < n_states:
        raise ValueError(
            f"need at least n_states={n_states} observations, got {len(observations)}"
        )

    obs = np.array(observations, dtype=np.float64)
    n = obs.size
    k = n_states

    # Initialise state means by linear quantiles, sigmas by overall SD,
    # transition matrix uniform, pi uniform.
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    mus = np.quantile(obs, quantiles)
    sigmas = np.full(k, max(float(np.std(obs)), 1e-6))
    pi = np.full(k, 1.0 / k)
    A = np.full((k, k), 1.0 / k)

    if seed is not None:
        rng = np.random.default_rng(seed)
        mus = mus + rng.normal(scale=1e-3, size=k)

    prev_ll = -np.inf
    converged = False
    iter_used = 0
    for it in range(1, max_iter + 1):
        iter_used = it
        log_alpha, log_beta, ll = _forward_backward(obs, pi, A, mus, sigmas)
        log_gamma = log_alpha + log_beta - ll
        gamma = np.exp(log_gamma)

        log_emit = np.empty((n, k))
        for s in range(k):
            log_emit[:, s] = _log_gauss(obs, float(mus[s]), float(sigmas[s]))
        log_xi_acc = np.full((k, k), -np.inf)
        log_A = np.log(A + 1e-300)
        for t in range(n - 1):
            term = (
                log_alpha[t][:, None]
                + log_A
                + log_emit[t + 1][None, :]
                + log_beta[t + 1][None, :]
            ) - ll
            log_xi_acc = np.logaddexp(log_xi_acc, term)
        xi = np.exp(log_xi_acc)

        pi = gamma[0] / max(float(np.sum(gamma[0])), 1e-12)
        denom = np.sum(gamma[:-1], axis=0)
        denom = np.where(denom > 0.0, denom, 1e-12)
        A = xi / denom[:, None]
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)

        gamma_sum = gamma.sum(axis=0)
        gamma_sum = np.where(gamma_sum > 0.0, gamma_sum, 1e-12)
        new_mus = (gamma * obs[:, None]).sum(axis=0) / gamma_sum
        new_vars = (gamma * (obs[:, None] - new_mus[None, :]) ** 2).sum(axis=0) / gamma_sum
        new_sigmas = np.sqrt(np.maximum(new_vars, 1e-9))
        mus = new_mus
        sigmas = new_sigmas

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    states = _viterbi(obs, pi, A, mus, sigmas)
    log_likelihood = _forward_backward(obs, pi, A, mus, sigmas)[2]

    segments: list[HMMSegment] = []
    start = 0
    for i in range(1, n):
        if states[i] != states[start]:
            segments.append(
                HMMSegment(
                    start_index=start,
                    end_index=i,
                    state=int(states[start]),
                    mean_observation=float(np.mean(obs[start:i])),
                )
            )
            start = i
    segments.append(
        HMMSegment(
            start_index=start,
            end_index=n,
            state=int(states[start]),
            mean_observation=float(np.mean(obs[start:n])),
        )
    )

    return GaussianHMMFit(
        pi=tuple(float(x) for x in pi),
        transition_matrix=tuple(tuple(float(v) for v in row) for row in A),
        means=tuple(float(m) for m in mus),
        sigmas=tuple(float(s) for s in sigmas),
        log_likelihood=float(log_likelihood),
        state_sequence=tuple(int(s) for s in states),
        segments=tuple(segments),
        n=int(n),
        n_iterations=iter_used,
        converged=converged,
    )
