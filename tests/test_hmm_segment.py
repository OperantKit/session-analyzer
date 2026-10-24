"""Tests for Gaussian HMM segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.hmm_segment import (
    GaussianHMMFit,
    HMMSegment,
    _forward_backward,
    _viterbi,
    fit_gaussian_hmm,
)


@pytest.mark.unit
def test_two_state_segmentation_recovers_known_blocks() -> None:
    rng = np.random.default_rng(0)
    block1 = rng.normal(loc=1.0, scale=0.3, size=80)
    block2 = rng.normal(loc=5.0, scale=0.3, size=80)
    block3 = rng.normal(loc=1.0, scale=0.3, size=80)
    obs = np.concatenate([block1, block2, block3]).tolist()

    result = fit_gaussian_hmm(obs, n_states=2)

    assert isinstance(result, GaussianHMMFit)
    assert result.converged
    assert sorted(result.means) == pytest.approx(
        sorted([1.0, 5.0]), abs=0.3
    )
    # State sequence should switch at least twice (3 blocks).
    assert len(result.segments) >= 3
    # Middle block should have a different state from edges.
    middle_state = max(set(result.state_sequence[80:160]), key=list(result.state_sequence[80:160]).count)
    edge_state = max(set(result.state_sequence[:80]), key=list(result.state_sequence[:80]).count)
    assert middle_state != edge_state


@pytest.mark.unit
def test_segments_are_contiguous_and_cover_observations() -> None:
    obs = [1.0] * 30 + [10.0] * 30
    result = fit_gaussian_hmm(obs, n_states=2)
    assert all(isinstance(s, HMMSegment) for s in result.segments)
    # First segment starts at 0, last segment ends at len(obs).
    assert result.segments[0].start_index == 0
    assert result.segments[-1].end_index == len(obs)
    # Segments are contiguous.
    for prev, curr in zip(result.segments, result.segments[1:], strict=False):
        assert prev.end_index == curr.start_index


@pytest.mark.unit
def test_transition_matrix_rows_sum_to_one() -> None:
    obs = list(np.random.default_rng(1).normal(size=200))
    result = fit_gaussian_hmm(obs, n_states=2)
    for row in result.transition_matrix:
        assert sum(row) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_raises_on_too_few_states() -> None:
    with pytest.raises(ValueError, match="n_states"):
        fit_gaussian_hmm([1.0, 2.0, 3.0], n_states=1)


@pytest.mark.unit
def test_raises_on_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least n_states"):
        fit_gaussian_hmm([1.0, 2.0], n_states=3)


@pytest.mark.unit
def test_raises_on_invalid_max_iter_or_tol() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        fit_gaussian_hmm([1.0, 2.0, 3.0], n_states=2, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        fit_gaussian_hmm([1.0, 2.0, 3.0], n_states=2, tol=0.0)


# ---------------------------------------------------------------------------
# Reference implementation (S-166).
#
# Expected values come from hmmlearn 0.3.3 ``GaussianHMM`` (1-D,
# ``covariance_type="diag"``, ``covars_prior=0``, ``covars_weight=1``,
# ``min_covar=1e-12`` so the M-step is plain maximum likelihood), started
# from the same initial values as fit_gaussian_hmm (quantile means with the
# seed-0 jitter, overall SD, uniform pi and A). The series is 300 draws from
# ``numpy.random.default_rng(166)`` in alternating blocks of N(0.5, 0.4) and
# N(3.0, 0.8).
# ---------------------------------------------------------------------------


def _reference_series() -> np.ndarray:
    rng = np.random.default_rng(166)
    blocks = [(0, 40), (1, 30), (0, 50), (1, 20), (0, 60), (1, 40), (0, 60)]
    mu, sd = (0.5, 3.0), (0.4, 0.8)
    return np.concatenate([rng.normal(mu[s], sd[s], n) for s, n in blocks])


def _block_path() -> tuple[int, ...]:
    blocks = [(0, 40), (1, 30), (0, 50), (1, 20), (0, 60), (1, 40), (0, 60)]
    return tuple(s for s, n in blocks for _ in range(n))


@pytest.mark.unit
@pytest.mark.reference(
    module="hmm_segment",
    verdict="reference-implementation",
    source="Rabiner (1989)",
)
def test_fit_matches_hmmlearn_maximum_likelihood() -> None:
    """Converged parameters (rel 1e-8) and log-likelihood (rel 1e-10)."""
    fit = fit_gaussian_hmm(_reference_series(), n_states=2, tol=1e-12, max_iter=5000)
    assert fit.converged
    assert fit.means == pytest.approx((0.5052751772842693, 2.9511930430855795), rel=1e-8)
    assert fit.sigmas == pytest.approx((0.4102330255239148, 0.7974978095968991), rel=1e-8)
    a = fit.transition_matrix
    assert a[0] == pytest.approx((0.9854095149174681, 0.014590485082531992), rel=1e-8)
    assert a[1] == pytest.approx((0.03389141072551079, 0.9661085892744892), rel=1e-8)
    assert fit.pi[0] == pytest.approx(1.0, abs=1e-12)
    assert fit.log_likelihood == pytest.approx(-247.07573596938101, rel=1e-10)
    # hmmlearn's Viterbi path equals the generating block structure here.
    assert fit.state_sequence == _block_path()


@pytest.mark.unit
def test_forward_and_viterbi_match_hmmlearn_at_fixed_parameters() -> None:
    """pi (0.6, 0.4), A [[.95, .05], [.1, .9]], N(0.5, 0.4) / N(3.0, 0.8).

    hmmlearn ``score`` = -254.16971080896866; ``decode`` returns the
    generating block path.
    """
    obs = _reference_series()
    pi = np.array([0.6, 0.4])
    a = np.array([[0.95, 0.05], [0.1, 0.9]])
    mus = np.array([0.5, 3.0])
    sds = np.array([0.4, 0.8])
    ll = _forward_backward(obs, pi, a, mus, sds)[2]
    assert ll == pytest.approx(-254.16971080896866, rel=1e-12)
    assert tuple(int(s) for s in _viterbi(obs, pi, a, mus, sds)) == _block_path()
