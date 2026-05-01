"""Tests for Gaussian HMM segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.hmm_segment import (
    GaussianHMMFit,
    HMMSegment,
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
