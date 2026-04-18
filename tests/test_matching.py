"""Tests for Generalized Matching Law fitting.

Strategy: synthesize ratio data with known sensitivity ``a`` and bias
``log b`` and verify ``estimate_matching_parameters`` recovers the
parameters within tight tolerance. Also exercises error cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.matching import (
    MatchingParameters,
    estimate_matching_parameters,
    generalized_matching_law,
)


@pytest.mark.unit
def test_recovers_perfect_matching_parameters() -> None:
    """When data lies exactly on log B = log R (a=1, log b=0), fit must recover it."""
    reinforcement_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    responses_ratios = list(reinforcement_ratios)  # B/B' = R/R'

    result = estimate_matching_parameters(responses_ratios, reinforcement_ratios)

    assert isinstance(result, MatchingParameters)
    assert result.sensitivity == pytest.approx(1.0, abs=1e-6)
    assert result.bias == pytest.approx(0.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.n == 5


@pytest.mark.unit
def test_recovers_undermatching_with_bias() -> None:
    """Synthesize a=0.7, log b=0.4 ratios and recover them."""
    a_true = 0.7
    log_b_true = 0.4
    log_r = np.linspace(np.log(0.1), np.log(10.0), 20)
    log_b_responses = generalized_matching_law(log_r, a_true, log_b_true)

    reinforcement_ratios = np.exp(log_r).tolist()
    responses_ratios = np.exp(log_b_responses).tolist()

    result = estimate_matching_parameters(responses_ratios, reinforcement_ratios)
    assert result.sensitivity == pytest.approx(a_true, abs=1e-3)
    assert result.bias == pytest.approx(log_b_true, abs=1e-3)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_recovers_overmatching() -> None:
    a_true = 1.4
    log_b_true = -0.2
    log_r = np.linspace(np.log(0.2), np.log(5.0), 15)
    responses_ratios = np.exp(generalized_matching_law(log_r, a_true, log_b_true)).tolist()
    reinforcement_ratios = np.exp(log_r).tolist()

    result = estimate_matching_parameters(responses_ratios, reinforcement_ratios)
    assert result.sensitivity == pytest.approx(a_true, abs=1e-3)
    assert result.bias == pytest.approx(log_b_true, abs=1e-3)


@pytest.mark.unit
def test_r_squared_decreases_with_noise() -> None:
    rng = np.random.default_rng(99)
    a_true = 0.85
    log_b_true = 0.0
    log_r = np.linspace(np.log(0.1), np.log(10.0), 30)
    log_b_responses = generalized_matching_law(log_r, a_true, log_b_true)
    log_b_noisy = log_b_responses + rng.normal(scale=0.3, size=len(log_b_responses))

    result = estimate_matching_parameters(np.exp(log_b_noisy).tolist(), np.exp(log_r).tolist())
    assert 0.5 < result.r_squared < 1.0
    # Sensitivity should still be in the right ballpark
    assert result.sensitivity == pytest.approx(a_true, abs=0.2)


@pytest.mark.unit
def test_filters_non_positive_pairs() -> None:
    """Pairs containing zero or negative ratios must be discarded."""
    responses_ratios = [1.0, 2.0, 0.0, 4.0, -1.0]
    reinforcement_ratios = [1.0, 2.0, 1.5, 4.0, 0.5]

    result = estimate_matching_parameters(responses_ratios, reinforcement_ratios)
    assert result.n == 3
    assert result.sensitivity == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_matching_parameters([1.0, 2.0], [1.0, 2.0, 3.0])


@pytest.mark.unit
def test_raises_on_insufficient_data() -> None:
    # Only one valid pair after filtering -> not enough to fit a line
    with pytest.raises(ValueError, match="at least 2"):
        estimate_matching_parameters([1.0, 0.0, -1.0], [1.0, 1.0, 1.0])


@pytest.mark.unit
def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        estimate_matching_parameters([], [])
