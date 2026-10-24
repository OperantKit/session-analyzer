"""Tests for Generalized Matching Law fitting.

Strategy: synthesize ratio data with known sensitivity ``a`` and bias
``log b`` and verify ``estimate_matching_parameters`` recovers the
parameters within tight tolerance. Also exercises error cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.matching import (
    MatchingParameters,
    estimate_matching_parameters,
    estimate_single_observation_gml,
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


# ---------------------------------------------------------------------------
# estimate_single_observation_gml
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_obs_gml_strict_matching_when_b_equals_r() -> None:
    # B1/B2 = R1/R2 = 2 ⇒ a=1.0, b=1.0.
    result = estimate_single_observation_gml(20, 10, 4, 2)
    assert result is not None
    a, b = result
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(1.0)


@pytest.mark.unit
def test_single_obs_gml_equal_responses_and_reinforcers_returns_one_one() -> None:
    # Equal everything: convention a=1, b=1.
    result = estimate_single_observation_gml(10, 10, 5, 5)
    assert result == (1.0, 1.0)


@pytest.mark.unit
def test_single_obs_gml_undermatching() -> None:
    # B1/B2 = 2 with R1/R2 = 4 ⇒ a = log(2)/log(4) = 0.5.
    result = estimate_single_observation_gml(20, 10, 8, 2)
    assert result is not None
    a, _b = result
    assert a == pytest.approx(0.5)


@pytest.mark.unit
def test_single_obs_gml_overmatching() -> None:
    # B1/B2 = 4 with R1/R2 = 2 ⇒ a = log(4)/log(2) = 2.0.
    result = estimate_single_observation_gml(20, 5, 4, 2)
    assert result is not None
    a, _b = result
    assert a == pytest.approx(2.0)


@pytest.mark.unit
def test_single_obs_gml_negative_counts_returns_none() -> None:
    assert estimate_single_observation_gml(-1, 10, 5, 5) is None
    assert estimate_single_observation_gml(10, 10, -1, 5) is None


@pytest.mark.unit
def test_single_obs_gml_both_responses_zero_returns_none() -> None:
    assert estimate_single_observation_gml(0, 0, 5, 5) is None


@pytest.mark.unit
def test_single_obs_gml_both_reinforcements_zero_returns_none() -> None:
    assert estimate_single_observation_gml(10, 10, 0, 0) is None


@pytest.mark.unit
def test_single_obs_gml_b_denominator_zero_returns_none() -> None:
    # responses_b = 0: cannot form a finite response ratio.
    assert estimate_single_observation_gml(10, 0, 5, 5) is None
    # reinforcements_b = 0: cannot form a finite reinforcement ratio.
    assert estimate_single_observation_gml(10, 10, 5, 0) is None


@pytest.mark.unit
def test_single_obs_gml_unequal_responses_with_equal_reinforcers_returns_none() -> None:
    # R1 == R2 but B1 != B2 ⇒ sensitivity is undetermined ⇒ None.
    assert estimate_single_observation_gml(20, 10, 5, 5) is None



# --- Reference computations ---------------------------------------------
#
# Published values: Baum (1974) Table 1 (per-dataset a and log k) would be
# the natural target, but the article body (PMC1333261, scanned PDF) could
# not be retrieved; the abstract gives the equation only. No published
# value is asserted here. Instead the estimator is checked against ordinary
# least squares (numpy.polyfit, degree 1) on log ratios, which is the
# standard GML estimator, and the single-observation solver by hand.


@pytest.mark.unit
@pytest.mark.reference(
    module="matching",
    verdict="not-retrieved",
    source="Baum (1974)",
)
def test_gml_fit_equals_ordinary_least_squares() -> None:
    rng = np.random.default_rng(20261024)
    x = np.linspace(-1.5, 1.5, 9)
    y = 0.82 * x + 0.11 + rng.normal(0.0, 0.08, size=x.size)
    result = estimate_matching_parameters(np.exp(y).tolist(), np.exp(x).tolist())
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    assert result.sensitivity == pytest.approx(slope, rel=1e-9)
    assert result.bias == pytest.approx(intercept, abs=1e-9)
    assert result.r_squared == pytest.approx(r**2, rel=1e-9)


@pytest.mark.unit
def test_gml_bias_is_reported_in_natural_log_units() -> None:
    """Baum (1974) and most matching papers report log k in base 10. The
    estimator works on natural logs, so ``bias = ln b = ln(10) * log10 b``;
    sensitivity is base-invariant."""
    b = 1.6  # log10 b = 0.2041
    x = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    responses = (b * x**0.9).tolist()
    result = estimate_matching_parameters(responses, x.tolist())
    assert result.sensitivity == pytest.approx(0.9, abs=1e-9)
    assert result.bias == pytest.approx(np.log(10.0) * np.log10(b), abs=1e-9)


@pytest.mark.unit
def test_single_obs_gml_hand_computation() -> None:
    """B1/B2 = 60/20 = 3, R1/R2 = 40/10 = 4:
    a = ln 3 / ln 4 = 0.792481..., log b = 0 so b = 1."""
    a, b = estimate_single_observation_gml(60, 20, 40, 10)  # type: ignore[misc]
    assert a == pytest.approx(math.log(3.0) / math.log(4.0), rel=1e-12)
    assert a == pytest.approx(0.7924812503605781, rel=1e-12)
    assert b == pytest.approx(1.0, abs=1e-12)
