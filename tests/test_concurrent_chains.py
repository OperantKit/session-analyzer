"""Tests for concurrent-chains choice prediction."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.concurrent_chains import (
    ConcurrentChainsPrediction,
    predict_concurrent_chains_choice,
)


@pytest.mark.unit
def test_equal_amounts_and_delays_predict_indifference() -> None:
    result = predict_concurrent_chains_choice(
        left_amount=1.0,
        left_delay=10.0,
        right_amount=1.0,
        right_delay=10.0,
        left_responses=50,
        right_responses=50,
    )
    assert isinstance(result, ConcurrentChainsPrediction)
    assert result.predicted_proportion_left == pytest.approx(0.5)
    assert result.observed_proportion_left == pytest.approx(0.5)
    assert result.deviation == pytest.approx(0.0)


@pytest.mark.unit
def test_shorter_delay_raises_predicted_proportion() -> None:
    result = predict_concurrent_chains_choice(
        left_amount=1.0,
        left_delay=2.0,
        right_amount=1.0,
        right_delay=10.0,
        left_responses=0,
        right_responses=0,
        k=1.0,
    )
    # V_L = 1/3, V_R = 1/11 -> P_L = (1/3) / (1/3 + 1/11) = 11/14.
    assert result.predicted_proportion_left == pytest.approx(11.0 / 14.0, rel=1e-6)


@pytest.mark.unit
def test_larger_amount_raises_predicted_proportion() -> None:
    result = predict_concurrent_chains_choice(
        left_amount=3.0,
        left_delay=10.0,
        right_amount=1.0,
        right_delay=10.0,
        left_responses=0,
        right_responses=0,
        k=1.0,
    )
    # V_L = 3/11, V_R = 1/11 -> P_L = 0.75.
    assert result.predicted_proportion_left == pytest.approx(0.75)


@pytest.mark.unit
def test_zero_responses_yields_observed_half() -> None:
    result = predict_concurrent_chains_choice(
        left_amount=1.0,
        left_delay=1.0,
        right_amount=1.0,
        right_delay=1.0,
        left_responses=0,
        right_responses=0,
    )
    assert result.observed_proportion_left == 0.5


@pytest.mark.unit
def test_raises_on_invalid_parameters() -> None:
    base = dict(left_delay=1.0, right_delay=1.0, left_responses=10, right_responses=10)
    with pytest.raises(ValueError, match="amounts"):
        predict_concurrent_chains_choice(left_amount=0.0, right_amount=1.0, **base)
    with pytest.raises(ValueError, match="delays"):
        predict_concurrent_chains_choice(
            left_amount=1.0,
            right_amount=1.0,
            left_delay=-1.0,
            right_delay=1.0,
            left_responses=10,
            right_responses=10,
        )
    with pytest.raises(ValueError, match="response counts"):
        predict_concurrent_chains_choice(
            left_amount=1.0,
            right_amount=1.0,
            left_delay=1.0,
            right_delay=1.0,
            left_responses=-1,
            right_responses=10,
        )
    with pytest.raises(ValueError, match="k"):
        predict_concurrent_chains_choice(
            left_amount=1.0,
            right_amount=1.0,
            left_delay=1.0,
            right_delay=1.0,
            left_responses=10,
            right_responses=10,
            k=0.0,
        )
