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


# ---------------------------------------------------------------------------
# Closed forms (Mazur, 2001: V = A / (1 + kD))
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.reference(
    module="concurrent_chains",
    verdict="properties",
    source="Mazur (2001)",
)
def test_values_and_choice_match_hyperbolic_closed_form() -> None:
    result = predict_concurrent_chains_choice(
        left_amount=1.0,
        left_delay=10.0,
        right_amount=1.0,
        right_delay=20.0,
        left_responses=60,
        right_responses=40,
        k=0.2,
    )
    assert result.v_left == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert result.v_right == pytest.approx(1.0 / 5.0, rel=1e-12)
    assert result.predicted_proportion_left == pytest.approx(0.625, rel=1e-12)
    assert result.deviation == pytest.approx(0.6 - 0.625, rel=1e-12)


@pytest.mark.unit
def test_amount_delay_tradeoff_at_equal_value_predicts_indifference() -> None:
    # A_R / A_L = (1 + k D_R) / (1 + k D_L) makes the two values equal.
    k, d_l, d_r, a_l = 0.5, 2.0, 8.0, 2.0
    a_r = a_l * (1.0 + k * d_r) / (1.0 + k * d_l)
    result = predict_concurrent_chains_choice(
        left_amount=a_l,
        left_delay=d_l,
        right_amount=a_r,
        right_delay=d_r,
        left_responses=0,
        right_responses=0,
        k=k,
    )
    assert result.predicted_proportion_left == pytest.approx(0.5, rel=1e-12)
