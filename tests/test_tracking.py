"""Tests for PCA sign-tracking / goal-tracking index."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.tracking import (
    TrackingIndex,
    TrackingType,
    compute_tracking_index,
)


@pytest.mark.unit
def test_pure_sign_tracker_classified_at_plus_one() -> None:
    result = compute_tracking_index(
        contacts_cs=50,
        contacts_food=0,
        latency_cs=0.5,
        latency_food=8.0,
        probability_cs=1.0,
        probability_food=0.0,
        cs_duration=8.0,
    )
    assert isinstance(result, TrackingIndex)
    assert result.response_bias == pytest.approx(1.0)
    assert result.probability_score == pytest.approx(1.0)
    assert result.latency_score == pytest.approx((8.0 - 0.5) / 8.0)
    assert result.index > 0.9
    assert result.classification is TrackingType.SIGN_TRACKER


@pytest.mark.unit
def test_pure_goal_tracker_classified_at_minus_one() -> None:
    result = compute_tracking_index(
        contacts_cs=0,
        contacts_food=50,
        latency_cs=8.0,
        latency_food=0.5,
        probability_cs=0.0,
        probability_food=1.0,
        cs_duration=8.0,
    )
    assert result.response_bias == pytest.approx(-1.0)
    assert result.probability_score == pytest.approx(-1.0)
    assert result.index < -0.9
    assert result.classification is TrackingType.GOAL_TRACKER


@pytest.mark.unit
def test_equal_behavior_is_intermediate() -> None:
    result = compute_tracking_index(
        contacts_cs=10,
        contacts_food=10,
        latency_cs=4.0,
        latency_food=4.0,
        probability_cs=0.5,
        probability_food=0.5,
        cs_duration=8.0,
    )
    assert result.index == pytest.approx(0.0)
    assert result.classification is TrackingType.INTERMEDIATE


@pytest.mark.unit
def test_zero_contacts_response_bias_is_zero() -> None:
    result = compute_tracking_index(
        contacts_cs=0,
        contacts_food=0,
        latency_cs=8.0,
        latency_food=8.0,
        probability_cs=0.0,
        probability_food=0.0,
        cs_duration=8.0,
    )
    assert result.response_bias == 0.0
    assert result.latency_score == 0.0


@pytest.mark.unit
def test_custom_thresholds_shift_classification() -> None:
    # Index ≈ 0.4 normally intermediate; with sign_threshold=0.3 becomes ST.
    high_sign = compute_tracking_index(
        contacts_cs=6,
        contacts_food=4,
        latency_cs=3.0,
        latency_food=5.0,
        probability_cs=0.6,
        probability_food=0.4,
        cs_duration=8.0,
        sign_threshold=0.2,
        goal_threshold=-0.2,
    )
    assert high_sign.classification is TrackingType.SIGN_TRACKER


@pytest.mark.unit
def test_raises_on_invalid_inputs() -> None:
    base = dict(
        contacts_food=10,
        latency_cs=1.0,
        latency_food=1.0,
        probability_cs=0.5,
        probability_food=0.5,
        cs_duration=8.0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        compute_tracking_index(contacts_cs=-1, **base)
    with pytest.raises(ValueError, match="probabilities"):
        compute_tracking_index(
            contacts_cs=10,
            contacts_food=10,
            latency_cs=1.0,
            latency_food=1.0,
            probability_cs=1.5,
            probability_food=0.5,
            cs_duration=8.0,
        )
    with pytest.raises(ValueError, match="cs_duration"):
        compute_tracking_index(
            contacts_cs=10,
            contacts_food=10,
            latency_cs=1.0,
            latency_food=1.0,
            probability_cs=0.5,
            probability_food=0.5,
            cs_duration=0.0,
        )
    with pytest.raises(ValueError, match="goal_threshold"):
        compute_tracking_index(
            contacts_cs=10,
            contacts_food=10,
            latency_cs=1.0,
            latency_food=1.0,
            probability_cs=0.5,
            probability_food=0.5,
            cs_duration=8.0,
            sign_threshold=0.1,
            goal_threshold=0.2,
        )


# ---------------------------------------------------------------------------
# Published definition (S-166).
#
# Meyer, P. J., Lovic, V., Saunders, B. T., Yager, L. M., Flagel, S. B.,
# Morrow, J. D., & Robinson, T. E. (2012). PLoS ONE, 7(6), e38987.
# https://doi.org/10.1371/journal.pone.0038987
# Table 1: Response Bias = (lever - cup) / (lever + cup); Probability
# Difference = p(lever) - p(cup); Latency Score = (mean cup latency - mean
# lever latency) / 8 (CS duration 8 s); PCA Score = mean of the three.
# Classification (Results, Fig. 5): ST if the score is 0.5 or above, GT if
# -0.5 or less. The paper reports no worked numeric example, so the values
# below are hand computations of Table 1.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_matches_meyer_2012_table_1_formulas() -> None:
    # 60 lever presses, 20 cup entries; p 0.9 vs 0.4; latencies 2.0 vs 6.0 s.
    # RB = 40/80 = 0.5; PD = 0.5; LS = (6 - 2)/8 = 0.5; PCA = 0.5 -> ST.
    r = compute_tracking_index(
        contacts_cs=60,
        contacts_food=20,
        latency_cs=2.0,
        latency_food=6.0,
        probability_cs=0.9,
        probability_food=0.4,
        cs_duration=8.0,
    )
    assert r.response_bias == pytest.approx(0.5, abs=1e-12)
    assert r.probability_score == pytest.approx(0.5, abs=1e-12)
    assert r.latency_score == pytest.approx(0.5, abs=1e-12)
    assert r.index == pytest.approx(0.5, abs=1e-12)
    assert r.classification is TrackingType.SIGN_TRACKER


@pytest.mark.unit
def test_meyer_2012_cutoff_minus_half_is_goal_tracker() -> None:
    # Mirror image: RB = -0.5, PD = -0.5, LS = -0.5 -> PCA = -0.5 -> GT.
    r = compute_tracking_index(
        contacts_cs=20,
        contacts_food=60,
        latency_cs=6.0,
        latency_food=2.0,
        probability_cs=0.4,
        probability_food=0.9,
        cs_duration=8.0,
    )
    assert r.index == pytest.approx(-0.5, abs=1e-12)
    assert r.classification is TrackingType.GOAL_TRACKER
