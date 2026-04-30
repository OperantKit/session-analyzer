"""Tests for observing-response / conditioned-reinforcement analysis."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.observing import (
    ConditionedReinforcementStrength,
    ObservingMetrics,
    compute_conditioned_reinforcement_strength,
    compute_observing_metrics,
)


@pytest.mark.unit
def test_observing_metrics_basic() -> None:
    metrics = compute_observing_metrics(
        observing_count=120,
        observing_duration_sec=600.0,
        s_plus_productions=80,
        s_minus_productions=40,
    )
    assert isinstance(metrics, ObservingMetrics)
    # 120 / (600/60) = 12 / min
    assert metrics.observing_rate_per_min == pytest.approx(12.0)
    assert metrics.s_plus_proportion == pytest.approx(80 / 120)
    assert metrics.good_news_dominance is True


@pytest.mark.unit
def test_observing_metrics_balanced_productions() -> None:
    metrics = compute_observing_metrics(
        observing_count=100,
        observing_duration_sec=600.0,
        s_plus_productions=50,
        s_minus_productions=50,
    )
    assert metrics.s_plus_proportion == pytest.approx(0.5)
    assert metrics.good_news_dominance is False


@pytest.mark.unit
def test_observing_metrics_validates() -> None:
    with pytest.raises(ValueError):
        compute_observing_metrics(
            observing_count=10,
            observing_duration_sec=0.0,
            s_plus_productions=5,
            s_minus_productions=5,
        )
    with pytest.raises(ValueError):
        compute_observing_metrics(
            observing_count=-1,
            observing_duration_sec=10.0,
            s_plus_productions=0,
            s_minus_productions=0,
        )
    with pytest.raises(ValueError):
        # productions > observing_count is allowed (one obs may produce one stimulus
        # period, with multiple stimulus types over that period). But mismatch in totals
        # should be permitted; instead, validate non-negativity:
        compute_observing_metrics(
            observing_count=10,
            observing_duration_sec=10.0,
            s_plus_productions=-1,
            s_minus_productions=0,
        )


@pytest.mark.unit
def test_conditioned_reinforcement_strength_above_chance() -> None:
    metrics = compute_conditioned_reinforcement_strength(
        observing_rate_correlated=12.0,
        observing_rate_uncorrelated=2.0,
    )
    assert isinstance(metrics, ConditionedReinforcementStrength)
    assert metrics.strength_index == pytest.approx(6.0)
    assert metrics.is_conditioned_reinforcer is True


@pytest.mark.unit
def test_conditioned_reinforcement_strength_no_effect() -> None:
    metrics = compute_conditioned_reinforcement_strength(
        observing_rate_correlated=5.0,
        observing_rate_uncorrelated=4.8,
    )
    assert metrics.strength_index == pytest.approx(5.0 / 4.8, rel=1e-6)
    assert metrics.is_conditioned_reinforcer is False


@pytest.mark.unit
def test_conditioned_reinforcement_strength_validates() -> None:
    with pytest.raises(ValueError):
        compute_conditioned_reinforcement_strength(
            observing_rate_correlated=-1.0,
            observing_rate_uncorrelated=1.0,
        )
    with pytest.raises(ValueError):
        compute_conditioned_reinforcement_strength(
            observing_rate_correlated=1.0,
            observing_rate_uncorrelated=0.0,
        )
