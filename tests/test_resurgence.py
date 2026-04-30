"""Tests for resurgence quantification (3-phase target/alternative paradigm)."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.resurgence import (
    ResurgenceMetrics,
    analyze_resurgence,
)


@pytest.mark.unit
def test_resurgence_classic_pattern() -> None:
    """Target response rate decreases under DRA, recovers under test."""
    metrics = analyze_resurgence(
        target_rates_phase1=[40, 42, 41, 43, 44],
        target_rates_phase2=[30, 15, 5, 2, 1],
        target_rates_phase3=[20, 10, 5, 2, 1],
        alt_rates_phase2=[5, 25, 38, 41, 42],
        alt_rates_phase3=[40, 20, 5, 2, 1],
        baseline_window=3,
        test_window=2,
    )
    assert isinstance(metrics, ResurgenceMetrics)
    # Baseline ~ 42.67 (mean of last 3 of phase 1)
    assert metrics.baseline_target_rate == pytest.approx(42.67, abs=0.1)
    # Phase-2 terminal target rate ~ 1.5 (last 3 sessions mean)
    assert metrics.suppression_target_rate == pytest.approx(2.67, abs=0.1)
    # Phase-3 test rate (first 2 sessions) ~ 15
    assert metrics.test_target_rate == pytest.approx(15.0, abs=0.1)
    # Resurgence magnitude = test - phase2 terminal
    assert metrics.resurgence_magnitude == pytest.approx(15.0 - 2.67, abs=0.1)
    # Resurgence ratio = test / baseline
    assert metrics.resurgence_ratio == pytest.approx(15.0 / 42.67, rel=0.01)
    assert metrics.resurged is True


@pytest.mark.unit
def test_resurgence_no_recovery() -> None:
    metrics = analyze_resurgence(
        target_rates_phase1=[40, 42, 44],
        target_rates_phase2=[30, 15, 1],
        target_rates_phase3=[1, 1, 1, 1],
        alt_rates_phase2=[5, 25, 40],
        alt_rates_phase3=[30, 10, 2, 1],
        baseline_window=3,
        test_window=2,
    )
    assert metrics.resurgence_magnitude < 1.0
    assert metrics.resurged is False


@pytest.mark.unit
def test_resurgence_alternative_suppression_index() -> None:
    """The alternative-response index quantifies the alternative's
    suppression in phase 3 relative to its phase 2 terminal level."""
    metrics = analyze_resurgence(
        target_rates_phase1=[40, 40, 40],
        target_rates_phase2=[20, 5, 1],
        target_rates_phase3=[10, 5, 2],
        alt_rates_phase2=[10, 30, 40],
        alt_rates_phase3=[35, 10, 2],
        baseline_window=2,
        test_window=2,
    )
    # alt_baseline (phase 2 terminal) ~ 35, alt_test (phase 3 early) ~ 22.5
    assert metrics.alt_baseline_rate == pytest.approx(35.0, abs=0.1)
    assert metrics.alt_test_rate == pytest.approx(22.5, abs=0.1)


@pytest.mark.unit
def test_resurgence_validates_window() -> None:
    with pytest.raises(ValueError):
        analyze_resurgence(
            target_rates_phase1=[1, 2],
            target_rates_phase2=[1],
            target_rates_phase3=[1],
            alt_rates_phase2=[1],
            alt_rates_phase3=[1],
            baseline_window=5,
            test_window=1,
        )


@pytest.mark.unit
def test_resurgence_validates_phase_alignment() -> None:
    with pytest.raises(ValueError):
        analyze_resurgence(
            target_rates_phase1=[40, 40],
            target_rates_phase2=[10, 5],
            target_rates_phase3=[2, 3],
            alt_rates_phase2=[5, 25, 30],  # length mismatch
            alt_rates_phase3=[2, 1],
            baseline_window=2,
            test_window=1,
        )


@pytest.mark.unit
def test_resurgence_default_windows() -> None:
    """Default windows: full phase 1 baseline; first session of phase 3."""
    metrics = analyze_resurgence(
        target_rates_phase1=[40, 40, 40],
        target_rates_phase2=[10, 5, 1],
        target_rates_phase3=[20, 10, 5],
        alt_rates_phase2=[10, 30, 40],
        alt_rates_phase3=[30, 10, 2],
    )
    assert metrics.baseline_target_rate == pytest.approx(40.0)
    # Default test_window=1 -> first session only
    assert metrics.test_target_rate == pytest.approx(20.0)
