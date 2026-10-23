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


# ---------------------------------------------------------------------------
# Hand computation
# ---------------------------------------------------------------------------
#
# The module is descriptive (Epstein, 1985); Shahan & Sweeney (2011) model
# predictions are out of scope, so there is no published value to match.


@pytest.mark.unit
def test_metrics_match_hand_computation_with_windows() -> None:
    m = analyze_resurgence(
        target_rates_phase1=[40.0, 50.0, 60.0],
        target_rates_phase2=[20.0, 10.0, 5.0, 3.0],
        target_rates_phase3=[12.0, 8.0, 4.0],
        alt_rates_phase2=[5.0, 20.0, 30.0, 32.0],
        alt_rates_phase3=[10.0, 4.0, 1.0],
        baseline_window=2,
        test_window=2,
    )
    assert m.baseline_target_rate == 55.0  # mean(50, 60)
    assert m.suppression_target_rate == 4.0  # mean(5, 3)
    assert m.test_target_rate == 10.0  # mean(12, 8)
    assert m.resurgence_magnitude == 6.0
    assert m.resurgence_ratio == round(10.0 / 55.0, 4)
    assert m.alt_baseline_rate == 31.0
    assert m.alt_test_rate == 7.0
    assert m.resurged


@pytest.mark.unit
def test_resurged_threshold_is_inclusive() -> None:
    m = analyze_resurgence(
        target_rates_phase1=[50.0],
        target_rates_phase2=[2.0],
        target_rates_phase3=[5.0],
        alt_rates_phase2=[30.0],
        alt_rates_phase3=[1.0],
    )
    assert m.resurgence_ratio == 0.1
    assert m.resurged
