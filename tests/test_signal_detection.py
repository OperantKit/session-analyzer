"""Tests for signal detection theory (SDT) analysis."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.signal_detection import (
    ROCCurve,
    SignalDetectionMetrics,
    compute_roc,
    compute_sdt,
)


@pytest.mark.unit
def test_sdt_unbiased_observer_zero_criterion() -> None:
    # H = 0.84, F = 0.16  -> d' ~ 2.0, c ~ 0
    metrics = compute_sdt(hits=84, misses=16, false_alarms=16, correct_rejections=84)
    assert isinstance(metrics, SignalDetectionMetrics)
    # log-linear correction shifts (84.5/101, 16.5/101) ~ (0.837, 0.163)
    assert metrics.hit_rate == pytest.approx(0.837, abs=5e-3)
    assert metrics.false_alarm_rate == pytest.approx(0.163, abs=5e-3)
    assert metrics.d_prime == pytest.approx(2.0, abs=0.05)
    assert metrics.criterion == pytest.approx(0.0, abs=0.05)
    assert metrics.beta == pytest.approx(1.0, abs=0.1)


@pytest.mark.unit
def test_sdt_chance_performance() -> None:
    metrics = compute_sdt(hits=50, misses=50, false_alarms=50, correct_rejections=50)
    assert metrics.d_prime == pytest.approx(0.0, abs=1e-9)
    assert metrics.a_prime == pytest.approx(0.5, abs=1e-9)


@pytest.mark.unit
def test_sdt_log_linear_correction_ceiling() -> None:
    """Hit rate of 1.0 should be corrected via log-linear (Hautus 1995)."""
    metrics = compute_sdt(hits=100, misses=0, false_alarms=0, correct_rejections=100)
    # Both rates corrected toward (n+0.5)/(n+1); finite d_prime
    assert math.isfinite(metrics.d_prime)
    assert metrics.d_prime > 3.0
    assert metrics.hit_rate < 1.0
    assert metrics.false_alarm_rate > 0.0


@pytest.mark.unit
def test_sdt_conservative_bias_positive_c() -> None:
    # H low, F low -> conservative (high criterion, c > 0)
    metrics = compute_sdt(hits=30, misses=70, false_alarms=5, correct_rejections=95)
    assert metrics.criterion > 0.0
    assert metrics.beta > 1.0


@pytest.mark.unit
def test_sdt_liberal_bias_negative_c() -> None:
    metrics = compute_sdt(hits=95, misses=5, false_alarms=70, correct_rejections=30)
    assert metrics.criterion < 0.0
    assert metrics.beta < 1.0


@pytest.mark.unit
def test_sdt_validates_inputs() -> None:
    with pytest.raises(ValueError):
        compute_sdt(hits=0, misses=0, false_alarms=0, correct_rejections=10)
    with pytest.raises(ValueError):
        compute_sdt(hits=10, misses=0, false_alarms=0, correct_rejections=0)
    with pytest.raises(ValueError):
        compute_sdt(hits=-1, misses=0, false_alarms=0, correct_rejections=10)


@pytest.mark.unit
def test_roc_monotonic() -> None:
    rates = [
        (0.05, 0.01),
        (0.30, 0.10),
        (0.60, 0.30),
        (0.85, 0.55),
        (0.99, 0.90),
    ]
    roc = compute_roc(rates)
    assert isinstance(roc, ROCCurve)
    # Includes endpoints (0,0) and (1,1)
    assert roc.fa_rates[0] == 0.0
    assert roc.hit_rates[0] == 0.0
    assert roc.fa_rates[-1] == 1.0
    assert roc.hit_rates[-1] == 1.0
    # Monotonic non-decreasing in FA
    for i in range(1, len(roc.fa_rates)):
        assert roc.fa_rates[i] >= roc.fa_rates[i - 1]
    # AUC > 0.5 for above-chance points
    assert roc.auc > 0.5


@pytest.mark.unit
def test_roc_chance_diagonal() -> None:
    rates = [(p, p) for p in (0.2, 0.5, 0.8)]
    roc = compute_roc(rates)
    assert roc.auc == pytest.approx(0.5, abs=0.05)


@pytest.mark.unit
def test_roc_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_roc([])
