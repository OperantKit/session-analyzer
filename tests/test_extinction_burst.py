"""Tests for extinction-burst detection."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.extinction_burst import (
    ExtinctionBurstResult,
    detect_extinction_burst,
)


@pytest.mark.unit
def test_detects_burst_when_initial_rate_exceeds_baseline() -> None:
    # Baseline: 60 responses in 120 s -> 0.5 r/s.
    # First-minute extinction: 60 responses -> 1.0 r/s. Ratio = 2.0.
    baseline = [float(i) * 2.0 for i in range(60)]
    extinction = [120.0 + i for i in range(60)]
    result = detect_extinction_burst(
        baseline_response_times=baseline,
        baseline_duration_sec=120.0,
        extinction_start_time=120.0,
        extinction_response_times=extinction,
        burst_window_sec=60.0,
    )
    assert isinstance(result, ExtinctionBurstResult)
    assert result.burst_ratio == pytest.approx(2.0)
    assert result.burst_detected


@pytest.mark.unit
def test_no_burst_when_extinction_rate_below_threshold() -> None:
    baseline = [float(i) for i in range(120)]
    extinction = [120.0 + i * 1.5 for i in range(40)]  # ~ 0.67 r/s
    result = detect_extinction_burst(
        baseline_response_times=baseline,
        baseline_duration_sec=120.0,
        extinction_start_time=120.0,
        extinction_response_times=extinction,
        burst_window_sec=60.0,
    )
    # baseline_rate = 1.0 r/s, burst_rate = 60 / 60 if all 40 responses in
    # first 60 s; check actual count: extinction times are 120, 121.5, ..., so
    # all 40 are within the first 60 s -> 0.667 r/s -> ratio < 1.0.
    assert not result.burst_detected
    assert result.burst_ratio < 1.0


@pytest.mark.unit
def test_zero_baseline_with_extinction_responses_yields_inf_ratio() -> None:
    result = detect_extinction_burst(
        baseline_response_times=[],
        baseline_duration_sec=60.0,
        extinction_start_time=60.0,
        extinction_response_times=[60.5, 61.0],
        burst_window_sec=30.0,
    )
    assert math.isinf(result.burst_ratio)
    assert result.burst_detected


@pytest.mark.unit
def test_zero_baseline_and_zero_extinction_yields_zero_ratio() -> None:
    result = detect_extinction_burst(
        baseline_response_times=[],
        baseline_duration_sec=60.0,
        extinction_start_time=60.0,
        extinction_response_times=[],
        burst_window_sec=30.0,
    )
    assert result.burst_ratio == 0.0
    assert not result.burst_detected


@pytest.mark.unit
def test_raises_on_invalid_inputs() -> None:
    base = dict(
        baseline_response_times=[],
        extinction_start_time=0.0,
        extinction_response_times=[],
    )
    with pytest.raises(ValueError, match="baseline_duration_sec"):
        detect_extinction_burst(baseline_duration_sec=0.0, **base)
    with pytest.raises(ValueError, match="burst_window_sec"):
        detect_extinction_burst(
            baseline_duration_sec=10.0, burst_window_sec=0.0, **base
        )
    with pytest.raises(ValueError, match="ratio_threshold"):
        detect_extinction_burst(
            baseline_duration_sec=10.0,
            ratio_threshold=0.0,
            **base,
        )


@pytest.mark.unit
def test_raises_on_extinction_time_before_start() -> None:
    with pytest.raises(ValueError, match=">= extinction_start_time"):
        detect_extinction_burst(
            baseline_response_times=[],
            baseline_duration_sec=10.0,
            extinction_start_time=10.0,
            extinction_response_times=[5.0, 11.0],
            burst_window_sec=10.0,
        )


# ---------------------------------------------------------------------------
# Boundaries by hand
# ---------------------------------------------------------------------------
#
# Lerman & Iwata (1995) define the burst as an initial increase in the target
# response under extinction (abstract; they found it in 24% of 113 data sets)
# without a fixed window or ratio, so the 60-s window and 1.5 threshold are
# this module's parameters. The window is closed: a response exactly
# burst_window_sec after onset counts, and a ratio equal to the threshold is a
# burst.


@pytest.mark.unit
def test_ratio_at_threshold_and_closed_window() -> None:
    baseline = [float(t) for t in range(60)]  # 1 response/s for 60 s
    start = 100.0
    extinction = [start + (i + 1) * 60.0 / 90.0 for i in range(90)]  # last at 160.0
    result = detect_extinction_burst(
        baseline_response_times=baseline,
        baseline_duration_sec=60.0,
        extinction_start_time=start,
        extinction_response_times=extinction + [160.5],
    )
    assert extinction[-1] == pytest.approx(160.0)
    assert result.baseline_rate == pytest.approx(1.0)
    assert result.burst_rate == pytest.approx(1.5)
    assert result.burst_ratio == pytest.approx(1.5)
    assert result.burst_detected
