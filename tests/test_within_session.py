"""Tests for within-session response-rate dynamics (McSweeney 1994)."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.within_session import (
    WithinSessionDynamics,
    analyze_within_session,
    bin_response_rates,
)


@pytest.mark.unit
def test_bin_response_rates_basic() -> None:
    timestamps = [0.5, 1.5, 2.5, 3.5, 4.5]
    rates = bin_response_rates(timestamps, session_duration=5.0, n_bins=5)
    # one response per second-long bin -> 1.0 r/s in each bin
    assert rates == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.unit
def test_bin_response_rates_drops_out_of_range() -> None:
    timestamps = [-0.5, 0.5, 1.5, 5.5]
    rates = bin_response_rates(timestamps, session_duration=2.0, n_bins=2)
    # bin0: ts=0.5 -> 1 response in [0,1) -> 1.0 r/s
    # bin1: ts=1.5 -> 1 response in [1,2) -> 1.0 r/s
    assert rates == pytest.approx([1.0, 1.0])


@pytest.mark.unit
def test_bin_response_rates_validates() -> None:
    with pytest.raises(ValueError):
        bin_response_rates([1.0], session_duration=0.0, n_bins=5)
    with pytest.raises(ValueError):
        bin_response_rates([1.0], session_duration=10.0, n_bins=0)


@pytest.mark.unit
def test_within_session_inverted_u() -> None:
    """McSweeney-style inverted-U: warmup, peak, satiation."""
    rng = np.random.default_rng(0)
    # Generate timestamps from an inhomogeneous Poisson with parabolic rate
    n_bins = 12
    duration = 1200.0  # 20 min
    bin_centers = (np.arange(n_bins) + 0.5) * (duration / n_bins)
    bin_rate = 0.5 + 1.5 * np.sin(np.pi * np.arange(n_bins) / (n_bins - 1))
    timestamps: list[float] = []
    for k, c in enumerate(bin_centers):
        n = rng.poisson(bin_rate[k] * (duration / n_bins))
        timestamps.extend(rng.uniform(c - duration / (2 * n_bins), c + duration / (2 * n_bins), n))

    metrics = analyze_within_session(timestamps, session_duration=duration, n_bins=n_bins)
    assert isinstance(metrics, WithinSessionDynamics)
    # Peak bin should be near the middle
    assert 3 <= metrics.peak_bin <= 8
    assert metrics.rise_slope > 0.0
    assert metrics.fall_slope < 0.0
    assert metrics.peak_rate > metrics.rates[0]
    assert metrics.peak_rate > metrics.rates[-1]
    assert metrics.n_bins == n_bins


@pytest.mark.unit
def test_within_session_monotonic_warmup() -> None:
    """Pure rising rate -> peak at last bin."""
    timestamps: list[float] = []
    for bin_idx in range(10):
        # bin_idx responses within bin
        for k in range(bin_idx + 1):
            timestamps.append(bin_idx * 60.0 + k * 0.5)

    metrics = analyze_within_session(timestamps, session_duration=600.0, n_bins=10)
    assert metrics.peak_bin == 9
    assert metrics.rise_slope > 0.0


@pytest.mark.unit
def test_within_session_validates_inputs() -> None:
    with pytest.raises(ValueError):
        analyze_within_session([], session_duration=100.0, n_bins=10)
    with pytest.raises(ValueError):
        analyze_within_session([1.0], session_duration=100.0, n_bins=1)
