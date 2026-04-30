"""Tests for peak-procedure analysis."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.peak import (
    PeakTrialBinnedMetrics,
    PeakTrialMLEMetrics,
    analyze_peak_trial_binned,
    analyze_peak_trial_mle,
)


@pytest.mark.unit
def test_binned_returns_none_for_no_responses() -> None:
    assert analyze_peak_trial_binned([], trial_duration=60.0) is None


@pytest.mark.unit
def test_binned_recovers_peak_from_gaussian_responses() -> None:
    rng = np.random.default_rng(7)
    # Responses concentrated around t=30 (simulate FI 30 peak trial).
    times = np.sort(
        np.clip(rng.normal(loc=30.0, scale=5.0, size=400), 0.0, 120.0)
    )

    result = analyze_peak_trial_binned(
        times.tolist(), trial_duration=120.0, bin_width=2.0
    )
    assert isinstance(result, PeakTrialBinnedMetrics)
    assert result.peak_time == pytest.approx(30.0, abs=2.0)
    assert result.start_time < result.peak_time < result.stop_time
    assert result.middle_time == pytest.approx(30.0, abs=3.0)
    assert result.n_responses == len(times)


@pytest.mark.unit
def test_binned_rejects_non_positive_trial_duration() -> None:
    with pytest.raises(ValueError, match="trial_duration"):
        analyze_peak_trial_binned([1.0], trial_duration=0.0)


@pytest.mark.unit
def test_binned_rejects_non_positive_bin_width() -> None:
    with pytest.raises(ValueError, match="bin_width"):
        analyze_peak_trial_binned([1.0], trial_duration=10.0, bin_width=0.0)


@pytest.mark.unit
def test_binned_rejects_out_of_range_response() -> None:
    with pytest.raises(ValueError, match="within"):
        analyze_peak_trial_binned([150.0], trial_duration=60.0)


@pytest.mark.unit
def test_mle_returns_none_for_too_few_responses() -> None:
    assert analyze_peak_trial_mle([], trial_duration=60.0) is None
    assert analyze_peak_trial_mle([15.0], trial_duration=60.0) is None


@pytest.mark.unit
def test_mle_recovers_start_stop_from_three_rate_mixture() -> None:
    rng = np.random.default_rng(11)
    T = 120.0
    # Low rate 0.02/s on [0,20) and (60,120]; high rate 1.0/s on [20,60].
    low_1 = rng.uniform(0.0, 20.0, size=rng.poisson(0.02 * 20.0))
    mid = rng.uniform(20.0, 60.0, size=rng.poisson(1.0 * 40.0))
    low_2 = rng.uniform(60.0, 120.0, size=rng.poisson(0.02 * 60.0))
    times = np.sort(np.concatenate([low_1, mid, low_2]))

    result = analyze_peak_trial_mle(times.tolist(), trial_duration=T)
    assert isinstance(result, PeakTrialMLEMetrics)
    assert result.start_time == pytest.approx(20.0, abs=3.0)
    assert result.stop_time == pytest.approx(60.0, abs=3.0)
    assert result.high_rate > result.low_rate
    assert result.middle_time == pytest.approx(
        (result.start_time + result.stop_time) / 2.0
    )


@pytest.mark.unit
def test_mle_rejects_out_of_range_response() -> None:
    with pytest.raises(ValueError, match="within"):
        analyze_peak_trial_mle([-1.0, 5.0], trial_duration=10.0)
