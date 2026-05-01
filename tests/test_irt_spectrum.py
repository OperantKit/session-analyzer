"""Tests for IRT autocorrelation and spectrum analyses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.irt_spectrum import (
    IRTAutocorrelation,
    IRTSpectrum,
    compute_irt_autocorrelation,
    compute_irt_spectrum,
)


@pytest.mark.unit
def test_autocorrelation_at_lag_zero_is_one() -> None:
    rng = np.random.default_rng(1)
    irts = list(rng.gamma(2.0, 0.5, size=200))
    result = compute_irt_autocorrelation(irts, max_lag=10)
    assert isinstance(result, IRTAutocorrelation)
    assert result.values[0] == pytest.approx(1.0, abs=1e-9)
    assert result.lags == tuple(range(11))


@pytest.mark.unit
def test_autocorrelation_detects_alternating_series() -> None:
    # Period-2 alternation -> rho(1) ~ -1.
    series = [1.0, 2.0] * 100
    result = compute_irt_autocorrelation(series, max_lag=4)
    # Biased estimator on N=200 series gives rho(1) ~= -0.995, rho(2) ~= 0.99.
    assert result.values[1] == pytest.approx(-1.0, abs=0.02)
    assert result.values[2] == pytest.approx(1.0, abs=0.02)


@pytest.mark.unit
def test_autocorrelation_rejects_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        compute_irt_autocorrelation([1.0] * 20, max_lag=5)


@pytest.mark.unit
def test_autocorrelation_rejects_invalid_lag() -> None:
    with pytest.raises(ValueError, match="max_lag"):
        compute_irt_autocorrelation([1.0, 2.0, 3.0], max_lag=0)
    with pytest.raises(ValueError, match="max_lag"):
        compute_irt_autocorrelation([1.0, 2.0, 3.0], max_lag=10)


@pytest.mark.unit
def test_spectrum_recovers_dominant_frequency() -> None:
    # Sample a sinusoid at fs = 100 Hz, frequency 5 Hz.
    fs = 100.0
    n = 256
    t = np.arange(n) / fs
    series = np.sin(2 * math.pi * 5.0 * t)
    result = compute_irt_spectrum(series.tolist(), sampling_rate=fs)
    assert isinstance(result, IRTSpectrum)
    assert result.peak_frequency == pytest.approx(5.0, abs=0.5)


@pytest.mark.unit
def test_spectrum_default_sampling_rate_is_one_step() -> None:
    series = [1.0, -1.0] * 64  # period 2 in samples -> peak at 0.5 cyc/step
    result = compute_irt_spectrum(series)
    assert result.sampling_rate is None
    assert result.peak_frequency == pytest.approx(0.5, abs=1e-3)


@pytest.mark.unit
def test_spectrum_rejects_short_input_or_bad_fs() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        compute_irt_spectrum([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="sampling_rate"):
        compute_irt_spectrum([1.0, 2.0, 3.0, 4.0], sampling_rate=0.0)
