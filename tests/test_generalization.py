"""Tests for stimulus generalization gradient fits."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.generalization import (
    ExponentialGradientParameters,
    GaussianGradientParameters,
    fit_exponential_gradient,
    fit_gaussian_gradient,
)


@pytest.mark.unit
def test_gaussian_recovers_known_parameters() -> None:
    r0_true = 120.0
    sigma_true = 2.5
    distances = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    responses = [r0_true * np.exp(-(d**2) / (2 * sigma_true**2)) for d in distances]

    result = fit_gaussian_gradient(distances, responses)

    assert isinstance(result, GaussianGradientParameters)
    assert result.peak_response == pytest.approx(r0_true, rel=1e-4)
    assert result.sigma == pytest.approx(sigma_true, rel=1e-4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.n == 7


@pytest.mark.unit
def test_exponential_recovers_known_parameters() -> None:
    r0_true = 80.0
    lam_true = 4.0
    distances = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    responses = [r0_true * np.exp(-d / lam_true) for d in distances]

    result = fit_exponential_gradient(distances, responses)

    assert isinstance(result, ExponentialGradientParameters)
    assert result.peak_response == pytest.approx(r0_true, rel=1e-4)
    assert result.length_scale == pytest.approx(lam_true, rel=1e-4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_gaussian_robust_to_noise() -> None:
    rng = np.random.default_rng(3)
    r0_true = 100.0
    sigma_true = 3.0
    distances = np.linspace(0.0, 12.0, 13)
    clean = r0_true * np.exp(-(distances**2) / (2 * sigma_true**2))
    noisy = np.clip(clean + rng.normal(scale=2.0, size=clean.shape), 0.0, None)

    result = fit_gaussian_gradient(distances.tolist(), noisy.tolist())
    assert result.sigma == pytest.approx(sigma_true, rel=0.2)
    assert result.r_squared > 0.9


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_gaussian_gradient([0.0, 1.0], [10.0, 5.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        fit_exponential_gradient([0.0, 1.0], [10.0])


@pytest.mark.unit
def test_raises_on_insufficient_data() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_gaussian_gradient([0.0, 1.0], [10.0, 5.0])
    with pytest.raises(ValueError, match="at least 3"):
        fit_exponential_gradient([0.0, 1.0], [10.0, 5.0])


@pytest.mark.unit
def test_raises_on_negative_distance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fit_gaussian_gradient([-1.0, 0.0, 1.0], [1.0, 2.0, 1.0])


@pytest.mark.unit
def test_raises_on_negative_response() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fit_exponential_gradient([0.0, 1.0, 2.0], [10.0, -1.0, 3.0])
