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


# ---------------------------------------------------------------------------
# Least-squares reference (S-166). Guttman & Kalish (1956) and Shepard
# (1987) report no table this module reproduces (G&K's gradient counts are
# in an APA paywalled article), so the fits are checked against an
# independent least-squares solution: for a fixed width the best R0 is
# sum(r g) / sum(g^2), and the width is found by a bounded 1-D search on the
# profiled residual sum of squares.
# ---------------------------------------------------------------------------

_D = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
# Seeded noisy gradient: 100 * exp(-d^2 / (2 * 22^2)) + N(0, 4).
_R = 100.0 * np.exp(-(_D**2) / (2 * 22.0**2)) + np.random.default_rng(166).normal(0, 4, _D.size)
_R = np.clip(_R, 0.0, None)


def _profiled_fit(kernel) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    from scipy.optimize import minimize_scalar

    def rss(width: float) -> float:
        g = kernel(_D, width)
        r0 = float(np.dot(_R, g) / np.dot(g, g))
        return float(np.sum((_R - r0 * g) ** 2))

    res = minimize_scalar(rss, bounds=(1.0, 200.0), method="bounded", options={"xatol": 1e-10})
    g = kernel(_D, res.x)
    return float(np.dot(_R, g) / np.dot(g, g)), float(res.x)


@pytest.mark.unit
@pytest.mark.reference(
    module="generalization",
    verdict="not-retrieved",
    source="Guttman & Kalish (1956)",
)
def test_gaussian_fit_matches_profiled_least_squares() -> None:
    r0, sigma = _profiled_fit(lambda d, s: np.exp(-(d**2) / (2 * s**2)))
    fit = fit_gaussian_gradient(list(_D), list(_R))
    assert fit.sigma == pytest.approx(sigma, rel=1e-5)
    assert fit.peak_response == pytest.approx(r0, rel=1e-5)


@pytest.mark.unit
def test_exponential_fit_matches_profiled_least_squares() -> None:
    r0, lam = _profiled_fit(lambda d, s: np.exp(-d / s))
    fit = fit_exponential_gradient(list(_D), list(_R))
    assert fit.length_scale == pytest.approx(lam, rel=1e-5)
    assert fit.peak_response == pytest.approx(r0, rel=1e-5)
