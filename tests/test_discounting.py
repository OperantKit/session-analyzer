"""Tests for delay-discounting model fitting."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.discounting import (
    HyperbolicDiscountingParameters,
    HyperboloidDiscountingParameters,
    fit_hyperbolic_discounting,
    fit_hyperboloid_discounting,
)


@pytest.mark.unit
def test_recovers_known_hyperbolic_k() -> None:
    amount = 100.0
    k_true = 0.05
    delays = [0.0, 1.0, 2.0, 7.0, 30.0, 90.0, 180.0, 365.0]
    values = [amount / (1.0 + k_true * d) for d in delays]

    result = fit_hyperbolic_discounting(delays, values, amount=amount)

    assert isinstance(result, HyperbolicDiscountingParameters)
    assert result.k == pytest.approx(k_true, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.n == len(delays)


@pytest.mark.unit
def test_hyperbolic_fit_robust_to_noise() -> None:
    rng = np.random.default_rng(42)
    amount = 100.0
    k_true = 0.02
    delays = np.array([0.0, 1.0, 7.0, 30.0, 90.0, 180.0, 365.0])
    clean = amount / (1.0 + k_true * delays)
    noisy = clean + rng.normal(scale=1.5, size=clean.shape)

    result = fit_hyperbolic_discounting(delays.tolist(), noisy.tolist(), amount=amount)
    assert result.k == pytest.approx(k_true, rel=0.3)
    assert result.r_squared > 0.9


@pytest.mark.unit
def test_hyperbolic_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_hyperbolic_discounting([0.0, 1.0], [100.0, 90.0, 80.0], amount=100.0)


@pytest.mark.unit
def test_hyperbolic_raises_on_insufficient_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_hyperbolic_discounting([0.0], [100.0], amount=100.0)


@pytest.mark.unit
def test_hyperbolic_raises_on_non_positive_amount() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_hyperbolic_discounting([0.0, 1.0], [100.0, 90.0], amount=0.0)


@pytest.mark.unit
def test_recovers_known_hyperboloid_parameters() -> None:
    amount = 100.0
    k_true = 0.03
    s_true = 0.8
    delays = [0.1, 1.0, 7.0, 30.0, 90.0, 180.0, 365.0]
    values = [amount / (1.0 + k_true * d) ** s_true for d in delays]

    result = fit_hyperboloid_discounting(delays, values, amount=amount)

    assert isinstance(result, HyperboloidDiscountingParameters)
    assert result.k == pytest.approx(k_true, abs=1e-3)
    assert result.s == pytest.approx(s_true, abs=1e-3)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_hyperboloid_reduces_to_hyperbolic_when_s_is_one() -> None:
    amount = 50.0
    k_true = 0.1
    delays = [0.0, 2.0, 10.0, 50.0, 100.0]
    values = [amount / (1.0 + k_true * d) for d in delays]

    hyper = fit_hyperbolic_discounting(delays, values, amount=amount)
    mazur = fit_hyperboloid_discounting(delays, values, amount=amount)

    assert mazur.s == pytest.approx(1.0, abs=1e-2)
    assert mazur.k == pytest.approx(hyper.k, rel=1e-2)
