"""Tests for unit-price and cross-price elasticity analyses."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.unit_price import (
    CrossPriceElasticity,
    UnitPriceElasticity,
    compute_cross_price_elasticity,
    compute_unit_prices,
    fit_unit_price_elasticity,
)


@pytest.mark.unit
def test_compute_unit_prices_basic() -> None:
    prices = [10.0, 20.0, 30.0]
    doses = [1.0, 2.0, 1.5]
    up = compute_unit_prices(prices, doses)
    assert up == pytest.approx([10.0, 10.0, 20.0])


@pytest.mark.unit
def test_compute_unit_prices_validates() -> None:
    with pytest.raises(ValueError):
        compute_unit_prices([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        compute_unit_prices([10.0, 20.0], [1.0, 0.0])


@pytest.mark.unit
def test_unit_price_elasticity_log_log_slope() -> None:
    """Q = k * P^alpha; recover alpha = -1.2 exactly from clean data."""
    alpha = -1.2
    prices = [1.0, 2.0, 5.0, 10.0, 20.0]
    consumption = [100.0 * p**alpha for p in prices]
    fit = fit_unit_price_elasticity(unit_prices=prices, consumption=consumption)
    assert isinstance(fit, UnitPriceElasticity)
    assert fit.elasticity == pytest.approx(-1.2, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
    assert fit.n == 5


@pytest.mark.unit
def test_unit_price_elasticity_inelastic_demand() -> None:
    # |alpha| < 1 -> inelastic
    alpha = -0.4
    prices = [1.0, 2.0, 4.0, 8.0]
    consumption = [50.0 * p**alpha for p in prices]
    fit = fit_unit_price_elasticity(unit_prices=prices, consumption=consumption)
    assert fit.elasticity == pytest.approx(-0.4, abs=1e-6)
    assert fit.is_inelastic
    assert fit.is_elastic is False


@pytest.mark.unit
def test_cross_price_elasticity_substitutes() -> None:
    # Q_A increases with P_B -> positive cross-elasticity (substitutes)
    p_b = [1.0, 2.0, 4.0, 8.0]
    q_a = [10.0 * pb**0.6 for pb in p_b]
    fit = compute_cross_price_elasticity(prices_other=p_b, consumption_target=q_a)
    assert isinstance(fit, CrossPriceElasticity)
    assert fit.elasticity == pytest.approx(0.6, abs=1e-6)
    assert fit.relationship == "substitutes"


@pytest.mark.unit
def test_cross_price_elasticity_complements() -> None:
    p_b = [1.0, 2.0, 4.0, 8.0]
    q_a = [10.0 * pb**(-0.5) for pb in p_b]
    fit = compute_cross_price_elasticity(prices_other=p_b, consumption_target=q_a)
    assert fit.elasticity == pytest.approx(-0.5, abs=1e-6)
    assert fit.relationship == "complements"


@pytest.mark.unit
def test_cross_price_elasticity_independent() -> None:
    p_b = [1.0, 2.0, 4.0, 8.0]
    q_a = [10.0] * 4
    fit = compute_cross_price_elasticity(prices_other=p_b, consumption_target=q_a)
    assert fit.elasticity == pytest.approx(0.0, abs=1e-6)
    assert fit.relationship == "independent"


@pytest.mark.unit
def test_unit_price_elasticity_filters_non_positive() -> None:
    fit = fit_unit_price_elasticity(
        unit_prices=[1.0, 2.0, 0.0, 4.0],
        consumption=[100.0, 50.0, 999.0, 25.0],
    )
    assert fit.n == 3


@pytest.mark.unit
def test_unit_price_elasticity_validates_lengths() -> None:
    with pytest.raises(ValueError):
        fit_unit_price_elasticity(unit_prices=[1.0], consumption=[1.0, 2.0])
    with pytest.raises(ValueError):
        fit_unit_price_elasticity(unit_prices=[1.0], consumption=[1.0])
