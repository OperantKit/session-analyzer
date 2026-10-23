"""Tests for unit-price and cross-price elasticity analyses."""

from __future__ import annotations

import math

import numpy as np
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


# ---------------------------------------------------------------------------
# Reference values (closed forms and OLS)
# ---------------------------------------------------------------------------
#
# Hursh (1980, 1991) define unit price as cost per unit of reinforcer; the
# module docstring's own example (FR50 / 100 uL and FR10 / 20 uL share a unit
# price of 0.5) is checked here. No published table of fitted elasticities was
# retrieved (Hursh & Roma, 2013 is paywalled), so the slopes are checked against
# a known power law and against numpy.polyfit.


@pytest.mark.unit
def test_unit_price_equivalence_of_docstring_example() -> None:
    assert compute_unit_prices([50.0, 10.0], [100.0, 20.0]) == [0.5, 0.5]


@pytest.mark.unit
def test_own_price_elasticity_recovers_power_law_exponent() -> None:
    prices = [1.0, 2.0, 4.0, 8.0, 16.0]
    consumption = [100.0 * p**-1.3 for p in prices]
    fit = fit_unit_price_elasticity(unit_prices=prices, consumption=consumption)
    assert fit.elasticity == pytest.approx(-1.3, abs=1e-6)
    assert fit.intercept == pytest.approx(math.log(100.0), abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
    assert fit.is_elastic and not fit.is_inelastic


@pytest.mark.unit
def test_elasticities_agree_with_numpy_polyfit_on_noisy_data() -> None:
    prices = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    consumption = [52.0, 40.0, 33.0, 21.0, 14.0, 8.5]
    slope, intercept = np.polyfit(np.log(prices), np.log(consumption), 1)
    own = fit_unit_price_elasticity(unit_prices=prices, consumption=consumption)
    cross = compute_cross_price_elasticity(
        prices_other=prices, consumption_target=consumption
    )
    assert own.elasticity == pytest.approx(slope, abs=1e-6)
    assert own.intercept == pytest.approx(intercept, abs=1e-6)
    assert cross.elasticity == own.elasticity


@pytest.mark.unit
def test_cross_price_elasticity_recovers_power_law_exponent() -> None:
    prices_b = [1.0, 2.0, 4.0, 8.0]
    q_a = [5.0 * p**0.4 for p in prices_b]
    fit = compute_cross_price_elasticity(prices_other=prices_b, consumption_target=q_a)
    assert fit.elasticity == pytest.approx(0.4, abs=1e-6)
    assert fit.relationship == "substitutes"
