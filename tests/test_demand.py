"""Tests for exponential demand-curve fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.demand import (
    DemandParameters,
    fit_exponential_demand,
)


def _synth(prices: list[float], q0: float, alpha: float, k: float) -> list[float]:
    """Generate consumption via Hursh & Silberberg (2008):
    log10 Q = log10 Q0 + k * (exp(-alpha * Q0 * C) - 1).
    """
    log_q = np.log10(q0) + k * (np.exp(-alpha * q0 * np.array(prices)) - 1.0)
    return (10.0**log_q).tolist()


@pytest.mark.unit
def test_recovers_known_parameters_with_k_fixed() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    q0_true = 50.0
    alpha_true = 0.005
    k_fixed = 3.0

    consumption = _synth(prices, q0_true, alpha_true, k_fixed)
    result = fit_exponential_demand(prices, consumption, k=k_fixed)

    assert isinstance(result, DemandParameters)
    assert result.q0 == pytest.approx(q0_true, rel=1e-3)
    assert result.alpha == pytest.approx(alpha_true, rel=1e-3)
    assert result.k == pytest.approx(k_fixed, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_recovers_known_parameters_with_k_estimated() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    q0_true = 30.0
    alpha_true = 0.01
    k_true = 2.5

    consumption = _synth(prices, q0_true, alpha_true, k_true)
    result = fit_exponential_demand(prices, consumption, k=None)

    assert result.q0 == pytest.approx(q0_true, rel=1e-2)
    assert result.alpha == pytest.approx(alpha_true, rel=1e-2)
    assert result.k == pytest.approx(k_true, rel=1e-2)
    assert result.r_squared > 0.99


@pytest.mark.unit
def test_pmax_and_omax_are_consistent() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    q0 = 100.0
    alpha = 0.002
    k = 3.0
    consumption = _synth(prices, q0, alpha, k)
    result = fit_exponential_demand(prices, consumption, k=k)

    # P_max: where d(Q*P)/dP = 0. Closed form: solution of
    # alpha*Q0*P * ln(10)*k * exp(-alpha*Q0*P) = 1  (Hursh & Roma 2013).
    # Validate numerically: expenditure at P_max must exceed neighbours.
    p_max = result.p_max
    assert p_max > 0.0
    q_at_pmax = q0 * 10 ** (k * (math.exp(-alpha * q0 * p_max) - 1.0))
    o_at_pmax = p_max * q_at_pmax
    assert result.o_max == pytest.approx(o_at_pmax, rel=1e-3)

    # Expenditure is unimodal in P; values on both sides should be lower.
    for factor in (0.5, 2.0):
        p = p_max * factor
        q = q0 * 10 ** (k * (math.exp(-alpha * q0 * p) - 1.0))
        assert p * q < o_at_pmax + 1e-6


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_exponential_demand([1.0, 2.0], [10.0])


@pytest.mark.unit
def test_raises_on_insufficient_data() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_exponential_demand([1.0, 2.0], [10.0, 5.0])


@pytest.mark.unit
def test_raises_on_non_positive_prices_or_consumption() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_exponential_demand([0.0, 1.0, 2.0], [10.0, 5.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        fit_exponential_demand([1.0, 2.0, 3.0], [10.0, 0.0, 2.0])
