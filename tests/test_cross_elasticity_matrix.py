"""Tests for multi-good cross-elasticity matrix."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.cross_elasticity_matrix import (
    CrossElasticityMatrix,
    compute_cross_elasticity_matrix,
)
from session_analyzer.analytics.unit_price import (
    compute_cross_price_elasticity,
    fit_unit_price_elasticity,
)


@pytest.mark.unit
def test_diagonal_recovers_own_price_elasticity() -> None:
    # Q_A = 100 / P_A^0.5, Q_B = 50 / P_B^1.2, no substitution.
    prices = {
        "A": [1.0, 2.0, 4.0, 8.0, 16.0],
        "B": [1.0, 2.0, 4.0, 8.0, 16.0],
    }
    cons = {
        "A": [100.0 / (p**0.5) for p in prices["A"]],
        "B": [50.0 / (p**1.2) for p in prices["B"]],
    }
    result = compute_cross_elasticity_matrix(cons, prices)
    assert isinstance(result, CrossElasticityMatrix)
    a_idx = result.goods.index("A")
    b_idx = result.goods.index("B")
    assert result.matrix[a_idx][a_idx] == pytest.approx(-0.5, abs=1e-6)
    assert result.matrix[b_idx][b_idx] == pytest.approx(-1.2, abs=1e-6)


@pytest.mark.unit
def test_off_diagonal_positive_for_substitutes() -> None:
    # Substitutes: Q_A = 10 * P_B, Q_B = 10 * P_A. Cross-elasticities = +1.
    prices = {
        "A": [1.0, 2.0, 4.0, 8.0],
        "B": [1.0, 2.0, 4.0, 8.0],
    }
    cons = {
        "A": [10.0 * pb for pb in prices["B"]],
        "B": [10.0 * pa for pa in prices["A"]],
    }
    result = compute_cross_elasticity_matrix(cons, prices)
    a, b = result.goods.index("A"), result.goods.index("B")
    assert result.matrix[a][b] == pytest.approx(1.0, abs=1e-6)
    assert result.matrix[b][a] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_nan_when_fewer_than_two_valid_pairs() -> None:
    prices = {"A": [0.0, 0.0, 1.0], "B": [1.0, 2.0, 3.0]}
    cons = {"A": [10.0, 5.0, 1.0], "B": [10.0, 5.0, 1.0]}
    result = compute_cross_elasticity_matrix(cons, prices)
    a = result.goods.index("A")
    # epsilon_{A,A}: only one pair has P_A > 0 -> NaN.
    assert math.isnan(result.matrix[a][a])


@pytest.mark.unit
def test_raises_on_mismatched_keys() -> None:
    with pytest.raises(ValueError, match="same set of goods"):
        compute_cross_elasticity_matrix(
            {"A": [1.0, 2.0]}, {"A": [1.0, 2.0], "B": [1.0, 2.0]}
        )


@pytest.mark.unit
def test_raises_on_too_few_conditions() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compute_cross_elasticity_matrix({"A": [1.0]}, {"A": [1.0]})


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_cross_elasticity_matrix(
            {"A": [1.0, 2.0], "B": [1.0]},
            {"A": [1.0, 2.0], "B": [1.0, 2.0]},
        )


# ---------------------------------------------------------------------------
# Reference values (OLS and agreement with unit_price)
# ---------------------------------------------------------------------------
#
# Each entry is a simple (one-regressor) log-log OLS slope. With a full
# factorial price design the log prices are uncorrelated, so the simple slopes
# equal the exponents of a multiplicative demand system
# Q_i = c_i * P_A^e_iA * P_B^e_iB exactly. No published matrix was retrieved
# (Hursh & Roma, 2013 is paywalled).


@pytest.mark.unit
def test_factorial_design_recovers_partial_elasticities() -> None:
    levels = [1.0, 2.0, 4.0]
    p_a = [a for a in levels for _ in levels]
    p_b = [b for _ in levels for b in levels]
    q_a = [10.0 * a**-0.8 * b**0.3 for a, b in zip(p_a, p_b, strict=True)]
    q_b = [6.0 * a**-0.2 * b**-1.1 for a, b in zip(p_a, p_b, strict=True)]
    result = compute_cross_elasticity_matrix({"A": q_a, "B": q_b}, {"A": p_a, "B": p_b})
    expected = ((-0.8, 0.3), (-0.2, -1.1))
    for i in range(2):
        for j in range(2):
            assert result.matrix[i][j] == pytest.approx(expected[i][j], abs=1e-12)


@pytest.mark.unit
def test_entries_agree_with_unit_price_and_polyfit() -> None:
    p_a = [1.0, 2.0, 3.0, 5.0, 8.0]
    p_b = [4.0, 2.5, 6.0, 1.5, 3.0]
    q_a = [30.0, 18.0, 15.5, 8.0, 6.2]
    q_b = [9.0, 12.0, 7.5, 16.0, 11.0]
    result = compute_cross_elasticity_matrix({"A": q_a, "B": q_b}, {"A": p_a, "B": p_b})
    own_a = fit_unit_price_elasticity(unit_prices=p_a, consumption=q_a)
    cross_ab = compute_cross_price_elasticity(prices_other=p_b, consumption_target=q_a)
    assert result.matrix[0][0] == pytest.approx(own_a.elasticity, abs=1e-6)
    assert result.matrix[0][1] == pytest.approx(cross_ab.elasticity, abs=1e-6)
    slope_ba, _ = np.polyfit(np.log(p_a), np.log(q_b), 1)
    assert result.matrix[1][0] == pytest.approx(slope_ba, rel=1e-12)
