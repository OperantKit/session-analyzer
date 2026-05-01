"""Tests for multi-good cross-elasticity matrix."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.cross_elasticity_matrix import (
    CrossElasticityMatrix,
    compute_cross_elasticity_matrix,
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
