"""Tests for Killeen (1994) MPR fitting."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.killeen_law import (
    KilleenLawParameters,
    fit_killeen_law,
)


def _killeen(r: float, k: float, a: float, delta: float) -> float:
    return (k * a * r) / (a * r * delta + 1.0)


@pytest.mark.unit
def test_recovers_known_parameters_with_delta_fixed() -> None:
    k, a, delta = 2.0, 0.5, 0.02
    rs = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    bs = [_killeen(r, k, a, delta) for r in rs]

    result = fit_killeen_law(rs, bs, delta=delta)

    assert isinstance(result, KilleenLawParameters)
    assert result.k == pytest.approx(k, rel=1e-3)
    assert result.a == pytest.approx(a, rel=1e-3)
    assert result.delta == pytest.approx(delta, abs=1e-12)
    assert result.r_squared > 0.999
    assert result.n == 7


@pytest.mark.unit
def test_predicts_monotonically_increasing_toward_asymptote() -> None:
    k, a, delta = 2.0, 1.0, 0.01
    asymptote = k / delta  # 200
    rs = [1.0, 5.0, 20.0, 100.0, 500.0]
    bs = [_killeen(r, k, a, delta) for r in rs]
    assert all(bs[i] < bs[i + 1] for i in range(len(bs) - 1))
    assert bs[-1] < asymptote


@pytest.mark.unit
def test_raises_on_non_positive_delta() -> None:
    with pytest.raises(ValueError, match="delta"):
        fit_killeen_law([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], delta=0.0)


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_killeen_law([1.0, 2.0], [10.0], delta=0.05)


@pytest.mark.unit
def test_raises_on_insufficient_data() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_killeen_law([1.0], [10.0], delta=0.05)
