"""Tests for adjusting-amount indifference-point estimation."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.indifference import (
    IndifferencePoint,
    estimate_indifference_point,
)


@pytest.mark.unit
def test_indifference_point_stable_oscillation() -> None:
    """Adjusting amount oscillates ±1 around 50.0 in last 6 trials -> stable."""
    amounts = [100.0, 80.0, 60.0, 50.0, 51.0, 49.0, 50.0, 51.0, 49.0, 50.0]
    result = estimate_indifference_point(
        adjusting_amounts=amounts,
        stable_window=6,
        stability_tolerance=2.0,
    )
    assert isinstance(result, IndifferencePoint)
    assert result.indifference_value == pytest.approx(50.0, abs=1.0)
    assert result.stable is True
    assert result.range_in_window == pytest.approx(2.0)


@pytest.mark.unit
def test_indifference_point_unstable_drifting() -> None:
    """Drifting amounts -> unstable."""
    amounts = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    result = estimate_indifference_point(
        adjusting_amounts=[float(x) for x in amounts],
        stable_window=5,
        stability_tolerance=2.0,
    )
    assert result.stable is False
    assert result.range_in_window > 2.0


@pytest.mark.unit
def test_indifference_point_default_window() -> None:
    """Default stable_window=6 with enough data."""
    amounts = [50.0] * 10
    result = estimate_indifference_point(adjusting_amounts=amounts)
    assert result.indifference_value == pytest.approx(50.0)
    assert result.stable is True


@pytest.mark.unit
def test_indifference_point_validates_window() -> None:
    with pytest.raises(ValueError):
        estimate_indifference_point(adjusting_amounts=[1.0, 2.0], stable_window=5)
    with pytest.raises(ValueError):
        estimate_indifference_point(adjusting_amounts=[1.0, 2.0, 3.0], stable_window=0)


@pytest.mark.unit
def test_indifference_point_n_trials() -> None:
    amounts = [100.0, 50.0, 50.0, 50.0]
    result = estimate_indifference_point(adjusting_amounts=amounts, stable_window=3)
    assert result.n_trials == 4
    assert result.indifference_value == pytest.approx(50.0)
