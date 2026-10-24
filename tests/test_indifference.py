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


# ---------------------------------------------------------------------------
# Hand computations
# ---------------------------------------------------------------------------
#
# Richards et al. (1997) take the indifference point as the mean adjusting
# amount over stable trials; the window length and tolerance are parameters of
# this module. A titration against a fixed delayed value V = A / (1 + kD)
# oscillates within one step of V, so its trailing mean lies within half a step.


@pytest.mark.unit
@pytest.mark.reference(
    module="indifference",
    verdict="properties",
    source="Mazur (1987)",
)
def test_window_mean_and_range_by_hand() -> None:
    result = estimate_indifference_point(
        adjusting_amounts=[2.0, 4.0, 10.0, 12.0, 11.0, 9.0, 10.0, 12.0],
        stable_window=6,
        stability_tolerance=3.0,
    )
    assert result.indifference_value == pytest.approx(64.0 / 6.0, abs=1e-6)
    assert result.range_in_window == 3.0
    assert result.stable  # range equal to the tolerance counts as stable
    unstable = estimate_indifference_point(
        adjusting_amounts=[2.0, 4.0, 10.0, 12.0, 11.0, 9.0, 10.0, 12.0],
        stable_window=6,
        stability_tolerance=2.999,
    )
    assert not unstable.stable


@pytest.mark.unit
def test_titration_converges_to_hyperbolic_value() -> None:
    delayed_value = 10.0 / (1.0 + 0.25 * 4.0)  # = 5.0
    step = 0.5
    amount = 1.0
    amounts = []
    for _ in range(60):
        amounts.append(amount)
        amount += -step if amount >= delayed_value else step
    result = estimate_indifference_point(adjusting_amounts=amounts, stable_window=10)
    assert abs(result.indifference_value - delayed_value) <= step / 2
    assert result.stable
