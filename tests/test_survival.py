"""Tests for Kaplan-Meier survival estimation."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.survival import SurvivalCurve, kaplan_meier


@pytest.mark.unit
def test_survival_steps_on_uncensored_events() -> None:
    # 4 subjects, all events at 1, 2, 3, 4. S(t) = 0.75, 0.5, 0.25, 0.
    result = kaplan_meier([1.0, 2.0, 3.0, 4.0])
    assert isinstance(result, SurvivalCurve)
    assert result.times == (1.0, 2.0, 3.0, 4.0)
    assert result.survival == pytest.approx((0.75, 0.5, 0.25, 0.0))
    assert result.at_risk == (4, 3, 2, 1)
    assert result.events == (1, 1, 1, 1)
    assert result.median_survival_time == 2.0


@pytest.mark.unit
def test_censored_observations_remove_from_risk_without_stepping() -> None:
    times = [1.0, 2.0, 3.0, 4.0]
    censored = [False, True, False, False]
    result = kaplan_meier(times, censored)
    # At t=1: S = 1 - 1/4 = 0.75. At t=2: censored, no step.
    # At t=3: at risk = 2, S = 0.75 * (1 - 1/2) = 0.375.
    # At t=4: at risk = 1, S = 0.375 * (1 - 1/1) = 0.
    assert result.times == (1.0, 3.0, 4.0)
    assert result.survival == pytest.approx((0.75, 0.375, 0.0))
    assert result.at_risk == (4, 2, 1)
    assert result.n_events == 3


@pytest.mark.unit
def test_median_none_when_curve_never_reaches_half() -> None:
    # All but one subject censored -> curve only drops once.
    times = [1.0, 2.0, 3.0, 4.0]
    censored = [False, True, True, True]
    result = kaplan_meier(times, censored)
    # S(1) = 0.75; no further events -> median is None.
    assert result.median_survival_time is None


@pytest.mark.unit
def test_tied_event_times_aggregate() -> None:
    result = kaplan_meier([2.0, 2.0, 5.0])
    # At t=2: 2 events / 3 at risk -> S = 1/3. At t=5: 1/1 -> 0.
    assert result.times == (2.0, 5.0)
    assert result.survival == pytest.approx((1.0 / 3.0, 0.0))
    assert result.events == (2, 1)


@pytest.mark.unit
def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        kaplan_meier([])


@pytest.mark.unit
def test_raises_on_non_positive_times() -> None:
    with pytest.raises(ValueError, match="positive"):
        kaplan_meier([0.0, 1.0])


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        kaplan_meier([1.0, 2.0], [False])
