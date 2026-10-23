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


# ---------------------------------------------------------------------------
# Reference implementation (S-166).
#
# Data: the 6-MP arm of the leukaemia remission trial of Freireich et al.
# (1963), as tabulated by Gehan, E. A. (1965). A generalized Wilcoxon test
# for comparing arbitrarily singly-censored samples. Biometrika, 52(1/2),
# 203-224. https://doi.org/10.1093/biomet/52.1-2.203 ('+' = censored).
# Expected values: lifelines 0.30.3 ``KaplanMeierFitter`` on the same data
# (event table and ``median_survival_time_``). A censoring tied with an
# event (6 and 6+) counts in the risk set at that time, as in Kaplan &
# Meier (1958).
# ---------------------------------------------------------------------------

_GEHAN_6MP = "6 6 6 6+ 7 9+ 10 10+ 11+ 13 16 17+ 19+ 20+ 22 23 25+ 32+ 32+ 34+ 35+"


@pytest.mark.unit
def test_matches_lifelines_on_freireich_6mp_arm() -> None:
    raw = _GEHAN_6MP.split()
    times = [float(x.rstrip("+")) for x in raw]
    censored = [x.endswith("+") for x in raw]
    curve = kaplan_meier(times, censored)
    assert curve.times == (6.0, 7.0, 10.0, 13.0, 16.0, 22.0, 23.0)
    assert curve.at_risk == (21, 17, 15, 12, 11, 7, 6)
    assert curve.events == (3, 1, 1, 1, 1, 1, 1)
    assert curve.survival == pytest.approx(
        (
            0.857142857143,
            0.806722689076,
            0.752941176471,
            0.690196078431,
            0.627450980392,
            0.537815126050,
            0.448179271709,
        ),
        abs=5e-13,
    )
    assert curve.median_survival_time == 23.0
    assert curve.n == 21
    assert curve.n_events == 9
