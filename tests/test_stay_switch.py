"""Tests for stay-switch and COD-compliance analyses."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.stay_switch import (
    CODComplianceResult,
    StaySwitchMetrics,
    analyze_stay_switch,
    check_cod_compliance,
)


@pytest.mark.unit
def test_analyze_returns_none_for_lt_two_responses() -> None:
    assert analyze_stay_switch([], []) is None
    assert analyze_stay_switch(["L"], [0.0]) is None


@pytest.mark.unit
def test_analyze_counts_changeovers_and_stays() -> None:
    sides = ["L", "L", "L", "R", "R", "L", "R", "R"]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    result = analyze_stay_switch(sides, times)
    assert isinstance(result, StaySwitchMetrics)
    # Bouts: LLL, RR, L, RR -> 4 stays, 3 changeovers.
    assert result.n_changeovers == 3
    assert result.n_stays == 4
    assert result.mean_stay_responses == pytest.approx(8 / 4)


@pytest.mark.unit
def test_analyze_changeover_rate_per_minute() -> None:
    sides = ["L", "R", "L", "R"]
    times = [0.0, 30.0, 60.0, 120.0]
    result = analyze_stay_switch(sides, times)
    assert result is not None
    # 3 changeovers across 120 s -> 1.5 / min.
    assert result.changeover_rate_per_min == pytest.approx(1.5)


@pytest.mark.unit
def test_analyze_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        analyze_stay_switch(["L"], [0.0, 1.0])


@pytest.mark.unit
def test_analyze_raises_on_unsorted_times() -> None:
    with pytest.raises(ValueError, match="sorted"):
        analyze_stay_switch(["L", "R"], [5.0, 1.0])


@pytest.mark.unit
def test_cod_compliance_counts_reinforcers_after_cod_window() -> None:
    # Changeover at t=10 and t=30. COD = 5 s.
    # Reinforcers at t=12 (2 s after CO -> non-compliant), t=20 (10 s after
    # CO -> compliant), t=33 (3 s after CO -> non-compliant).
    sides = ["L", "L", "R", "R", "R", "L", "L"]
    times = [0.0, 5.0, 10.0, 15.0, 25.0, 30.0, 35.0]
    reinforcers = [12.0, 20.0, 33.0]
    result = check_cod_compliance(sides, times, reinforcers, cod_sec=5.0)
    assert isinstance(result, CODComplianceResult)
    assert result.n_reinforcers == 3
    assert result.n_compliant == 1
    assert result.compliance_rate == pytest.approx(1 / 3)


@pytest.mark.unit
def test_cod_compliance_treats_pre_changeover_reinforcers_as_compliant() -> None:
    sides = ["L", "L", "R"]
    times = [0.0, 1.0, 10.0]
    # Reinforcer at t=5 — before any changeover.
    result = check_cod_compliance(sides, times, [5.0], cod_sec=2.0)
    assert result.n_compliant == 1


@pytest.mark.unit
def test_cod_compliance_zero_reinforcers_is_full_compliance() -> None:
    result = check_cod_compliance(["L", "R"], [0.0, 1.0], [], cod_sec=2.0)
    assert result.compliance_rate == 1.0
    assert result.n_reinforcers == 0


@pytest.mark.unit
def test_cod_compliance_raises_on_invalid_cod() -> None:
    with pytest.raises(ValueError, match="cod_sec"):
        check_cod_compliance(["L", "R"], [0.0, 1.0], [2.0], cod_sec=0.0)
