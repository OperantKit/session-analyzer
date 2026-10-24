"""Tests for Schneider (1969) two-state break-and-run segmentation."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.schneider_segment import (
    SchneiderTwoStateResult,
    schneider_two_state,
)


@pytest.mark.unit
def test_recovers_break_time_for_clear_two_state_pattern() -> None:
    # FI 60 s. Pause: 0-30 s (no responses). Run: 30-60 s, response every 1 s.
    response_times = [30.0 + i for i in range(30)]
    result = schneider_two_state(
        response_times, interval_start=0.0, interval_end=60.0
    )
    assert isinstance(result, SchneiderTwoStateResult)
    # Break should land at ~30 s (the first response is at t=30).
    assert result.break_time == pytest.approx(31.0, abs=2.0)
    assert result.high_rate > result.low_rate
    assert result.n_low + result.n_high == len(response_times)


@pytest.mark.unit
def test_returns_none_when_fewer_than_2_per_segment() -> None:
    # Only 3 responses with default min=2 -> need >=4 -> None.
    result = schneider_two_state(
        [10.0, 20.0, 30.0], interval_start=0.0, interval_end=60.0
    )
    assert result is None


@pytest.mark.unit
def test_constant_rate_gives_equal_low_and_high_rates() -> None:
    # One response per second throughout: both fitted lines have slope 1,
    # wherever the breakpoint falls.
    result = schneider_two_state(
        [float(i) for i in range(1, 61)],
        interval_start=0.0,
        interval_end=60.0,
    )
    assert result is not None
    assert result.low_rate == pytest.approx(1.0, rel=0.05)
    assert result.high_rate == pytest.approx(1.0, rel=0.05)
    assert result.slope_difference == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_switch_at_two_thirds_is_found_by_least_squares() -> None:
    # 0.1 resp/s for 0-40 s, then 1 resp/s for 40-60 s: the breakpoint is
    # at 2/3 of the interval, not at a short high-rate pair near the end.
    low = [10.0 * i for i in range(1, 5)]
    high = [40.0 + i for i in range(1, 21)]
    result = schneider_two_state(
        low + high, interval_start=0.0, interval_end=60.0
    )
    assert result is not None
    assert result.break_time == pytest.approx(40.0, abs=1.5)
    assert result.low_rate == pytest.approx(0.1, abs=0.03)
    assert result.high_rate == pytest.approx(1.0, rel=0.1)
    # The response at 40 s lies on the switch itself and may go either way.
    assert result.n_low in (3, 4)
    assert result.n_low + result.n_high == 24


@pytest.mark.unit
def test_break_time_is_on_the_clock_of_the_responses() -> None:
    # The same interval shifted by 100 s moves break_time by 100 s.
    low = [10.0 * i for i in range(1, 5)]
    high = [40.0 + i for i in range(1, 21)]
    base = schneider_two_state(low + high, interval_start=0.0, interval_end=60.0)
    shifted = schneider_two_state(
        [t + 100.0 for t in low + high], interval_start=100.0, interval_end=160.0
    )
    assert base is not None and shifted is not None
    assert shifted.break_time == pytest.approx(base.break_time + 100.0)
    assert shifted.pause_duration == pytest.approx(base.pause_duration)


@pytest.mark.unit
def test_state_one_may_hold_no_responses() -> None:
    # Schneider (1969): state 1 has "little or no responding". With no
    # response before 30 s, every breakpoint in [29, 30] fits the counts
    # exactly (low rate 0 to 1/30); the fit must land there with a
    # non-negative low rate, not somewhere earlier with a negative one.
    result = schneider_two_state(
        [30.0 + i for i in range(30)], interval_start=0.0, interval_end=60.0
    )
    assert result is not None
    assert 29.0 - 1e-9 <= result.break_time <= 30.0 + 1e-9
    assert 0.0 <= result.low_rate <= 1.0 / 30.0 + 1e-9
    assert result.high_rate == pytest.approx(1.0)
    assert result.n_low <= 1


@pytest.mark.unit
def test_raises_on_invalid_interval_bounds() -> None:
    with pytest.raises(ValueError, match="interval_end"):
        schneider_two_state([1.0], interval_start=10.0, interval_end=10.0)


@pytest.mark.unit
def test_raises_on_unsorted_responses() -> None:
    with pytest.raises(ValueError, match="sorted"):
        schneider_two_state(
            [5.0, 1.0, 3.0], interval_start=0.0, interval_end=10.0
        )


@pytest.mark.unit
def test_raises_on_responses_outside_interval() -> None:
    with pytest.raises(ValueError, match="within"):
        schneider_two_state(
            [-1.0, 5.0], interval_start=0.0, interval_end=10.0
        )
    with pytest.raises(ValueError, match="within"):
        schneider_two_state(
            [5.0, 11.0], interval_start=0.0, interval_end=10.0
        )


@pytest.mark.unit
def test_raises_on_invalid_grid() -> None:
    with pytest.raises(ValueError, match="grid"):
        schneider_two_state(
            [1.0, 2.0, 3.0, 4.0], interval_start=0.0, interval_end=10.0, grid=1
        )


@pytest.mark.unit
def test_raises_on_invalid_min_responses() -> None:
    with pytest.raises(ValueError, match="min_responses_per_segment"):
        schneider_two_state(
            [1.0, 2.0],
            interval_start=0.0,
            interval_end=10.0,
            min_responses_per_segment=0,
        )


# ---------------------------------------------------------------------------
# Recovery of a known two-line cumulative record (property check)
# ---------------------------------------------------------------------------
#
# Published values: Schneider (1969) reports breakpoints for FI 16-512 s,
# on average about two-thirds of the interval (abstract), but the per-FI
# table is only in the scanned PMC PDF (PMC1338670), which was not
# retrieved. Instead the cumulative record below lies exactly on two lines
# (0.1 resp/s up to 30 s, then 2 resp/s), so the least-squares breakpoint is
# 30 s with zero residual and is unique on the 0.5-s grid.


@pytest.mark.unit
@pytest.mark.reference(
    module="schneider_segment",
    verdict="not-retrieved",
    source="Schneider (1969)",
)
def test_recovers_known_breakpoint_and_rates_from_exact_two_line_record() -> None:
    low = [10.0, 20.0, 30.0]  # count k at t = k / 0.1
    high = [30.0 + j / 2.0 for j in range(1, 101)]  # count 3 + j at 30 + j / 2
    result = schneider_two_state(
        low + high, interval_start=0.0, interval_end=100.0, grid=200
    )
    assert result is not None
    assert result.break_time == pytest.approx(30.0, abs=1e-9)
    assert result.low_rate == pytest.approx(0.1, rel=1e-9)
    assert result.high_rate == pytest.approx(2.0, rel=1e-9)
    assert result.n_low == 3
    assert result.n_high == 100
