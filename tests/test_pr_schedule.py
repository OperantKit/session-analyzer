"""Tests for progressive-ratio schedule classification."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.pr_schedule import (
    PRClassification,
    classify_pr_schedule,
    extract_step_ratios,
)


# ---------------------------------------------------------------------------
# extract_step_ratios
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_step_ratios_empty_reinforcements() -> None:
    assert extract_step_ratios([1.0, 2.0, 3.0], []) == []


@pytest.mark.unit
def test_extract_step_ratios_counts_responses_per_reinforcer() -> None:
    # 3 responses (at t=1,2,3), reinforcer at t=3 ⇒ first step = 3.
    # 5 more responses (at t=4..8), reinforcer at t=8 ⇒ second step = 5.
    response_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    reinforcement_times = [3.0, 8.0]
    assert extract_step_ratios(response_times, reinforcement_times) == [3, 5]


@pytest.mark.unit
def test_extract_step_ratios_rf_at_response_time_inclusive() -> None:
    # A response emitted exactly at the reinforcement time is treated as
    # completing this step (bisect_right semantics).
    response_times = [1.0, 2.0]
    reinforcement_times = [2.0]
    assert extract_step_ratios(response_times, reinforcement_times) == [2]


@pytest.mark.unit
def test_extract_step_ratios_no_responses() -> None:
    assert extract_step_ratios([], [1.0, 2.0]) == [0, 0]


# ---------------------------------------------------------------------------
# classify_pr_schedule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_empty_returns_none_step_type() -> None:
    result = classify_pr_schedule([])
    assert isinstance(result, PRClassification)
    assert result.is_pr is False
    assert result.step_type == "none"
    assert result.n_completed == 0
    assert result.last_completed_ratio == 0
    assert result.current_requirement == 1


@pytest.mark.unit
def test_classify_too_few_steps_is_unknown() -> None:
    # 2 steps < min 3 ⇒ unknown.
    result = classify_pr_schedule([1, 2])
    assert result.is_pr is False
    assert result.step_type == "unknown"
    assert result.last_completed_ratio == 2
    assert result.n_completed == 2


@pytest.mark.unit
def test_classify_non_monotone_is_not_pr() -> None:
    # 1, 2, 1 is not monotone increasing.
    result = classify_pr_schedule([1, 2, 1])
    assert result.is_pr is False
    assert result.step_type == "none"
    assert result.last_completed_ratio == 1


@pytest.mark.unit
def test_classify_linear_arithmetic_progression() -> None:
    # 5, 10, 15, 20 ⇒ linear (start=5, inc=5).
    result = classify_pr_schedule([5, 10, 15, 20])
    assert result.is_pr is True
    assert result.step_type == "linear"
    # Next step extrapolates start + inc * n = 5 + 5*4 = 25.
    assert result.current_requirement == 25


@pytest.mark.unit
def test_classify_hodos_within_tolerance() -> None:
    # The Hodos formula: max(1, round(5 * exp(0.2 * n) - 5))
    # n = 1..6 ⇒ 1, 2, 4, 6, 9, 12
    hodos = [max(1, round(5 * math.exp(0.2 * n) - 5)) for n in range(1, 7)]
    result = classify_pr_schedule(hodos)
    assert result.is_pr is True
    assert result.step_type == "hodos"
    # Next step: n=7
    expected_next = max(1, round(5 * math.exp(0.2 * 7) - 5))
    assert result.current_requirement == expected_next


@pytest.mark.unit
def test_classify_unknown_monotone_not_linear_or_hodos() -> None:
    # Strictly increasing but non-arithmetic, non-Hodos.
    result = classify_pr_schedule([1, 7, 50, 200])
    assert result.is_pr is True
    assert result.step_type == "unknown"
    # Next requirement = last + (last - second_to_last) = 200 + 150 = 350.
    assert result.current_requirement == 350
