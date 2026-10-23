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


def _pr_with_timeout_presses(
    requirements: list[int], *, stray_delay: float = 0.5
) -> tuple[list[float], list[float]]:
    """Synthesize a PR session with one stray press shortly after each reinforcer.

    Each step's ``n`` counted presses are spaced 1 s apart and the
    reinforcer coincides with the last one. After every reinforcer, one
    extra press occurs ``stray_delay`` s later (inside a post-reinforcement
    timeout, so the programme does not count it toward the next ratio).
    """
    responses: list[float] = []
    reinforcers: list[float] = []
    t = 0.0
    for n in requirements:
        if reinforcers:
            responses.append(reinforcers[-1] + stray_delay)
            t = reinforcers[-1] + 10.0
        for _ in range(n):
            t += 1.0
            responses.append(t)
        reinforcers.append(t)
    return responses, reinforcers


@pytest.mark.unit
def test_extract_step_ratios_default_counts_timeout_presses() -> None:
    responses, reinforcers = _pr_with_timeout_presses([1, 4, 7, 10, 13])
    steps = extract_step_ratios(responses, reinforcers)
    assert steps == [1, 5, 8, 11, 14]
    assert classify_pr_schedule(steps).step_type == "unknown"


@pytest.mark.unit
def test_extract_step_ratios_timeout_excludes_post_reinforcer_presses() -> None:
    responses, reinforcers = _pr_with_timeout_presses([1, 4, 7, 10, 13])
    steps = extract_step_ratios(responses, reinforcers, timeout_sec=5.0)
    assert steps == [1, 4, 7, 10, 13]
    result = classify_pr_schedule(steps)
    assert result.step_type == "linear"
    assert result.current_requirement == 16


@pytest.mark.unit
def test_extract_step_ratios_timeout_keeps_press_at_reinforcer_time() -> None:
    # The press that coincides with the reinforcer still completes that step.
    responses = [1.0, 2.0, 3.0, 10.0]
    assert extract_step_ratios(responses, [2.0, 10.0], timeout_sec=5.0) == [2, 1]


@pytest.mark.unit
def test_extract_step_ratios_timeout_boundary_is_open() -> None:
    # A press exactly timeout_sec after the reinforcer counts again.
    assert extract_step_ratios([1.0, 6.0, 7.0], [1.0, 7.0], timeout_sec=5.0) == [
        1,
        2,
    ]


@pytest.mark.unit
def test_extract_step_ratios_reinforcer_inside_previous_timeout() -> None:
    # Second reinforcer falls inside the first timeout: zero counted
    # responses, and later steps are not double-counted.
    responses = [1.0, 2.0, 3.0, 8.0, 9.0]
    assert extract_step_ratios(responses, [1.0, 3.0, 9.0], timeout_sec=5.0) == [
        1,
        0,
        2,
    ]


@pytest.mark.unit
def test_extract_step_ratios_negative_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_sec"):
        extract_step_ratios([1.0], [1.0], timeout_sec=-1.0)


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


# ---------------------------------------------------------------------------
# Published ratio series (Richardson & Roberts, 1996)
# ---------------------------------------------------------------------------
#
# Richardson & Roberts (1996) define the exponential PR series
# ratio(n) = round(5 * exp(0.2 n) - 5). The article body is paywalled and was
# not retrieved. Corroboration:
# - Steps 1-10 (1, 2, 4, 6, 9, 12, 15, 20, 25, 32): quoted verbatim, citing
#   Richardson & Roberts (1996), by the ConductScience "Progressive Ratio"
#   protocol page (https://conductscience.com/conductmaze/protocols/progressive-ratio).
# - Steps 11-20 (40, 50, 62, 77, 95, 118, 145, 178, 219, 268): the widely
#   reproduced continuation; not checked against a retrieved source.

_RR1996_FIRST_10 = [1, 2, 4, 6, 9, 12, 15, 20, 25, 32]
_RR1996_11_TO_20 = [40, 50, 62, 77, 95, 118, 145, 178, 219, 268]


@pytest.mark.unit
def test_hodos_series_matches_richardson_roberts_first_10_steps() -> None:
    from session_analyzer.analytics.pr_schedule import _hodos_requirement

    assert [_hodos_requirement(n) for n in range(1, 11)] == _RR1996_FIRST_10


@pytest.mark.unit
def test_hodos_series_matches_richardson_roberts_steps_11_to_20() -> None:
    from session_analyzer.analytics.pr_schedule import _hodos_requirement

    assert [_hodos_requirement(n) for n in range(11, 21)] == _RR1996_11_TO_20


@pytest.mark.unit
def test_classify_full_richardson_roberts_series_extrapolates_step_21() -> None:
    """Twenty completed steps classify as the exponential series; the
    requirement in progress is step 21 = round(5 e^4.2 - 5) = 328."""
    result = classify_pr_schedule(_RR1996_FIRST_10 + _RR1996_11_TO_20)
    assert result.is_pr is True
    assert result.step_type == "hodos"
    assert result.last_completed_ratio == 268
    assert result.current_requirement == round(5 * math.exp(4.2) - 5) == 328
