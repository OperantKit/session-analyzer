"""Tests for progressive-ratio breakpoint detection."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.breakpoint import (
    BreakpointResult,
    IRTBreakpointResult,
    detect_breakpoint,
    estimate_irt_breakpoint,
)


@pytest.mark.unit
def test_detects_breakpoint_from_terminal_pause() -> None:
    # Completed 1, 2, 4, 8. Then 10-minute pause with no further completion.
    completions = [(1, 5.0), (2, 20.0), (4, 60.0), (8, 180.0)]
    result = detect_breakpoint(
        completions,
        session_end_time=180.0 + 600.0,
        pause_threshold_sec=300.0,
    )

    assert isinstance(result, BreakpointResult)
    assert result.breakpoint_ratio == 8
    assert result.n_ratios_completed == 4
    assert result.terminal_pause_sec == pytest.approx(600.0)
    assert result.method == "pause_threshold"


@pytest.mark.unit
def test_no_breakpoint_when_session_ends_before_threshold() -> None:
    # Session ended 60 s after the last completion, below 300 s threshold.
    completions = [(1, 5.0), (2, 20.0), (4, 60.0)]
    result = detect_breakpoint(
        completions,
        session_end_time=120.0,
        pause_threshold_sec=300.0,
    )

    assert result.breakpoint_ratio is None
    assert result.method == "session_ended"
    assert result.n_ratios_completed == 3
    assert result.terminal_pause_sec == pytest.approx(60.0)


@pytest.mark.unit
def test_empty_completions_returns_null_result() -> None:
    result = detect_breakpoint([], session_end_time=300.0)
    assert result.breakpoint_ratio is None
    assert result.n_ratios_completed == 0
    assert result.method == "no_completions"


@pytest.mark.unit
def test_raises_on_unsorted_completion_times() -> None:
    with pytest.raises(ValueError, match="sorted"):
        detect_breakpoint([(1, 10.0), (2, 5.0)], session_end_time=100.0)


@pytest.mark.unit
def test_raises_on_session_end_before_last_completion() -> None:
    with pytest.raises(ValueError, match="session_end_time"):
        detect_breakpoint(
            [(1, 5.0), (2, 120.0)],
            session_end_time=60.0,
        )


@pytest.mark.unit
def test_default_threshold_is_300_sec() -> None:
    """Richardson & Roberts (1996) conventional 5-minute no-response criterion."""
    completions = [(1, 0.0)]
    result = detect_breakpoint(completions, session_end_time=301.0)
    assert result.breakpoint_ratio == 1
    assert result.method == "pause_threshold"


# ---------------------------------------------------------------------------
# IRT-within-current-step (live) breakpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_irt_breakpoint_status_active_when_irts_well_below_threshold() -> None:
    # Reinforcer at t=10. Then 4 responses 1 s apart inside the current step.
    response_times = [1.0, 2.0, 3.0, 4.0, 5.0, 11.0, 12.0, 13.0, 14.0]
    reinforcement_times = [10.0]
    result = estimate_irt_breakpoint(
        response_times,
        reinforcement_times,
        irt_threshold_sec=60.0,
    )
    assert isinstance(result, IRTBreakpointResult)
    assert result.status == "active"
    assert result.breakpoint_reached is False
    assert result.last_irt_sec == pytest.approx(1.0)


@pytest.mark.unit
def test_irt_breakpoint_status_approaching_when_last_irt_above_half_threshold() -> None:
    # Last IRT = 35 s, threshold = 60 s ⇒ status "approaching" (35 >= 30).
    response_times = [11.0, 12.0, 47.0]
    reinforcement_times = [10.0]
    result = estimate_irt_breakpoint(
        response_times,
        reinforcement_times,
        irt_threshold_sec=60.0,
    )
    assert result.status == "approaching"
    assert result.breakpoint_reached is False


@pytest.mark.unit
def test_irt_breakpoint_status_reached_when_any_irt_exceeds_threshold() -> None:
    # IRT 100 s exceeds threshold 60 s ⇒ reached.
    response_times = [11.0, 12.0, 112.0]
    reinforcement_times = [10.0]
    result = estimate_irt_breakpoint(
        response_times,
        reinforcement_times,
        irt_threshold_sec=60.0,
    )
    assert result.status == "reached"
    assert result.breakpoint_reached is True
    assert result.last_irt_sec == pytest.approx(100.0)


@pytest.mark.unit
def test_irt_breakpoint_insufficient_data_when_zero_or_one_response_in_step() -> None:
    response_times = [11.0]  # only one response after the last reinforcer
    reinforcement_times = [10.0]
    result = estimate_irt_breakpoint(response_times, reinforcement_times)
    assert result.status == "insufficient_data"
    assert result.last_irt_sec is None


@pytest.mark.unit
def test_irt_breakpoint_uses_all_responses_when_no_reinforcement_yet() -> None:
    # No reinforcement yet ⇒ inspect all responses.
    response_times = [0.0, 1.0, 2.0, 100.0]
    reinforcement_times: list[float] = []
    result = estimate_irt_breakpoint(
        response_times,
        reinforcement_times,
        irt_threshold_sec=60.0,
    )
    assert result.status == "reached"
    assert result.n_completed == 0


@pytest.mark.unit
def test_irt_breakpoint_reports_metadata_unchanged() -> None:
    response_times = [11.0, 12.0]
    reinforcement_times = [10.0]
    result = estimate_irt_breakpoint(
        response_times,
        reinforcement_times,
        irt_threshold_sec=60.0,
        last_completed_ratio=8,
        current_requirement=12,
    )
    assert result.last_completed_ratio == 8
    assert result.current_requirement == 12
    assert result.n_completed == 1


@pytest.mark.unit
def test_irt_breakpoint_raises_on_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_irt_breakpoint([1.0, 2.0], [], irt_threshold_sec=0.0)
    with pytest.raises(ValueError, match="positive"):
        estimate_irt_breakpoint([1.0, 2.0], [], irt_threshold_sec=-1.0)



# ---------------------------------------------------------------------------
# Richardson & Roberts (1996) operationalization (property checks)
# ---------------------------------------------------------------------------
#
# Breakpoint = last ratio completed before a pause at least as long as the
# criterion (Richardson & Roberts, 1996; the criterion length is
# procedure-dependent, 5 min is the module default). No numeric dataset is
# published for this
# definition, so the checks build completions on the published exponential
# series (1, 2, 4, 6, 9, 12, 15, 20, 25, 32, 40, 50, 62, 77, 95, ...).

_RR_SERIES = [1, 2, 4, 6, 9, 12, 15, 20, 25, 32, 40, 50, 62, 77, 95]


@pytest.mark.unit
@pytest.mark.reference(
    module="breakpoint",
    verdict="properties",
    source="Richardson & Roberts (1996)",
)
def test_breakpoint_is_last_completed_ratio_on_published_series() -> None:
    completions = [(r, 30.0 * i) for i, r in enumerate(_RR_SERIES, start=1)]
    last_time = completions[-1][1]
    result = detect_breakpoint(
        completions, session_end_time=last_time + 3600.0, pause_threshold_sec=3600.0
    )
    assert result.method == "pause_threshold"
    assert result.breakpoint_ratio == 95
    assert result.n_ratios_completed == 15


@pytest.mark.unit
def test_breakpoint_pause_criterion_is_inclusive_at_threshold() -> None:
    completions = [(r, 10.0 * i) for i, r in enumerate(_RR_SERIES[:5], start=1)]
    at = detect_breakpoint(completions, session_end_time=50.0 + 300.0)
    below = detect_breakpoint(completions, session_end_time=50.0 + 299.999)
    assert at.breakpoint_ratio == 9
    assert below.breakpoint_ratio is None
    assert below.method == "session_ended"
