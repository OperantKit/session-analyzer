"""Tests for progressive-ratio breakpoint detection."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.breakpoint import (
    BreakpointResult,
    detect_breakpoint,
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
