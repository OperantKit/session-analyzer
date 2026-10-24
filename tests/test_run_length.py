"""Tests for FR run-length analysis."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.run_length import RunMetrics, analyze_fr_runs


@pytest.mark.unit
def test_none_when_fewer_than_two_reinforcers() -> None:
    assert analyze_fr_runs([1.0, 2.0, 3.0], [5.0]) is None


@pytest.mark.unit
def test_decomposes_runs_between_reinforcers() -> None:
    # Reinforcers at t=0, 10, 20.
    # Run 1 (t in (0,10]): responses 2.0, 2.5, 3.0 (pause=2.0, duration=1.0).
    # Run 2 (t in (10,20]): responses 15.0, 15.4, 15.8, 16.2 (pause=5.0, duration=1.2).
    reinforcers = [0.0, 10.0, 20.0]
    responses = [2.0, 2.5, 3.0, 15.0, 15.4, 15.8, 16.2]

    result = analyze_fr_runs(responses, reinforcers)
    assert isinstance(result, RunMetrics)
    assert result.n_runs == 2
    assert result.mean_run_length == pytest.approx(3.5)
    assert result.mean_run_duration == pytest.approx(1.1)
    assert result.mean_pause_duration == pytest.approx(3.5)
    assert result.pause_to_run_ratio == pytest.approx(3.5 / 1.1)
    # Total 7 responses over 1.0 + 1.2 = 2.2 s => ~3.18 resp/s
    assert result.mean_run_rate == pytest.approx(7.0 / 2.2, rel=1e-6)


@pytest.mark.unit
def test_skips_intervals_with_fewer_than_two_responses() -> None:
    reinforcers = [0.0, 10.0, 20.0, 30.0]
    responses = [5.0, 15.0, 25.0, 25.5]  # first two intervals have single response
    result = analyze_fr_runs(responses, reinforcers)
    assert result is not None
    assert result.n_runs == 1


@pytest.mark.unit
def test_none_when_no_valid_runs() -> None:
    reinforcers = [0.0, 10.0, 20.0]
    responses = [5.0, 15.0]  # each interval has only one response
    assert analyze_fr_runs(responses, reinforcers) is None


@pytest.mark.unit
def test_raises_on_unsorted_inputs() -> None:
    with pytest.raises(ValueError, match="sorted"):
        analyze_fr_runs([3.0, 1.0], [0.0, 5.0])
    with pytest.raises(ValueError, match="sorted"):
        analyze_fr_runs([1.0, 2.0], [5.0, 0.0])


# ---------------------------------------------------------------------------
# Hand computation
# ---------------------------------------------------------------------------
#
# Schlinger, Derenne, & Baron (2008) review pause and run measures; the
# module's decomposition is a definition, so the check is by hand. The
# response that completes the ratio (t = 10) shares its time with the
# reinforcer and belongs to the run that ends there.


@pytest.mark.unit
@pytest.mark.reference(
    module="run_length",
    verdict="properties",
    source="Schlinger, Derenne & Baron (2008)",
)
def test_runs_match_hand_computation() -> None:
    m = analyze_fr_runs(
        [2.0, 3.0, 4.0, 10.0, 13.0, 15.0, 17.0, 19.0, 25.0],
        [0.0, 10.0, 25.0],
    )
    assert m is not None
    assert m.n_runs == 2
    assert m.mean_run_length == pytest.approx(4.5)  # (4 + 5) / 2
    assert m.mean_pause_duration == pytest.approx(2.5)  # (2 + 3) / 2
    assert m.mean_run_duration == pytest.approx(10.0)  # (8 + 12) / 2
    assert m.mean_run_rate == pytest.approx(9.0 / 20.0)
    assert m.pause_to_run_ratio == pytest.approx(0.25)
