"""Tests for response latency analysis."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.latency import (
    LatencyMetrics,
    analyze_latencies,
    compute_latencies,
)


@pytest.mark.unit
def test_compute_latencies_pairs_each_reference_with_next_response() -> None:
    reference_times = [0.0, 10.0, 20.0]
    response_times = [1.5, 12.0, 13.0, 21.2, 30.0]

    latencies = compute_latencies(reference_times, response_times)

    assert latencies == pytest.approx([1.5, 2.0, 1.2])


@pytest.mark.unit
def test_compute_latencies_skips_missing_response_after_reference() -> None:
    reference_times = [0.0, 10.0, 20.0]
    # No response after t=20 -> latency for that trial is None / omitted.
    response_times = [1.0, 11.0]

    latencies = compute_latencies(reference_times, response_times)

    assert latencies == pytest.approx([1.0, 1.0])


@pytest.mark.unit
def test_compute_latencies_ignores_responses_before_first_reference() -> None:
    reference_times = [5.0, 10.0]
    response_times = [0.5, 1.0, 6.0, 11.5]

    latencies = compute_latencies(reference_times, response_times)

    assert latencies == pytest.approx([1.0, 1.5])


@pytest.mark.unit
def test_compute_latencies_applies_max_latency_cutoff() -> None:
    # First reference has no response within 60 s (next response at t=100);
    # second reference has a response 1 s later.
    reference_times = [0.0, 99.0]
    response_times = [100.0, 200.0]

    latencies = compute_latencies(reference_times, response_times, max_latency_sec=60.0)

    assert latencies == pytest.approx([1.0])


@pytest.mark.unit
def test_compute_latencies_raises_on_unsorted_inputs() -> None:
    with pytest.raises(ValueError, match="sorted"):
        compute_latencies([0.0, 5.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="sorted"):
        compute_latencies([0.0], [3.0, 1.0])


@pytest.mark.unit
def test_analyze_latencies_returns_summary_statistics() -> None:
    result = analyze_latencies([1.0, 2.0, 3.0, 4.0, 5.0])

    assert isinstance(result, LatencyMetrics)
    assert result.n == 5
    assert result.mean == pytest.approx(3.0)
    assert result.median == pytest.approx(3.0)
    assert result.min == pytest.approx(1.0)
    assert result.max == pytest.approx(5.0)
    assert result.std == pytest.approx(1.4142, abs=1e-3)


@pytest.mark.unit
def test_analyze_latencies_returns_none_for_empty_input() -> None:
    assert analyze_latencies([]) is None


@pytest.mark.unit
def test_analyze_latencies_filters_non_positive_values() -> None:
    result = analyze_latencies([-1.0, 0.0, 2.0, 4.0])
    assert result is not None
    assert result.n == 2
    assert result.mean == pytest.approx(3.0)
