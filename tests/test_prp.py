"""Tests for post-reinforcement pause analysis."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.prp import (
    PRPMetrics,
    analyze_prps,
    compute_prps,
    compute_quarter_life,
)


@pytest.mark.unit
def test_compute_prps_pairs_reinforcers_with_next_response() -> None:
    r = [10.0, 30.0, 60.0]
    responses = [5.0, 11.5, 15.0, 35.0, 62.5]
    prps = compute_prps(r, responses)
    assert prps == pytest.approx([1.5, 5.0, 2.5])


@pytest.mark.unit
def test_compute_prps_skips_terminal_reinforcer_without_response() -> None:
    r = [10.0, 30.0]
    responses = [12.0]
    assert compute_prps(r, responses) == pytest.approx([2.0])


@pytest.mark.unit
def test_compute_prps_raises_on_unsorted() -> None:
    with pytest.raises(ValueError, match="sorted"):
        compute_prps([10.0, 5.0], [1.0])
    with pytest.raises(ValueError, match="sorted"):
        compute_prps([1.0], [5.0, 2.0])


@pytest.mark.unit
def test_analyze_prps_returns_summary() -> None:
    result = analyze_prps([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(result, PRPMetrics)
    assert result.n == 5
    assert result.mean == pytest.approx(3.0)
    assert result.median == pytest.approx(3.0)


@pytest.mark.unit
def test_analyze_prps_returns_none_when_empty() -> None:
    assert analyze_prps([]) is None
    assert analyze_prps([-1.0, 0.0]) is None


@pytest.mark.unit
def test_quarter_life_uniform_responding_near_quantile() -> None:
    # Reinforcers at 0 and 100; 10 responses uniformly at 10, 20, ..., 100.
    reinforcers = [0.0, 100.0]
    responses = [10.0 * (i + 1) for i in range(10)]
    ql = compute_quarter_life(reinforcers, responses, quantile=0.25)
    # Ceil(0.25 * 10) = 3 -> 3rd response at t=30.0 -> fraction = 0.30.
    assert ql == pytest.approx([0.30])


@pytest.mark.unit
def test_quarter_life_scallop_pattern_exceeds_quantile() -> None:
    # Responses concentrated near end of interval (FI scallop).
    reinforcers = [0.0, 100.0]
    responses = [95.0, 96.0, 97.0, 98.0, 99.0]
    ql = compute_quarter_life(reinforcers, responses, quantile=0.25)
    # ceil(0.25*5)=2, 2nd response at t=96 -> fraction=0.96 >> 0.25.
    assert ql[0] > 0.9


@pytest.mark.unit
def test_quarter_life_skips_intervals_with_one_response() -> None:
    reinforcers = [0.0, 10.0, 20.0, 30.0]
    responses = [5.0, 15.0, 25.0]  # one response per interval
    assert compute_quarter_life(reinforcers, responses) == []


@pytest.mark.unit
def test_quarter_life_raises_on_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        compute_quarter_life([0.0, 10.0], [1.0, 2.0], quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        compute_quarter_life([0.0, 10.0], [1.0, 2.0], quantile=1.0)


@pytest.mark.unit
def test_quarter_life_raises_on_unsorted() -> None:
    with pytest.raises(ValueError, match="sorted"):
        compute_quarter_life([10.0, 5.0], [1.0])


# ---------------------------------------------------------------------------
# Analytic values (property checks)
# ---------------------------------------------------------------------------
#
# Published values: Felton & Lyon (1966) report pause means per FR value and
# Fry, Kelleher, & Cook (1960) define the quarter-life on FI performance,
# but neither body was retrieved (PMC scanned PDFs; Fry et al. has no
# abstract on PubMed). The checks below use closed-form cases instead.
# If the cumulative response count in an interval of length T follows
# N(t) = N * (t / T)^p, the q-th response quantile is reached at
# t / T = q^(1/p): p = 1 (uniform) gives 0.25, p = 2 (linearly
# accelerating rate) gives 0.5, p = 3 gives 0.25^(1/3).


def _power_cumulative_interval(n: int, p: float, start: float, length: float) -> list[float]:
    return [start + length * (k / n) ** (1.0 / p) for k in range(1, n + 1)]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("p", "expected"),
    [(1.0, 0.25), (2.0, 0.5), (3.0, 0.25 ** (1.0 / 3.0))],
)
@pytest.mark.reference(
    module="prp",
    verdict="properties",
    source="Fry, Kelleher & Cook (1960)",
)
def test_quarter_life_matches_closed_form_for_power_cumulative(
    p: float, expected: float
) -> None:
    n = 400  # quantile index ceil(0.25 * 400) = 100 lands exactly on k/n = 0.25
    responses = _power_cumulative_interval(n, p, 0.0, 120.0) + _power_cumulative_interval(
        n, p, 120.0, 120.0
    )
    ql = compute_quarter_life([0.0, 120.0, 240.0], responses)
    assert ql == pytest.approx([expected, expected], rel=1e-12)


@pytest.mark.unit
def test_analyze_prps_matches_statistics_module() -> None:
    import statistics

    prps = [3.2, 7.9, 1.1, 12.4, 5.5, 5.5, 9.0]
    m = analyze_prps(prps)
    assert m is not None
    assert m.mean == pytest.approx(statistics.fmean(prps), rel=1e-12)
    assert m.median == pytest.approx(statistics.median(prps), rel=1e-12)
    assert m.std == pytest.approx(statistics.pstdev(prps), rel=1e-12)
    assert (m.min, m.max, m.n) == (1.1, 12.4, 7)
