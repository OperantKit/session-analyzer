"""Tests for the learning-curve change-point analysis (Gallistel, Fairhurst & Balsam, 2004)."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.change_point import (
    AcquisitionSummary,
    ChangePointResult,
    acquisition_summary,
    change_points,
    change_points_from_times,
)


@pytest.mark.unit
@pytest.mark.reference(
    module="change_point",
    verdict="properties",
    source="Gallistel, Fairhurst & Balsam (2004)",
)
def test_recovers_a_known_step_in_binary_records() -> None:
    near = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        x = np.concatenate([rng.binomial(1, 0.1, 150), rng.binomial(1, 0.8, 150)])
        found = change_points(x.tolist()).change_points
        assert found, f"seed {seed}: no change point"
        if 150 - 30 <= found[0] <= 150 + 5:
            near += 1
    assert near >= 12


@pytest.mark.unit
def test_recovers_a_step_in_count_records() -> None:
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.poisson(0.5, 40), rng.poisson(12.0, 60)])
    result = change_points(x.tolist())
    assert result.change_points
    assert 35 <= result.change_points[0] <= 41
    assert result.levels[0] < 1.0 < 10.0 < result.levels[1]


@pytest.mark.unit
def test_stationary_records_rarely_give_a_change_point_at_the_default_criterion() -> None:
    # The test is repeated at every point, so stationary records are not free of
    # change points: this seed gives 1 of 200 at criterion 4 and 77 of 200 at 2.
    rng = np.random.default_rng(0)
    records = [rng.binomial(1, 0.3, 300).tolist() for _ in range(200)]
    at_default = sum(bool(change_points(r).change_points) for r in records)
    at_two = sum(bool(change_points(r, criterion=2.0).change_points) for r in records)
    assert at_default <= 2
    assert at_two > 40


@pytest.mark.unit
def test_default_criterion_is_four() -> None:
    assert change_points([0, 1, 0, 1]).criterion == 4.0


@pytest.mark.unit
def test_all_zero_and_short_records_have_no_change_point() -> None:
    zero = change_points([0] * 50)
    assert zero.change_points == () and zero.levels == (0.0,)
    one = change_points([3])
    assert one.change_points == () and one.levels == (3.0,)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[], [1, -1, 2], [1, 0.5, 2], [1, float("nan")]])
def test_rejects_empty_negative_and_non_integer_counts(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        change_points(bad)


@pytest.mark.unit
@pytest.mark.parametrize("criterion", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_non_positive_or_non_finite_criterion(criterion: float) -> None:
    with pytest.raises(ValueError):
        change_points([0, 1, 1], criterion=criterion)


@pytest.mark.unit
@pytest.mark.reference(
    module="change_point",
    verdict="properties",
    source="Gallistel, Fairhurst & Balsam (2004); timed form has no retrieved input",
)
def test_event_times_recover_a_rate_step() -> None:
    rng = np.random.default_rng(3)
    slow = np.cumsum(rng.exponential(10.0, 30))  # 0.1 per s
    fast = slow[-1] + np.cumsum(rng.exponential(1.0, 120))  # 1 per s
    result = change_points_from_times(np.concatenate([slow, fast]).tolist())
    assert isinstance(result, ChangePointResult)
    assert result.change_points and 27 <= result.change_points[0] <= 33
    assert result.levels[0] < 0.3 and result.levels[1] > 0.6


@pytest.mark.unit
def test_event_times_of_a_poisson_process_rarely_give_a_change_point() -> None:
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(50):
        times = np.cumsum(rng.exponential(1.0, 200))
        hits += bool(change_points_from_times(times.tolist()).change_points)
    assert hits <= 3  # this seed gives 2 of 50


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[], [0.0, 1.0], [1.0, 0.5], [-1.0, 2.0], [1.0, float("inf")]])
def test_event_times_must_be_positive_and_increasing(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        change_points_from_times(bad)


@pytest.mark.unit
def test_acquisition_summary_of_a_known_step() -> None:
    values = [0] * 20 + [5] * 40
    result = change_points(values)
    assert result.change_points == (20,)
    summary = acquisition_summary(values, result)
    assert isinstance(summary, AcquisitionSummary)
    assert summary.onset_latency == 20
    assert summary.asymptote == pytest.approx(5.0)
    assert summary.dynamic_interval == 0
    assert summary.first_fraction == pytest.approx(1.0)
    assert summary.levels == (0.0, 5.0)


@pytest.mark.unit
def test_acquisition_summary_counts_units_to_eighty_percent() -> None:
    values = [0] * 10 + [1] * 10 + [2] * 10 + [5] * 30
    summary = acquisition_summary(values, (10, 20, 30))
    assert summary.onset_latency == 10
    assert summary.asymptote == pytest.approx(5.0)
    assert summary.dynamic_interval == 20  # 10 -> 30, the first level >= 4.0
    assert summary.first_fraction == pytest.approx(0.2)
    assert summary.levels == pytest.approx((0.0, 1.0, 2.0, 5.0))


@pytest.mark.unit
def test_acquisition_summary_without_upward_change() -> None:
    values = [3] * 10 + [0] * 10
    summary = acquisition_summary(values, (10,))
    assert summary.onset_latency is None
    assert summary.dynamic_interval is None
    assert summary.asymptote == 0.0
    assert summary.first_fraction is None  # the second-half mean is 0


@pytest.mark.unit
def test_acquisition_summary_without_change_points() -> None:
    summary = acquisition_summary([2] * 10, ())
    assert summary.onset_latency is None
    assert summary.first_fraction is None
    assert summary.levels == (2.0,)


@pytest.mark.unit
def test_acquisition_summary_accepts_an_asymptote_and_reports_unreached() -> None:
    values = [0] * 10 + [2] * 10
    summary = acquisition_summary(values, (10,), asymptote=10.0)
    assert summary.asymptote == 10.0
    assert summary.onset_latency == 10
    assert summary.dynamic_interval is None
    assert summary.first_fraction == pytest.approx(0.2)


@pytest.mark.unit
@pytest.mark.parametrize("cps", [(0,), (5, 5), (12, 4), (20,)])
def test_acquisition_summary_rejects_bad_change_points(cps: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        acquisition_summary([1] * 20, cps)


@pytest.mark.unit
def test_results_are_frozen() -> None:
    result = change_points([0] * 10 + [1] * 10)
    with pytest.raises(AttributeError):
        result.criterion = 2.0  # type: ignore[misc]
