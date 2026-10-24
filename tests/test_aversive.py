"""Tests for ``session_analyzer.analytics.aversive``.

Free-operant avoidance (Sidman, 1953) and punishment (Azrin, 1960) summaries.
Hand computations pin the formulas; Azrin (1960) supplies printed counts that
the suppression ratio reproduces.
"""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics import (
    AvoidanceMetrics,
    PunishmentMetrics,
    analyze_avoidance,
    analyze_punishment,
)


def _evenly(n: int, duration: float) -> list[float]:
    """``n`` response times spread inside ``[0, duration)``."""
    return [duration * (i + 0.5) / n for i in range(n)]


# ---------------------------------------------------------------------------
# Published values: Azrin (1960). Kept first so that the coverage table reads
# this module's strongest verdict.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.reference(
    module="aversive",
    verdict="published",
    source="Azrin (1960) JEAB 3:123-142, stable state under 50-V punishment",
)
def test_suppression_ratio_reproduces_azrin_1960_daily_counts() -> None:
    # Azrin (1960): about 900 responses per day under punishment against 2800
    # without; sessions of one hour each.
    m = analyze_punishment(
        _evenly(2800, 3600.0), _evenly(900, 3600.0), 3600.0, 3600.0
    )
    assert m.suppression_ratio == pytest.approx(900 / 2800, abs=1e-12)


@pytest.mark.unit
@pytest.mark.reference(
    module="aversive",
    verdict="published",
    source="Azrin (1960) JEAB 3:123-142, first 30-V session",
)
def test_suppression_ratio_below_azrin_1960_first_session_bound() -> None:
    # Azrin (1960): from 40 responses per minute to less than 15 in the first
    # punished session.
    m = analyze_punishment(_evenly(40, 60.0), _evenly(14, 60.0), 60.0, 60.0)
    assert m.baseline_rate == pytest.approx(40.0, abs=1e-12)
    assert m.punished_rate < 15.0
    assert m.suppression_ratio < 15 / 40


# ---------------------------------------------------------------------------
# Avoidance
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.reference(
    module="aversive",
    verdict="properties",
    source="Sidman (1953); Clark & Hull (1966)",
)
def test_avoidance_rates_and_avoided_fraction_by_hand() -> None:
    # 3 shocks and 10 responses in 60 s; with S-S 10 s, 6 shocks were possible.
    m = analyze_avoidance(
        _evenly(10, 60.0), [5.0, 25.0, 45.0], 60.0, shock_shock_interval=10.0
    )
    assert isinstance(m, AvoidanceMetrics)
    assert m.shock_rate == pytest.approx(3.0, abs=1e-12)
    assert m.response_rate == pytest.approx(10.0, abs=1e-12)
    assert m.responses_per_shock == pytest.approx(10 / 3, abs=1e-12)
    assert m.shocks_avoided_fraction == pytest.approx(0.5, abs=1e-12)
    assert (m.n_shocks, m.n_responses, m.duration) == (3, 10, 60.0)


@pytest.mark.unit
def test_avoidance_shock_free_fraction_counts_bins_without_shock() -> None:
    # bins of 20 s: [0, 20) and [40, 60) hold shocks, [20, 40) does not.
    m = analyze_avoidance([], [3.0, 41.0, 59.0], 60.0, bin_width=20.0)
    assert m.shock_free_fraction == pytest.approx(1 / 3, abs=1e-12)
    assert m.shocks_avoided_fraction is None


@pytest.mark.unit
def test_avoidance_without_shocks() -> None:
    m = analyze_avoidance(_evenly(4, 60.0), [], 60.0, shock_shock_interval=10.0)
    assert m.shock_rate == 0.0
    assert math.isinf(m.responses_per_shock)
    assert m.shock_free_fraction == 1.0
    assert m.shocks_avoided_fraction == 1.0


@pytest.mark.unit
def test_avoidance_counts_the_trailing_partial_bin() -> None:
    # 50 s with 20 s bins: [0, 20), [20, 40), [40, 50) — a shock only in the last.
    m = analyze_avoidance([], [45.0], 50.0, bin_width=20.0)
    assert m.shock_free_fraction == pytest.approx(2 / 3, abs=1e-12)


# ---------------------------------------------------------------------------
# Punishment
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_punishment_suppression_and_recovery_by_hand() -> None:
    m = analyze_punishment(
        _evenly(30, 60.0),
        _evenly(6, 60.0),
        60.0,
        60.0,
        recovery_response_times=_evenly(24, 60.0),
        recovery_duration=60.0,
        punisher_times=_evenly(6, 60.0),
    )
    assert isinstance(m, PunishmentMetrics)
    assert m.baseline_rate == pytest.approx(30.0, abs=1e-12)
    assert m.punished_rate == pytest.approx(6.0, abs=1e-12)
    assert m.suppression_ratio == pytest.approx(0.2, abs=1e-12)
    assert m.recovery_rate == pytest.approx(24.0, abs=1e-12)
    assert m.recovery_ratio == pytest.approx(0.8, abs=1e-12)
    assert m.n_punishers == 6


@pytest.mark.unit
def test_punishment_without_recovery_phase() -> None:
    m = analyze_punishment(_evenly(10, 60.0), _evenly(5, 60.0), 60.0, 60.0)
    assert m.recovery_rate is None
    assert m.recovery_ratio is None
    assert m.n_punishers == 0


@pytest.mark.unit
def test_punishment_zero_baseline_gives_nan_ratio() -> None:
    m = analyze_punishment([], [], 60.0, 60.0)
    assert math.isnan(m.suppression_ratio)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan])
def test_avoidance_rejects_non_positive_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        analyze_avoidance([], [], duration)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("responses", "shocks"), [([61.0], []), ([], [-0.1]), ([math.nan], [])]
)
def test_avoidance_rejects_times_outside_the_session(
    responses: list[float], shocks: list[float]
) -> None:
    with pytest.raises(ValueError):
        analyze_avoidance(responses, shocks, 60.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs", [{"shock_shock_interval": 0.0}, {"bin_width": 0.0}]
)
def test_avoidance_rejects_non_positive_intervals(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        analyze_avoidance([], [], 60.0, **kwargs)


@pytest.mark.unit
def test_punishment_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        analyze_punishment([], [], 0.0, 60.0)
    with pytest.raises(ValueError):
        analyze_punishment([70.0], [], 60.0, 60.0)
    with pytest.raises(ValueError):
        analyze_punishment([], [], 60.0, 60.0, recovery_response_times=[1.0])
    with pytest.raises(ValueError):
        analyze_punishment([], [], 60.0, 60.0, recovery_duration=60.0)
    with pytest.raises(ValueError):
        analyze_punishment([], [], 60.0, 60.0, punisher_times=[61.0])
