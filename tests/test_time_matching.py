"""Tests for time-allocation matching law (Baum & Rachlin, 1969)."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.time_matching import (
    TimeMatchingParameters,
    estimate_time_matching_parameters,
)


@pytest.mark.unit
def test_time_matching_perfect_fit() -> None:
    # Time allocation exactly tracks reinforcement (a=1, b=1)
    times = [(2, 1), (3, 1), (1, 2), (4, 1), (1, 4)]
    reinf = [(2, 1), (3, 1), (1, 2), (4, 1), (1, 4)]
    p = estimate_time_matching_parameters(
        time_ratios=[t1 / t2 for t1, t2 in times],
        reinforcement_ratios=[r1 / r2 for r1, r2 in reinf],
    )
    assert isinstance(p, TimeMatchingParameters)
    assert p.sensitivity == pytest.approx(1.0, abs=1e-6)
    assert p.bias == pytest.approx(0.0, abs=1e-6)
    assert p.r_squared == pytest.approx(1.0, abs=1e-6)
    assert p.n == 5


@pytest.mark.unit
def test_time_matching_undermatching() -> None:
    """a < 1 indicates undermatching (typical for time allocation)."""
    import math

    # Construct y = 0.7 * x exactly
    rr = [0.5, 1.0, 2.0, 3.0, 4.0]
    tr = [math.exp(0.7 * math.log(r)) for r in rr]
    p = estimate_time_matching_parameters(
        time_ratios=tr,
        reinforcement_ratios=rr,
    )
    assert p.sensitivity == pytest.approx(0.7, abs=1e-6)
    assert p.bias == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_time_matching_input_validation() -> None:
    with pytest.raises(ValueError):
        estimate_time_matching_parameters(time_ratios=[1.0], reinforcement_ratios=[1.0, 2.0])
    with pytest.raises(ValueError):
        # all non-positive values are filtered, leaving < 2
        estimate_time_matching_parameters(
            time_ratios=[0.0, -1.0, 1.0], reinforcement_ratios=[1.0, 2.0, 0.0]
        )


@pytest.mark.unit
def test_time_matching_filters_non_positive() -> None:
    p = estimate_time_matching_parameters(
        time_ratios=[1.0, 2.0, 0.0, 4.0],
        reinforcement_ratios=[1.0, 2.0, 1.0, 4.0],
    )
    # The (0.0, 1.0) pair is dropped
    assert p.n == 3
