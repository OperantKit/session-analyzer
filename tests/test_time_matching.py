"""Tests for time-allocation matching law (Baum & Rachlin, 1969)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.matching import estimate_matching_parameters
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


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------
#
# Baum & Rachlin (1969): the time ratio was directly proportional to the
# reinforcement ratio with a constant of proportionality below 1 for every
# pigeon (bias toward the right side). Only the abstract was retrieved (the
# PMC scan is behind a download check), so the per-pigeon table is not used.
# The fit is the same OLS as the response-allocation matching law, and the
# bias is a natural logarithm (log10 b = bias / ln 10).


@pytest.mark.unit
def test_proportional_time_allocation_gives_unit_slope_and_negative_bias() -> None:
    reinforcement = [0.25, 0.5, 1.0, 2.0, 4.0]
    time = [0.8 * r for r in reinforcement]
    fit = estimate_time_matching_parameters(time, reinforcement)
    assert fit.sensitivity == pytest.approx(1.0, abs=1e-6)
    assert fit.bias == pytest.approx(math.log(0.8), abs=1e-6)
    assert fit.bias < 0.0


@pytest.mark.unit
def test_agrees_with_response_matching_and_polyfit() -> None:
    reinforcement = [0.2, 0.5, 1.0, 3.0, 6.0]
    time = [0.35, 0.6, 1.1, 2.2, 3.4]
    fit = estimate_time_matching_parameters(time, reinforcement)
    gml = estimate_matching_parameters(time, reinforcement)
    slope, intercept = np.polyfit(np.log(reinforcement), np.log(time), 1)
    assert fit.sensitivity == pytest.approx(gml.sensitivity, abs=1e-6)
    assert fit.bias == pytest.approx(gml.bias, abs=1e-6)
    assert fit.sensitivity == pytest.approx(slope, abs=1e-6)
    assert fit.bias == pytest.approx(intercept, abs=1e-6)
