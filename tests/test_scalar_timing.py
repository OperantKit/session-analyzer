"""Tests for scalar-timing analyses (SET)."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.scalar_timing import (
    ScalarPropertyTest,
    ScalarTimingResult,
    analyze_scalar_timing,
    assess_scalar_property,
)


@pytest.mark.unit
def test_analyze_returns_none_for_fewer_than_two_responses() -> None:
    assert analyze_scalar_timing([]) is None
    assert analyze_scalar_timing([5.0]) is None


@pytest.mark.unit
def test_analyze_computes_cv() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=10.0, scale=2.0, size=2000).clip(min=1e-3).tolist()
    result = analyze_scalar_timing(samples)
    assert isinstance(result, ScalarTimingResult)
    assert result.mean == pytest.approx(10.0, abs=0.2)
    assert result.std == pytest.approx(2.0, abs=0.2)
    assert result.coefficient_of_variation == pytest.approx(0.2, abs=0.05)


@pytest.mark.unit
def assess_scalar_property_holds_for_constant_cv_across_criteria() -> None:
    rng = np.random.default_rng(7)
    cv_target = 0.25
    samples = {}
    for criterion in (5.0, 15.0, 30.0, 60.0):
        sd = cv_target * criterion
        samples[criterion] = (
            rng.normal(loc=criterion, scale=sd, size=2000).clip(min=1e-3).tolist()
        )

    result = assess_scalar_property(samples, tolerance=0.20)

    assert isinstance(result, ScalarPropertyTest)
    assert result.is_scalar
    assert result.mean_cv == pytest.approx(cv_target, abs=0.05)
    assert result.cv_of_cvs < 0.10


@pytest.mark.unit
def assess_scalar_property_fails_when_cv_varies_strongly() -> None:
    rng = np.random.default_rng(2)
    samples = {
        5.0: rng.normal(5.0, 0.1, size=500).clip(min=1e-3).tolist(),  # CV ~ 0.02
        60.0: rng.normal(60.0, 30.0, size=500).clip(min=1e-3).tolist(),  # CV ~ 0.5
    }
    result = assess_scalar_property(samples, tolerance=0.20)
    assert not result.is_scalar
    assert result.cv_of_cvs > 0.20


@pytest.mark.unit
def assess_scalar_property_raises_if_too_few_criteria() -> None:
    with pytest.raises(ValueError, match="at least 2 criteria"):
        assess_scalar_property({5.0: [4.5, 5.5]})
