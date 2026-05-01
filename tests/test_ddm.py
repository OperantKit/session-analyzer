"""Tests for EZ-diffusion parameter recovery."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.ddm import (
    EZDiffusionParameters,
    fit_ez_diffusion,
)


def _forward_ez(v: float, a: float, ter: float, s: float = 0.1) -> tuple[float, float, float]:
    """Generate (Pc, MRTc, VRTc) from EZ-DDM parameters (Wagenmakers et al., 2007)."""
    y = math.exp(-v * a / (s * s))
    pc = 1.0 / (1.0 + y)
    mdt = (a / (2.0 * v)) * (1.0 - y) / (1.0 + y)
    mrt = mdt + ter
    # VRT formula from Wagenmakers et al. (2007), eq. 5:
    vrt = (
        a / (2.0 * v**3)
        * (1.0 - 2.0 * v * a / (s * s) * y - y * y)
        / ((1.0 + y) ** 2)
    ) * (s * s)
    return pc, mrt, vrt


@pytest.mark.unit
def test_recovers_parameters_from_synthesised_summary_statistics() -> None:
    v_true, a_true, ter_true = 0.2, 0.12, 0.30
    pc, mrt, vrt = _forward_ez(v_true, a_true, ter_true)

    result = fit_ez_diffusion(
        proportion_correct=pc, mean_rt_correct=mrt, var_rt_correct=vrt
    )
    assert isinstance(result, EZDiffusionParameters)
    assert result.v == pytest.approx(v_true, rel=1e-4)
    assert result.a == pytest.approx(a_true, rel=1e-4)
    assert result.ter == pytest.approx(ter_true, abs=1e-4)


@pytest.mark.unit
def test_negative_drift_when_below_chance() -> None:
    # Build a "harder" scenario where Pc < 0.5 -> negative drift.
    pc = 0.3
    mrt = 0.5
    vrt = 0.04
    result = fit_ez_diffusion(
        proportion_correct=pc, mean_rt_correct=mrt, var_rt_correct=vrt
    )
    assert result.v < 0.0
    assert result.a > 0.0


@pytest.mark.unit
def test_chance_performance_returns_zero_drift() -> None:
    result = fit_ez_diffusion(
        proportion_correct=0.5, mean_rt_correct=0.6, var_rt_correct=0.05
    )
    assert result.v == 0.0


@pytest.mark.unit
def test_raises_on_invalid_proportion() -> None:
    with pytest.raises(ValueError, match="proportion_correct"):
        fit_ez_diffusion(proportion_correct=0.0, mean_rt_correct=0.5, var_rt_correct=0.04)
    with pytest.raises(ValueError, match="proportion_correct"):
        fit_ez_diffusion(proportion_correct=1.0, mean_rt_correct=0.5, var_rt_correct=0.04)


@pytest.mark.unit
def test_raises_on_non_positive_rt_or_var() -> None:
    with pytest.raises(ValueError, match="mean_rt_correct"):
        fit_ez_diffusion(proportion_correct=0.7, mean_rt_correct=0.0, var_rt_correct=0.04)
    with pytest.raises(ValueError, match="var_rt_correct"):
        fit_ez_diffusion(proportion_correct=0.7, mean_rt_correct=0.5, var_rt_correct=0.0)
