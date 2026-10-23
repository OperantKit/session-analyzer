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


# ---------------------------------------------------------------------------
# Published values (S-166).
#
# Source: Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P.
# (2007). An EZ-diffusion model for response time and accuracy. Psychonomic
# Bulletin & Review, 14(1), 3-22. https://doi.org/10.3758/BF03194023
# Appendix ("R Code for the EZ-Diffusion Model"): the process v = 0.1,
# a = 0.14, Ter = 0.300, s = 0.1 yields MRT = 0.723, VRT = 0.112,
# Pc = .802 (rounded), and the reference R function called with these
# rounded inputs prints v = 0.09993853, a = 0.1399702, Ter = 0.30003.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_matches_wagenmakers_2007_appendix_output() -> None:
    """Rounded inputs (.802, .112, .723) reproduce the printed R output.

    Tolerance: half a unit in the last printed place of each value
    (5e-9 for v, 5e-8 for a, 5e-6 for Ter).
    """
    fit = fit_ez_diffusion(
        proportion_correct=0.802, mean_rt_correct=0.723, var_rt_correct=0.112
    )
    assert fit.v == pytest.approx(0.09993853, abs=5e-9)
    assert fit.a == pytest.approx(0.1399702, abs=5e-8)
    assert fit.ter == pytest.approx(0.30003, abs=5e-6)


@pytest.mark.unit
def test_forward_model_matches_wagenmakers_2007_appendix_summary() -> None:
    """v = 0.1, a = 0.14, Ter = 0.3 gives Pc .802, MRT 0.723, VRT 0.112.

    The appendix reports these three values rounded to three decimals.
    """
    pc, mrt, vrt = _forward_ez(0.1, 0.14, 0.3)
    assert round(pc, 3) == pytest.approx(0.802)
    assert round(mrt, 3) == pytest.approx(0.723)
    assert round(vrt, 3) == pytest.approx(0.112)
