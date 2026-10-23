"""Tests for hierarchical GML fitting."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.hierarchical_gml import (
    HierarchicalGMLFit,
    fit_hierarchical_gml,
)


@pytest.mark.unit
def test_recovers_group_mean_when_subjects_share_parameters() -> None:
    rng = np.random.default_rng(7)
    mu_a, mu_b = 0.85, 0.0
    tau_a, tau_b = 0.05, 0.05
    sigma_eps = 0.05

    x_grid = np.linspace(np.log(0.1), np.log(10.0), 8)

    xs = {}
    ys = {}
    for s in range(8):
        a_i = rng.normal(mu_a, tau_a)
        b_i = rng.normal(mu_b, tau_b)
        y = a_i * x_grid + b_i + rng.normal(0.0, sigma_eps, size=x_grid.size)
        xs[f"S{s}"] = x_grid.tolist()
        ys[f"S{s}"] = y.tolist()

    result = fit_hierarchical_gml(ys, xs)

    assert isinstance(result, HierarchicalGMLFit)
    assert result.mu_sensitivity == pytest.approx(mu_a, abs=0.1)
    assert result.mu_bias == pytest.approx(mu_b, abs=0.1)
    assert result.n_subjects == 8
    assert result.n_observations == 8 * x_grid.size
    lo, hi = result.sensitivity_ci
    assert lo <= mu_a <= hi


@pytest.mark.unit
def test_raises_on_subject_set_mismatch() -> None:
    with pytest.raises(ValueError, match="same set of subjects"):
        fit_hierarchical_gml({"A": [0.0, 1.0]}, {"B": [0.0, 1.0]})


@pytest.mark.unit
def test_raises_on_too_few_subjects() -> None:
    with pytest.raises(ValueError, match="at least 2 subjects"):
        fit_hierarchical_gml({"A": [0.0, 1.0]}, {"A": [0.0, 1.0]})


@pytest.mark.unit
def test_raises_on_too_few_observations_per_subject() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        fit_hierarchical_gml(
            {"A": [0.0], "B": [0.0, 1.0]},
            {"A": [0.0], "B": [0.0, 1.0]},
        )


@pytest.mark.unit
def test_raises_on_within_subject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_hierarchical_gml(
            {"A": [0.0, 1.0], "B": [0.0]},
            {"A": [0.0, 1.0, 2.0], "B": [0.0]},
        )


# ---------------------------------------------------------------------------
# Reference implementation: statsmodels MixedLM (maximum likelihood)
# ---------------------------------------------------------------------------
#
# Expected values come from statsmodels 0.15.0
# ``smf.mixedlm("y ~ x", df, groups="subject", re_formula="1",
# vc_formula={"x": "0 + x"}).fit(reml=False)`` on the data below: a random
# intercept and an independent random slope, the same model this module
# fits (statsmodels is not a package dependency, so the values are pinned).
# Gelman & Hill (2007) give no numerical example of this model, so there is
# no published value to match.

_REF_X = [-1.0, -0.5, 0.0, 0.5, 1.0]
_REF_Y = {
    "s1": [-0.95, -0.52, 0.08, 0.41, 0.93],
    "s2": [-0.71, -0.33, 0.12, 0.47, 0.88],
    "s3": [-1.02, -0.49, -0.07, 0.38, 0.79],
    "s4": [-0.61, -0.18, 0.21, 0.62, 1.04],
    "s5": [-0.84, -0.47, -0.01, 0.33, 0.69],
    "s6": [-0.90, -0.36, 0.05, 0.55, 0.97],
}


@pytest.mark.unit
def test_matches_statsmodels_mixedlm_ml_estimates() -> None:
    fit = fit_hierarchical_gml(_REF_Y, dict.fromkeys(_REF_Y, _REF_X))
    assert fit.mu_sensitivity == pytest.approx(0.8589999999999997, rel=1e-6)
    assert fit.mu_bias == pytest.approx(0.03533333333333338, rel=1e-6)
    assert fit.sd_sensitivity == pytest.approx(0.06107048608485036, rel=1e-4)
    assert fit.sd_bias == pytest.approx(0.09923734782100294, rel=1e-4)
    assert fit.sigma_obs == pytest.approx(0.038100734413604766, rel=1e-4)
    assert fit.log_likelihood == pytest.approx(38.784592659681415, rel=1e-8)
    assert fit.n_subjects == 6
    assert fit.n_observations == 30


@pytest.mark.unit
def test_interval_half_width_uses_statsmodels_standard_error() -> None:
    """95 % Wald interval = estimate +/- 1.96 * SE, with SE from the GLS
    information matrix at the ML estimate (statsmodels ``bse_fe``):
    0.026802582760442815 for mu_sensitivity, 0.04110633445735158 for
    mu_bias. The SE includes within-subject noise, not only tau."""
    fit = fit_hierarchical_gml(_REF_Y, dict.fromkeys(_REF_Y, _REF_X))
    half_a = (fit.sensitivity_ci[1] - fit.sensitivity_ci[0]) / 2.0
    half_b = (fit.bias_ci[1] - fit.bias_ci[0]) / 2.0
    assert half_a == pytest.approx(1.96 * 0.026802582760442815, rel=1e-4)
    assert half_b == pytest.approx(1.96 * 0.04110633445735158, rel=1e-4)
