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
