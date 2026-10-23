"""Tests for Rescorla-Wagner simulation and fitting."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.rescorla_wagner import (
    RescorlaWagnerParameters,
    fit_rescorla_wagner,
    simulate_rescorla_wagner,
)


@pytest.mark.unit
def test_simulate_approaches_lambda_under_reinforcement() -> None:
    trace = simulate_rescorla_wagner(
        [True] * 50, alpha_beta=0.3, lambda_=1.0, initial_v=0.0
    )
    assert trace[-1] == pytest.approx(1.0, abs=1e-3)
    # Monotonically increasing under all-reinforcement.
    assert all(b >= a - 1e-12 for a, b in zip(trace, trace[1:], strict=False))


@pytest.mark.unit
def test_simulate_decays_under_extinction() -> None:
    trace = simulate_rescorla_wagner(
        [False] * 30, alpha_beta=0.3, lambda_=1.0, initial_v=1.0
    )
    assert trace[-1] == pytest.approx(0.0, abs=1e-3)
    assert all(b <= a + 1e-12 for a, b in zip(trace, trace[1:], strict=False))


@pytest.mark.unit
def test_simulate_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="alpha_beta"):
        simulate_rescorla_wagner([True], alpha_beta=0.0, lambda_=1.0)
    with pytest.raises(ValueError, match="alpha_beta"):
        simulate_rescorla_wagner([True], alpha_beta=1.5, lambda_=1.0)


@pytest.mark.unit
def test_fit_recovers_known_parameters_from_synthetic_trace() -> None:
    ab_true, lam_true = 0.25, 0.8
    us = [True] * 20 + [False] * 20 + [True] * 20
    observed = simulate_rescorla_wagner(us, alpha_beta=ab_true, lambda_=lam_true)

    result = fit_rescorla_wagner(observed, us)

    assert isinstance(result, RescorlaWagnerParameters)
    assert result.alpha_beta == pytest.approx(ab_true, abs=0.02)
    assert result.lambda_ == pytest.approx(lam_true, abs=0.02)
    assert result.r_squared > 0.999
    assert result.n_trials == 60


@pytest.mark.unit
def test_fit_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_rescorla_wagner([0.1, 0.2], [True])


@pytest.mark.unit
def test_fit_raises_on_insufficient_trials() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_rescorla_wagner([0.5], [True])


# ---------------------------------------------------------------------------
# Closed-form values
# ---------------------------------------------------------------------------
#
# Rescorla & Wagner (1972) give the model, not a table of fitted values.
# Under constant reinforcement V_n = lambda (1 - (1 - ab)^n) from V_0 = 0;
# under extinction V_n = V_0 (1 - ab)^n.


@pytest.mark.unit
def test_acquisition_matches_closed_form() -> None:
    trace = simulate_rescorla_wagner([True] * 20, alpha_beta=0.2, lambda_=1.0)
    expected = [1.0 - 0.8**n for n in range(1, 21)]
    assert trace == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_extinction_matches_closed_form() -> None:
    trace = simulate_rescorla_wagner([False] * 15, alpha_beta=0.3, lambda_=1.0, initial_v=0.8)
    assert trace == pytest.approx([0.8 * 0.7**n for n in range(1, 16)], rel=1e-12)


@pytest.mark.unit
def test_fit_recovers_parameters_from_closed_form_trace() -> None:
    us = [True] * 15 + [False] * 10
    observed = simulate_rescorla_wagner(us, alpha_beta=0.25, lambda_=0.9)
    fit = fit_rescorla_wagner(observed, us)
    assert fit.alpha_beta == pytest.approx(0.25, rel=1e-4)
    assert fit.lambda_ == pytest.approx(0.9, rel=1e-4)
