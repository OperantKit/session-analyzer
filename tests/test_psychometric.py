"""Tests for psychometric-function fitting."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.psychometric import (
    PsychometricFit,
    fit_psychometric,
)


@pytest.mark.unit
def test_logit_recovers_known_threshold_and_slope() -> None:
    rng = np.random.default_rng(1)
    mu_true, sigma_true = 5.0, 1.5
    stimuli = list(np.linspace(-2.0, 12.0, 15))
    n_total = [200] * len(stimuli)
    p_true = 1.0 / (1.0 + np.exp(-(np.array(stimuli) - mu_true) / sigma_true))
    n_correct = list(rng.binomial(n_total, p_true))

    result = fit_psychometric(stimuli, n_correct, n_total, link="logit")

    assert isinstance(result, PsychometricFit)
    assert result.mu == pytest.approx(mu_true, abs=0.3)
    assert result.sigma == pytest.approx(sigma_true, abs=0.3)


@pytest.mark.unit
def test_probit_recovers_known_threshold_and_slope() -> None:
    rng = np.random.default_rng(2)
    from scipy.stats import norm
    mu_true, sigma_true = 0.0, 2.0
    stimuli = list(np.linspace(-6.0, 6.0, 13))
    n_total = [200] * len(stimuli)
    p_true = norm.cdf((np.array(stimuli) - mu_true) / sigma_true)
    n_correct = list(rng.binomial(n_total, p_true))

    result = fit_psychometric(stimuli, n_correct, n_total, link="probit")

    assert result.mu == pytest.approx(mu_true, abs=0.3)
    assert result.sigma == pytest.approx(sigma_true, abs=0.3)


@pytest.mark.unit
def test_2afc_design_with_gamma_half() -> None:
    rng = np.random.default_rng(3)
    mu_true, sigma_true = 4.0, 1.0
    stimuli = list(np.linspace(0.0, 8.0, 9))
    n_total = [400] * len(stimuli)
    inner = 1.0 / (1.0 + np.exp(-(np.array(stimuli) - mu_true) / sigma_true))
    p_true = 0.5 + 0.5 * inner
    n_correct = list(rng.binomial(n_total, p_true))

    result = fit_psychometric(
        stimuli, n_correct, n_total, link="logit", gamma=0.5
    )
    assert result.gamma == 0.5
    assert result.mu == pytest.approx(mu_true, abs=0.3)


@pytest.mark.unit
def test_fit_lapse_recovers_upper_asymptote_shortfall() -> None:
    rng = np.random.default_rng(4)
    mu_true, sigma_true, lapse_true = 0.0, 1.0, 0.05
    stimuli = list(np.linspace(-5.0, 5.0, 21))
    n_total = [800] * len(stimuli)
    inner = 1.0 / (1.0 + np.exp(-(np.array(stimuli) - mu_true) / sigma_true))
    p_true = (1.0 - lapse_true) * inner
    n_correct = list(rng.binomial(n_total, p_true))

    result = fit_psychometric(
        stimuli, n_correct, n_total, link="logit", fit_lapse=True
    )
    assert result.lapse == pytest.approx(lapse_true, abs=0.02)


@pytest.mark.unit
def test_raises_on_invalid_link() -> None:
    with pytest.raises(ValueError, match="link"):
        fit_psychometric([1.0, 2.0, 3.0], [1, 2, 3], [10, 10, 10], link="bad")  # type: ignore[arg-type]


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_psychometric([1.0, 2.0], [1, 2, 3], [10, 10, 10])


@pytest.mark.unit
def test_raises_on_invalid_counts() -> None:
    with pytest.raises(ValueError, match="n_correct"):
        fit_psychometric([1.0, 2.0, 3.0], [11, 5, 5], [10, 10, 10])
