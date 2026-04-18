"""Tests for bout-decomposition EM algorithm.

Strategy: synthesize IRTs from a known 2-component log-normal mixture and
verify ``analyze_bouts`` recovers the parameters within tolerance. Also
exercises edge cases (insufficient data, all-zero IRTs).
"""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.bout import BoutMetrics, _em_2_lognormal, analyze_bouts


@pytest.mark.unit
def test_returns_none_when_too_few_irts() -> None:
    assert analyze_bouts([0.5] * 9) is None


@pytest.mark.unit
def test_returns_none_when_all_irts_non_positive() -> None:
    assert analyze_bouts([0.0, -1.0, -0.5] * 5) is None


@pytest.mark.unit
def test_filters_non_positive_irts() -> None:
    rng = np.random.default_rng(42)
    valid = rng.lognormal(mean=-1.0, sigma=0.3, size=20).tolist()
    mixed = [*valid, 0.0, -1.0, -2.5]

    result = analyze_bouts(mixed)
    assert result is not None
    assert result.n_irts == len(valid)


@pytest.mark.unit
def test_recovers_known_mixture_parameters() -> None:
    """EM should recover means of a well-separated 2-component mixture."""
    rng = np.random.default_rng(0)
    # Within-bout: short IRTs ~ exp(-1.0) = 0.37 s
    within = rng.lognormal(mean=-1.0, sigma=0.3, size=600)
    # Inter-bout: long IRTs ~ exp(2.0) = 7.4 s
    inter = rng.lognormal(mean=2.0, sigma=0.5, size=200)
    irts = np.concatenate([within, inter]).tolist()

    result = analyze_bouts(irts)
    assert result is not None
    assert isinstance(result, BoutMetrics)
    assert result.converged

    # within_bout_irt_sec ~ exp(-1.0) ~= 0.37
    assert result.within_bout_irt_sec == pytest.approx(np.exp(-1.0), rel=0.15)
    # mean_inter_bout_interval_sec ~ exp(2.0) ~= 7.39
    assert result.mean_inter_bout_interval_sec == pytest.approx(np.exp(2.0), rel=0.20)
    # within_bout_rate ~ 1/0.37 ~= 2.7 resp/sec
    assert result.within_bout_rate == pytest.approx(1.0 / np.exp(-1.0), rel=0.15)
    # bout_initiation_rate ~ 60/7.4 ~= 8.1 bouts/min
    assert result.bout_initiation_rate == pytest.approx(60.0 / np.exp(2.0), rel=0.20)
    # pi_within reflects the 600/800 = 0.75 mixing weight
    assert 0.5 <= result.pi_within <= 0.95


@pytest.mark.unit
def test_em_short_component_is_smaller_than_long() -> None:
    """Internal invariant: component 1 must be the short (within-bout) one."""
    rng = np.random.default_rng(7)
    within = rng.lognormal(mean=-1.0, sigma=0.3, size=400)
    inter = rng.lognormal(mean=2.0, sigma=0.5, size=150)
    log_irts = np.log(np.concatenate([within, inter]))

    mu1, sigma1, mu2, sigma2, pi1, converged = _em_2_lognormal(log_irts)
    assert mu1 < mu2
    assert sigma1 > 0.0
    assert sigma2 > 0.0
    assert 0.0 < pi1 < 1.0
    assert converged is True


@pytest.mark.unit
def test_metrics_are_finite_and_rounded() -> None:
    rng = np.random.default_rng(123)
    irts = rng.lognormal(mean=0.0, sigma=1.0, size=100).tolist()

    result = analyze_bouts(irts)
    assert result is not None
    for value in (
        result.bout_initiation_rate,
        result.within_bout_rate,
        result.mean_inter_bout_interval_sec,
        result.within_bout_irt_sec,
    ):
        assert np.isfinite(value)
    assert 0.0 <= result.pi_within <= 1.0
