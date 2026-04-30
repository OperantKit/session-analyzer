"""Tests for log-survivor analysis of inter-response times.

The log-survivor function S(t) = P(IRT > t) plotted on a log scale is a
standard JEAB visualization (Shull, 1991; Shull et al., 2001) for
diagnosing whether responding follows a single Poisson process (linear
log-survivor) or has bout structure (bilinear / broken-line).
"""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.log_survivor import (
    LogSurvivorBilinearFit,
    LogSurvivorCurve,
    compute_log_survivor,
    fit_bilinear_log_survivor,
)


@pytest.mark.unit
def test_log_survivor_basic_shape() -> None:
    irts = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    curve = compute_log_survivor(irts)

    assert isinstance(curve, LogSurvivorCurve)
    # Times are sorted ascending
    assert list(curve.times) == sorted(curve.times)
    # Survival starts at 1 and ends at >= 1/n
    assert curve.survival[0] == pytest.approx(1.0)
    assert curve.survival[-1] >= 1.0 / len(irts) - 1e-9
    # Strictly non-increasing
    diffs = np.diff(curve.survival)
    assert np.all(diffs <= 1e-12)
    # log_survival aligns with survival
    assert curve.log_survival[0] == pytest.approx(0.0)
    assert np.allclose(np.exp(curve.log_survival), curve.survival, atol=1e-9)


@pytest.mark.unit
def test_log_survivor_filters_non_positive() -> None:
    irts = [0.0, -1.0, 0.5, 1.0, 2.0]
    curve = compute_log_survivor(irts)
    assert curve.n_irts == 3


@pytest.mark.unit
def test_log_survivor_returns_none_for_empty() -> None:
    assert compute_log_survivor([]) is None
    assert compute_log_survivor([0.0, -1.0]) is None


@pytest.mark.unit
def test_log_survivor_single_exponential_is_linear() -> None:
    """Pure Poisson IRTs yield a linear log-survivor with slope -rate."""
    rng = np.random.default_rng(42)
    rate = 2.0
    irts = rng.exponential(scale=1.0 / rate, size=2000).tolist()
    curve = compute_log_survivor(irts)
    assert curve is not None

    # Linear regression on the middle 80 % of the curve (avoid tail noise)
    n = len(curve.times)
    lo, hi = int(n * 0.05), int(n * 0.85)
    slope = np.polyfit(curve.times[lo:hi], curve.log_survival[lo:hi], 1)[0]
    assert slope == pytest.approx(-rate, rel=0.15)


@pytest.mark.unit
def test_bilinear_fit_recovers_breakpoint() -> None:
    """Two-component IRTs should produce a bilinear log-survivor with a
    knot near the regime boundary."""
    rng = np.random.default_rng(0)
    fast = rng.exponential(scale=0.2, size=600)  # within-bout, rate=5
    slow = rng.exponential(scale=5.0, size=200)  # between-bout, rate=0.2
    irts = np.concatenate([fast, slow]).tolist()

    fit = fit_bilinear_log_survivor(irts)
    assert isinstance(fit, LogSurvivorBilinearFit)
    # Knot should fall between the two regime time-scales
    assert 0.2 < fit.knot_time_sec < 5.0
    # First slope steeper than second
    assert fit.slope_fast < fit.slope_slow < 0.0
    # SSE finite
    assert np.isfinite(fit.sse)


@pytest.mark.unit
def test_bilinear_fit_returns_none_when_too_few() -> None:
    assert fit_bilinear_log_survivor([0.1, 0.2]) is None
