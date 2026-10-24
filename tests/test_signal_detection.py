"""Tests for signal detection theory (SDT) analysis."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.signal_detection import (
    ROCCurve,
    SignalDetectionMetrics,
    compute_roc,
    compute_sdt,
)


@pytest.mark.unit
def test_sdt_unbiased_observer_zero_criterion() -> None:
    # H = 0.84, F = 0.16  -> d' ~ 2.0, c ~ 0
    metrics = compute_sdt(hits=84, misses=16, false_alarms=16, correct_rejections=84)
    assert isinstance(metrics, SignalDetectionMetrics)
    # log-linear correction shifts (84.5/101, 16.5/101) ~ (0.837, 0.163)
    assert metrics.hit_rate == pytest.approx(0.837, abs=5e-3)
    assert metrics.false_alarm_rate == pytest.approx(0.163, abs=5e-3)
    assert metrics.d_prime == pytest.approx(2.0, abs=0.05)
    assert metrics.criterion == pytest.approx(0.0, abs=0.05)
    assert metrics.beta == pytest.approx(1.0, abs=0.1)


@pytest.mark.unit
def test_sdt_chance_performance() -> None:
    metrics = compute_sdt(hits=50, misses=50, false_alarms=50, correct_rejections=50)
    assert metrics.d_prime == pytest.approx(0.0, abs=1e-9)
    assert metrics.a_prime == pytest.approx(0.5, abs=1e-9)


@pytest.mark.unit
def test_sdt_log_linear_correction_ceiling() -> None:
    """Hit rate of 1.0 should be corrected via log-linear (Hautus 1995)."""
    metrics = compute_sdt(hits=100, misses=0, false_alarms=0, correct_rejections=100)
    # Both rates corrected toward (n+0.5)/(n+1); finite d_prime
    assert math.isfinite(metrics.d_prime)
    assert metrics.d_prime > 3.0
    assert metrics.hit_rate < 1.0
    assert metrics.false_alarm_rate > 0.0


@pytest.mark.unit
def test_sdt_conservative_bias_positive_c() -> None:
    # H low, F low -> conservative (high criterion, c > 0)
    metrics = compute_sdt(hits=30, misses=70, false_alarms=5, correct_rejections=95)
    assert metrics.criterion > 0.0
    assert metrics.beta > 1.0


@pytest.mark.unit
def test_sdt_liberal_bias_negative_c() -> None:
    metrics = compute_sdt(hits=95, misses=5, false_alarms=70, correct_rejections=30)
    assert metrics.criterion < 0.0
    assert metrics.beta < 1.0


@pytest.mark.unit
def test_sdt_validates_inputs() -> None:
    with pytest.raises(ValueError):
        compute_sdt(hits=0, misses=0, false_alarms=0, correct_rejections=10)
    with pytest.raises(ValueError):
        compute_sdt(hits=10, misses=0, false_alarms=0, correct_rejections=0)
    with pytest.raises(ValueError):
        compute_sdt(hits=-1, misses=0, false_alarms=0, correct_rejections=10)


@pytest.mark.unit
def test_roc_monotonic() -> None:
    rates = [
        (0.05, 0.01),
        (0.30, 0.10),
        (0.60, 0.30),
        (0.85, 0.55),
        (0.99, 0.90),
    ]
    roc = compute_roc(rates)
    assert isinstance(roc, ROCCurve)
    # Includes endpoints (0,0) and (1,1)
    assert roc.fa_rates[0] == 0.0
    assert roc.hit_rates[0] == 0.0
    assert roc.fa_rates[-1] == 1.0
    assert roc.hit_rates[-1] == 1.0
    # Monotonic non-decreasing in FA
    for i in range(1, len(roc.fa_rates)):
        assert roc.fa_rates[i] >= roc.fa_rates[i - 1]
    # AUC > 0.5 for above-chance points
    assert roc.auc > 0.5


@pytest.mark.unit
def test_roc_chance_diagonal() -> None:
    rates = [(p, p) for p in (0.2, 0.5, 0.8)]
    roc = compute_roc(rates)
    assert roc.auc == pytest.approx(0.5, abs=0.05)


@pytest.mark.unit
def test_roc_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_roc([])


# ---------------------------------------------------------------------------
# Published values (S-166).
#
# Source: Stanislaw, H., & Todorov, N. (1999). Calculation of signal
# detection theory measures. Behavior Research Methods, Instruments, &
# Computers, 31(1), 137-149. https://doi.org/10.3758/BF03207704
# Figure 1: equal-variance normal noise N(0, 1) and signal N(2, 1), criterion
# 0.5 SD above the noise mean, so H = Phi(1.5) and F = 1 - Phi(0.5). The text
# reports d' = 2.00, c = -0.50, ln(beta) = -1.00, A' = .89, Grier's B'' = -0.55
# (two decimals). A' follows Eq. 2 and B'' follows Eq. 9,
# B'' = sign(H - F) [H(1-H) - F(1-F)] / [H(1-H) + F(1-F)].
#
# compute_sdt always applies the Hautus (1995) log-linear correction, so the
# figure's rates are entered as counts out of 10^6 trials per stimulus; the
# correction then moves each rate by less than 1e-6.
# ---------------------------------------------------------------------------

_N = 1_000_000


def _counts(h: float, f: float) -> dict[str, int]:
    hits = round(h * _N)
    fas = round(f * _N)
    return {
        "hits": hits,
        "misses": _N - hits,
        "false_alarms": fas,
        "correct_rejections": _N - fas,
    }


@pytest.mark.unit
@pytest.mark.reference(
    module="signal_detection",
    verdict="published",
    source="Stanislaw & Todorov (1999)",
)
def test_matches_stanislaw_todorov_1999_figure_1() -> None:
    """Figure 1 values, tolerance half a unit in the second decimal."""
    from scipy.stats import norm

    h = float(norm.cdf(1.5))
    f = float(1.0 - norm.cdf(0.5))
    m = compute_sdt(**_counts(h, f))
    assert m.d_prime == pytest.approx(2.00, abs=0.005)
    assert m.criterion == pytest.approx(-0.50, abs=0.005)
    assert math.log(m.beta) == pytest.approx(-1.00, abs=0.005)
    assert m.a_prime == pytest.approx(0.89, abs=0.005)
    assert m.b_double_prime == pytest.approx(-0.55, abs=0.005)


@pytest.mark.unit
def test_b_double_prime_keeps_sign_below_chance() -> None:
    """Eq. 9 multiplies by sign(H - F), so swapping H and F leaves B'' as is.

    Swapping negates both sign(H - F) and the numerator. Stanislaw &
    Todorov (1999) note that applying the H >= F formula when H < F gives
    the correct magnitude with the wrong sign.
    """
    from scipy.stats import norm

    h = float(norm.cdf(1.5))
    f = float(1.0 - norm.cdf(0.5))
    above = compute_sdt(**_counts(h, f))
    below = compute_sdt(**_counts(f, h))
    assert below.b_double_prime == pytest.approx(
        above.b_double_prime, abs=1e-6
    )
    # A' for the swapped pair follows the H < F branch of Eq. 2.
    assert below.a_prime == pytest.approx(1.0 - above.a_prime, abs=1e-6)


@pytest.mark.unit
def test_log_linear_correction_adds_half_to_each_cell() -> None:
    """Hautus (1995): each cell frequency is increased by 0.5.

    Hautus, M. J. (1995). Behavior Research Methods, Instruments, &
    Computers, 27(1), 46-51. https://doi.org/10.3758/BF03203619 (abstract:
    "each cell frequency in the contingency table is increased by 0.5").
    """
    m = compute_sdt(hits=10, misses=0, false_alarms=0, correct_rejections=20)
    assert m.hit_rate == pytest.approx(round(10.5 / 11, 6), abs=1e-12)
    assert m.false_alarm_rate == pytest.approx(round(0.5 / 21, 6), abs=1e-12)
    from scipy.stats import norm

    assert m.d_prime == pytest.approx(
        norm.ppf(10.5 / 11) - norm.ppf(0.5 / 21), abs=1e-6
    )
