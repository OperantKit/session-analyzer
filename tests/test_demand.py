"""Tests for exponential demand-curve fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.demand import (
    DemandParameters,
    compute_pmax,
    fit_exponential_demand,
)


def _synth(prices: list[float], q0: float, alpha: float, k: float) -> list[float]:
    """Generate consumption via Hursh & Silberberg (2008):
    log10 Q = log10 Q0 + k * (exp(-alpha * Q0 * C) - 1).
    """
    log_q = np.log10(q0) + k * (np.exp(-alpha * q0 * np.array(prices)) - 1.0)
    return (10.0**log_q).tolist()


@pytest.mark.unit
def test_recovers_known_parameters_with_k_fixed() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    q0_true = 50.0
    alpha_true = 0.005
    k_fixed = 3.0

    consumption = _synth(prices, q0_true, alpha_true, k_fixed)
    result = fit_exponential_demand(prices, consumption, k=k_fixed)

    assert isinstance(result, DemandParameters)
    assert result.q0 == pytest.approx(q0_true, rel=1e-3)
    assert result.alpha == pytest.approx(alpha_true, rel=1e-3)
    assert result.k == pytest.approx(k_fixed, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_recovers_known_parameters_with_k_estimated() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    q0_true = 30.0
    alpha_true = 0.01
    k_true = 2.5

    consumption = _synth(prices, q0_true, alpha_true, k_true)
    result = fit_exponential_demand(prices, consumption, k=None)

    assert result.q0 == pytest.approx(q0_true, rel=1e-2)
    assert result.alpha == pytest.approx(alpha_true, rel=1e-2)
    assert result.k == pytest.approx(k_true, rel=1e-2)
    assert result.r_squared > 0.99


@pytest.mark.unit
def test_pmax_and_omax_are_consistent() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    q0 = 100.0
    alpha = 0.002
    k = 3.0
    consumption = _synth(prices, q0, alpha, k)
    result = fit_exponential_demand(prices, consumption, k=k)

    # P_max: where d(Q*P)/dP = 0. Closed form: solution of
    # alpha*Q0*P * ln(10)*k * exp(-alpha*Q0*P) = 1  (Hursh & Roma 2013).
    # Validate numerically: expenditure at P_max must exceed neighbours.
    p_max = result.p_max
    assert p_max > 0.0
    q_at_pmax = q0 * 10 ** (k * (math.exp(-alpha * q0 * p_max) - 1.0))
    o_at_pmax = p_max * q_at_pmax
    assert result.o_max == pytest.approx(o_at_pmax, rel=1e-3)

    # Expenditure is unimodal in P; values on both sides should be lower.
    for factor in (0.5, 2.0):
        p = p_max * factor
        q = q0 * 10 ** (k * (math.exp(-alpha * q0 * p) - 1.0))
        assert p * q < o_at_pmax + 1e-6


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_exponential_demand([1.0, 2.0], [10.0])


@pytest.mark.unit
def test_raises_on_insufficient_data() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_exponential_demand([1.0, 2.0], [10.0, 5.0])


@pytest.mark.unit
def test_raises_on_non_positive_prices_or_consumption() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_exponential_demand([0.0, 1.0, 2.0], [10.0, 5.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        fit_exponential_demand([1.0, 2.0, 3.0], [10.0, 0.0, 2.0])


# --- Published reference values (beezdemand) -------------------------------
#
# Source: the beezdemand R package README (Kaplan, Gilroy, Reed, Koffarnus,
# & Hursh, 2019; package README as distributed on CRAN/GitHub), example
# dataset ``apt`` (alcohol purchase task), subject id 19. The README lists
# this subject's 16 (price, consumption) rows and the output of
# ``fit_demand_fixed(apt, equation = "hs")`` (k fixed at 2):
#     Q0 = 10.158665, alpha = 0.00205 (printed to 3 significant figures).
# The "hs" equation is Equation 2 of Kaplan et al. (2019): nonlinear least
# squares on log10(Q), with the free price (C = 0) included.

_APT_ID19_PRICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0,
                    8.0, 9.0, 10.0, 15.0, 20.0]
_APT_ID19_CONSUMPTION = [10.0, 10.0, 10.0, 8.0, 8.0, 8.0, 7.0, 7.0, 7.0,
                         6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0]


@pytest.mark.unit
@pytest.mark.reference(
    module="demand",
    verdict="published",
    source="Hursh & Silberberg (2008); Kaplan et al. (2019)",
)
def test_log10_fit_matches_beezdemand_hs_output_for_apt_id19() -> None:
    """beezdemand ``fit_demand_fixed(apt, equation="hs")``, id 19, k = 2.

    Expected: Q0 = 10.158665 (README prints 6 decimals; tolerance 1e-5
    relative), alpha = 0.00205 (README prints 3 significant figures;
    tolerance = half a unit in the last printed digit, 5e-6 absolute).
    """
    result = fit_exponential_demand(
        _APT_ID19_PRICES, _APT_ID19_CONSUMPTION, k=2.0, fit_space="log10"
    )
    assert result.q0 == pytest.approx(10.158665, rel=1e-5)
    assert result.alpha == pytest.approx(0.00205, abs=5e-6)


@pytest.mark.unit
def test_linear_fit_is_a_different_estimator_from_beezdemand_hs() -> None:
    """The default (least squares on linear Q) does not reproduce the
    beezdemand "hs" value; it is a different estimator (Equation 3 error
    structure applied to the Hursh & Silberberg curve). Pinned so that the
    default cannot silently change."""
    prices = _APT_ID19_PRICES[1:]  # linear space keeps the strict-positive rule
    consumption = _APT_ID19_CONSUMPTION[1:]
    result = fit_exponential_demand(prices, consumption, k=2.0)
    assert abs(result.q0 - 10.158665) / 10.158665 > 1e-3


@pytest.mark.unit
def test_log10_fit_accepts_free_price() -> None:
    result = fit_exponential_demand([0.0, 1.0, 5.0, 10.0], [10.0, 9.0, 5.0, 2.0],
                                    k=2.0, fit_space="log10")
    assert result.q0 > 0.0


@pytest.mark.unit
def test_rejects_unknown_fit_space() -> None:
    with pytest.raises(ValueError, match="fit_space"):
        fit_exponential_demand([1.0, 2.0, 3.0], [10.0, 5.0, 2.0],
                               fit_space="ln")  # type: ignore[arg-type]


# Source: beezdemand README, ``ExtraF(dat = apt, equation = "koff", k = 2,
# groupcol = "group")``, shared-alpha row for group "a":
#     Q0d = 8.489634, Alpha = 0.0040198, K = 2,
#     Pmaxd = 8.453799, Omaxd = 22.63159, Omaxa = 22.63190.
# Pmaxd is beezdemand's derived approximation
#     P_max ~= (0.083 k + 0.65) / (Q0 alpha k^1.5)
# (attributed in Kaplan et al., 2019 to Hursh, 2014 and Hursh & Roma, 2013;
# the original texts were not retrieved, so the formula is pinned against
# the beezdemand output, not against the papers). Omaxd = Pmaxd * Q(Pmaxd);
# Omaxa is the true maximum of expenditure.
_KOFF_Q0, _KOFF_ALPHA, _KOFF_K = 8.489634, 0.0040198, 2.0


@pytest.mark.unit
def test_approximate_pmax_matches_beezdemand_pmaxd() -> None:
    """Tolerance 1e-6 relative: the inputs are printed to 7 and 5
    significant figures, which bounds the reproducible precision."""
    p_max, o_max = compute_pmax(_KOFF_Q0, _KOFF_ALPHA, _KOFF_K, method="approximate")
    assert p_max == pytest.approx(8.453799, rel=1e-6)
    assert o_max == pytest.approx(22.63159, rel=1e-6)


@pytest.mark.unit
def test_exact_pmax_reaches_beezdemand_omaxa() -> None:
    """Exact P_max solves u k ln(10) exp(-u) = 1 for u = alpha Q0 P on the
    principal Lambert-W branch: P_max = -W0(-1 / (k ln 10)) / (alpha Q0).
    Its expenditure is the maximum, matching beezdemand Omaxa = 22.63190."""
    p_max, o_max = compute_pmax(_KOFF_Q0, _KOFF_ALPHA, _KOFF_K, method="exact")
    from scipy.special import lambertw

    u = -lambertw(-1.0 / (_KOFF_K * math.log(10.0)), 0).real
    assert p_max == pytest.approx(u / (_KOFF_ALPHA * _KOFF_Q0), rel=1e-8)
    assert o_max == pytest.approx(22.63190, rel=1e-6)


@pytest.mark.unit
def test_fit_reports_requested_pmax_definition() -> None:
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    consumption = _synth(prices, 50.0, 0.005, 3.0)
    default = fit_exponential_demand(prices, consumption, k=3.0)
    approx = fit_exponential_demand(prices, consumption, k=3.0, pmax_method="approximate")
    expected_approx, _ = compute_pmax(default.q0, default.alpha, 3.0, method="approximate")
    assert approx.p_max == pytest.approx(expected_approx, rel=1e-9)
    exact, _ = compute_pmax(default.q0, default.alpha, 3.0, method="exact")
    assert default.p_max == pytest.approx(exact, rel=1e-9)


@pytest.mark.unit
def test_compute_pmax_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        compute_pmax(1.0, 0.01, 2.0, method="lambert")  # type: ignore[arg-type]
