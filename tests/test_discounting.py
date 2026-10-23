"""Tests for delay-discounting model fitting."""

from __future__ import annotations

import numpy as np
import pytest

from session_analyzer.analytics.discounting import (
    HyperbolicDiscountingParameters,
    HyperboloidDiscountingParameters,
    fit_hyperbolic_discounting,
    fit_hyperboloid_discounting,
)


@pytest.mark.unit
def test_recovers_known_hyperbolic_k() -> None:
    amount = 100.0
    k_true = 0.05
    delays = [0.0, 1.0, 2.0, 7.0, 30.0, 90.0, 180.0, 365.0]
    values = [amount / (1.0 + k_true * d) for d in delays]

    result = fit_hyperbolic_discounting(delays, values, amount=amount)

    assert isinstance(result, HyperbolicDiscountingParameters)
    assert result.k == pytest.approx(k_true, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.n == len(delays)


@pytest.mark.unit
def test_hyperbolic_fit_robust_to_noise() -> None:
    rng = np.random.default_rng(42)
    amount = 100.0
    k_true = 0.02
    delays = np.array([0.0, 1.0, 7.0, 30.0, 90.0, 180.0, 365.0])
    clean = amount / (1.0 + k_true * delays)
    noisy = clean + rng.normal(scale=1.5, size=clean.shape)

    result = fit_hyperbolic_discounting(delays.tolist(), noisy.tolist(), amount=amount)
    assert result.k == pytest.approx(k_true, rel=0.3)
    assert result.r_squared > 0.9


@pytest.mark.unit
def test_hyperbolic_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_hyperbolic_discounting([0.0, 1.0], [100.0, 90.0, 80.0], amount=100.0)


@pytest.mark.unit
def test_hyperbolic_raises_on_insufficient_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_hyperbolic_discounting([0.0], [100.0], amount=100.0)


@pytest.mark.unit
def test_hyperbolic_raises_on_non_positive_amount() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_hyperbolic_discounting([0.0, 1.0], [100.0, 90.0], amount=0.0)


@pytest.mark.unit
def test_recovers_known_hyperboloid_parameters() -> None:
    amount = 100.0
    k_true = 0.03
    s_true = 0.8
    delays = [0.1, 1.0, 7.0, 30.0, 90.0, 180.0, 365.0]
    values = [amount / (1.0 + k_true * d) ** s_true for d in delays]

    result = fit_hyperboloid_discounting(delays, values, amount=amount)

    assert isinstance(result, HyperboloidDiscountingParameters)
    assert result.k == pytest.approx(k_true, abs=1e-3)
    assert result.s == pytest.approx(s_true, abs=1e-3)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_hyperboloid_reduces_to_hyperbolic_when_s_is_one() -> None:
    amount = 50.0
    k_true = 0.1
    delays = [0.0, 2.0, 10.0, 50.0, 100.0]
    values = [amount / (1.0 + k_true * d) for d in delays]

    hyper = fit_hyperbolic_discounting(delays, values, amount=amount)
    mazur = fit_hyperboloid_discounting(delays, values, amount=amount)

    assert mazur.s == pytest.approx(1.0, abs=1e-2)
    assert mazur.k == pytest.approx(hyper.k, rel=1e-2)


# --- Reference implementation (discountingtools) ---------------------------
#
# Source: the discountingtools R package README (Gilroy; demo
# ``test_single_fits_recovery.R``). Data are generated with
# ``dd_discount_func_mazur(delays, lnk)`` = 1 / (1 + exp(lnk) * D)
# (maxValue = 1) on the delay grid 1, 30, 180, 540, 1080, 2160, 4320, 8640,
# with k drawn from N(0.07, 0.03) and Gaussian noise SD 0.025 added, then
# ``fit_dd_curves(plan = "mazur")`` recovers k. R's random streams are not
# reproducible here, so the check uses the same generator, grid and k
# distribution centre (mean and mean +/- 1 SD) without noise: recovery
# must then be exact (no published fitted values exist for this demo).

_DDTOOLS_DELAYS = [1.0, 30.0, 180.0, 540.0, 1080.0, 2160.0, 4320.0, 8640.0]


def _dd_discount_func_mazur(delays: list[float], lnk: float) -> list[float]:
    return [1.0 / (1.0 + np.exp(lnk) * d) for d in delays]


@pytest.mark.unit
@pytest.mark.parametrize("k_true", [0.04, 0.07, 0.10])
def test_recovers_k_on_discountingtools_grid(k_true: float) -> None:
    values = _dd_discount_func_mazur(_DDTOOLS_DELAYS, float(np.log(k_true)))
    result = fit_hyperbolic_discounting(_DDTOOLS_DELAYS, values, amount=1.0)
    assert result.k == pytest.approx(k_true, rel=1e-6)


@pytest.mark.unit
def test_hyperboloid_is_myerson_green_form_not_rachlin_form() -> None:
    """Definition check. The fitted hyperboloid is Myerson & Green (1995) /
    Green & Myerson (2004): V = A / (1 + kD)^s ("raising the denominator to a
    power", Myerson & Green 1995 abstract). Rachlin (2006) instead raises
    delay itself to a power, V = A / (1 + kD^s) (abstract: "kD^s"). Data
    from the Myerson-Green form are recovered exactly; data from the Rachlin
    form with s != 1 are not."""
    delays = [0.0, 1.0, 7.0, 30.0, 90.0, 180.0, 365.0, 730.0]
    mg = [100.0 / (1.0 + 0.02 * d) ** 0.6 for d in delays]
    fit_mg = fit_hyperboloid_discounting(delays, mg, amount=100.0)
    assert fit_mg.k == pytest.approx(0.02, rel=1e-4)
    assert fit_mg.s == pytest.approx(0.6, rel=1e-4)

    rachlin = [100.0 / (1.0 + 0.02 * d**0.6) for d in delays]
    fit_r = fit_hyperboloid_discounting(delays, rachlin, amount=100.0)
    assert fit_r.r_squared < 1.0 - 1e-6
