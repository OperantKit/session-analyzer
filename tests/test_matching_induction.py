"""Tests for the matching-with-induction fit (Baum & Grace, 2020)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.matching_induction import (
    MatchingInductionParameters,
    fit_matching_with_induction,
    predict_log_ratio_with_induction,
)


class TestPredictLogRatioWithInduction:
    def test_strict_matching_with_zero_induction(self) -> None:
        """w = 0 reduces to the strict generalized matching law."""
        log_r_ratio = np.array([math.log(2.0), math.log(0.5)])
        r1 = np.array([2.0, 1.0])
        r2 = np.array([1.0, 2.0])
        out = predict_log_ratio_with_induction(
            log_r_ratio, r1, r2, a=1.0, log_b=0.0, w=0.0,
        )
        # under strict matching log(B1/B2) == log(R1/R2)
        assert math.isclose(out[0], math.log(2.0), abs_tol=1e-9)
        assert math.isclose(out[1], math.log(0.5), abs_tol=1e-9)

    def test_induction_amplifies_richer_alternative(self) -> None:
        """w > 0 increases the predicted ratio on the richer side."""
        log_r_ratio = np.array([math.log(3.0)])
        r1 = np.array([3.0])
        r2 = np.array([1.0])
        strict = predict_log_ratio_with_induction(
            log_r_ratio, r1, r2, a=1.0, log_b=0.0, w=0.0,
        )
        induced = predict_log_ratio_with_induction(
            log_r_ratio, r1, r2, a=1.0, log_b=0.0, w=0.5,
        )
        assert induced[0] > strict[0]


class TestFitMatchingWithInduction:
    def test_perfect_matching_recovers_a_1_w_0(self) -> None:
        """Data generated under strict matching recovers a=1, log_b=0, w=0."""
        r1 = [4.0, 3.0, 2.0, 1.0, 0.5]
        r2 = [0.5, 1.0, 2.0, 3.0, 4.0]
        responses_ratios = [a / b for a, b in zip(r1, r2, strict=False)]
        reinforcement_ratios = responses_ratios[:]

        result = fit_matching_with_induction(
            responses_ratios=responses_ratios,
            reinforcement_ratios=reinforcement_ratios,
            reinforcement_rates_1=r1,
            reinforcement_rates_2=r2,
        )
        assert isinstance(result, MatchingInductionParameters)
        assert math.isclose(result.sensitivity, 1.0, abs_tol=1e-3)
        assert math.isclose(result.bias, 0.0, abs_tol=1e-3)
        assert math.isclose(result.induction_weight, 0.0, abs_tol=1e-3)
        assert result.r_squared > 0.99

    def test_recovers_induction_when_present(self) -> None:
        """Synthetic data with w=0.4 recovers w close to 0.4."""
        r1_list = [4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5]
        r2_list = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
        a_true = 1.0
        log_b_true = 0.0
        w_true = 0.4
        log_r = np.log(np.array(r1_list) / np.array(r2_list))
        r1 = np.array(r1_list)
        r2 = np.array(r2_list)
        log_b = predict_log_ratio_with_induction(
            log_r, r1, r2, a=a_true, log_b=log_b_true, w=w_true,
        )
        responses_ratios = [float(math.exp(v)) for v in log_b]
        reinforcement_ratios = [float(r1_list[i] / r2_list[i]) for i in range(len(r1_list))]

        result = fit_matching_with_induction(
            responses_ratios=responses_ratios,
            reinforcement_ratios=reinforcement_ratios,
            reinforcement_rates_1=r1_list,
            reinforcement_rates_2=r2_list,
        )
        assert math.isclose(result.sensitivity, a_true, abs_tol=0.05)
        assert math.isclose(result.bias, log_b_true, abs_tol=0.05)
        assert math.isclose(result.induction_weight, w_true, abs_tol=0.05)

    def test_input_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_matching_with_induction(
                responses_ratios=[1.0, 2.0],
                reinforcement_ratios=[1.0],  # mismatched
                reinforcement_rates_1=[1.0, 2.0],
                reinforcement_rates_2=[1.0, 1.0],
            )

    def test_too_few_points_raises(self) -> None:
        """Fitting 3 parameters requires at least 3 data points."""
        with pytest.raises(ValueError):
            fit_matching_with_induction(
                responses_ratios=[1.0, 2.0],
                reinforcement_ratios=[1.0, 2.0],
                reinforcement_rates_1=[1.0, 2.0],
                reinforcement_rates_2=[1.0, 1.0],
            )


# ---------------------------------------------------------------------------
# Closed forms and recovery
# ---------------------------------------------------------------------------
#
# The induction term (R1 - R2) / (R1 + R2) equals (r - 1) / (r + 1) =
# tanh(ln(r) / 2) with r = R1 / R2, so it depends on the reinforcement ratio
# only, not on the absolute rates. The three parameters are identifiable only
# through the curvature of tanh(x / 2) against x, which needs a wide range of
# ratios. Baum & Grace (2020) give no values for this parameterization.


@pytest.mark.unit
def test_induction_term_depends_on_ratio_only() -> None:
    ratios = np.array([0.25, 0.5, 2.0, 8.0])
    log_r = np.log(ratios)
    slow = predict_log_ratio_with_induction(
        log_r, ratios * 1.0, np.ones(4), a=0.9, log_b=0.1, w=0.4
    )
    fast = predict_log_ratio_with_induction(
        log_r, ratios * 30.0, np.full(4, 30.0), a=0.9, log_b=0.1, w=0.4
    )
    assert slow == pytest.approx(fast, rel=1e-12)
    assert slow == pytest.approx(0.9 * log_r + 0.1 + 0.4 * np.tanh(log_r / 2.0), rel=1e-12)


@pytest.mark.unit
def test_recovers_known_parameters_over_wide_ratio_range() -> None:
    ratios = [1 / 27, 1 / 9, 1 / 3, 1.0, 3.0, 9.0, 27.0]
    r2 = [10.0] * len(ratios)
    r1 = [r * 10.0 for r in ratios]
    b = np.exp(
        predict_log_ratio_with_induction(
            np.log(ratios), np.array(r1), np.array(r2), a=0.8, log_b=0.15, w=0.5
        )
    )
    fit = fit_matching_with_induction(
        responses_ratios=list(b),
        reinforcement_ratios=ratios,
        reinforcement_rates_1=r1,
        reinforcement_rates_2=r2,
    )
    assert fit.sensitivity == pytest.approx(0.8, abs=1e-6)
    assert fit.bias == pytest.approx(0.15, abs=1e-6)
    assert fit.induction_weight == pytest.approx(0.5, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
