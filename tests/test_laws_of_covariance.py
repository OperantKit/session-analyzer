"""Tests for laws-of-covariance resurgence prediction (after Podlesnik & Baum, 2024).
"""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.laws_of_covariance import (
    LawsOfCovariancePrediction,
    predict_resurgence_laws_of_covariance,
)


class TestPredictLawsOfCovariance:
    def test_full_withdrawal_predicts_full_recovery(self) -> None:
        """If alternative reinforcement is fully withdrawn, target recovery
        equals the baseline rate scaled by the covariance coefficient."""
        result = predict_resurgence_laws_of_covariance(
            baseline_target_rate=100.0,
            alt_baseline_rate=80.0,
            alt_test_rate=0.0,
            covariance_coefficient=1.0,
        )
        assert isinstance(result, LawsOfCovariancePrediction)
        assert math.isclose(result.predicted_target_rate, 100.0, abs_tol=1e-9)
        assert math.isclose(result.fraction_alt_withdrawn, 1.0, abs_tol=1e-9)

    def test_no_withdrawal_predicts_no_recovery(self) -> None:
        """If alternative reinforcement is unchanged, no target recovery."""
        result = predict_resurgence_laws_of_covariance(
            baseline_target_rate=100.0,
            alt_baseline_rate=80.0,
            alt_test_rate=80.0,
            covariance_coefficient=1.0,
        )
        assert math.isclose(result.predicted_target_rate, 0.0, abs_tol=1e-9)
        assert math.isclose(result.fraction_alt_withdrawn, 0.0, abs_tol=1e-9)

    def test_half_withdrawal_predicts_proportional_recovery(self) -> None:
        result = predict_resurgence_laws_of_covariance(
            baseline_target_rate=100.0,
            alt_baseline_rate=80.0,
            alt_test_rate=40.0,
            covariance_coefficient=1.0,
        )
        assert math.isclose(result.predicted_target_rate, 50.0, abs_tol=1e-9)
        assert math.isclose(result.fraction_alt_withdrawn, 0.5, abs_tol=1e-9)

    def test_covariance_coefficient_scales_prediction(self) -> None:
        full = predict_resurgence_laws_of_covariance(
            baseline_target_rate=100.0,
            alt_baseline_rate=80.0,
            alt_test_rate=0.0,
            covariance_coefficient=0.6,
        )
        assert math.isclose(full.predicted_target_rate, 60.0, abs_tol=1e-9)

    def test_alt_baseline_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            predict_resurgence_laws_of_covariance(
                baseline_target_rate=100.0,
                alt_baseline_rate=0.0,
                alt_test_rate=0.0,
            )

    def test_baseline_target_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            predict_resurgence_laws_of_covariance(
                baseline_target_rate=-1.0,
                alt_baseline_rate=10.0,
                alt_test_rate=0.0,
            )

    def test_alt_test_exceeds_baseline_raises(self) -> None:
        """Alt-test above baseline implies *more* alt reinforcement than the
        DRA phase, which is outside the resurgence paradigm."""
        with pytest.raises(ValueError):
            predict_resurgence_laws_of_covariance(
                baseline_target_rate=100.0,
                alt_baseline_rate=80.0,
                alt_test_rate=100.0,
            )

    def test_covariance_coefficient_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            predict_resurgence_laws_of_covariance(
                baseline_target_rate=100.0,
                alt_baseline_rate=80.0,
                alt_test_rate=0.0,
                covariance_coefficient=-0.1,
            )


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------
#
# Podlesnik & Baum (2024) discuss resurgence with the laws of allocation,
# induction, and covariance qualitatively (abstract); the point prediction
# B_test = c * B_baseline * (1 - R_alt_test / R_alt_baseline) is this module's
# own formula, so it is checked as a closed form only.


@pytest.mark.unit
@pytest.mark.reference(
    module="laws_of_covariance",
    verdict="properties",
    source="Podlesnik & Baum (2024)",
)
def test_partial_withdrawal_by_hand() -> None:
    result = predict_resurgence_laws_of_covariance(
        baseline_target_rate=40.0,
        alt_baseline_rate=2.0,
        alt_test_rate=0.5,
        covariance_coefficient=0.8,
    )
    assert result.fraction_alt_withdrawn == pytest.approx(0.75, rel=1e-12)
    assert result.predicted_target_rate == pytest.approx(24.0, rel=1e-12)
