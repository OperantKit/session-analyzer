"""Tests for stimulus-control analyses (discrimination ratio + peak shift)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from session_analyzer.analytics.stimulus_control import (
    DiscriminationRatio,
    PeakShiftResult,
    compute_discrimination_ratio,
    detect_peak_shift,
)


@pytest.mark.unit
def test_discrimination_ratio_perfect() -> None:
    dr = compute_discrimination_ratio(s_plus_count=100, s_minus_count=0)
    assert isinstance(dr, DiscriminationRatio)
    assert dr.discrimination_ratio == pytest.approx(1.0)
    assert dr.suppression_ratio == pytest.approx(0.0)
    assert dr.is_discriminated


@pytest.mark.unit
def test_discrimination_ratio_chance() -> None:
    dr = compute_discrimination_ratio(s_plus_count=50, s_minus_count=50)
    assert dr.discrimination_ratio == pytest.approx(0.5)
    assert dr.is_discriminated is False


@pytest.mark.unit
def test_discrimination_ratio_suppression() -> None:
    # Suppression = SR(S-) / [SR(S-) + SR(S+)]
    dr = compute_discrimination_ratio(s_plus_count=80, s_minus_count=20)
    assert dr.suppression_ratio == pytest.approx(20 / 100)


@pytest.mark.unit
def test_discrimination_ratio_validates() -> None:
    with pytest.raises(ValueError):
        compute_discrimination_ratio(s_plus_count=-1, s_minus_count=10)
    with pytest.raises(ValueError):
        compute_discrimination_ratio(s_plus_count=0, s_minus_count=0)


@pytest.mark.unit
def test_peak_shift_classical_pattern() -> None:
    """Train S+=550, S-=540; peak shifts away from S- (toward higher values)."""
    s_plus = 550.0
    s_minus = 540.0
    # Empirical gradient with peak at 560 (shifted away from S-=540)
    stimulus_values = [520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0]
    sigma = 12.0
    peak_loc = 560.0
    rates = [50.0 * math.exp(-((v - peak_loc) ** 2) / (2 * sigma**2)) for v in stimulus_values]

    result = detect_peak_shift(
        stimulus_values=stimulus_values,
        response_rates=rates,
        s_plus=s_plus,
        s_minus=s_minus,
    )
    assert isinstance(result, PeakShiftResult)
    assert result.empirical_peak_value == pytest.approx(560.0, abs=1.0)
    assert result.fitted_peak_value == pytest.approx(560.0, abs=2.0)
    # Shift away from S- means in direction (S+ - S-) = +10, fitted peak > S+
    assert result.shift_magnitude > 0.0
    assert result.shift_direction == "away_from_s_minus"


@pytest.mark.unit
def test_peak_shift_no_shift_at_s_plus() -> None:
    """Peak at S+ -> magnitude = 0, direction = none."""
    s_plus = 550.0
    s_minus = 540.0
    stimulus_values = list(np.linspace(520, 580, 7))
    sigma = 15.0
    rates = [50.0 * math.exp(-((v - s_plus) ** 2) / (2 * sigma**2)) for v in stimulus_values]
    result = detect_peak_shift(
        stimulus_values=stimulus_values,
        response_rates=rates,
        s_plus=s_plus,
        s_minus=s_minus,
    )
    assert result.shift_magnitude == pytest.approx(0.0, abs=1.0)
    assert result.shift_direction == "none"


@pytest.mark.unit
def test_peak_shift_toward_s_minus_unusual() -> None:
    """Peak between S- and S+ -> classified as 'toward_s_minus' (atypical)."""
    s_plus = 550.0
    s_minus = 540.0
    stimulus_values = list(np.linspace(520, 580, 13))
    sigma = 12.0
    peak_loc = 545.0  # between S- and S+
    rates = [50.0 * math.exp(-((v - peak_loc) ** 2) / (2 * sigma**2)) for v in stimulus_values]
    result = detect_peak_shift(
        stimulus_values=stimulus_values,
        response_rates=rates,
        s_plus=s_plus,
        s_minus=s_minus,
    )
    assert result.shift_direction == "toward_s_minus"


@pytest.mark.unit
def test_peak_shift_validates_inputs() -> None:
    with pytest.raises(ValueError):
        detect_peak_shift(stimulus_values=[1.0], response_rates=[1.0, 2.0], s_plus=1.0, s_minus=0.0)
    with pytest.raises(ValueError):
        detect_peak_shift(
            stimulus_values=[1.0, 2.0, 3.0],
            response_rates=[1.0, 2.0, 3.0],
            s_plus=1.0,
            s_minus=1.0,
        )


# ---------------------------------------------------------------------------
# Hand computations (S-166). Annau & Kamin (1961) and Hanson (1959) are APA
# paywalled; their tables are not reproduced.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_suppression_ratio_in_cer_form() -> None:
    """CER suppression ratio B / (A + B), B = CS responses, A = pre-CS.

    Passing the pre-CS count as ``s_plus_count`` and the CS count as
    ``s_minus_count``: A = 40, B = 10 gives 10 / 50 = 0.2.
    """
    r = compute_discrimination_ratio(s_plus_count=40, s_minus_count=10)
    assert r.suppression_ratio == pytest.approx(0.2, abs=1e-12)
    assert r.discrimination_ratio == pytest.approx(0.8, abs=1e-12)
    # threshold 0.75 is inclusive
    assert compute_discrimination_ratio(s_plus_count=3, s_minus_count=1).is_discriminated


@pytest.mark.unit
def test_peak_shift_recovers_exact_gaussian_peak() -> None:
    """Noise-free gradient peaking at 545 with S+ = 550, S- = 570.

    The fitted peak is 545 (shift +5 away from S-, 25 % of |S+ - S-|).
    """
    x = np.arange(500.0, 601.0, 10.0)
    y = 80.0 * np.exp(-((x - 545.0) ** 2) / (2 * 15.0**2))
    r = detect_peak_shift(
        stimulus_values=list(x), response_rates=list(y), s_plus=550.0, s_minus=570.0
    )
    assert r.fitted_peak_value == pytest.approx(545.0, abs=1e-5)
    assert r.shift_magnitude == pytest.approx(5.0, abs=1e-5)
    assert r.shift_direction == "away_from_s_minus"
    assert r.fitted_sigma == pytest.approx(15.0, abs=1e-5)
