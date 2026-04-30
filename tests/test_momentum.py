"""Tests for Nevin (1992) resistance-to-change estimation."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.matching import (
    SingleAlternativeMatchingParameters,
    fit_single_alternative_matching,
)
from session_analyzer.analytics.momentum import (
    MomentumParameters,
    estimate_resistance_to_change,
)


@pytest.mark.unit
def test_recovers_known_resistance_parameter() -> None:
    baseline = 60.0  # responses/min
    m_true = 5.0
    magnitudes = [1.0, 2.0, 3.0, 5.0, 10.0]
    disrupted = [baseline * 10.0 ** (-x / m_true) for x in magnitudes]

    result = estimate_resistance_to_change(baseline, disrupted, magnitudes)

    assert isinstance(result, MomentumParameters)
    assert result.m == pytest.approx(m_true, rel=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.n == 5


@pytest.mark.unit
def test_higher_m_means_more_resistance() -> None:
    baseline = 100.0
    magnitudes = [1.0, 2.0, 4.0]
    fragile = [baseline * 10.0 ** (-x / 1.0) for x in magnitudes]  # m=1
    robust = [baseline * 10.0 ** (-x / 20.0) for x in magnitudes]  # m=20

    f = estimate_resistance_to_change(baseline, fragile, magnitudes)
    r = estimate_resistance_to_change(baseline, robust, magnitudes)

    assert r.m > f.m


@pytest.mark.unit
def test_raises_on_non_positive_baseline() -> None:
    with pytest.raises(ValueError, match="baseline_rate"):
        estimate_resistance_to_change(0.0, [1.0, 2.0], [1.0, 2.0])


@pytest.mark.unit
def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_resistance_to_change(10.0, [5.0], [1.0, 2.0])


@pytest.mark.unit
def test_filters_non_positive_rates_and_zero_magnitudes() -> None:
    baseline = 10.0
    # Include invalid pairs that must be dropped.
    magnitudes = [0.0, 1.0, 2.0, 3.0]
    disrupted = [10.0, 5.0, 2.5, 1.25]

    result = estimate_resistance_to_change(baseline, disrupted, magnitudes)
    assert result.n == 3  # x=0 pair dropped


@pytest.mark.unit
def test_raises_on_insufficient_pairs() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        estimate_resistance_to_change(10.0, [5.0], [1.0])


@pytest.mark.unit
def test_infinite_m_when_behavior_does_not_decrease() -> None:
    baseline = 10.0
    magnitudes = [1.0, 2.0, 3.0]
    disrupted = [12.0, 15.0, 18.0]  # behavior increased under disruption

    result = estimate_resistance_to_change(baseline, disrupted, magnitudes)
    assert math.isinf(result.m)


@pytest.mark.unit
def test_single_alternative_recovers_known_parameters() -> None:
    k_true = 100.0
    r_e_true = 15.0
    rein_rates = [5.0, 15.0, 30.0, 60.0, 120.0, 240.0]
    resp_rates = [k_true * r / (r + r_e_true) for r in rein_rates]

    result = fit_single_alternative_matching(rein_rates, resp_rates)

    assert isinstance(result, SingleAlternativeMatchingParameters)
    assert result.k == pytest.approx(k_true, rel=1e-4)
    assert result.r_e == pytest.approx(r_e_true, rel=1e-4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.n == 6


@pytest.mark.unit
def test_single_alternative_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_single_alternative_matching([1.0, 2.0], [10.0])


@pytest.mark.unit
def test_single_alternative_raises_on_insufficient_data() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_single_alternative_matching([1.0, -1.0], [10.0, 20.0])
