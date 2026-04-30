"""Tests for operant-variability (U-value) analysis."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.variability import (
    VariabilityMetrics,
    compute_u_value,
)


@pytest.mark.unit
def test_single_sequence_repeated_yields_u_value_zero() -> None:
    sequences = ["LLLL"] * 16
    result = compute_u_value(sequences, alphabet_size=2, sequence_length=4)
    assert isinstance(result, VariabilityMetrics)
    assert result.u_value == 0.0
    assert result.entropy_bits == 0.0
    assert result.n_unique == 1
    assert result.n_possible == 16


@pytest.mark.unit
def test_uniform_distribution_yields_u_value_one() -> None:
    all_sequences = [
        "LLLL", "LLLR", "LLRL", "LLRR",
        "LRLL", "LRLR", "LRRL", "LRRR",
        "RLLL", "RLLR", "RLRL", "RLRR",
        "RRLL", "RRLR", "RRRL", "RRRR",
    ]
    result = compute_u_value(all_sequences, alphabet_size=2, sequence_length=4)
    assert result.u_value == pytest.approx(1.0)
    assert result.entropy_bits == pytest.approx(4.0)
    assert result.n_unique == 16


@pytest.mark.unit
def test_partial_variability_between_zero_and_one() -> None:
    # Two sequences, each occurring 4 times: H = 1 bit; Hmax = 4 bits -> U = 0.25.
    sequences = ["LLLL"] * 4 + ["RRRR"] * 4
    result = compute_u_value(sequences, alphabet_size=2, sequence_length=4)
    assert result.entropy_bits == pytest.approx(1.0)
    assert result.u_value == pytest.approx(0.25)


@pytest.mark.unit
def test_accepts_tuple_sequences() -> None:
    sequences = [(0, 1), (1, 0), (0, 1), (1, 1)]
    result = compute_u_value(sequences, alphabet_size=2, sequence_length=2)
    assert result.n_sequences == 4
    assert result.n_unique == 3
    # Probabilities (2/4, 1/4, 1/4) -> H = 1.5, Hmax = 2 -> U = 0.75.
    assert result.u_value == pytest.approx(0.75)


@pytest.mark.unit
def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_u_value([], alphabet_size=2, sequence_length=4)


@pytest.mark.unit
def test_raises_on_invalid_alphabet() -> None:
    with pytest.raises(ValueError, match="alphabet_size"):
        compute_u_value(["LL"], alphabet_size=1, sequence_length=2)


@pytest.mark.unit
def test_raises_on_mismatched_sequence_length() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_u_value(["LL", "LLL"], alphabet_size=2, sequence_length=2)


@pytest.mark.unit
def test_entropy_equals_log_of_unique_when_uniform() -> None:
    sequences = ["A", "B", "C", "D"]
    result = compute_u_value(sequences, alphabet_size=4, sequence_length=1)
    assert result.entropy_bits == pytest.approx(math.log2(4))
    assert result.u_value == pytest.approx(1.0)
