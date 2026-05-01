"""Tests for n-gram entropy and Markov-order estimation."""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.ngram_entropy import (
    BlockEntropy,
    ConditionalEntropy,
    MarkovOrderEstimate,
    compute_block_entropy,
    compute_conditional_entropy,
    estimate_markov_order,
)


@pytest.mark.unit
def test_block_entropy_uniform_alphabet() -> None:
    # 4-symbol uniform sequence: H_1 = log2(4) = 2.
    seq = list("ABCD" * 100)
    result = compute_block_entropy(seq, order=1)
    assert isinstance(result, BlockEntropy)
    assert result.entropy_bits == pytest.approx(2.0, abs=1e-12)


@pytest.mark.unit
def test_block_entropy_collapses_for_constant_sequence() -> None:
    seq = ["A"] * 50
    result = compute_block_entropy(seq, order=1)
    assert result.entropy_bits == 0.0
    assert result.n_unique_blocks == 1


@pytest.mark.unit
def test_conditional_entropy_zero_order_equals_marginal() -> None:
    seq = list("ABCD" * 50)
    cond = compute_conditional_entropy(seq, order=0)
    assert isinstance(cond, ConditionalEntropy)
    assert cond.h_n_bits == pytest.approx(2.0, abs=1e-12)


@pytest.mark.unit
def test_conditional_entropy_first_order_zero_for_perfect_pattern() -> None:
    # Period-4 pattern -> next symbol fully determined by previous symbol.
    # The biased plug-in estimator has a small finite-sample residual due
    # to boundary blocks; require it to be negligibly small but allow
    # ~ log(N)/N order of magnitude.
    seq = list("ABCD" * 80)
    cond = compute_conditional_entropy(seq, order=1)
    assert cond.h_n_bits == pytest.approx(0.0, abs=0.01)


@pytest.mark.unit
def test_conditional_entropy_iid_unchanged_with_order() -> None:
    # Pseudo-iid binary sequence: h_0 ~ h_1 ~ 1 bit.
    import random
    rng = random.Random(0)
    seq = [rng.choice("LR") for _ in range(2000)]
    h0 = compute_conditional_entropy(seq, order=0).h_n_bits
    h1 = compute_conditional_entropy(seq, order=1).h_n_bits
    assert abs(h0 - 1.0) < 0.05
    assert abs(h1 - 1.0) < 0.05


@pytest.mark.unit
def test_estimate_markov_order_detects_first_order_dependency() -> None:
    # First-order Markov: switching pattern with strong stay bias.
    import random
    rng = random.Random(42)
    seq = ["L"]
    p_stay = 0.9
    for _ in range(2000):
        if seq[-1] == "L":
            seq.append("L" if rng.random() < p_stay else "R")
        else:
            seq.append("R" if rng.random() < p_stay else "L")
    result = estimate_markov_order(seq, max_order=3, tolerance=0.05)
    assert isinstance(result, MarkovOrderEstimate)
    assert result.estimated_order in (1, 2)
    assert result.converged


@pytest.mark.unit
def test_block_entropy_raises_for_too_short_sequence() -> None:
    with pytest.raises(ValueError, match="at least"):
        compute_block_entropy(["A"], order=2)


@pytest.mark.unit
def test_estimate_markov_order_rejects_invalid_args() -> None:
    seq = list("AB" * 50)
    with pytest.raises(ValueError, match="max_order"):
        estimate_markov_order(seq, max_order=0)
    with pytest.raises(ValueError, match="tolerance"):
        estimate_markov_order(seq, max_order=2, tolerance=0.0)
