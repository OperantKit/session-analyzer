"""Tests for Lag-N variability metrics."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.lag_variability import (
    LagVariabilityMetrics,
    analyze_lag_variability,
    compute_lag_pass_rate,
)


@pytest.mark.unit
def test_lag_pass_rate_lag1_alternating() -> None:
    sequences = ["LL", "RR", "LL", "RR", "LL"]
    rate = compute_lag_pass_rate(sequences, lag=1)
    # First trial undefined; trials 2-5 differ from previous -> 4/4 = 1.0
    assert rate == pytest.approx(1.0)


@pytest.mark.unit
def test_lag_pass_rate_lag1_repeating() -> None:
    sequences = ["LL", "LL", "LL", "LL"]
    rate = compute_lag_pass_rate(sequences, lag=1)
    # All differ-from-previous fail -> 0/3 = 0.0
    assert rate == pytest.approx(0.0)


@pytest.mark.unit
def test_lag_pass_rate_lag2() -> None:
    # Lag-2: differ from each of last 2
    sequences = ["AA", "BB", "CC", "AA", "BB", "AA"]
    # Trial 3: differ from {AA, BB} -> CC -> pass
    # Trial 4: differ from {BB, CC} -> AA -> pass
    # Trial 5: differ from {CC, AA} -> BB -> pass
    # Trial 6: differ from {AA, BB} -> AA fails (matches AA at i-2)
    # 3 / 4 = 0.75
    rate = compute_lag_pass_rate(sequences, lag=2)
    assert rate == pytest.approx(0.75)


@pytest.mark.unit
def test_lag_pass_rate_validates() -> None:
    with pytest.raises(ValueError):
        compute_lag_pass_rate([], lag=1)
    with pytest.raises(ValueError):
        compute_lag_pass_rate(["a", "b"], lag=0)
    with pytest.raises(ValueError):
        # Lag exceeds available history
        compute_lag_pass_rate(["a", "b"], lag=5)


@pytest.mark.unit
def test_analyze_lag_variability_combined() -> None:
    sequences = ["LL", "LR", "RL", "RR", "LL", "LR", "RL", "RR"]
    metrics = analyze_lag_variability(
        sequences=sequences,
        alphabet_size=2,
        sequence_length=2,
        lag=2,
    )
    assert isinstance(metrics, LagVariabilityMetrics)
    assert 0.0 <= metrics.u_value <= 1.0
    # Uniform over 4 unique sequences -> U should be 1.0
    assert metrics.u_value == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= metrics.lag_pass_rate <= 1.0
    assert metrics.mean_unique_in_window > 1.0
    assert metrics.lag == 2


@pytest.mark.unit
def test_analyze_lag_variability_low_variability() -> None:
    sequences = ["LL"] * 10
    metrics = analyze_lag_variability(
        sequences=sequences,
        alphabet_size=2,
        sequence_length=2,
        lag=2,
    )
    assert metrics.u_value == pytest.approx(0.0)
    assert metrics.lag_pass_rate == pytest.approx(0.0)
    assert metrics.mean_unique_in_window == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Hand computations and closed forms
# ---------------------------------------------------------------------------
#
# Under Lag N a sequence is reinforced only if it differs from each of the
# previous N (Page & Neuringer, 1985; the article is paywalled, so the
# definition is checked by hand). A cycle through K distinct sequences passes
# every trial at lag N < K and fails every trial at lag N = K.


@pytest.mark.unit
@pytest.mark.reference(
    module="lag_variability",
    verdict="properties",
    source="Page & Neuringer (1985)",
)
def test_pass_rate_and_window_uniqueness_by_hand() -> None:
    seqs = ["A", "B", "A", "C", "C", "B"]
    # i=2 A in {A,B} fail; i=3 C pass; i=4 C in {A,C} fail; i=5 B in {C} pass.
    assert compute_lag_pass_rate(seqs, lag=2) == pytest.approx(0.5)
    m = analyze_lag_variability(sequences=seqs, alphabet_size=3, sequence_length=1, lag=2)
    # windows {A,B} {B,A} {A,C} {C} {C,B}: 2+2+2+1+2 = 9 over 5
    assert m.mean_unique_in_window == pytest.approx(1.8)


@pytest.mark.unit
def test_cycle_of_k_sequences_closed_form() -> None:
    cycle = ["LL", "LR", "RL"] * 10
    assert compute_lag_pass_rate(cycle, lag=1) == 1.0
    assert compute_lag_pass_rate(cycle, lag=2) == 1.0
    assert compute_lag_pass_rate(cycle, lag=3) == 0.0
