"""Tests for tabular TD(0) simulation."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.td_learning import (
    TDLearningTrace,
    simulate_td_zero,
)


@pytest.mark.unit
def test_td_zero_recovers_terminal_reward_value() -> None:
    """Simple chain s0 -> s1 -> s2 (terminal), reward 1 only at the end.

    With gamma=1, alpha=1, after enough passes V(s0) and V(s1) approach 1.
    """
    states = [0, 1, 2]
    rewards = [0.0, 1.0]
    last: TDLearningTrace | None = None
    for _ in range(50):
        last = simulate_td_zero(
            states, rewards,
            alpha=1.0, gamma=1.0,
            n_states=3, terminal_state=2,
            initial_values=last.values if last else None,
        )
    assert last is not None
    assert last.values[0] == pytest.approx(1.0, abs=1e-9)
    assert last.values[1] == pytest.approx(1.0, abs=1e-9)
    assert last.values[2] == pytest.approx(0.0)


@pytest.mark.unit
def test_td_zero_discount_propagates_value_backward() -> None:
    states = [0, 1, 2]
    rewards = [0.0, 1.0]
    last = None
    for _ in range(200):
        last = simulate_td_zero(
            states, rewards, alpha=0.5, gamma=0.9,
            n_states=3, terminal_state=2,
            initial_values=last.values if last else None,
        )
    assert last is not None
    # V(s1) -> 1, V(s0) -> 0.9.
    assert last.values[1] == pytest.approx(1.0, abs=1e-3)
    assert last.values[0] == pytest.approx(0.9, abs=1e-3)


@pytest.mark.unit
def test_td_zero_records_prediction_errors() -> None:
    trace = simulate_td_zero(
        [0, 1, 2], [0.0, 1.0],
        alpha=0.5, gamma=1.0, n_states=3, terminal_state=2,
    )
    # First TD error from s0 with V(s1)=0, r=0 -> 0.
    # Second TD error from s1 with V(terminal)=0, r=1 -> 1.
    assert trace.errors == pytest.approx((0.0, 1.0))


@pytest.mark.unit
def test_td_zero_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        simulate_td_zero([0, 1], [0.5], alpha=0.0, gamma=0.9, n_states=2)
    with pytest.raises(ValueError, match="alpha"):
        simulate_td_zero([0, 1], [0.5], alpha=1.5, gamma=0.9, n_states=2)


@pytest.mark.unit
def test_td_zero_rejects_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        simulate_td_zero([0, 1], [0.5], alpha=0.5, gamma=-0.1, n_states=2)
    with pytest.raises(ValueError, match="gamma"):
        simulate_td_zero([0, 1], [0.5], alpha=0.5, gamma=1.5, n_states=2)


@pytest.mark.unit
def test_td_zero_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="len\\(states\\)"):
        simulate_td_zero([0, 1, 2], [0.5], alpha=0.5, gamma=0.5, n_states=3)
