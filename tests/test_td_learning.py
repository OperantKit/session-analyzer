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


# ---------------------------------------------------------------------------
# Closed forms (S-166). Sutton (1988) and Schultz, Dayan & Montague (1997)
# report no numeric table that this simulator reproduces, so the checks are
# closed forms of tabular TD(0) on a deterministic chain 0 -> 1 -> 2 -> T
# with reward 1 on entering T.
# ---------------------------------------------------------------------------


def _run_episodes(n_episodes: int, *, alpha: float, gamma: float) -> tuple[float, ...]:
    values: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    for _ in range(n_episodes):
        trace = simulate_td_zero(
            [0, 1, 2, 3],
            [0.0, 0.0, 1.0],
            alpha=alpha,
            gamma=gamma,
            initial_values=values,
            terminal_state=3,
        )
        values = trace.values
    return values


@pytest.mark.unit
@pytest.mark.reference(
    module="td_learning",
    verdict="properties",
    source="Sutton (1988)",
)
def test_converges_to_discounted_return_on_deterministic_chain() -> None:
    """V(s) -> gamma^(steps to reward - 1): 0.81, 0.9, 1 for gamma = 0.9."""
    v = _run_episodes(300, alpha=0.5, gamma=0.9)
    assert v[:3] == pytest.approx((0.81, 0.9, 1.0), abs=1e-12)
    assert v[3] == 0.0


@pytest.mark.unit
def test_last_state_follows_closed_form_per_episode() -> None:
    """V_n(2) = 1 - (1 - alpha)^n after n episodes (V_0 = 0)."""
    for n in (1, 2, 5):
        v = _run_episodes(n, alpha=0.3, gamma=0.9)
        assert v[2] == pytest.approx(1.0 - 0.7**n, abs=1e-12)


@pytest.mark.unit
def test_terminal_state_is_not_updated_when_episodes_are_concatenated() -> None:
    """Docstring: V(terminal) = 0 and is not updated.

    Two episodes in one trajectory: 0 -> 1 -> T -> 0 -> 1 -> T. The
    transition out of T starts the next episode, so it carries no TD error
    and leaves V(T) at 0.
    """
    trace = simulate_td_zero(
        [0, 1, 2, 0, 1, 2],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        alpha=0.5,
        gamma=0.9,
        initial_values=(0.2, 0.0, 0.0),
        terminal_state=2,
    )
    assert trace.values[2] == 0.0
    assert trace.errors[2] == 0.0
    # Same result as two separate episodes.
    first = simulate_td_zero(
        [0, 1, 2], [0.0, 1.0], alpha=0.5, gamma=0.9,
        initial_values=(0.2, 0.0, 0.0), terminal_state=2,
    )
    second = simulate_td_zero(
        [0, 1, 2], [0.0, 1.0], alpha=0.5, gamma=0.9,
        initial_values=first.values, terminal_state=2,
    )
    assert trace.values == pytest.approx(second.values, abs=1e-15)
