"""Temporal-difference (TD) learning trace simulation.

For a state sequence ``s_0, s_1, ..., s_T`` and rewards ``r_1, ..., r_T``,
TD(0) updates state-value estimates according to

    delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)
    V(s_t) <- V(s_t) + alpha * delta_t

This module provides a tabular TD(0) simulator suitable for fitting
behavioural trajectories to a reinforcement-learning trace (e.g., the
Schultz, Dayan, & Montague, 1997 dopamine-as-prediction-error account).

Reference
---------
Sutton, R. S. (1988). Learning to predict by the methods of temporal
    differences. *Machine Learning*, *3*(1), 9-44.
    https://doi.org/10.1007/BF00115009

Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate
    of prediction and reward. *Science*, *275*(5306), 1593-1599.
    https://doi.org/10.1126/science.275.5306.1593
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TDLearningTrace:
    """Result of running TD(0) over a fixed trajectory.

    Attributes
    ----------
    values : tuple of float
        Final state-value estimates indexed by state id.
    errors : tuple of float
        Per-step TD prediction errors.
    alpha : float
    gamma : float
    n_steps : int
    """

    values: tuple[float, ...]
    errors: tuple[float, ...]
    alpha: float
    gamma: float
    n_steps: int


def simulate_td_zero(
    states: Sequence[int],
    rewards: Sequence[float],
    *,
    alpha: float,
    gamma: float,
    n_states: int | None = None,
    initial_values: Sequence[float] | None = None,
    terminal_state: int | None = None,
) -> TDLearningTrace:
    """Run tabular TD(0) over the given trajectory.

    Parameters
    ----------
    states : sequence of int
        State sequence ``s_0, s_1, ..., s_T``. ``len(states)`` must be
        ``len(rewards) + 1``.
    rewards : sequence of float
        Reward sequence ``r_1, ..., r_T`` (the reward observed at the
        transition into ``states[t+1]``).
    alpha : float
        Learning rate, in ``(0, 1]``.
    gamma : float
        Discount factor, in ``[0, 1]``.
    n_states : int, optional
        Total number of distinct states. Defaults to ``max(states) + 1``.
    initial_values : sequence of float, optional
        Initial value estimates. Defaults to zeros.
    terminal_state : int, optional
        State id treated as terminal: ``V(terminal) = 0`` and is not
        updated. A transition out of the terminal state (episodes
        concatenated in one trajectory) starts a new episode and records
        a TD error of 0. Defaults to none.

    Returns
    -------
    TDLearningTrace
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    if len(states) != len(rewards) + 1:
        raise ValueError(
            "len(states) must equal len(rewards) + 1 "
            f"(got {len(states)} and {len(rewards)})"
        )
    if not states:
        raise ValueError("states must be non-empty")

    inferred_n = max(states) + 1
    n = inferred_n if n_states is None else n_states
    if n < inferred_n:
        raise ValueError(
            f"n_states={n} is smaller than max state id {inferred_n - 1}"
        )

    values = (
        [0.0] * n
        if initial_values is None
        else list(initial_values)
    )
    if len(values) != n:
        raise ValueError(
            f"initial_values length {len(values)} != n_states {n}"
        )

    errors: list[float] = []
    for t, r in enumerate(rewards):
        s = states[t]
        s_next = states[t + 1]
        if terminal_state is not None and s == terminal_state:
            # Leaving the terminal state starts a new episode: no TD
            # error and no update, so V(terminal) stays as given.
            errors.append(0.0)
            continue
        v_next = 0.0 if s_next == terminal_state else values[s_next]
        delta = r + gamma * v_next - values[s]
        errors.append(delta)
        values[s] = values[s] + alpha * delta

    return TDLearningTrace(
        values=tuple(values),
        errors=tuple(errors),
        alpha=alpha,
        gamma=gamma,
        n_steps=len(rewards),
    )
