"""Concurrent-chains choice prediction (Mazur, 2001).

Under a concurrent-chains schedule, responses during the initial link
on either alternative produce entry into a terminal link characterized
by an amount and a delay to reinforcement. The *hyperbolic-decay model*
(Mazur, 1984, 2001) assigns each alternative a value

    V = A / (1 + k * D)

and predicts initial-link choice allocation by the relative value

    P(left) = V_L / (V_L + V_R).

This module compares observed initial-link allocation with the
hyperbolic-decay prediction, returning both side-by-side so users can
assess model fit without committing to any particular k.

Reference
---------
Mazur, J. E. (2001). Hyperbolic value addition and general models of
    animal choice. *Psychological Review*, *108*(1), 96-112.
    https://doi.org/10.1037/0033-295X.108.1.96
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConcurrentChainsPrediction:
    """Comparison of observed vs. hyperbolic-decay-predicted allocation.

    Attributes
    ----------
    v_left, v_right : float
        Hyperbolic-decay values for the two terminal links.
    predicted_proportion_left : float
        ``V_L / (V_L + V_R)``.
    observed_proportion_left : float
        Observed proportion of initial-link responses on the left
        alternative. ``0.5`` when total responses are zero.
    deviation : float
        ``observed - predicted`` (negative => under-preference for left).
    k : float
        The discounting rate used.
    """

    v_left: float
    v_right: float
    predicted_proportion_left: float
    observed_proportion_left: float
    deviation: float
    k: float


def predict_concurrent_chains_choice(
    *,
    left_amount: float,
    left_delay: float,
    right_amount: float,
    right_delay: float,
    left_responses: int,
    right_responses: int,
    k: float = 1.0,
) -> ConcurrentChainsPrediction:
    """Predict initial-link choice via Mazur's hyperbolic-decay model.

    Parameters
    ----------
    left_amount, right_amount : float
        Reinforcer magnitudes for each terminal link (strictly positive).
    left_delay, right_delay : float
        Terminal-link delays in seconds (non-negative).
    left_responses, right_responses : int
        Observed initial-link response counts for each alternative
        (non-negative).
    k : float
        Hyperbolic discounting rate (strictly positive). Default 1.0.

    Returns
    -------
    ConcurrentChainsPrediction

    Raises
    ------
    ValueError
        If amounts are non-positive, delays are negative, responses are
        negative, or ``k`` is non-positive.
    """
    if left_amount <= 0.0 or right_amount <= 0.0:
        raise ValueError("amounts must be strictly positive")
    if left_delay < 0.0 or right_delay < 0.0:
        raise ValueError("delays must be non-negative")
    if left_responses < 0 or right_responses < 0:
        raise ValueError("response counts must be non-negative")
    if k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")

    v_l = left_amount / (1.0 + k * left_delay)
    v_r = right_amount / (1.0 + k * right_delay)
    predicted = v_l / (v_l + v_r)

    total = left_responses + right_responses
    observed = left_responses / total if total > 0 else 0.5

    return ConcurrentChainsPrediction(
        v_left=v_l,
        v_right=v_r,
        predicted_proportion_left=predicted,
        observed_proportion_left=observed,
        deviation=observed - predicted,
        k=k,
    )
