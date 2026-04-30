"""Sign-tracking vs. goal-tracking classification.

In Pavlovian conditioned-approach (PCA) paradigms, subjects diverge
into *sign-trackers* (ST) who preferentially approach and interact
with the conditioned stimulus (e.g. a retractable lever) and
*goal-trackers* (GT) who preferentially approach the site of
reinforcement (e.g. the food cup). The PCA tracking index
aggregates three behavioural components into a single
``[-1, +1]``-bounded score (Meyer et al., 2012):

    index = (response_bias + latency_score + probability_score) / 3

where each component is normalized so that ``+1`` indicates pure
sign-tracking and ``-1`` indicates pure goal-tracking.

Conventional classification cut-offs (Meyer et al., 2012):

- ``index >= 0.5``  -> sign-tracker
- ``index <= -0.5`` -> goal-tracker
- otherwise         -> intermediate

References
----------
Flagel, S. B., Watson, S. J., Robinson, T. E., & Akil, H. (2007).
    Individual differences in the propensity to approach signals vs
    goals promote different adaptations in the dopamine system of
    rats. *Psychopharmacology*, *191*(3), 599-607.
    https://doi.org/10.1007/s00213-006-0535-8

Meyer, P. J., Lovic, V., Saunders, B. T., Yager, L. M., Flagel, S. B.,
    Morrow, J. D., & Robinson, T. E. (2012). Quantifying individual
    variation in the propensity to attribute incentive salience to
    reward cues. *PLoS ONE*, *7*(6), e38987.
    https://doi.org/10.1371/journal.pone.0038987
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackingType(Enum):
    """Tracking classification (Meyer et al., 2012 cut-offs)."""

    SIGN_TRACKER = "sign_tracker"
    GOAL_TRACKER = "goal_tracker"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True)
class TrackingIndex:
    """PCA tracking index and classification.

    Attributes
    ----------
    response_bias : float
        ``(contacts_cs - contacts_food) / (contacts_cs + contacts_food)``
        in ``[-1, +1]``. ``0`` when both are zero.
    latency_score : float
        ``(latency_food - latency_cs) / cs_duration`` in ``[-1, +1]``.
        ``+1`` means the first CS contact preceded the first food-cup
        contact by the full CS duration; ``-1`` is the reverse.
    probability_score : float
        ``probability_cs - probability_food`` in ``[-1, +1]``, where
        each probability is the fraction of trials containing at
        least one contact of the respective type.
    index : float
        Mean of the three components, in ``[-1, +1]``.
    classification : TrackingType
    """

    response_bias: float
    latency_score: float
    probability_score: float
    index: float
    classification: TrackingType


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_tracking_index(
    *,
    contacts_cs: int,
    contacts_food: int,
    latency_cs: float,
    latency_food: float,
    probability_cs: float,
    probability_food: float,
    cs_duration: float,
    sign_threshold: float = 0.5,
    goal_threshold: float = -0.5,
) -> TrackingIndex:
    """Compute the PCA tracking index from trial aggregates.

    Parameters
    ----------
    contacts_cs, contacts_food : int
        Total contacts on the CS and food-cup across all trials.
    latency_cs, latency_food : float
        Mean latency (seconds) to first contact within a trial. Use
        ``cs_duration`` for trials without a contact of that type,
        which is the convention in Meyer et al. (2012).
    probability_cs, probability_food : float
        Proportion of trials with at least one CS / food-cup contact,
        each in ``[0, 1]``.
    cs_duration : float
        CS presentation length in seconds (strictly positive).
    sign_threshold, goal_threshold : float
        Classification cut-offs on the aggregate index. Defaults are
        the Meyer et al. (2012) values of ``+0.5`` and ``-0.5``.

    Returns
    -------
    TrackingIndex

    Raises
    ------
    ValueError
        If counts are negative, probabilities are outside ``[0, 1]``,
        or ``cs_duration`` is non-positive.
    """
    if contacts_cs < 0 or contacts_food < 0:
        raise ValueError("contacts must be non-negative")
    if not 0.0 <= probability_cs <= 1.0 or not 0.0 <= probability_food <= 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    if cs_duration <= 0.0:
        raise ValueError(f"cs_duration must be positive, got {cs_duration}")
    if latency_cs < 0.0 or latency_food < 0.0:
        raise ValueError("latencies must be non-negative")
    if goal_threshold >= sign_threshold:
        raise ValueError("goal_threshold must be < sign_threshold")

    total_contacts = contacts_cs + contacts_food
    if total_contacts == 0:
        response_bias = 0.0
    else:
        response_bias = (contacts_cs - contacts_food) / total_contacts

    latency_score = _clip((latency_food - latency_cs) / cs_duration)
    probability_score = _clip(probability_cs - probability_food)

    index = (response_bias + latency_score + probability_score) / 3.0

    if index >= sign_threshold:
        kind = TrackingType.SIGN_TRACKER
    elif index <= goal_threshold:
        kind = TrackingType.GOAL_TRACKER
    else:
        kind = TrackingType.INTERMEDIATE

    return TrackingIndex(
        response_bias=float(response_bias),
        latency_score=float(latency_score),
        probability_score=float(probability_score),
        index=float(index),
        classification=kind,
    )
