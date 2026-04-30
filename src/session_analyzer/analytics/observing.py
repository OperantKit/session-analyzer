"""Observing-response and conditioned-reinforcement analysis.

In an *observing-response* procedure (Wyckoff, 1952), the subject can
emit a response that produces stimuli signalling whether the imminent
schedule period is reinforcement (S+) or extinction (S-). Observing
responses are not themselves required for primary reinforcement, so
their rate is taken as an index of the *conditioned reinforcement*
strength of the discriminative stimuli.

Two summaries:

- :class:`ObservingMetrics` — rate of observing responses and the
  proportion of those that produced S+ vs S-. The latter is the
  empirical anchor of Dinsmoor's *good-news* hypothesis: observing is
  maintained primarily by S+ presentations.

- :class:`ConditionedReinforcementStrength` — ratio of observing rate
  in a fully correlated condition (S+ / S- predict reinforcement /
  extinction) to a yoked or uncorrelated control. A ratio meaningfully
  greater than 1 is taken as evidence that the stimulus has acquired
  conditioned-reinforcer function.

References
----------
Dinsmoor, J. A. (1983). Observing and conditioned reinforcement.
*Behavioral and Brain Sciences*, 6(4), 693-704.
https://doi.org/10.1017/S0140525X00017969

Wyckoff, L. B. (1952). The role of observing responses in
discrimination learning. Part I. *Psychological Review*, 59(6),
431-442. https://doi.org/10.1037/h0053932
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservingMetrics:
    """Rate and S+/S- composition of observing responses.

    Attributes
    ----------
    observing_rate_per_min : float
    s_plus_proportion : float
        Fraction of observing responses (or stimulus productions) that
        produced S+. Bounded in ``[0, 1]``.
    good_news_dominance : bool
        ``True`` if ``s_plus_proportion >= 0.6``, the conventional
        threshold for invoking the good-news interpretation.
    observing_count : int
    s_plus_productions : int
    s_minus_productions : int
    observing_duration_sec : float
    """

    observing_rate_per_min: float
    s_plus_proportion: float
    good_news_dominance: bool
    observing_count: int
    s_plus_productions: int
    s_minus_productions: int
    observing_duration_sec: float


@dataclass(frozen=True)
class ConditionedReinforcementStrength:
    """Conditioned-reinforcement strength as a between-condition ratio.

    Attributes
    ----------
    strength_index : float
        ``observing_rate_correlated / observing_rate_uncorrelated``.
        Values > 1 indicate the stimulus has acquired conditioned
        reinforcer function.
    observing_rate_correlated : float
    observing_rate_uncorrelated : float
    is_conditioned_reinforcer : bool
        ``True`` if ``strength_index >= 2.0`` (heuristic).
    """

    strength_index: float
    observing_rate_correlated: float
    observing_rate_uncorrelated: float
    is_conditioned_reinforcer: bool


def compute_observing_metrics(
    *,
    observing_count: int,
    observing_duration_sec: float,
    s_plus_productions: int,
    s_minus_productions: int,
    good_news_threshold: float = 0.60,
) -> ObservingMetrics:
    """Summarise rate and S+/S- composition of observing responses."""
    if observing_count < 0 or s_plus_productions < 0 or s_minus_productions < 0:
        raise ValueError("counts must be non-negative")
    if observing_duration_sec <= 0.0:
        raise ValueError("observing_duration_sec must be > 0")
    rate = observing_count / (observing_duration_sec / 60.0)
    total_productions = s_plus_productions + s_minus_productions
    s_plus_prop = (
        s_plus_productions / total_productions if total_productions > 0 else 0.0
    )
    return ObservingMetrics(
        observing_rate_per_min=round(rate, 6),
        s_plus_proportion=round(s_plus_prop, 6),
        good_news_dominance=s_plus_prop >= good_news_threshold,
        observing_count=observing_count,
        s_plus_productions=s_plus_productions,
        s_minus_productions=s_minus_productions,
        observing_duration_sec=observing_duration_sec,
    )


def compute_conditioned_reinforcement_strength(
    *,
    observing_rate_correlated: float,
    observing_rate_uncorrelated: float,
    threshold: float = 2.0,
) -> ConditionedReinforcementStrength:
    """Quantify the conditioned-reinforcer strength as a ratio of
    observing rates between fully-correlated and uncorrelated
    conditions.

    Parameters
    ----------
    observing_rate_correlated, observing_rate_uncorrelated : float
        Observing-response rates (responses / minute) in the two
        conditions. The uncorrelated rate must be > 0 to define the
        ratio.
    threshold : float
        Minimum ratio at which the stimulus is flagged as a
        conditioned reinforcer (default 2x).
    """
    if observing_rate_correlated < 0.0 or observing_rate_uncorrelated < 0.0:
        raise ValueError("observing rates must be non-negative")
    if observing_rate_uncorrelated == 0.0:
        raise ValueError("observing_rate_uncorrelated must be > 0 to form a ratio")
    ratio = observing_rate_correlated / observing_rate_uncorrelated
    return ConditionedReinforcementStrength(
        strength_index=round(ratio, 6),
        observing_rate_correlated=observing_rate_correlated,
        observing_rate_uncorrelated=observing_rate_uncorrelated,
        is_conditioned_reinforcer=ratio >= threshold,
    )
