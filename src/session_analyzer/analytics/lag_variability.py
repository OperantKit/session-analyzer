"""Lag-N variability analysis.

Under a Lag-N variability schedule, a response sequence is reinforced
only if it differs from each of the preceding N sequences. The
empirical question is how variability changes with the imposed lag
requirement (Page & Neuringer, 1985; Neuringer, 2002).

This module provides two complementary statistics on top of the basic
U-value (see :mod:`session_analyzer.analytics.variability`):

- :func:`compute_lag_pass_rate` — the fraction of trials whose
  sequence differed from each of the previous N. This equals the
  reinforcement rate under a strict Lag-N schedule.
- :func:`analyze_lag_variability` — bundles the U-value, the
  Lag-N pass rate, and the average number of unique sequences in
  trailing length-``N`` windows.

References
----------
Page, S., & Neuringer, A. (1985). Variability is an operant.
*Journal of Experimental Psychology: Animal Behavior Processes*,
11(3), 429-452. https://doi.org/10.1037/0097-7403.11.3.429

Neuringer, A. (2002). Operant variability: Evidence, functions, and
theory. *Psychonomic Bulletin & Review*, 9(4), 672-705.
https://doi.org/10.3758/BF03196324
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from .variability import compute_u_value


@dataclass(frozen=True)
class LagVariabilityMetrics:
    """Lag-N variability summary.

    Attributes
    ----------
    u_value : float
        Overall U-value (Page & Neuringer, 1985), in ``[0, 1]``.
    lag_pass_rate : float
        Fraction of trials at index ``i >= lag`` whose sequence
        differs from each of ``sequences[i - lag : i]``.
    mean_unique_in_window : float
        Average number of unique sequences in the ``lag``-length
        sliding window over indices ``[lag - 1, n)``.
    lag : int
    n_sequences : int
    """

    u_value: float
    lag_pass_rate: float
    mean_unique_in_window: float
    lag: int
    n_sequences: int


def compute_lag_pass_rate(sequences: Sequence[Hashable], *, lag: int) -> float:
    """Fraction of trials at index ``>= lag`` that differ from each of
    the previous ``lag`` sequences.

    Parameters
    ----------
    sequences : sequence of hashable
    lag : int
        Strictly positive lag depth. Must be ``< len(sequences)``.
    """
    n = len(sequences)
    if n == 0:
        raise ValueError("sequences must be non-empty")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if lag >= n:
        raise ValueError(f"lag ({lag}) must be less than len(sequences) ({n})")

    passes = 0
    eligible = n - lag
    for i in range(lag, n):
        recent = set(sequences[i - lag : i])
        if sequences[i] not in recent:
            passes += 1
    return passes / eligible


def analyze_lag_variability(
    *,
    sequences: Sequence[Hashable],
    alphabet_size: int,
    sequence_length: int,
    lag: int,
) -> LagVariabilityMetrics:
    """Bundle the overall U-value, Lag-N pass rate, and per-window
    uniqueness for a sequence stream.
    """
    base = compute_u_value(
        sequences,
        alphabet_size=alphabet_size,
        sequence_length=sequence_length,
    )
    n = len(sequences)
    if n <= lag:
        raise ValueError(f"need more than lag={lag} sequences; got {n}")

    pass_rate = compute_lag_pass_rate(sequences, lag=lag)

    # Rolling unique count: window covers indices [i - lag + 1, i] inclusive
    unique_counts = [
        len(set(sequences[max(0, i - lag + 1) : i + 1])) for i in range(lag - 1, n)
    ]
    mean_unique = sum(unique_counts) / len(unique_counts)

    return LagVariabilityMetrics(
        u_value=round(base.u_value, 6),
        lag_pass_rate=round(pass_rate, 6),
        mean_unique_in_window=round(mean_unique, 6),
        lag=lag,
        n_sequences=n,
    )
