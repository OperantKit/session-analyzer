"""Choice allocation analysis for concurrent schedules.

Provides data types and functions for analyzing choice behavior in
concurrent schedules: allocation proportions, exclusive-choice detection,
and choice-mode classification.

Design rationale
----------------
These functions operate on **observed behavioral data** (response counts),
not on schedule specifications. The contingency-dsl specifies the procedure
(what the experimenter arranges); this module characterizes the effect
(what the organism does).

Exclusive-choice threshold
--------------------------
The default ``threshold=0.95`` is a conventional operationalization of
"nearly all responses allocated to one alternative." No single paper
defines this exact value. The rationale:

- Herrnstein & Loveland (1975) describe exclusive preference qualitatively
  as "exclusive preference for the smaller ratio" under unequal concurrent
  VR-VR, without specifying a numerical criterion.
- Baum, Aparicio, & Alonso-Alvarez (2022) report that actual behavior
  under concurrent VR-VR falls between probability matching and
  optimization, with large individual differences — implying that any
  fixed threshold is an approximation.
- The 0.95 value is chosen as a conservative threshold: an organism
  allocating >= 95% of responses to one alternative has, for practical
  purposes, ceased responding on the other. This is stricter than 0.90
  (which may include moderate preference) and more permissive than 0.99
  (which excludes stochastic noise in short sessions).

Users should treat this as a configurable analysis parameter, not a
theoretical constant. The threshold should be reported alongside results,
and sensitivity analyses across multiple thresholds are recommended.

References
----------
Baum, W. M., Aparicio, C. F., & Alonso-Alvarez, B. (2022). Rate matching,
    probability matching, and optimization in concurrent ratio schedules.
    *Journal of the Experimental Analysis of Behavior*, *118*(1).
    https://doi.org/10.1002/jeab.771

Herrnstein, R. J. (1961). Relative and absolute strength of response as a
    function of frequency of reinforcement.
    *Journal of the Experimental Analysis of Behavior*, *4*(3), 267-272.
    https://doi.org/10.1901/jeab.1961.4-267

Herrnstein, R. J., & Loveland, D. H. (1975). Maximizing and matching on
    concurrent ratio schedules.
    *Journal of the Experimental Analysis of Behavior*, *24*(1), 107-116.
    https://doi.org/10.1901/jeab.1975.24-107
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class ChoiceMode(Enum):
    """Classification of choice allocation patterns.

    - ``EXCLUSIVE``: one alternative captures >= threshold of all responses.
    - ``MATCHING``: allocations roughly proportional (spread <= tolerance).
    - ``INTERMEDIATE``: neither exclusive nor matching.
    """

    EXCLUSIVE = "exclusive"
    MATCHING = "matching"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True)
class ChoiceAllocation:
    """Per-operandum allocation proportions.

    Attributes
    ----------
    proportions : tuple[float, ...]
        Proportion of responses allocated to each operandum. Sums to 1.0.
    counts : tuple[int, ...]
        Raw response counts per operandum.
    total : int
        Total responses across all operanda.
    """

    proportions: tuple[float, ...]
    counts: tuple[int, ...]
    total: int


@dataclass(frozen=True)
class ExclusiveChoiceResult:
    """Result of exclusive-choice detection.

    Attributes
    ----------
    is_exclusive : bool
        True if one operandum captures >= threshold of all responses.
    dominant_operandum : int or None
        Index of the dominant operandum, or None if not exclusive.
    dominant_proportion : float or None
        Proportion captured by the dominant operandum.
    threshold : float
        The threshold used for detection.
    """

    is_exclusive: bool
    dominant_operandum: Optional[int]
    dominant_proportion: Optional[float]
    threshold: float


def compute_choice_allocation(
    response_counts: Sequence[int],
) -> ChoiceAllocation:
    """Compute allocation proportions from response counts.

    Parameters
    ----------
    response_counts : sequence of int
        Number of responses on each operandum.

    Returns
    -------
    ChoiceAllocation
        Allocation proportions and raw counts. If total is 0, returns
        uniform proportions.

    Raises
    ------
    ValueError
        If fewer than 2 operanda are provided.
    """
    counts = tuple(response_counts)
    if len(counts) < 2:
        raise ValueError("At least 2 operanda required for choice analysis")

    total = sum(counts)
    if total == 0:
        n = len(counts)
        proportions = tuple(1.0 / n for _ in counts)
    else:
        proportions = tuple(c / total for c in counts)

    return ChoiceAllocation(proportions=proportions, counts=counts, total=total)


_DEFAULT_EXCLUSIVE_THRESHOLD: float = 0.95


def detect_exclusive_choice(
    allocation: ChoiceAllocation,
    *,
    threshold: float = _DEFAULT_EXCLUSIVE_THRESHOLD,
) -> ExclusiveChoiceResult:
    """Detect whether one operandum captures exclusive choice.

    Parameters
    ----------
    allocation : ChoiceAllocation
        Output of :func:`compute_choice_allocation`.
    threshold : float
        Proportion at or above which one operandum is considered to
        have captured exclusive choice. Default 0.95. See module
        docstring for rationale and literature context.

    Returns
    -------
    ExclusiveChoiceResult
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    for i, p in enumerate(allocation.proportions):
        if p >= threshold:
            return ExclusiveChoiceResult(
                is_exclusive=True,
                dominant_operandum=i,
                dominant_proportion=p,
                threshold=threshold,
            )

    return ExclusiveChoiceResult(
        is_exclusive=False,
        dominant_operandum=None,
        dominant_proportion=None,
        threshold=threshold,
    )


_DEFAULT_MATCHING_TOLERANCE: float = 0.10


def classify_choice_mode(
    allocation: ChoiceAllocation,
    *,
    exclusive_threshold: float = _DEFAULT_EXCLUSIVE_THRESHOLD,
    matching_tolerance: float = _DEFAULT_MATCHING_TOLERANCE,
) -> ChoiceMode:
    """Classify choice allocation as exclusive, matching, or intermediate.

    Parameters
    ----------
    allocation : ChoiceAllocation
        Output of :func:`compute_choice_allocation`.
    exclusive_threshold : float
        Proportion for exclusive-choice detection. Default 0.95.
    matching_tolerance : float
        Maximum spread (max - min) between allocation proportions to
        classify as matching. Default 0.10 (10 percentage points).

    Returns
    -------
    ChoiceMode
    """
    exclusive = detect_exclusive_choice(
        allocation, threshold=exclusive_threshold
    )
    if exclusive.is_exclusive:
        return ChoiceMode.EXCLUSIVE

    # From counts, so that a spread of exactly the tolerance (55 / 45 with
    # 0.10) is not pushed above it by 0.55 - 0.45 in floating point.
    if allocation.total > 0:
        spread = (max(allocation.counts) - min(allocation.counts)) / allocation.total
    else:
        spread = max(allocation.proportions) - min(allocation.proportions)
    if spread <= matching_tolerance:
        return ChoiceMode.MATCHING

    return ChoiceMode.INTERMEDIATE
