"""Progressive-ratio (PR) schedule classification.

Detects whether a sequence of completed step ratios is consistent with a
Progressive Ratio schedule and, if so, which step algorithm generates it
(Hodos / Richardson & Roberts vs linear vs unknown). Complements
:mod:`session_analyzer.analytics.breakpoint`, which handles the
break-point question from the same data.

The two operations are intentionally separate:

- *Classification* asks "what schedule produced these completions?" and
  is used to decide which downstream analyses apply (e.g. the Hodos
  step formula vs an arithmetic series).
- *Break-point detection* asks "did the subject stop responding?" and
  may be answered by either the terminal-pause criterion
  (:func:`session_analyzer.analytics.breakpoint.detect_breakpoint`) or
  the IRT-within-current-step criterion
  (:func:`session_analyzer.analytics.breakpoint.estimate_irt_breakpoint`).

References
----------
Hodos, W. (1961). Progressive ratio as a measure of reward strength.
    *Science*, *134*(3483), 943-944.
    https://doi.org/10.1126/science.134.3483.943

Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio
    schedules in drug self-administration studies in rats.
    *Journal of Neuroscience Methods*, *66*(1), 1-11.
    https://doi.org/10.1016/0165-0270(95)00153-0
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_MIN_STEPS_FOR_DETECTION = 3
"""Minimum completed steps required before classifying a session as PR."""

StepType = Literal["hodos", "linear", "unknown", "none"]


@dataclass(frozen=True)
class PRClassification:
    """Result of PR schedule classification.

    Attributes
    ----------
    is_pr : bool
        True when the data are consistent with a Progressive Ratio
        schedule (monotone-increasing step ratios with at least
        :data:`_MIN_STEPS_FOR_DETECTION` completed steps).
    step_ratios : list of int
        The completed response requirement at each step, one entry per
        reinforcer delivered.
    last_completed_ratio : int
        Ratio requirement of the last completed step (the BP-2 measure).
        ``0`` if no reinforcers have been delivered.
    current_requirement : int
        Estimated ratio requirement for the current (incomplete) step.
        Computed by extrapolating the detected step formula; falls back
        to the last observed delta when the step type is ``unknown``.
    n_completed : int
        Number of reinforcers delivered (= number of completed steps).
    step_type : "hodos" | "linear" | "unknown" | "none"
        Detected step algorithm. ``"none"`` means no completions; other
        values are reported even when ``is_pr`` is False (e.g. a
        non-monotone or too-short sequence still has a step type of
        ``"none"`` or ``"unknown"`` for downstream display).
    """

    is_pr: bool
    step_ratios: list[int]
    last_completed_ratio: int
    current_requirement: int
    n_completed: int
    step_type: StepType


# ---------------------------------------------------------------------------
# Step-formula primitives
# ---------------------------------------------------------------------------


def _hodos_requirement(n: int) -> int:
    """Hodos / Richardson & Roberts (1996) ratio for step *n* (1-based)."""
    return max(1, round(5 * math.exp(0.2 * n) - 5))


def _is_hodos(ratios: Sequence[int]) -> bool:
    """Return True when *ratios* match the Hodos formula within +/-15% (or +/-1)."""
    for i, observed in enumerate(ratios):
        expected = _hodos_requirement(i + 1)
        tolerance = max(1, round(expected * 0.15))
        if abs(observed - expected) > tolerance:
            return False
    return True


def _detect_linear(ratios: Sequence[int]) -> tuple[bool, int, int]:
    """Check whether *ratios* form an arithmetic sequence.

    Returns
    -------
    (is_linear, start, increment)
        ``is_linear`` is True iff the sequence has at least 2 entries
        and is arithmetic with a positive common difference (within
        +/-1 tolerance per entry). ``start`` and ``increment`` describe
        that arithmetic.
    """
    if len(ratios) < 2:
        return False, 0, 0
    increment = ratios[1] - ratios[0]
    if increment <= 0:
        return False, 0, 0
    start = ratios[0]
    for i, r in enumerate(ratios):
        if abs(r - (start + increment * i)) > 1:
            return False, 0, 0
    return True, start, increment


def _is_monotone_increasing(ratios: Sequence[int]) -> bool:
    return all(b > a for a, b in zip(ratios, ratios[1:], strict=False))


def _next_requirement(
    ratios: Sequence[int],
    step_type: StepType,
    lin_start: int = 1,
    lin_inc: int = 5,
) -> int:
    """Estimate the response requirement for the next (currently incomplete) step."""
    n = len(ratios) + 1  # 1-based index of the upcoming step
    if step_type == "hodos":
        return _hodos_requirement(n)
    if step_type == "linear":
        return lin_start + lin_inc * (n - 1)
    if len(ratios) >= 2:
        return max(1, ratios[-1] + (ratios[-1] - ratios[-2]))
    if ratios:
        return ratios[-1] + 1
    return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_step_ratios(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    *,
    timeout_sec: float = 0.0,
) -> list[int]:
    """Extract the response requirement completed at each reinforcement.

    For each reinforcement event, the step ratio equals the number of
    responses that occurred between the previous reinforcement (or
    session start) and this reinforcement.

    Both inputs MUST be sorted ascending.

    Parameters
    ----------
    response_times : sequence of float
        Response timestamps (seconds), sorted ascending.
    reinforcement_times : sequence of float
        Reinforcement-delivery timestamps (seconds), sorted ascending.
    timeout_sec : float, optional
        Post-reinforcement timeout. Responses with
        ``rf < t < rf + timeout_sec`` after a reinforcer at ``rf`` are not
        counted toward the next ratio (many PR programmes ignore
        responses during the reinforcer delivery / timeout). A response at
        exactly ``rf`` still completes the step that ``rf`` reinforces.
        Default ``0.0`` counts every response.

    Returns
    -------
    list of int
        One entry per reinforcer; the value is the response count that
        completed that step.

    Raises
    ------
    ValueError
        If ``timeout_sec`` is negative.
    """
    if timeout_sec < 0.0:
        raise ValueError(f"timeout_sec must be >= 0, got {timeout_sec}")
    if not reinforcement_times:
        return []
    ratios: list[int] = []
    prev_idx = 0
    for rf_time in reinforcement_times:
        # bisect_right counts responses with timestamp <= rf_time when
        # the response stream is sorted ascending. (A response emitted
        # at exactly rf_time is treated as completing this step.)
        idx = bisect_right(response_times, rf_time)
        ratios.append(max(0, idx - prev_idx))
        prev_idx = max(prev_idx, idx)
        if timeout_sec > 0.0:
            # Skip responses strictly inside (rf_time, rf_time + timeout_sec).
            skip_to = bisect_left(response_times, rf_time + timeout_sec)
            prev_idx = max(prev_idx, skip_to)
    return ratios


def classify_pr_schedule(step_ratios: Sequence[int]) -> PRClassification:
    """Classify completed step ratios as Hodos / linear / unknown PR.

    Parameters
    ----------
    step_ratios : sequence of int
        Response requirements completed at each step, in order. Use
        :func:`extract_step_ratios` to derive these from raw timestamp
        streams.

    Returns
    -------
    PRClassification
        Detection outcome plus derived fields (last completed ratio,
        next-step requirement estimate, step-type label).
    """
    ratios = list(step_ratios)
    n = len(ratios)

    if n == 0:
        return PRClassification(
            is_pr=False,
            step_ratios=[],
            last_completed_ratio=0,
            current_requirement=1,
            n_completed=0,
            step_type="none",
        )

    last_completed = ratios[-1]

    if n < _MIN_STEPS_FOR_DETECTION:
        return PRClassification(
            is_pr=False,
            step_ratios=ratios,
            last_completed_ratio=last_completed,
            current_requirement=_next_requirement(ratios, "unknown"),
            n_completed=n,
            step_type="unknown",
        )

    if not _is_monotone_increasing(ratios):
        return PRClassification(
            is_pr=False,
            step_ratios=ratios,
            last_completed_ratio=last_completed,
            current_requirement=last_completed,
            n_completed=n,
            step_type="none",
        )

    linear_ok, lin_start, lin_inc = _detect_linear(ratios)
    if linear_ok:
        step_type: StepType = "linear"
        current_req = _next_requirement(ratios, step_type, lin_start, lin_inc)
    elif _is_hodos(ratios):
        step_type = "hodos"
        current_req = _next_requirement(ratios, step_type)
    else:
        step_type = "unknown"
        current_req = _next_requirement(ratios, step_type)

    return PRClassification(
        is_pr=True,
        step_ratios=ratios,
        last_completed_ratio=last_completed,
        current_requirement=current_req,
        n_completed=n,
        step_type=step_type,
    )
