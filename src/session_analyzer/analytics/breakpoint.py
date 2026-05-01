"""Progressive-ratio (PR) breakpoint detection.

The **breakpoint** is the highest ratio a subject completes before
abandoning responding. Three conventional operationalizations are
supported here:

1. **Terminal-pause criterion** (Richardson & Roberts, 1996), via
   :func:`detect_breakpoint`: the session ends when the subject fails
   to complete the next ratio within a fixed window (commonly 5 min).
   The breakpoint is the last completed ratio. Requires a known
   ``session_end_time`` and is a post-hoc measure.
2. **Final completed ratio at scheduled session end**: if the session
   ended on time (not by pause), the breakpoint is simply the highest
   completed ratio — i.e. the subject did not "break." Reported via the
   ``method = "session_ended"`` outcome of :func:`detect_breakpoint`.
3. **IRT-within-current-step criterion**, via
   :func:`estimate_irt_breakpoint`: the break point is declared when an
   inter-response time inside the *current* (not yet completed) ratio
   step exceeds a configurable threshold. Works on an ongoing session
   without knowing ``session_end_time`` and is the preferred live
   indicator on a dashboard.

References
----------
Hodos, W. (1961). Progressive ratio as a measure of reward strength.
    *Science*, *134*(3483), 943-944.
    https://doi.org/10.1126/science.134.3483.943

Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio
    schedules in drug self-administration studies in rats: a method to
    evaluate reinforcing efficacy. *Journal of Neuroscience Methods*,
    *66*(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0

Roane, H. S. (2008). On the applied use of progressive-ratio schedules
    of reinforcement. *Journal of Applied Behavior Analysis*, *41*(2),
    155-161. https://doi.org/10.1901/jaba.2008.41-155
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

_DEFAULT_PAUSE_THRESHOLD_SEC: float = 300.0  # Richardson & Roberts (1996)
_DEFAULT_IRT_THRESHOLD_SEC: float = 60.0  # standard PR pharmacology threshold
_APPROACHING_FACTOR: float = 0.5  # IRT >= this * threshold ⇒ "approaching"


@dataclass(frozen=True)
class BreakpointResult:
    """Progressive-ratio breakpoint detection result.

    Attributes
    ----------
    breakpoint_ratio : int or None
        The ratio value of the last completed ratio when a breakpoint is
        detected by the pause criterion. ``None`` if the session ended
        before the pause threshold (no break inferred) or if no ratios
        were completed.
    n_ratios_completed : int
        Total number of ratios completed during the session.
    terminal_pause_sec : float
        Elapsed time between the last completion and session end.
    threshold_sec : float
        The pause threshold used for detection.
    method : str
        One of ``"pause_threshold"``, ``"session_ended"``,
        ``"no_completions"``.
    """

    breakpoint_ratio: int | None
    n_ratios_completed: int
    terminal_pause_sec: float
    threshold_sec: float
    method: str


def detect_breakpoint(
    completions: Sequence[tuple[int, float]],
    *,
    session_end_time: float,
    pause_threshold_sec: float = _DEFAULT_PAUSE_THRESHOLD_SEC,
) -> BreakpointResult:
    """Detect the breakpoint from a sequence of ratio completions.

    Parameters
    ----------
    completions : sequence of (int, float)
        Ordered sequence of ``(ratio_value, completion_time_sec)`` pairs,
        one per completed ratio, sorted ascending by completion time.
    session_end_time : float
        Session end time (seconds since session start). Must be greater
        than or equal to the last completion time.
    pause_threshold_sec : float
        Pause duration that qualifies as a breakpoint. Default 300 s
        (Richardson & Roberts, 1996).

    Returns
    -------
    BreakpointResult

    Raises
    ------
    ValueError
        If completion times are not sorted ascending, or if
        ``session_end_time`` is before the last completion.
    """
    if pause_threshold_sec <= 0.0:
        raise ValueError(
            f"pause_threshold_sec must be positive, got {pause_threshold_sec}"
        )

    if not completions:
        return BreakpointResult(
            breakpoint_ratio=None,
            n_ratios_completed=0,
            terminal_pause_sec=session_end_time,
            threshold_sec=pause_threshold_sec,
            method="no_completions",
        )

    times = [t for _, t in completions]
    if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
        raise ValueError("completion times must be sorted ascending")

    last_ratio, last_time = completions[-1]
    if session_end_time < last_time:
        raise ValueError(
            "session_end_time must be >= the last completion time "
            f"({session_end_time} < {last_time})"
        )

    terminal_pause = session_end_time - last_time
    if terminal_pause >= pause_threshold_sec:
        return BreakpointResult(
            breakpoint_ratio=int(last_ratio),
            n_ratios_completed=len(completions),
            terminal_pause_sec=terminal_pause,
            threshold_sec=pause_threshold_sec,
            method="pause_threshold",
        )

    return BreakpointResult(
        breakpoint_ratio=None,
        n_ratios_completed=len(completions),
        terminal_pause_sec=terminal_pause,
        threshold_sec=pause_threshold_sec,
        method="session_ended",
    )


# ---------------------------------------------------------------------------
# IRT-within-current-step (live) breakpoint estimate
# ---------------------------------------------------------------------------


IRTBreakpointStatus = Literal["active", "approaching", "reached", "insufficient_data"]


@dataclass(frozen=True)
class IRTBreakpointResult:
    """Live break-point estimate using the IRT-within-current-step criterion.

    Complementary to :class:`BreakpointResult` (terminal-pause): this
    method works on an ongoing session without knowing the session end
    time, and reports a four-level status suitable for a live
    dashboard indicator.

    Attributes
    ----------
    breakpoint_reached : bool
        True when any IRT inside the current (incomplete) ratio step
        exceeds ``irt_threshold_sec``.
    last_irt_sec : float | None
        Most recent IRT (s) within the current step, or ``None`` when
        fewer than two responses have occurred since the last
        reinforcer.
    irt_threshold_sec : float
        IRT threshold used to declare the behavioural break point.
    last_completed_ratio : int
        Ratio requirement of the last completed step (BP-2).
    current_requirement : int
        Estimated ratio requirement for the current step. Pass ``0``
        if unknown.
    n_completed : int
        Total reinforcers delivered in this session so far.
    status : "active" | "approaching" | "reached" | "insufficient_data"
        Coarse-grained dashboard label.
    """

    breakpoint_reached: bool
    last_irt_sec: float | None
    irt_threshold_sec: float
    last_completed_ratio: int
    current_requirement: int
    n_completed: int
    status: IRTBreakpointStatus


def estimate_irt_breakpoint(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    *,
    irt_threshold_sec: float = _DEFAULT_IRT_THRESHOLD_SEC,
    last_completed_ratio: int = 0,
    current_requirement: int = 0,
) -> IRTBreakpointResult:
    """Estimate live break-point status from IRTs in the current ratio step.

    Inspects responses that occurred *after* the most recent
    reinforcement (or all responses, if no reinforcement has been
    delivered yet) and computes their IRTs. Declares
    ``breakpoint_reached`` when any of those IRTs is at least
    ``irt_threshold_sec``.

    Both ``response_times`` and ``reinforcement_times`` MUST be sorted
    ascending.

    Parameters
    ----------
    response_times : sequence of float
        Response timestamps (seconds), sorted ascending.
    reinforcement_times : sequence of float
        Reinforcement-delivery timestamps (seconds), sorted ascending.
    irt_threshold_sec : float, default 60.0
        IRT threshold (s) used to declare the break point. 60 s is
        standard in pharmacological PR studies; some labs use 120-300 s.
    last_completed_ratio : int, default 0
        Ratio requirement of the last completed step. Reported in the
        result for dashboard display; not used in the calculation.
    current_requirement : int, default 0
        Expected response requirement of the current step, e.g. from
        :func:`session_analyzer.analytics.pr_schedule.classify_pr_schedule`.
        Reported in the result; not used in the calculation.

    Returns
    -------
    IRTBreakpointResult

    Raises
    ------
    ValueError
        If ``irt_threshold_sec`` is not positive.
    """
    if irt_threshold_sec <= 0.0:
        raise ValueError(
            f"irt_threshold_sec must be positive, got {irt_threshold_sec}"
        )

    n_completed = len(reinforcement_times)
    last_rf = reinforcement_times[-1] if reinforcement_times else None
    if last_rf is not None:
        current = [t for t in response_times if t > last_rf]
    else:
        current = list(response_times)

    if len(current) < 2:
        return IRTBreakpointResult(
            breakpoint_reached=False,
            last_irt_sec=None,
            irt_threshold_sec=irt_threshold_sec,
            last_completed_ratio=last_completed_ratio,
            current_requirement=current_requirement,
            n_completed=n_completed,
            status="insufficient_data",
        )

    irts = [b - a for a, b in zip(current, current[1:], strict=False) if b > a]
    if not irts:
        return IRTBreakpointResult(
            breakpoint_reached=False,
            last_irt_sec=None,
            irt_threshold_sec=irt_threshold_sec,
            last_completed_ratio=last_completed_ratio,
            current_requirement=current_requirement,
            n_completed=n_completed,
            status="insufficient_data",
        )

    last_irt = irts[-1]
    breakpoint_reached = any(irt >= irt_threshold_sec for irt in irts)
    if breakpoint_reached:
        status: IRTBreakpointStatus = "reached"
    elif last_irt >= irt_threshold_sec * _APPROACHING_FACTOR:
        status = "approaching"
    else:
        status = "active"

    return IRTBreakpointResult(
        breakpoint_reached=breakpoint_reached,
        last_irt_sec=last_irt,
        irt_threshold_sec=irt_threshold_sec,
        last_completed_ratio=last_completed_ratio,
        current_requirement=current_requirement,
        n_completed=n_completed,
        status=status,
    )
