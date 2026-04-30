"""Progressive-ratio (PR) breakpoint detection.

The **breakpoint** is the highest ratio a subject completes before
abandoning responding. Two conventional operationalizations exist:

1. **Terminal-pause criterion** (Richardson & Roberts, 1996): the session
   ends when the subject fails to complete the next ratio within a
   fixed window (commonly 5 minutes). The breakpoint is the last
   completed ratio.
2. **Final completed ratio at scheduled session end**: if the session
   ended on time (not by pause), the breakpoint is simply the highest
   completed ratio — i.e. the subject did not "break."

This module supports both cases. The caller supplies completion times
and the session end time; the function classifies which regime applies.

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

from dataclasses import dataclass
from typing import Sequence


_DEFAULT_PAUSE_THRESHOLD_SEC: float = 300.0  # Richardson & Roberts (1996)


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
