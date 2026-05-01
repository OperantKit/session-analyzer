"""Schneider's (1969) two-state break-and-run segmentation.

Schneider analyzed FI cumulative records as a sequence of two
piecewise-linear segments: an initial low-rate "post-reinforcement
pause" and a subsequent high-rate "running" segment. Given response
times within a single inter-reinforcement interval, this module finds
the optimal break point ``t*`` that maximizes the difference between
the slopes of the two segments while explaining the observed counts.

The specific objective used here is the maximum increase in slope
across the candidate break point: for each candidate ``t*`` placed at
a response time, the slope below and above ``t*`` are computed, and
``t*`` is chosen to maximize ``slope_high - slope_low`` subject to a
minimum number of responses on each side.

Reference
---------
Schneider, B. A. (1969). A two-state analysis of fixed-interval
    responding in the pigeon. *Journal of the Experimental Analysis of
    Behavior*, *12*(5), 677-687. https://doi.org/10.1901/jeab.1969.12-677
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SchneiderTwoStateResult:
    """Two-state segmentation of one inter-reinforcement interval.

    Attributes
    ----------
    break_time : float
        Time within the interval at which the two-state model places
        the transition from low- to high-rate responding. Reported
        relative to ``interval_start``.
    pause_duration : float
        Duration of the low-rate phase (= ``break_time -
        interval_start``).
    run_duration : float
        Duration of the high-rate phase (= ``interval_end -
        break_time``).
    low_rate : float
        Responses per second during the pause phase.
    high_rate : float
        Responses per second during the run phase.
    slope_difference : float
        ``high_rate - low_rate``.
    n_low : int
    n_high : int
    """

    break_time: float
    pause_duration: float
    run_duration: float
    low_rate: float
    high_rate: float
    slope_difference: float
    n_low: int
    n_high: int


def schneider_two_state(
    response_times: Sequence[float],
    *,
    interval_start: float,
    interval_end: float,
    min_responses_per_segment: int = 2,
) -> SchneiderTwoStateResult | None:
    """Find the optimal two-state segmentation of a single FI interval.

    Parameters
    ----------
    response_times : sequence of float
        Response times within ``[interval_start, interval_end]``,
        sorted ascending.
    interval_start, interval_end : float
        Inter-reinforcement interval boundaries; ``end > start``.
    min_responses_per_segment : int
        Minimum number of responses required on each side of the break
        point. Default 2.

    Returns
    -------
    SchneiderTwoStateResult or None
        ``None`` if the interval contains fewer than
        ``2 * min_responses_per_segment`` responses (the model is
        under-identified below that).

    Raises
    ------
    ValueError
        On unsorted ``response_times``, out-of-range responses, or
        invalid interval bounds.
    """
    if interval_end <= interval_start:
        raise ValueError(
            "interval_end must be strictly greater than interval_start"
        )
    if min_responses_per_segment < 1:
        raise ValueError(
            "min_responses_per_segment must be >= 1"
        )
    if any(
        response_times[i] > response_times[i + 1]
        for i in range(len(response_times) - 1)
    ):
        raise ValueError("response_times must be sorted ascending")
    if response_times and (
        response_times[0] < interval_start or response_times[-1] > interval_end
    ):
        raise ValueError(
            "all response_times must lie within [interval_start, interval_end]"
        )

    n = len(response_times)
    if n < 2 * min_responses_per_segment:
        return None

    times = list(response_times)
    best: SchneiderTwoStateResult | None = None
    best_diff = -float("inf")

    # Candidate breaks: between response i and i+1, with at least
    # `min_responses_per_segment` responses on each side. We place
    # the break at the response time r_{i+1}, so responses 0..i fall
    # into the low segment and i+1..n-1 fall into the high segment.
    for i in range(min_responses_per_segment - 1, n - min_responses_per_segment):
        n_low = i + 1
        n_high = n - n_low
        break_time = times[i + 1]
        low_dur = break_time - interval_start
        high_dur = interval_end - break_time
        if low_dur <= 0.0 or high_dur <= 0.0:
            continue
        low_rate = n_low / low_dur
        high_rate = n_high / high_dur
        diff = high_rate - low_rate
        if diff > best_diff:
            best_diff = diff
            best = SchneiderTwoStateResult(
                break_time=float(break_time),
                pause_duration=float(low_dur),
                run_duration=float(high_dur),
                low_rate=float(low_rate),
                high_rate=float(high_rate),
                slope_difference=float(diff),
                n_low=n_low,
                n_high=n_high,
            )

    return best
