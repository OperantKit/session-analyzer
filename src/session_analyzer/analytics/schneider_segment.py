"""Schneider's (1969) two-state break-and-run segmentation.

Schneider analyzed FI cumulative records as two states: an initial
low-rate "post-reinforcement pause" and a subsequent high-rate
"running" state. The numerical procedure fits each interval's
cumulative record with two straight lines: one through the origin up to
the breakpoint, and a second one joined to it at the breakpoint. The
breakpoint is moved systematically through the interval, and the one
that minimizes the sum of squared deviations from the two lines is
taken as the breakpoint.

Here the cumulative record is the count ``1, 2, ..., n`` at each
response time, the breakpoint is moved along an evenly spaced grid of
the interval, and both fitted rates are kept non-negative (if the
unconstrained low rate is negative, the high-rate line alone is fitted
with the low rate fixed at 0, and vice versa).

Reference
---------
Schneider, B. A. (1969). A two-state analysis of fixed-interval
    responding in the pigeon. *Journal of the Experimental Analysis of
    Behavior*, *12*(5), 677-687. https://doi.org/10.1901/jeab.1969.12-677
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SchneiderTwoStateResult:
    """Two-state segmentation of one inter-reinforcement interval.

    Attributes
    ----------
    break_time : float
        Time at which the two-state model places the transition from
        low- to high-rate responding, on the same clock as the response
        times (``interval_start <= break_time <= interval_end``).
    pause_duration : float
        Duration of the low-rate phase (= ``break_time -
        interval_start``).
    run_duration : float
        Duration of the high-rate phase (= ``interval_end -
        break_time``).
    low_rate : float
        Fitted slope of the line before the breakpoint, in responses
        per second.
    high_rate : float
        Fitted slope of the line after the breakpoint, in responses per
        second.
    slope_difference : float
        ``high_rate - low_rate``.
    n_low : int
        Responses at or before ``break_time``.
    n_high : int
        Responses after ``break_time``.
    """

    break_time: float
    pause_duration: float
    run_duration: float
    low_rate: float
    high_rate: float
    slope_difference: float
    n_low: int
    n_high: int


def _fit_two_lines(
    x1: Sequence[float], x2: Sequence[float], y: Sequence[int]
) -> tuple[float, float, float] | None:
    """Least-squares ``y = low * x1 + high * x2`` with both slopes >= 0.

    Returns ``(sse, low, high)`` or ``None`` when no slope is identified.
    """
    s11 = sum(u * u for u in x1)
    s22 = sum(v * v for v in x2)
    s12 = sum(u * v for u, v in zip(x1, x2, strict=True))
    s1y = sum(u * c for u, c in zip(x1, y, strict=True))
    s2y = sum(v * c for v, c in zip(x2, y, strict=True))
    det = s11 * s22 - s12 * s12
    if det <= 0.0:
        return None
    low = (s1y * s22 - s2y * s12) / det
    high = (s2y * s11 - s1y * s12) / det
    if low < 0.0:
        low, high = 0.0, s2y / s22
    elif high < 0.0:
        low, high = s1y / s11, 0.0
    sse = sum((c - low * u - high * v) ** 2 for u, v, c in zip(x1, x2, y, strict=True))
    return sse, low, high


def schneider_two_state(
    response_times: Sequence[float],
    *,
    interval_start: float,
    interval_end: float,
    min_responses_per_segment: int = 2,
    grid: int = 200,
) -> SchneiderTwoStateResult | None:
    """Fit Schneider's (1969) two-state model to a single FI interval.

    Parameters
    ----------
    response_times : sequence of float
        Response times within ``[interval_start, interval_end]``,
        sorted ascending.
    interval_start, interval_end : float
        Inter-reinforcement interval boundaries; ``end > start``.
    min_responses_per_segment : int
        The interval must hold at least ``2 * min_responses_per_segment``
        responses. The fit itself does not require responses on each
        side of the breakpoint, because state 1 often holds none.
        Default 2.
    grid : int
        Number of equal steps of the interval along which the
        breakpoint is moved (candidates exclude both ends). Default 200.

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
        invalid interval bounds, ``min_responses_per_segment`` or
        ``grid``.
    """
    if interval_end <= interval_start:
        raise ValueError(
            "interval_end must be strictly greater than interval_start"
        )
    if min_responses_per_segment < 1:
        raise ValueError(
            "min_responses_per_segment must be >= 1"
        )
    if grid < 2:
        raise ValueError("grid must be >= 2")
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

    times = [t - interval_start for t in response_times]
    counts = range(1, n + 1)
    length = interval_end - interval_start
    best: tuple[float, float, float, float] | None = None
    for step in range(1, grid):
        b = length * step / grid
        x1 = [min(t, b) for t in times]
        x2 = [max(0.0, t - b) for t in times]
        fit = _fit_two_lines(x1, x2, counts)
        if fit is not None and (best is None or fit[0] < best[0]):
            best = (fit[0], b, fit[1], fit[2])
    if best is None:
        return None

    _, b, low, high = best
    n_low = sum(1 for t in times if t <= b)
    return SchneiderTwoStateResult(
        break_time=float(interval_start + b),
        pause_duration=float(b),
        run_duration=float(length - b),
        low_rate=float(low),
        high_rate=float(high),
        slope_difference=float(high - low),
        n_low=n_low,
        n_high=n - n_low,
    )
