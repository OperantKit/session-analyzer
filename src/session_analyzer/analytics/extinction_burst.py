"""Extinction-burst detection.

When previously reinforced responding is placed on extinction, response
rate often shows a transient *burst* — rate increases above the
baseline level for a short window before declining. This module
quantifies that burst by comparing initial-extinction rate to baseline
rate.

Lerman & Iwata (1995) describe the burst as an initial increase in the
target response under extinction and do not fix a window length or a
ratio; the 60-s window and the 1.5 threshold are this module's defaults.
The window is closed: a response exactly ``burst_window_sec`` after onset
is counted.

References
----------
Lerman, D. C., & Iwata, B. A. (1995). Prevalence of the extinction
    burst and its attenuation during treatment. *Journal of Applied
    Behavior Analysis*, *28*(1), 93-94.
    https://doi.org/10.1901/jaba.1995.28-93
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExtinctionBurstResult:
    """Extinction-burst detection result.

    Attributes
    ----------
    baseline_rate : float
        Responses per second during the baseline period.
    burst_rate : float
        Responses per second within the initial extinction window.
    burst_ratio : float
        ``burst_rate / baseline_rate``. ``+inf`` if baseline rate is 0
        and any extinction responses are observed.
    burst_detected : bool
        ``burst_ratio >= ratio_threshold``.
    burst_window_sec : float
    ratio_threshold : float
    """

    baseline_rate: float
    burst_rate: float
    burst_ratio: float
    burst_detected: bool
    burst_window_sec: float
    ratio_threshold: float


def detect_extinction_burst(
    *,
    baseline_response_times: Sequence[float],
    baseline_duration_sec: float,
    extinction_start_time: float,
    extinction_response_times: Sequence[float],
    burst_window_sec: float = 60.0,
    ratio_threshold: float = 1.5,
) -> ExtinctionBurstResult:
    """Compare baseline rate to initial-extinction rate.

    Parameters
    ----------
    baseline_response_times : sequence of float
        Response times during the baseline period (sorted ascending).
    baseline_duration_sec : float
        Length of the baseline period in seconds (strictly positive).
    extinction_start_time : float
        Absolute time at which extinction began.
    extinction_response_times : sequence of float
        Response times during extinction (sorted ascending; must be
        >= ``extinction_start_time``).
    burst_window_sec : float
        Length of the post-onset window over which burst rate is
        computed (default 60 s).
    ratio_threshold : float
        Minimum ``burst_rate / baseline_rate`` for the burst flag.
        Default 1.5 (50 % increase).

    Returns
    -------
    ExtinctionBurstResult

    Raises
    ------
    ValueError
        If durations are non-positive, sequences are not sorted, or
        any extinction time precedes ``extinction_start_time``.
    """
    if baseline_duration_sec <= 0.0:
        raise ValueError(
            f"baseline_duration_sec must be positive, got {baseline_duration_sec}"
        )
    if burst_window_sec <= 0.0:
        raise ValueError(
            f"burst_window_sec must be positive, got {burst_window_sec}"
        )
    if ratio_threshold <= 0.0:
        raise ValueError(
            f"ratio_threshold must be positive, got {ratio_threshold}"
        )
    if any(
        baseline_response_times[i] > baseline_response_times[i + 1]
        for i in range(len(baseline_response_times) - 1)
    ):
        raise ValueError("baseline_response_times must be sorted ascending")
    if any(
        extinction_response_times[i] > extinction_response_times[i + 1]
        for i in range(len(extinction_response_times) - 1)
    ):
        raise ValueError("extinction_response_times must be sorted ascending")
    if extinction_response_times and extinction_response_times[0] < extinction_start_time:
        raise ValueError(
            "extinction_response_times must all be >= extinction_start_time"
        )

    baseline_rate = len(baseline_response_times) / baseline_duration_sec

    end_time = extinction_start_time + burst_window_sec
    n_in_window = bisect.bisect_right(list(extinction_response_times), end_time)
    burst_rate = n_in_window / burst_window_sec

    if baseline_rate == 0.0:
        ratio = float("inf") if burst_rate > 0.0 else 0.0
    else:
        ratio = burst_rate / baseline_rate

    return ExtinctionBurstResult(
        baseline_rate=float(baseline_rate),
        burst_rate=float(burst_rate),
        burst_ratio=float(ratio),
        burst_detected=ratio >= ratio_threshold,
        burst_window_sec=float(burst_window_sec),
        ratio_threshold=float(ratio_threshold),
    )
