"""Response latency analysis.

Latency is operationalized as the elapsed time between a **reference event**
(typically the onset of a discriminative stimulus, trial start, or
reinforcer delivery) and the **first response** that follows it.

This module provides two functions:

- :func:`compute_latencies` — pair reference times with the next response
  and return per-trial latencies.
- :func:`analyze_latencies` — summarize a latency sample (mean, median,
  standard deviation, etc.).

The design keeps the operationalization of "reference event" at the call
site: `session-analyzer` does not assume a particular OKL v1 event type so
that latency can be computed for SD onset, trial start, post-reinforcer
pause, or any other experimentally-defined anchor.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LatencyMetrics:
    """Summary statistics for a sample of response latencies.

    Attributes
    ----------
    mean, median, std, min, max : float
        Descriptive statistics of the latency sample, in seconds.
    n : int
        Number of valid (positive) latencies used to compute the statistics.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int


def _is_sorted(values: Sequence[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def compute_latencies(
    reference_times: Sequence[float],
    response_times: Sequence[float],
    *,
    max_latency_sec: float | None = None,
) -> list[float]:
    """Compute per-trial latencies from reference events to next response.

    For each time ``t`` in ``reference_times``, the latency is
    ``r - t`` where ``r`` is the first element of ``response_times``
    strictly greater than ``t``. If no such response exists, the trial
    contributes no latency.

    Parameters
    ----------
    reference_times : sequence of float
        Times of reference events (e.g. SD onsets), sorted ascending.
    response_times : sequence of float
        Times of responses, sorted ascending.
    max_latency_sec : float, optional
        If provided, latencies greater than this value are discarded
        (common cut-off for "no response" trials, e.g. 60 s).

    Returns
    -------
    list of float
        Per-trial latencies in seconds.

    Raises
    ------
    ValueError
        If either input sequence is not sorted ascending.
    """
    if not _is_sorted(reference_times):
        raise ValueError("reference_times must be sorted ascending")
    if not _is_sorted(response_times):
        raise ValueError("response_times must be sorted ascending")

    sorted_responses = list(response_times)
    latencies: list[float] = []
    for t in reference_times:
        idx = bisect.bisect_right(sorted_responses, t)
        if idx >= len(sorted_responses):
            continue
        lat = sorted_responses[idx] - t
        if max_latency_sec is not None and lat > max_latency_sec:
            continue
        latencies.append(lat)
    return latencies


def analyze_latencies(latencies: Sequence[float]) -> LatencyMetrics | None:
    """Summarize a sample of response latencies.

    Parameters
    ----------
    latencies : sequence of float
        Per-trial latencies in seconds. Non-positive values are discarded.

    Returns
    -------
    LatencyMetrics or None
        Returns ``None`` when no positive latencies are available.
    """
    valid = [x for x in latencies if x > 0.0]
    if not valid:
        return None

    arr = np.array(valid, dtype=np.float64)
    return LatencyMetrics(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=0)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n=len(valid),
    )
