"""Post-reinforcement pause (PRP) and related ratio-schedule metrics.

Under fixed-ratio (FR) and fixed-interval (FI) schedules, responding
typically resumes after a pause following each reinforcer. This module
provides two analyses:

- :func:`compute_prps` — extract per-trial pauses as the latency from
  each reinforcer to the next response.
- :func:`analyze_prps` — summarize a PRP sample (mean, median, spread).
- :func:`compute_quarter_life` — Herrnstein & Morse (1957) quarter-life
  index: the proportion of the inter-reinforcement interval elapsed
  before 25% of the interval's responses have occurred. Values near
  0.25 indicate uniform responding; values well above 0.25 indicate an
  acceleration ("scallop") pattern typical of FI schedules.

References
----------
Felton, M., & Lyon, D. O. (1966). The post-reinforcement pause.
    *Journal of the Experimental Analysis of Behavior*, *9*(2),
    131-134. https://doi.org/10.1901/jeab.1966.9-131

Herrnstein, R. J., & Morse, W. H. (1957). A conjunctive schedule of
    reinforcement. *Journal of the Experimental Analysis of Behavior*,
    *1*(1), 15-24. https://doi.org/10.1901/jeab.1958.1-15
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PRPMetrics:
    """Summary statistics for a sample of post-reinforcement pauses.

    Attributes
    ----------
    mean, median, std, min, max : float
        Descriptive statistics of the PRP sample, in seconds.
    n : int
        Number of valid (positive) PRPs.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int


def compute_prps(
    reinforcement_times: Sequence[float],
    response_times: Sequence[float],
) -> list[float]:
    """Extract per-trial post-reinforcement pauses.

    For each reinforcement time, the PRP is the elapsed time until the
    next response. Reinforcers followed by no further responses are
    omitted.

    Parameters
    ----------
    reinforcement_times : sequence of float
        Times of reinforcer deliveries, sorted ascending.
    response_times : sequence of float
        Times of responses, sorted ascending.

    Returns
    -------
    list of float
        Per-trial PRPs in seconds.

    Raises
    ------
    ValueError
        If either sequence is not sorted ascending.
    """
    if any(
        reinforcement_times[i] > reinforcement_times[i + 1]
        for i in range(len(reinforcement_times) - 1)
    ):
        raise ValueError("reinforcement_times must be sorted ascending")
    if any(
        response_times[i] > response_times[i + 1]
        for i in range(len(response_times) - 1)
    ):
        raise ValueError("response_times must be sorted ascending")

    sorted_r = list(response_times)
    prps: list[float] = []
    import bisect

    for t in reinforcement_times:
        idx = bisect.bisect_right(sorted_r, t)
        if idx >= len(sorted_r):
            continue
        prps.append(sorted_r[idx] - t)
    return prps


def analyze_prps(prps: Sequence[float]) -> PRPMetrics | None:
    """Summarize a sample of post-reinforcement pauses."""
    valid = [x for x in prps if x > 0.0]
    if not valid:
        return None
    arr = np.array(valid, dtype=np.float64)
    return PRPMetrics(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=0)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n=len(valid),
    )


def compute_quarter_life(
    reinforcement_times: Sequence[float],
    response_times: Sequence[float],
    *,
    quantile: float = 0.25,
) -> list[float]:
    """Compute per-interval quarter-life indices (Herrnstein & Morse, 1957).

    For each inter-reinforcement interval, the quarter-life is the
    fraction of the interval elapsed by the time ``quantile`` (default
    0.25) of that interval's responses have occurred. Values near
    ``quantile`` (about 0.25 by default) indicate uniform responding;
    values well above it indicate an FI-style
    scallop (acceleration toward reinforcement). Intervals with fewer
    than two responses are skipped.

    Parameters
    ----------
    reinforcement_times : sequence of float
        Times of reinforcer deliveries, sorted ascending (at least 2
        required to define an interval).
    response_times : sequence of float
        Times of responses, sorted ascending.
    quantile : float
        Response-count quantile whose relative time is computed
        (default 0.25 for the classical quarter-life).

    Returns
    -------
    list of float
        Per-interval quarter-life values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``quantile`` is outside ``(0, 1)`` or either input sequence
        is not sorted.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if any(
        reinforcement_times[i] > reinforcement_times[i + 1]
        for i in range(len(reinforcement_times) - 1)
    ):
        raise ValueError("reinforcement_times must be sorted ascending")
    if any(
        response_times[i] > response_times[i + 1]
        for i in range(len(response_times) - 1)
    ):
        raise ValueError("response_times must be sorted ascending")

    results: list[float] = []
    r_arr = np.array(response_times, dtype=np.float64)
    for i in range(len(reinforcement_times) - 1):
        start = float(reinforcement_times[i])
        end = float(reinforcement_times[i + 1])
        if end <= start:
            continue
        mask = (r_arr > start) & (r_arr <= end)
        in_interval = r_arr[mask]
        if in_interval.size < 2:
            continue
        # Rank of the ceil(q * n)-th response (1-based index) gives the
        # first response at or past the q-th quantile of the count.
        target_index = max(int(np.ceil(quantile * in_interval.size)) - 1, 0)
        q_time = float(in_interval[target_index])
        fraction = (q_time - start) / (end - start)
        results.append(fraction)
    return results
