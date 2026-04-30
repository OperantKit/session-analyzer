"""Run-length analysis for ratio schedules.

Under fixed-ratio (FR) and variable-ratio (VR) schedules, responding
typically shows a "break-run" pattern: a post-reinforcement pause
followed by a rapid run of responses that completes the ratio. This
module decomposes each inter-reinforcement interval into a pause and
a run and reports per-run metrics.

A run is defined as the sequence of responses that occurs between one
reinforcer and the next. The pause is the time from the reinforcer to
the first response in the run; the run duration is the time from the
first to the last response in that run.

Reference
---------
Schlinger, H. D., Derenne, A., & Baron, A. (2008). What 50 years of
    research tell us about pausing under ratio schedules of
    reinforcement. *The Behavior Analyst*, *31*(1), 39-60.
    https://doi.org/10.1007/BF03392160
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RunMetrics:
    """Per-session FR run summary.

    Attributes
    ----------
    n_runs : int
        Number of runs with at least 2 responses (required to define
        a duration).
    mean_run_length : float
        Mean number of responses per run.
    mean_run_duration : float
        Mean run duration in seconds (first to last response in run).
    mean_run_rate : float
        Mean within-run response rate in responses/second.
    mean_pause_duration : float
        Mean pause in seconds (reinforcer to first response of run).
    pause_to_run_ratio : float
        Ratio ``mean_pause_duration / mean_run_duration``.
    """

    n_runs: int
    mean_run_length: float
    mean_run_duration: float
    mean_run_rate: float
    mean_pause_duration: float
    pause_to_run_ratio: float


def analyze_fr_runs(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
) -> RunMetrics | None:
    """Decompose responding into runs between reinforcers.

    Parameters
    ----------
    response_times : sequence of float
        Response times, sorted ascending.
    reinforcement_times : sequence of float
        Reinforcer delivery times, sorted ascending. At least 2 are
        required to define an inter-reinforcement interval.

    Returns
    -------
    RunMetrics or None
        ``None`` if fewer than 2 reinforcers are supplied or no run
        of at least 2 responses is observed.
    """
    if any(
        response_times[i] > response_times[i + 1]
        for i in range(len(response_times) - 1)
    ):
        raise ValueError("response_times must be sorted ascending")
    if any(
        reinforcement_times[i] > reinforcement_times[i + 1]
        for i in range(len(reinforcement_times) - 1)
    ):
        raise ValueError("reinforcement_times must be sorted ascending")

    if len(reinforcement_times) < 2:
        return None

    responses = list(response_times)

    run_lengths: list[int] = []
    run_durations: list[float] = []
    pause_durations: list[float] = []

    for i in range(len(reinforcement_times) - 1):
        start = float(reinforcement_times[i])
        end = float(reinforcement_times[i + 1])
        lo = bisect.bisect_right(responses, start)
        hi = bisect.bisect_right(responses, end)
        run = responses[lo:hi]
        if len(run) < 2:
            continue
        pause = run[0] - start
        duration = run[-1] - run[0]
        if duration <= 0.0:
            continue
        run_lengths.append(len(run))
        run_durations.append(duration)
        pause_durations.append(pause)

    if not run_lengths:
        return None

    lengths = np.array(run_lengths, dtype=np.float64)
    durations = np.array(run_durations, dtype=np.float64)
    pauses = np.array(pause_durations, dtype=np.float64)

    mean_len = float(np.mean(lengths))
    mean_dur = float(np.mean(durations))
    mean_pause = float(np.mean(pauses))
    # Within-run rate: use total responses / total duration to avoid
    # divide-by-zero amplification on short runs.
    total_responses = float(np.sum(lengths))
    total_duration = float(np.sum(durations))
    mean_rate = total_responses / total_duration if total_duration > 0.0 else 0.0

    ratio = mean_pause / mean_dur if mean_dur > 0.0 else float("inf")

    return RunMetrics(
        n_runs=len(run_lengths),
        mean_run_length=mean_len,
        mean_run_duration=mean_dur,
        mean_run_rate=float(mean_rate),
        mean_pause_duration=mean_pause,
        pause_to_run_ratio=float(ratio),
    )
