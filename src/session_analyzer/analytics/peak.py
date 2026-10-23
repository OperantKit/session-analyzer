"""Peak-procedure analysis (Roberts, 1981; Church, Meck, & Gibbon, 1994).

The **peak procedure** is an extension of fixed-interval (FI) schedules
in which occasional *peak trials* withhold reinforcement and extend the
trial well beyond the programmed interval. Response rate as a function
of time since trial onset typically shows a break-run-break pattern
peaking near the expected time of reinforcement.

Two complementary analyses are provided:

- :func:`analyze_peak_trial_binned` — bin-smoothed rate curve, peak
  time / rate, and half-max crossings (Roberts, 1981 conventional
  analysis).
- :func:`analyze_peak_trial_mle` — three-state maximum-likelihood
  analysis (Church, Meck, & Gibbon, 1994) that estimates start (s1)
  and stop (s2) times under the assumption of low-high-low Poisson
  response rates. The likelihood is this module's formulation of the
  low-high-low model; it has not been checked against the fitting
  criterion in the original article.

References
----------
Church, R. M., Meck, W. H., & Gibbon, J. (1994). Application of scalar
    timing theory to individual trials. *Journal of Experimental
    Psychology: Animal Behavior Processes*, *20*(2), 135-155.
    https://doi.org/10.1037/0097-7403.20.2.135

Roberts, S. (1981). Isolation of an internal clock. *Journal of
    Experimental Psychology: Animal Behavior Processes*, *7*(3),
    242-268. https://doi.org/10.1037/0097-7403.7.3.242
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PeakTrialBinnedMetrics:
    """Peak-trial metrics from binned rate curve (Roberts, 1981).

    Attributes
    ----------
    peak_time : float
        Time of maximum response rate, in seconds from trial onset.
    peak_rate : float
        Maximum response rate, in responses/second.
    start_time : float
        First time the rate curve exceeds half of ``peak_rate``.
    stop_time : float
        Last time the rate curve exceeds half of ``peak_rate``.
    spread : float
        ``stop_time - start_time``.
    middle_time : float
        ``(start_time + stop_time) / 2``.
    n_responses : int
        Total responses in the trial.
    bin_width : float
        Bin width used for rate estimation.
    """

    peak_time: float
    peak_rate: float
    start_time: float
    stop_time: float
    spread: float
    middle_time: float
    n_responses: int
    bin_width: float


@dataclass(frozen=True)
class PeakTrialMLEMetrics:
    """Peak-trial metrics from three-state MLE (Church, Meck, & Gibbon, 1994).

    Attributes
    ----------
    start_time : float
        Estimated time of transition from low to high rate (``s1``).
    stop_time : float
        Estimated time of transition from high to low rate (``s2``).
    high_rate : float
        MLE response rate during the high-rate (middle) period.
    low_rate : float
        MLE response rate during the two low-rate flanking periods.
    middle_time : float
        ``(start_time + stop_time) / 2``.
    spread : float
        ``stop_time - start_time``.
    n_responses : int
        Total responses in the trial.
    log_likelihood : float
        Maximised log-likelihood of the three-state model.
    """

    start_time: float
    stop_time: float
    high_rate: float
    low_rate: float
    middle_time: float
    spread: float
    n_responses: int
    log_likelihood: float


def _validate_trial(response_times: Sequence[float], trial_duration: float) -> np.ndarray:
    if trial_duration <= 0.0:
        raise ValueError(f"trial_duration must be positive, got {trial_duration}")
    arr = np.array(sorted(response_times), dtype=np.float64)
    if arr.size and (arr[0] < 0.0 or arr[-1] > trial_duration):
        raise ValueError(
            "all response times must lie within [0, trial_duration]"
        )
    return arr


def analyze_peak_trial_binned(
    response_times: Sequence[float],
    trial_duration: float,
    *,
    bin_width: float = 1.0,
) -> PeakTrialBinnedMetrics | None:
    """Bin-smoothed peak-procedure analysis (Roberts, 1981).

    Bins response counts with the given ``bin_width``, locates the bin
    with maximum rate, and returns the first/last bin exceeding
    half the peak as start/stop times.

    Parameters
    ----------
    response_times : sequence of float
        Response times within the trial, in seconds since trial onset.
    trial_duration : float
        Total trial duration, in seconds (strictly positive).
    bin_width : float
        Bin width for rate estimation (default 1 s).

    Returns
    -------
    PeakTrialBinnedMetrics or None
        ``None`` if no responses occurred in the trial.
    """
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    times = _validate_trial(response_times, trial_duration)
    if times.size == 0:
        return None

    n_bins = max(int(math.ceil(trial_duration / bin_width)), 1)
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_width
    counts, _ = np.histogram(times, bins=edges)
    rates = counts / bin_width

    peak_idx = int(np.argmax(rates))
    peak_rate = float(rates[peak_idx])
    peak_time = float(edges[peak_idx] + bin_width / 2.0)

    half = peak_rate / 2.0
    above = np.where(rates >= half)[0]
    if above.size == 0:
        start_time = peak_time
        stop_time = peak_time
    else:
        start_time = float(edges[above[0]])
        stop_time = float(edges[above[-1] + 1])

    return PeakTrialBinnedMetrics(
        peak_time=peak_time,
        peak_rate=peak_rate,
        start_time=start_time,
        stop_time=stop_time,
        spread=stop_time - start_time,
        middle_time=(start_time + stop_time) / 2.0,
        n_responses=int(times.size),
        bin_width=bin_width,
    )


def analyze_peak_trial_mle(
    response_times: Sequence[float],
    trial_duration: float,
) -> PeakTrialMLEMetrics | None:
    """Three-state MLE peak analysis (Church, Meck, & Gibbon, 1994).

    Assumes a piecewise-constant Poisson intensity with rate ``lambda_lo``
    on ``[0, s1) U (s2, T]`` and rate ``lambda_hi`` on ``[s1, s2]``.
    Estimates ``(s1, s2)`` by maximizing the profile log-likelihood;
    optimal break points necessarily fall at response times (or trial
    boundaries), so we search that O(n^2) discrete set.

    Parameters
    ----------
    response_times : sequence of float
        Response times within the trial, in seconds since trial onset.
    trial_duration : float
        Total trial duration, in seconds (strictly positive).

    Returns
    -------
    PeakTrialMLEMetrics or None
        ``None`` if fewer than 2 responses occurred (the model is
        under-identified below that).
    """
    times = _validate_trial(response_times, trial_duration)
    n = times.size
    if n < 2:
        return None

    # Candidate breakpoints: the trial start, each response time, the
    # trial end. Under a Poisson piecewise-constant model, the MLE
    # places s1 and s2 at event boundaries.
    anchors = np.concatenate(([0.0], times, [trial_duration]))
    anchors = np.unique(anchors)

    # Precompute cumulative response count F(t) = #{r_i <= t}.
    sorted_times = times

    def count_in(lo: float, hi: float) -> int:
        # Inclusive on lo, inclusive on hi (events are a measure-zero
        # set; boundary handling does not affect the MLE).
        left = int(np.searchsorted(sorted_times, lo, side="left"))
        right = int(np.searchsorted(sorted_times, hi, side="right"))
        return right - left

    best_ll = -math.inf
    best: tuple[float, float, float, float] | None = None

    for i in range(anchors.size):
        for j in range(i + 1, anchors.size):
            s1, s2 = float(anchors[i]), float(anchors[j])
            mid = s2 - s1
            flank = trial_duration - mid
            if mid <= 0.0 or flank <= 0.0:
                continue
            n_mid = count_in(s1, s2)
            n_flank = n - n_mid
            if n_mid == 0 and n_flank == 0:
                continue
            lam_hi = n_mid / mid if n_mid > 0 else 1e-12
            lam_lo = n_flank / flank if n_flank > 0 else 1e-12
            # Profile log-likelihood dropping constants:
            ll = 0.0
            if n_mid > 0:
                ll += n_mid * math.log(lam_hi) - lam_hi * mid
            else:
                ll -= lam_hi * mid
            if n_flank > 0:
                ll += n_flank * math.log(lam_lo) - lam_lo * flank
            else:
                ll -= lam_lo * flank
            if ll > best_ll:
                best_ll = ll
                best = (s1, s2, lam_hi, lam_lo)

    if best is None:
        return None

    s1, s2, lam_hi, lam_lo = best
    return PeakTrialMLEMetrics(
        start_time=s1,
        stop_time=s2,
        high_rate=lam_hi,
        low_rate=lam_lo,
        middle_time=(s1 + s2) / 2.0,
        spread=s2 - s1,
        n_responses=int(n),
        log_likelihood=float(best_ll),
    )
