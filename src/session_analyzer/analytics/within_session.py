"""Within-session response-rate dynamics (McSweeney, 1994).

Response rate typically follows an *inverted-U* across a session: a
brief warm-up rise, a peak, and a gradual satiation-driven decline
(McSweeney, Roll, & Cannon, 1994). This module bins responses into
``n_bins`` equal-duration windows and reports:

- the per-bin rate trajectory,
- the peak bin and its rate,
- linear slopes of the rising and falling limbs,
- the R^2 of a bilinear (broken-line) fit anchored at the peak bin.

References
----------
McSweeney, F. K., Roll, J. M., & Cannon, C. B. (1994). The generality
of within-session patterns of responding: Rate of reinforcement and
session length. *Animal Learning & Behavior*, 22(3), 252-266.
https://doi.org/10.3758/BF03209833
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WithinSessionDynamics:
    """Within-session response-rate trajectory.

    Attributes
    ----------
    rates : list[float]
        Response rate (responses / second) in each of ``n_bins``.
    peak_bin : int
        Zero-based index of the bin with the highest rate.
    peak_rate : float
        Rate (responses / second) at the peak bin.
    rise_slope : float
        OLS slope of ``rates[: peak_bin + 1]`` regressed on bin index.
    fall_slope : float
        OLS slope of ``rates[peak_bin :]`` regressed on bin index.
    bilinear_r_squared : float
        R^2 of the broken-line fit (rise + fall) against the empirical
        rates.
    n_bins : int
    n_responses : int
        Total number of responses falling inside ``[0, session_duration)``.
    """

    rates: list[float]
    peak_bin: int
    peak_rate: float
    rise_slope: float
    fall_slope: float
    bilinear_r_squared: float
    n_bins: int
    n_responses: int


def bin_response_rates(
    timestamps: list[float],
    *,
    session_duration: float,
    n_bins: int,
) -> list[float]:
    """Bin response timestamps into ``n_bins`` equal-duration windows.

    Returns rates in responses / second. Timestamps outside
    ``[0, session_duration)`` are dropped.
    """
    if session_duration <= 0.0:
        raise ValueError("session_duration must be > 0")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    bin_width = session_duration / n_bins
    counts = [0] * n_bins
    for t in timestamps:
        if 0.0 <= t < session_duration:
            idx = int(t / bin_width)
            if idx == n_bins:
                idx -= 1
            counts[idx] += 1
    return [c / bin_width for c in counts]


def _segment_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def analyze_within_session(
    timestamps: list[float],
    *,
    session_duration: float,
    n_bins: int = 12,
) -> WithinSessionDynamics:
    """Compute within-session dynamics from response timestamps.

    Parameters
    ----------
    timestamps : list of float
        Response times in seconds, relative to session start.
    session_duration : float
        Total session length in seconds.
    n_bins : int
        Number of equal-duration bins (default 12, i.e., 5-min bins for
        a 60-min session). Must be ``>= 2``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2 for a rise/fall decomposition")
    in_session = [t for t in timestamps if 0.0 <= t < session_duration]
    if not in_session:
        raise ValueError("no responses fall within [0, session_duration)")

    rates = bin_response_rates(
        in_session,
        session_duration=session_duration,
        n_bins=n_bins,
    )
    rates_arr = np.asarray(rates, dtype=np.float64)
    peak = int(np.argmax(rates_arr))
    peak_rate = float(rates_arr[peak])

    rise_slope = _segment_slope(rates_arr[: peak + 1])
    fall_slope = _segment_slope(rates_arr[peak:])

    # Broken-line predicted values
    rise_line = rates_arr[0] + rise_slope * np.arange(peak + 1)
    fall_line = rates_arr[peak] + fall_slope * np.arange(n_bins - peak)
    pred = np.concatenate([rise_line[:-1], fall_line]) if peak > 0 else fall_line
    if len(pred) != n_bins:
        # peak == 0 path keeps fall_line of full length
        pred = pred[:n_bins]
    ss_res = float(np.sum((rates_arr - pred) ** 2))
    ss_tot = float(np.sum((rates_arr - np.mean(rates_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    return WithinSessionDynamics(
        rates=[round(r, 6) for r in rates],
        peak_bin=peak,
        peak_rate=round(peak_rate, 6),
        rise_slope=round(rise_slope, 6),
        fall_slope=round(fall_slope, 6),
        bilinear_r_squared=round(float(r_squared), 6),
        n_bins=n_bins,
        n_responses=len(in_session),
    )
