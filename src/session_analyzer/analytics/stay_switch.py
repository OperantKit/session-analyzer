"""Changeover and COD analyses for concurrent schedules.

Two complementary measurements:

- :func:`analyze_stay_switch` — given a sequence of side-tagged
  responses, count changeovers and summarize stay-bout durations on
  each operandum.
- :func:`check_cod_compliance` — given a programmed COD (changeover
  delay), compute what fraction of reinforcers were preceded by a
  responding-on-the-same-operandum bout that **respected** the COD
  (i.e. ``last-changeover-time <= reinforcer-time - COD``).

References
----------
Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.),
    *Operant behavior: Areas of research and application* (pp. 213-270).
    Appleton-Century-Crofts.

Stubbs, D. A., & Pliskoff, S. S. (1969). Concurrent responding with
    fixed relative rate of reinforcement. *Journal of the Experimental
    Analysis of Behavior*, *12*(6), 887-895.
    https://doi.org/10.1901/jeab.1969.12-887
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StaySwitchMetrics:
    """Stay-switch summary.

    Attributes
    ----------
    n_changeovers : int
        Number of side switches between consecutive responses.
    n_responses : int
    n_stays : int
        Number of stay bouts (= ``n_changeovers + 1`` when there is at
        least one response, else 0).
    mean_stay_responses : float
        Mean number of responses within a stay bout.
    mean_stay_duration_sec : float
        Mean stay-bout duration (last - first response time in bout).
    changeover_rate_per_min : float
        Changeovers per minute of session time.
    session_duration_sec : float
        Time from first to last response.
    """

    n_changeovers: int
    n_responses: int
    n_stays: int
    mean_stay_responses: float
    mean_stay_duration_sec: float
    changeover_rate_per_min: float
    session_duration_sec: float


def analyze_stay_switch(
    sides: Sequence[str],
    response_times: Sequence[float],
) -> StaySwitchMetrics | None:
    """Decompose a side-tagged response sequence into stay bouts.

    Parameters
    ----------
    sides : sequence of str
        Side label ("L"/"R", or any string) per response, in temporal
        order.
    response_times : sequence of float
        Response times in seconds, sorted ascending. Must have the same
        length as ``sides``.

    Returns
    -------
    StaySwitchMetrics or None
        ``None`` when fewer than 2 responses are provided.

    Raises
    ------
    ValueError
        On length mismatch or unsorted ``response_times``.
    """
    if len(sides) != len(response_times):
        raise ValueError(
            "sides and response_times must have the same length "
            f"(got {len(sides)} and {len(response_times)})"
        )
    if any(
        response_times[i] > response_times[i + 1]
        for i in range(len(response_times) - 1)
    ):
        raise ValueError("response_times must be sorted ascending")
    if len(sides) < 2:
        return None

    bouts_responses: list[int] = []
    bouts_duration: list[float] = []
    bout_start = 0
    for i in range(1, len(sides)):
        if sides[i] != sides[bout_start]:
            n = i - bout_start
            duration = response_times[i - 1] - response_times[bout_start]
            bouts_responses.append(n)
            bouts_duration.append(duration)
            bout_start = i
    # Trailing bout
    n = len(sides) - bout_start
    duration = response_times[-1] - response_times[bout_start]
    bouts_responses.append(n)
    bouts_duration.append(duration)

    n_changeovers = len(bouts_responses) - 1
    session_duration = response_times[-1] - response_times[0]
    rate_per_min = (
        60.0 * n_changeovers / session_duration if session_duration > 0.0 else 0.0
    )
    mean_resp = float(np.mean(bouts_responses))
    mean_dur = float(np.mean(bouts_duration))

    return StaySwitchMetrics(
        n_changeovers=n_changeovers,
        n_responses=len(sides),
        n_stays=len(bouts_responses),
        mean_stay_responses=mean_resp,
        mean_stay_duration_sec=mean_dur,
        changeover_rate_per_min=float(rate_per_min),
        session_duration_sec=float(session_duration),
    )


@dataclass(frozen=True)
class CODComplianceResult:
    """COD compliance summary.

    Attributes
    ----------
    cod_sec : float
        Programmed COD duration.
    n_reinforcers : int
    n_compliant : int
        Reinforcers preceded by no changeover within the prior COD
        window.
    compliance_rate : float
        ``n_compliant / n_reinforcers`` (1.0 if there are no
        reinforcers).
    """

    cod_sec: float
    n_reinforcers: int
    n_compliant: int
    compliance_rate: float


def check_cod_compliance(
    sides: Sequence[str],
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    *,
    cod_sec: float,
) -> CODComplianceResult:
    """Compute the fraction of reinforcers respecting the COD window.

    For each reinforcer, the most recent changeover time is identified.
    The reinforcer is "compliant" iff the time elapsed since that
    changeover is at least ``cod_sec``. Reinforcers preceded by no
    changeover at all are counted as compliant.

    Parameters
    ----------
    sides, response_times : same as :func:`analyze_stay_switch`.
    reinforcement_times : sequence of float
        Reinforcer delivery times (sorted ascending).
    cod_sec : float
        COD duration in seconds (strictly positive).
    """
    if cod_sec <= 0.0:
        raise ValueError(f"cod_sec must be positive, got {cod_sec}")
    if len(sides) != len(response_times):
        raise ValueError("sides and response_times must have the same length")
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

    changeover_times: list[float] = []
    for i in range(1, len(sides)):
        if sides[i] != sides[i - 1]:
            changeover_times.append(response_times[i])

    n_reinforcers = len(reinforcement_times)
    if n_reinforcers == 0:
        return CODComplianceResult(
            cod_sec=cod_sec,
            n_reinforcers=0,
            n_compliant=0,
            compliance_rate=1.0,
        )

    n_compliant = 0
    for t in reinforcement_times:
        idx = bisect.bisect_right(changeover_times, t) - 1
        if idx < 0:
            n_compliant += 1
        else:
            elapsed = t - changeover_times[idx]
            if elapsed >= cod_sec:
                n_compliant += 1

    return CODComplianceResult(
        cod_sec=float(cod_sec),
        n_reinforcers=n_reinforcers,
        n_compliant=n_compliant,
        compliance_rate=n_compliant / n_reinforcers,
    )
