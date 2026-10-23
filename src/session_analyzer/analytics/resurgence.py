"""Resurgence quantification (3-phase paradigm).

Resurgence is the recovery of a previously extinguished operant response
when an alternative response, trained as part of differential
reinforcement of alternative behavior (DRA), is itself extinguished
(Epstein, 1985; Shahan & Sweeney, 2011).

The canonical 3-phase procedure:

1. **Phase 1 — Training**: target response is reinforced.
2. **Phase 2 — DRA**: target response is extinguished while an
   alternative response is reinforced. Target rate suppresses.
3. **Phase 3 — Test**: alternative reinforcement is also withdrawn.
   Target rate transiently recovers ("resurges").

This module summarises the magnitude of resurgence as descriptive
statistics on per-session response rates.

References
----------
Epstein, R. (1985). Extinction-induced resurgence: Preliminary
investigations and possible applications. *The Psychological Record*,
35(2), 143-153. https://doi.org/10.1007/BF03394918

Shahan, T. A., & Sweeney, M. M. (2011). A model of resurgence based on
behavioral momentum theory. *Journal of the Experimental Analysis of
Behavior*, 95(1), 91-108. https://doi.org/10.1901/jeab.2011.95-91
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class ResurgenceMetrics:
    """Descriptive statistics for a 3-phase resurgence procedure.

    All rates are in the same units the caller supplies (typically
    responses / minute or responses / session).

    Attributes
    ----------
    baseline_target_rate : float
        Mean target rate over the last ``baseline_window`` sessions of
        phase 1.
    suppression_target_rate : float
        Mean target rate over the last ``baseline_window`` sessions of
        phase 2 (after DRA-induced suppression).
    test_target_rate : float
        Mean target rate over the first ``test_window`` sessions of
        phase 3 (resurgence test).
    resurgence_magnitude : float
        ``test_target_rate - suppression_target_rate``. The recovery
        of the target response from its suppressed level.
    resurgence_ratio : float
        ``test_target_rate / baseline_target_rate``. The fraction of
        baseline responding recovered.
    alt_baseline_rate : float
        Mean alternative rate over the last ``baseline_window`` sessions
        of phase 2 (alternative is at asymptote).
    alt_test_rate : float
        Mean alternative rate over the first ``test_window`` sessions
        of phase 3 (alternative is being extinguished).
    resurged : bool
        Convenience flag: ``resurgence_magnitude > 0`` and the target
        rate is at least 10 % of baseline.
    n_phase1 : int
    n_phase2 : int
    n_phase3 : int
    """

    baseline_target_rate: float
    suppression_target_rate: float
    test_target_rate: float
    resurgence_magnitude: float
    resurgence_ratio: float
    alt_baseline_rate: float
    alt_test_rate: float
    resurged: bool
    n_phase1: int
    n_phase2: int
    n_phase3: int


def analyze_resurgence(
    *,
    target_rates_phase1: list[float],
    target_rates_phase2: list[float],
    target_rates_phase3: list[float],
    alt_rates_phase2: list[float],
    alt_rates_phase3: list[float],
    baseline_window: int = 0,
    test_window: int = 1,
    resurgence_threshold: float = 0.10,
) -> ResurgenceMetrics:
    """Quantify resurgence from per-session response rates.

    Parameters
    ----------
    target_rates_phase1, target_rates_phase2, target_rates_phase3 :
        Per-session target response rates for each phase (must be
        non-empty).
    alt_rates_phase2, alt_rates_phase3 :
        Per-session alternative response rates. Must have the same
        length as the corresponding ``target_rates_phaseN``.
    baseline_window :
        Number of trailing sessions of phases 1 and 2 to average.
        ``0`` (default) means *all* sessions of that phase.
    test_window :
        Number of leading sessions of phase 3 to average for the test.
    resurgence_threshold :
        Fraction of baseline that the test rate must exceed for
        ``resurged`` to be ``True`` (default 0.10).
    """
    if not target_rates_phase1 or not target_rates_phase2 or not target_rates_phase3:
        raise ValueError("each phase must have at least one session of target rates")
    if len(alt_rates_phase2) != len(target_rates_phase2):
        raise ValueError(
            "alt_rates_phase2 length must match target_rates_phase2"
        )
    if len(alt_rates_phase3) != len(target_rates_phase3):
        raise ValueError(
            "alt_rates_phase3 length must match target_rates_phase3"
        )
    if test_window < 1:
        raise ValueError("test_window must be >= 1")
    if test_window > len(target_rates_phase3):
        raise ValueError(
            "test_window exceeds the number of phase-3 sessions"
        )
    bw1 = len(target_rates_phase1) if baseline_window == 0 else baseline_window
    bw2 = len(target_rates_phase2) if baseline_window == 0 else baseline_window
    if bw1 > len(target_rates_phase1) or bw2 > len(target_rates_phase2):
        raise ValueError(
            "baseline_window exceeds the number of available sessions"
        )

    baseline_target = mean(target_rates_phase1[-bw1:])
    suppression_target = mean(target_rates_phase2[-bw2:])
    test_target = mean(target_rates_phase3[:test_window])
    alt_baseline = mean(alt_rates_phase2[-bw2:])
    alt_test = mean(alt_rates_phase3[:test_window])

    magnitude = test_target - suppression_target
    ratio = test_target / baseline_target if baseline_target > 0.0 else 0.0
    resurged = magnitude > 0.0 and ratio >= resurgence_threshold

    return ResurgenceMetrics(
        baseline_target_rate=round(baseline_target, 4),
        suppression_target_rate=round(suppression_target, 4),
        test_target_rate=round(test_target, 4),
        resurgence_magnitude=round(magnitude, 4),
        resurgence_ratio=round(ratio, 4),
        alt_baseline_rate=round(alt_baseline, 4),
        alt_test_rate=round(alt_test, 4),
        resurged=resurged,
        n_phase1=len(target_rates_phase1),
        n_phase2=len(target_rates_phase2),
        n_phase3=len(target_rates_phase3),
    )
