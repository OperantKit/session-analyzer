"""Signal detection theory (SDT) analysis.

Computes parametric (d', c, β) and non-parametric (A', B'') sensitivity
and bias indices from a 2x2 stimulus-response confusion matrix, plus
ROC-curve construction and AUC trapezoidal estimation.

The log-linear correction of Hautus (1995) is applied to handle hit
rates of 1.0 or false-alarm rates of 0.0 (otherwise z-transform diverges):
    H' = (hits + 0.5) / (hits + misses + 1)
    F' = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

References
----------
Green, D. M., & Swets, J. A. (1966). *Signal detection theory and
psychophysics*. Wiley.

Hautus, M. J. (1995). Corrections for extreme proportions and their
biasing effects on estimated values of d'. *Behavior Research Methods,
Instruments, & Computers*, 27(1), 46-51.
https://doi.org/10.3758/BF03203619

Macmillan, N. A., & Creelman, C. D. (2005). *Detection theory: A user's
guide* (2nd ed.). Lawrence Erlbaum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SignalDetectionMetrics:
    """Parametric and non-parametric SDT indices.

    Attributes
    ----------
    hit_rate : float
        Corrected P(respond | signal).
    false_alarm_rate : float
        Corrected P(respond | noise).
    d_prime : float
        Sensitivity ``z(H) - z(F)``.
    criterion : float
        Bias ``c = -0.5 * (z(H) + z(F))``. Positive = conservative.
    beta : float
        Likelihood-ratio bias ``exp(c * d')``. >1 conservative, <1 liberal.
    a_prime : float
        Non-parametric sensitivity (Pollack & Norman, 1964).
    b_double_prime : float
        Non-parametric bias (Donaldson, 1992). Positive = conservative.
    n_signal_trials : int
        ``hits + misses``.
    n_noise_trials : int
        ``false_alarms + correct_rejections``.
    """

    hit_rate: float
    false_alarm_rate: float
    d_prime: float
    criterion: float
    beta: float
    a_prime: float
    b_double_prime: float
    n_signal_trials: int
    n_noise_trials: int


@dataclass(frozen=True)
class ROCCurve:
    """Empirical ROC curve and trapezoidal AUC.

    Attributes
    ----------
    fa_rates : numpy.ndarray
        False-alarm rates (sorted ascending), with ``(0,0)`` and
        ``(1,1)`` anchors prepended/appended.
    hit_rates : numpy.ndarray
        Hit rates aligned with ``fa_rates``.
    auc : float
        Area under the curve (trapezoidal).
    """

    fa_rates: np.ndarray
    hit_rates: np.ndarray
    auc: float


def compute_sdt(
    *,
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
) -> SignalDetectionMetrics:
    """Compute SDT metrics from a 2x2 confusion matrix.

    All four counts must be non-negative integers; ``hits + misses`` and
    ``false_alarms + correct_rejections`` must each be positive.
    """
    for name, value in (
        ("hits", hits),
        ("misses", misses),
        ("false_alarms", false_alarms),
        ("correct_rejections", correct_rejections),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    n_signal = hits + misses
    n_noise = false_alarms + correct_rejections
    if n_signal == 0:
        raise ValueError("hits + misses must be > 0")
    if n_noise == 0:
        raise ValueError("false_alarms + correct_rejections must be > 0")

    # Hautus (1995) log-linear correction (always applied for stability).
    h = (hits + 0.5) / (n_signal + 1)
    f = (false_alarms + 0.5) / (n_noise + 1)

    z_h = float(stats.norm.ppf(h))
    z_f = float(stats.norm.ppf(f))

    d_prime = z_h - z_f
    criterion = -0.5 * (z_h + z_f)
    beta = float(np.exp(criterion * d_prime))

    # Non-parametric A' (Pollack & Norman, 1964).
    if h >= f:
        a_prime = 0.5 + ((h - f) * (1 + h - f)) / (4 * h * (1 - f))
    else:
        a_prime = 0.5 - ((f - h) * (1 + f - h)) / (4 * f * (1 - h))

    # Non-parametric B'' (Donaldson, 1992).
    numer = h * (1 - h) - f * (1 - f)
    denom = h * (1 - h) + f * (1 - f)
    b_double_prime = numer / denom if denom > 0.0 else 0.0

    return SignalDetectionMetrics(
        hit_rate=round(float(h), 6),
        false_alarm_rate=round(float(f), 6),
        d_prime=round(float(d_prime), 6),
        criterion=round(float(criterion), 6),
        beta=round(float(beta), 6),
        a_prime=round(float(a_prime), 6),
        b_double_prime=round(float(b_double_prime), 6),
        n_signal_trials=n_signal,
        n_noise_trials=n_noise,
    )


def compute_roc(rates: list[tuple[float, float]]) -> ROCCurve:
    """Construct an empirical ROC curve from a list of ``(H, F)`` pairs.

    The pairs typically come from operating points at different criteria
    (e.g., rating-scale data or multiple bias manipulations). Anchors
    ``(0, 0)`` and ``(1, 1)`` are inserted; AUC is the trapezoidal area
    after sorting by false-alarm rate.

    Parameters
    ----------
    rates : list of (hit_rate, false_alarm_rate)
        Each rate must be in ``[0, 1]``.
    """
    if not rates:
        raise ValueError("rates must not be empty")
    for h, f in rates:
        if not (0.0 <= h <= 1.0 and 0.0 <= f <= 1.0):
            raise ValueError(f"hit/false-alarm rates must be in [0, 1], got ({h}, {f})")

    fa = [0.0, *(f for _, f in rates), 1.0]
    hr = [0.0, *(h for h, _ in rates), 1.0]
    order = np.argsort(fa)
    fa_sorted = np.asarray(fa, dtype=np.float64)[order]
    hr_sorted = np.asarray(hr, dtype=np.float64)[order]
    auc = float(np.trapezoid(hr_sorted, fa_sorted))

    return ROCCurve(
        fa_rates=fa_sorted,
        hit_rates=hr_sorted,
        auc=round(auc, 6),
    )
