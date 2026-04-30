"""Stimulus-control analyses.

Two related but distinct measures of discriminative stimulus control:

- **Discrimination ratio**:
  ``DR = R(S+) / [R(S+) + R(S-)]``, in ``[0, 1]``.
  Equivalently, *suppression ratio* (Annau & Kamin, 1961) is
  ``SR = R(S-) / [R(S+) + R(S-)]`` and reports the same information
  with reversed polarity.

- **Peak shift** (Hanson, 1959):
  After intermixed S+/S- training, the peak of the generalization
  gradient shifts *away from* S-, often beyond S+ itself. The shift
  magnitude is operationalised here as the signed distance from S+ to
  the empirically-observed peak (positive = away from S-).

References
----------
Annau, Z., & Kamin, L. J. (1961). The conditioned emotional response
as a function of intensity of the US. *Journal of Comparative and
Physiological Psychology*, 54(4), 428-432.
https://doi.org/10.1037/h0042199

Hanson, H. M. (1959). Effects of discrimination training on stimulus
generalization. *Journal of Experimental Psychology*, 58(5), 321-334.
https://doi.org/10.1037/h0042606
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class DiscriminationRatio:
    """Discrimination and suppression ratios.

    Attributes
    ----------
    discrimination_ratio : float
        ``R(S+) / [R(S+) + R(S-)]``. Bounded in ``[0, 1]``; 1.0 is
        perfect S+ control, 0.5 is chance.
    suppression_ratio : float
        ``1 - discrimination_ratio``.
    s_plus_count : int
    s_minus_count : int
    is_discriminated : bool
        ``True`` if ``discrimination_ratio >= 0.75`` (heuristic).
    """

    discrimination_ratio: float
    suppression_ratio: float
    s_plus_count: int
    s_minus_count: int
    is_discriminated: bool


@dataclass(frozen=True)
class PeakShiftResult:
    """Empirical and fitted peak-shift summary.

    Attributes
    ----------
    empirical_peak_value : float
        Stimulus value with the maximum observed response rate (argmax).
    fitted_peak_value : float
        Location parameter ``mu`` of a free-peak Gaussian fit
        ``R(x) = R0 * exp(-(x - mu)^2 / (2 * sigma^2))``.
    fitted_sigma : float
        Width of the fitted Gaussian.
    fitted_peak_response : float
        Height ``R0`` of the fitted Gaussian.
    s_plus : float
    s_minus : float
    shift_magnitude : float
        Signed distance from S+ to the fitted peak, projected onto
        the ``(S+ - S-)`` direction. Positive = away from S-.
    shift_direction : {"away_from_s_minus", "toward_s_minus", "none"}
        Categorical interpretation of ``shift_magnitude``:
        ``"away_from_s_minus"`` if magnitude > 1 % of |S+ - S-|;
        ``"toward_s_minus"`` if magnitude < -1 %;
        ``"none"`` otherwise.
    r_squared : float
    n : int
    """

    empirical_peak_value: float
    fitted_peak_value: float
    fitted_sigma: float
    fitted_peak_response: float
    s_plus: float
    s_minus: float
    shift_magnitude: float
    shift_direction: str
    r_squared: float
    n: int


def compute_discrimination_ratio(
    *,
    s_plus_count: int,
    s_minus_count: int,
    threshold: float = 0.75,
) -> DiscriminationRatio:
    """Compute the discrimination and suppression ratios."""
    if s_plus_count < 0 or s_minus_count < 0:
        raise ValueError("response counts must be non-negative")
    total = s_plus_count + s_minus_count
    if total == 0:
        raise ValueError("at least one of s_plus_count or s_minus_count must be > 0")
    dr = s_plus_count / total
    sr = s_minus_count / total
    return DiscriminationRatio(
        discrimination_ratio=round(dr, 6),
        suppression_ratio=round(sr, 6),
        s_plus_count=s_plus_count,
        s_minus_count=s_minus_count,
        is_discriminated=dr >= threshold,
    )


def _free_gaussian(x: np.ndarray, peak: float, mu: float, sigma: float) -> np.ndarray:
    return peak * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def detect_peak_shift(
    *,
    stimulus_values: list[float],
    response_rates: list[float],
    s_plus: float,
    s_minus: float,
    classification_tolerance: float = 0.01,
) -> PeakShiftResult:
    """Fit a free-peak Gaussian to a generalization gradient and report
    the shift relative to S+.

    Parameters
    ----------
    stimulus_values, response_rates : list of float
        Aligned per-stimulus rate measurements.
    s_plus, s_minus : float
        Training-stimulus values along the generalization dimension.
        Must differ.
    classification_tolerance : float
        Threshold (as a fraction of ``|S+ - S-|``) below which the
        shift is classified as ``"none"``. Default 1 %.
    """
    if len(stimulus_values) != len(response_rates):
        raise ValueError(
            "stimulus_values and response_rates must have the same length"
        )
    if len(stimulus_values) < 3:
        raise ValueError("peak-shift fit requires at least 3 (stim, rate) points")
    if s_plus == s_minus:
        raise ValueError("s_plus and s_minus must differ")

    x = np.asarray(stimulus_values, dtype=np.float64)
    y = np.asarray(response_rates, dtype=np.float64)

    empirical_peak = float(x[int(np.argmax(y))])
    span = float(max(x) - min(x))

    p0 = [float(np.max(y)), empirical_peak, max(span / 4.0, 1e-3)]
    bounds = (
        [0.0, float(min(x)) - span, 1e-6],
        [np.inf, float(max(x)) + span, np.inf],
    )
    popt, _ = curve_fit(_free_gaussian, x, y, p0=p0, bounds=bounds, maxfev=20_000)
    peak_fit, mu_fit, sigma_fit = (float(p) for p in popt)

    pred = _free_gaussian(x, peak_fit, mu_fit, sigma_fit)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    direction_unit = 1.0 if s_plus > s_minus else -1.0
    shift = (mu_fit - s_plus) * direction_unit
    delta = abs(s_plus - s_minus)
    if abs(shift) < classification_tolerance * delta:
        direction = "none"
    elif shift > 0:
        direction = "away_from_s_minus"
    else:
        direction = "toward_s_minus"

    return PeakShiftResult(
        empirical_peak_value=round(empirical_peak, 6),
        fitted_peak_value=round(mu_fit, 6),
        fitted_sigma=round(sigma_fit, 6),
        fitted_peak_response=round(peak_fit, 6),
        s_plus=s_plus,
        s_minus=s_minus,
        shift_magnitude=round(shift, 6),
        shift_direction=direction,
        r_squared=round(float(r_squared), 6),
        n=len(stimulus_values),
    )
