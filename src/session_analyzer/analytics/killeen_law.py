"""Killeen's (1994) mathematical principles of reinforcement.

The MPR model expresses response rate as

    B = (k * a * R) / (a * R * delta + 1)

where ``R`` is reinforcement rate, ``k`` is the asymptotic-rate
parameter, ``a`` is the specific activation produced by reinforcement,
and ``delta`` is the response unit time (Killeen's "coupling"
parameter). The asymptote at ``R -> infinity`` equals ``k / delta``.

Identifiability note
--------------------
Substituting ``beta = a * k`` and ``gamma = a * delta`` reduces the
model to ``B = beta * R / (gamma * R + 1)``, a 2-parameter
saturating-rate function. ``a``, ``k``, ``delta`` are therefore not
separately identifiable from a single ``(R, B)`` curve; one of them
must be fixed by an external measurement (typically the response
unit time ``delta`` taken from the minimum observed inter-response
interval). This module fits ``a`` and ``k`` with ``delta`` supplied
by the caller.

Reference
---------
Killeen, P. R. (1994). Mathematical principles of reinforcement.
    *Behavioral and Brain Sciences*, *17*(1), 105-135.
    https://doi.org/10.1017/S0140525X00033628
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class KilleenLawParameters:
    """Killeen (1994) MPR fit result.

    Attributes
    ----------
    k : float
        Asymptotic response rate.
    a : float
        Specific activation produced by reinforcement.
    delta : float
        Response unit time / coupling parameter.
    r_squared : float
    n : int
    """

    k: float
    a: float
    delta: float
    r_squared: float
    n: int


def _killeen(r: np.ndarray, k: float, a: float, delta: float) -> np.ndarray:
    return (k * a * r) / (a * r * delta + 1.0)


def fit_killeen_law(
    reinforcement_rates: Sequence[float],
    response_rates: Sequence[float],
    *,
    delta: float,
) -> KilleenLawParameters:
    """Fit ``B = (k a R) / (a R delta + 1)`` with ``delta`` fixed.

    Parameters
    ----------
    reinforcement_rates : sequence of float
        Per-condition reinforcement rates ``R`` (non-negative).
    response_rates : sequence of float
        Per-condition response rates ``B`` (non-negative).
    delta : float
        Response unit time (strictly positive). Typically taken from
        the minimum observed inter-response interval, not estimated
        from the rate-vs-rate curve (see *Identifiability note* in the
        module docstring).

    Returns
    -------
    KilleenLawParameters

    Raises
    ------
    ValueError
        On length mismatch, ``delta <= 0``, or fewer than 2 valid
        pairs.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if len(reinforcement_rates) != len(response_rates):
        raise ValueError(
            "reinforcement_rates and response_rates must have the same length "
            f"(got {len(reinforcement_rates)} and {len(response_rates)})"
        )
    pairs = [
        (r, b)
        for r, b in zip(reinforcement_rates, response_rates, strict=False)
        if r >= 0.0 and b >= 0.0
    ]
    if len(pairs) < 2:
        raise ValueError(
            "Killeen MPR fit requires at least 2 non-negative pairs; "
            f"got {len(pairs)} after filtering"
        )

    r_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    b_arr = np.array([p[1] for p in pairs], dtype=np.float64)

    asymptote_estimate = (
        float(b_arr.max()) * 1.2 if b_arr.max() > 0.0 else 1.0
    )
    k_init = asymptote_estimate * delta
    a_init = 1.0

    def _model(r: np.ndarray, k: float, a: float) -> np.ndarray:
        return _killeen(r, k, a, delta)

    popt, _ = curve_fit(
        _model,
        r_arr,
        b_arr,
        p0=[k_init, a_init],
        bounds=([1e-9, 1e-9], [np.inf, np.inf]),
        maxfev=20_000,
    )
    k, a = float(popt[0]), float(popt[1])
    predicted = _killeen(r_arr, k, a, delta)
    ss_res = float(np.sum((b_arr - predicted) ** 2))
    ss_tot = float(np.sum((b_arr - np.mean(b_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return KilleenLawParameters(
        k=k,
        a=a,
        delta=float(delta),
        r_squared=float(r2),
        n=len(pairs),
    )
