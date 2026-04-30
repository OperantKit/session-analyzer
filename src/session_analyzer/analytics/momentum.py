"""Behavioral momentum / resistance to change (Nevin, 1992).

Behavioral momentum theory (Nevin, 1992) quantifies the resistance of
operant behavior to disruption. Given a baseline response rate ``B_o``
and disrupted response rates ``B_x`` at disruptor magnitudes ``x``
(e.g. pre-feeding time, extinction minutes, alternative reinforcer
magnitude), the theory models:

    log10(B_x / B_o) = -x / m

where ``m`` is the "mass" or resistance parameter. Larger ``m`` means
behavior is more resistant to change (more momentum) — a smaller
proportional decrement per unit disruption.

References
----------
Nevin, J. A. (1992). An integrative model for the study of behavioral
    momentum. *Journal of the Experimental Analysis of Behavior*,
    *57*(3), 301-316. https://doi.org/10.1901/jeab.1992.57-301

Nevin, J. A., & Grace, R. C. (2000). Behavioral momentum and the law
    of effect. *Behavioral and Brain Sciences*, *23*(1), 73-90.
    https://doi.org/10.1017/S0140525X00002405
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MomentumParameters:
    """Nevin (1992) resistance-to-change fit result.

    Attributes
    ----------
    m : float
        Mass / resistance parameter. Larger ``m`` indicates more
        resistance to disruption (greater behavioral momentum).
    r_squared : float
        Coefficient of determination of the linear fit on
        ``log10(B_x / B_o)`` vs ``x``.
    n : int
        Number of (disruptor_magnitude, disrupted_rate) pairs used.
    """

    m: float
    r_squared: float
    n: int


def estimate_resistance_to_change(
    baseline_rate: float,
    disrupted_rates: Sequence[float],
    disruptor_magnitudes: Sequence[float],
) -> MomentumParameters:
    """Estimate the resistance parameter ``m`` (Nevin, 1992).

    Fits ``log10(B_x / B_o) = -x / m`` via ordinary least squares on
    the linearized form, forcing the intercept to zero (since by
    definition ``B_x = B_o`` when ``x = 0``).

    Parameters
    ----------
    baseline_rate : float
        Undisrupted baseline response rate ``B_o`` (strictly positive).
    disrupted_rates : sequence of float
        Response rates ``B_x`` observed during disruption, strictly
        positive. Pairs with non-positive rates are discarded.
    disruptor_magnitudes : sequence of float
        Disruptor magnitudes ``x`` corresponding to each disrupted rate
        (non-negative, same length as ``disrupted_rates``).

    Returns
    -------
    MomentumParameters

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, if ``baseline_rate`` is
        non-positive, or if fewer than 2 valid pairs remain after
        filtering non-positive rates and zero magnitudes.
    """
    if baseline_rate <= 0.0:
        raise ValueError(f"baseline_rate must be positive, got {baseline_rate}")
    if len(disrupted_rates) != len(disruptor_magnitudes):
        raise ValueError(
            "disrupted_rates and disruptor_magnitudes must have the same length "
            f"(got {len(disrupted_rates)} and {len(disruptor_magnitudes)})"
        )

    pairs = [
        (x, b)
        for x, b in zip(disruptor_magnitudes, disrupted_rates, strict=False)
        if b > 0.0 and x > 0.0
    ]
    if len(pairs) < 2:
        raise ValueError(
            "resistance-to-change fit requires at least 2 valid pairs "
            f"(positive disrupted rate and positive magnitude); got {len(pairs)}"
        )

    x = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.log10(np.array([p[1] for p in pairs], dtype=np.float64) / baseline_rate)

    # Force through origin: slope = sum(x*y) / sum(x*x).
    slope = float(np.sum(x * y) / np.sum(x * x))
    if slope >= 0.0:
        # Behavior increased or did not decrease under disruption —
        # the model assumes monotonically decreasing ratio. Report
        # +inf momentum (infinite resistance) rather than negative m.
        m = float("inf")
    else:
        m = float(-1.0 / slope)

    predicted = slope * x
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum(y**2))  # intercept forced to 0
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    return MomentumParameters(
        m=m,
        r_squared=float(r_squared),
        n=len(pairs),
    )
