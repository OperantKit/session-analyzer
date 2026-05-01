"""EZ-diffusion model parameter recovery (Wagenmakers et al., 2007).

The drift-diffusion model (Ratcliff, 1978) describes two-choice
decision making as evidence accumulation between two thresholds. The
**EZ-diffusion** approximation is a closed-form mapping from three
summary statistics of an observed (response, RT) sample to three of
the model's primary parameters:

- ``v`` — drift rate (signed; positive favours the upper boundary)
- ``a`` — boundary separation
- ``Ter`` — non-decision (encoding + motor) time

EZ assumes no across-trial parameter variability and a starting point
at ``a / 2``. Inputs:

- ``Pc``  — proportion correct
- ``MRTc`` — mean correct RT (seconds)
- ``VRTc`` — variance of correct RT (seconds^2)

Closed-form formulas (Wagenmakers, van der Maas, & Grasman, 2007):

    L = log(Pc / (1 - Pc))
    x = L * (L * Pc^2 - L * Pc + Pc - 0.5) / VRTc
    v = sign(Pc - 0.5) * s * (x ^ (1/4))
    a = s^2 * L / v
    MDT = (a / (2 * v)) * (1 - exp(-v * a / s^2)) / (1 + exp(-v * a / s^2))
    Ter = MRTc - MDT

with ``s = 0.1`` the conventional within-trial diffusion-coefficient
scale used by Ratcliff and EZ.

Reference
---------
Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P.
    (2007). An EZ-diffusion model for response time and accuracy.
    *Psychonomic Bulletin & Review*, *14*(1), 3-22.
    https://doi.org/10.3758/BF03194023

Ratcliff, R. (1978). A theory of memory retrieval. *Psychological
    Review*, *85*(2), 59-108. https://doi.org/10.1037/0033-295X.85.2.59
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EZDiffusionParameters:
    """EZ-diffusion parameter estimates.

    Attributes
    ----------
    v : float
        Drift rate.
    a : float
        Boundary separation.
    ter : float
        Non-decision time (seconds).
    s : float
        Within-trial diffusion-coefficient scale used in the
        derivation (default ``0.1``).
    """

    v: float
    a: float
    ter: float
    s: float


def fit_ez_diffusion(
    *,
    proportion_correct: float,
    mean_rt_correct: float,
    var_rt_correct: float,
    s: float = 0.1,
) -> EZDiffusionParameters:
    """Recover EZ-DDM parameters from summary statistics.

    Parameters
    ----------
    proportion_correct : float
        Proportion of correct responses, strictly in ``(0, 1)``.
        ``0.5`` is treated as a special case yielding zero drift.
    mean_rt_correct : float
        Mean correct RT in seconds.
    var_rt_correct : float
        Variance of correct RT in seconds squared.
    s : float
        Within-trial diffusion coefficient (default 0.1).

    Returns
    -------
    EZDiffusionParameters

    Raises
    ------
    ValueError
        If the proportion correct is not in ``(0, 1)``, or if
        ``mean_rt_correct`` / ``var_rt_correct`` are non-positive.
    """
    pc = proportion_correct
    mrt = mean_rt_correct
    vrt = var_rt_correct
    if not 0.0 < pc < 1.0:
        raise ValueError(
            f"proportion_correct must lie in (0, 1), got {pc}"
        )
    if mrt <= 0.0:
        raise ValueError(f"mean_rt_correct must be positive, got {mrt}")
    if vrt <= 0.0:
        raise ValueError(f"var_rt_correct must be positive, got {vrt}")
    if s <= 0.0:
        raise ValueError(f"s must be positive, got {s}")

    if pc == 0.5:
        # Drift is undefined exactly at chance; return zero drift and
        # leave a, Ter undetermined. Use the convention v = 0, a = +inf
        # is awkward, so flag with a == 0 as well.
        return EZDiffusionParameters(v=0.0, a=0.0, ter=mrt, s=s)

    sign = 1.0 if pc > 0.5 else -1.0
    p = pc if pc > 0.5 else (1.0 - pc)
    L = math.log(p / (1.0 - p))
    inner = L * (L * p * p - L * p + p - 0.5) / vrt
    if inner <= 0.0:
        raise ValueError(
            "EZ-DDM precondition violated: derived intermediate quantity "
            "is non-positive (likely too small VRT for the observed Pc)"
        )
    v = sign * s * (inner ** 0.25)
    a = (s * s) * L / abs(v) * sign  # keep a positive
    a = abs(a)

    y = math.exp(-abs(v) * a / (s * s))
    mdt = (a / (2.0 * abs(v))) * (1.0 - y) / (1.0 + y)
    ter = mrt - mdt
    return EZDiffusionParameters(v=v, a=a, ter=ter, s=s)
