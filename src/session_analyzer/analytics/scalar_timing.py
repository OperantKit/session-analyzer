"""Scalar timing analyses.

Scalar Expectancy Theory (SET; Gibbon, 1977) predicts that response
distributions on temporal-discrimination tasks satisfy the
**scalar property**: standard deviation grows linearly with the mean,
so the coefficient of variation (CV) is constant across criterion
durations.

Two analyses:

- :func:`analyze_scalar_timing` — descriptive stats (mean, sd, CV) for
  a single criterion's response-time sample.
- :func:`assess_scalar_property` — across-criterion check that CV is
  approximately constant.

A learning-to-time (LeT; Machado, 1997) implementation would simulate
serial states; we leave that to the next session.

References
----------
Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal
    timing. *Psychological Review*, *84*(3), 279-325.
    https://doi.org/10.1037/0033-295X.84.3.279

Machado, A. (1997). Learning the temporal dynamics of behavior.
    *Psychological Review*, *104*(2), 241-265.
    https://doi.org/10.1037/0033-295X.104.2.241
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ScalarTimingResult:
    """Per-criterion scalar-timing summary.

    Attributes
    ----------
    mean : float
    std : float
    coefficient_of_variation : float
        ``std / mean`` (Weber fraction).
    n : int
    """

    mean: float
    std: float
    coefficient_of_variation: float
    n: int


@dataclass(frozen=True)
class ScalarPropertyTest:
    """Across-criterion test of the scalar property.

    Attributes
    ----------
    criteria : tuple of float
        Criterion durations in ascending order.
    cvs : tuple of float
        CV at each criterion.
    mean_cv : float
        Mean CV across criteria.
    cv_of_cvs : float
        ``std(cvs) / mean(cvs)`` — should be near zero if the scalar
        property holds.
    is_scalar : bool
        ``cv_of_cvs <= tolerance``.
    tolerance : float
    """

    criteria: tuple[float, ...]
    cvs: tuple[float, ...]
    mean_cv: float
    cv_of_cvs: float
    is_scalar: bool
    tolerance: float


def analyze_scalar_timing(
    response_times: Sequence[float],
) -> ScalarTimingResult | None:
    """Summarize a single criterion's response-time distribution.

    Returns ``None`` if there are fewer than two strictly positive
    response times (CV is undefined for n < 2 or for a zero mean).
    """
    valid = [x for x in response_times if x > 0.0]
    if len(valid) < 2:
        return None
    arr = np.array(valid, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    cv = std / mean if mean > 0.0 else 0.0
    return ScalarTimingResult(
        mean=mean,
        std=std,
        coefficient_of_variation=cv,
        n=len(valid),
    )


def assess_scalar_property(
    response_times_per_criterion: Mapping[float, Sequence[float]],
    *,
    tolerance: float = 0.20,
) -> ScalarPropertyTest:
    """Test whether CV is approximately constant across criteria.

    Parameters
    ----------
    response_times_per_criterion : mapping of float -> sequence of float
        Maps criterion durations to per-criterion response-time samples.
    tolerance : float
        Maximum ratio ``std(cvs) / mean(cvs)`` that still counts as
        scalar. Default 0.2.

    Returns
    -------
    ScalarPropertyTest

    Raises
    ------
    ValueError
        If fewer than two criteria have a computable CV.
    """
    sorted_keys = sorted(response_times_per_criterion.keys())
    cvs: list[float] = []
    keys: list[float] = []
    for k in sorted_keys:
        s = analyze_scalar_timing(response_times_per_criterion[k])
        if s is None:
            continue
        cvs.append(s.coefficient_of_variation)
        keys.append(k)
    if len(cvs) < 2:
        raise ValueError(
            "scalar-property test requires CVs from at least 2 criteria; "
            f"got {len(cvs)} after filtering"
        )
    arr = np.array(cvs, dtype=np.float64)
    mean_cv = float(np.mean(arr))
    cv_of_cvs = (
        float(np.std(arr, ddof=0) / mean_cv) if mean_cv > 0.0 else 0.0
    )
    return ScalarPropertyTest(
        criteria=tuple(keys),
        cvs=tuple(cvs),
        mean_cv=mean_cv,
        cv_of_cvs=cv_of_cvs,
        is_scalar=cv_of_cvs <= tolerance,
        tolerance=tolerance,
    )
