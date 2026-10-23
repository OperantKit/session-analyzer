"""Laws-of-covariance resurgence prediction (after Podlesnik & Baum, 2024).

Podlesnik & Baum (2024) apply the laws of allocation, induction, and
covariance to resurgence — an increase in an extinguished target
response when conditions for a more recently reinforced alternative
response worsen — and to other activity related or unrelated to the
reinforcer (phylogenetically important event). They also discuss how
induction could enter the resurgence-as-choice framework. The article
does not give the point prediction below; it is this module's own
minimal formula in the spirit of the covariance law.

For the canonical three-phase resurgence procedure (target trained →
target extinction + alternative reinforcement → alternative
extinction), the formula lets the target response recover in
proportion to the *reduction* in alternative reinforcement away from
its phase-2 asymptote. With ``R_alt^test = 0`` (full withdrawal), the
prediction is full recovery to the phase-1 baseline rate; with
``R_alt^test = R_alt^baseline`` (no withdrawal), the prediction is no
recovery.

This module implements a single-shot point prediction:

.. math::

    \\hat{B}_{target}^{test} = c \\, B_{target}^{baseline} \\,
        \\left( 1 - \\frac{R_{alt}^{test}}{R_{alt}^{baseline}} \\right)

where ``c`` is a unitless covariance coefficient in ``[0, 1]``
representing how completely the reduction in alternative reinforcement
is converted into target recovery (``c = 1`` is the parameter-free
limit). The prediction is intentionally minimal and is intended to be
compared with the dynamic Shahan & Sweeney (2011) prediction available
in :mod:`session_analyzer.analytics.resurgence`.

References
----------
Podlesnik, C. A., & Baum, W. M. (2024). Understanding resurgence and
    other emergent activity with the laws of allocation, induction, and
    covariance. *Journal of the Experimental Analysis of Behavior*,
    *122*(3), 375-391. https://doi.org/10.1002/jeab.4212

Shahan, T. A., & Sweeney, M. M. (2011). A model of resurgence based on
    behavioral momentum theory. *Journal of the Experimental Analysis
    of Behavior*, *95*(1), 91-108.
    https://doi.org/10.1901/jeab.2011.95-91
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LawsOfCovariancePrediction:
    """Laws-of-covariance point prediction for phase-3 target recovery.

    Attributes
    ----------
    predicted_target_rate : float
        Predicted phase-3 target response rate, in the same units as
        the input ``baseline_target_rate``.
    fraction_alt_withdrawn : float
        Fraction of phase-2 alternative reinforcement that was
        withdrawn in phase 3 (``1 - R_alt^test / R_alt^baseline``),
        bounded in ``[0, 1]``.
    covariance_coefficient : float
        The ``c`` value used for the prediction.
    """

    predicted_target_rate: float
    fraction_alt_withdrawn: float
    covariance_coefficient: float


def predict_resurgence_laws_of_covariance(
    *,
    baseline_target_rate: float,
    alt_baseline_rate: float,
    alt_test_rate: float,
    covariance_coefficient: float = 1.0,
) -> LawsOfCovariancePrediction:
    """Predict phase-3 target rate under the law of covariance.

    Parameters
    ----------
    baseline_target_rate :
        Phase-1 target response rate (non-negative).
    alt_baseline_rate :
        Phase-2 alternative reinforcement rate at asymptote (strictly
        positive — division by zero is not meaningful for the law of
        covariance, since there is no withdrawn rate to covary with).
    alt_test_rate :
        Phase-3 alternative reinforcement rate (typically ``0`` for
        full withdrawal). Must satisfy
        ``0 <= alt_test_rate <= alt_baseline_rate``.
    covariance_coefficient :
        Unitless coefficient ``c`` in ``[0, 1]``. ``c = 1`` (default)
        gives the parameter-free prediction; ``c < 1`` attenuates the
        predicted recovery, modelling residual extinction or
        decay-of-target-strength effects not captured by the
        covariance law alone.

    Returns
    -------
    LawsOfCovariancePrediction

    Raises
    ------
    ValueError
        If ``baseline_target_rate`` is negative, ``alt_baseline_rate``
        is non-positive, ``alt_test_rate`` is outside
        ``[0, alt_baseline_rate]``, or ``covariance_coefficient`` is
        outside ``[0, 1]``.
    """
    if baseline_target_rate < 0.0:
        raise ValueError(
            "baseline_target_rate must be non-negative; "
            f"got {baseline_target_rate}"
        )
    if alt_baseline_rate <= 0.0:
        raise ValueError(
            "alt_baseline_rate must be strictly positive (the law of "
            "covariance is undefined when there is no alternative "
            f"reinforcer to withdraw); got {alt_baseline_rate}"
        )
    if alt_test_rate < 0.0 or alt_test_rate > alt_baseline_rate:
        raise ValueError(
            "alt_test_rate must lie in [0, alt_baseline_rate]; "
            f"got alt_test_rate={alt_test_rate}, "
            f"alt_baseline_rate={alt_baseline_rate}"
        )
    if covariance_coefficient < 0.0 or covariance_coefficient > 1.0:
        raise ValueError(
            "covariance_coefficient must be in [0, 1]; "
            f"got {covariance_coefficient}"
        )

    fraction_withdrawn = 1.0 - (alt_test_rate / alt_baseline_rate)
    predicted = (
        covariance_coefficient * baseline_target_rate * fraction_withdrawn
    )
    return LawsOfCovariancePrediction(
        predicted_target_rate=predicted,
        fraction_alt_withdrawn=fraction_withdrawn,
        covariance_coefficient=covariance_coefficient,
    )
