"""Adjusting-amount indifference-point estimation.

In an *adjusting-amount* procedure (Mazur, 1987; Richards, Mitchell, de
Wit, & Seiden, 1997), the immediate amount of reinforcement is
incremented or decremented from trial to trial as a function of the
subject's choice between an immediate and a delayed (or larger)
reinforcer. The procedure converges on the value of the immediate
amount at which the subject is indifferent — equally likely to choose
either option.

The indifference value is conventionally taken as the mean of the
adjusting amount over a *stable* trailing window. Stability is
operationalised as the absolute range over the window remaining below
a tolerance.

The same procedure underlies *adjusting-delay* designs by simply
substituting a delay value for the amount.

References
----------
Mazur, J. E. (1987). An adjusting procedure for studying delayed
reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, &
H. Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5*
(pp. 55-73). Lawrence Erlbaum Associates.

Richards, J. B., Mitchell, S. H., de Wit, H., & Seiden, L. S. (1997).
Determination of discount functions in rats with an adjusting-amount
procedure. *Journal of the Experimental Analysis of Behavior*, 67(3),
353-366. https://doi.org/10.1901/jeab.1997.67-353
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class IndifferencePoint:
    """Estimated indifference value and stability summary.

    Attributes
    ----------
    indifference_value : float
        Mean adjusting amount over the trailing stable window.
    range_in_window : float
        ``max - min`` over the stable window. Smaller is more stable.
    stable : bool
        ``range_in_window <= stability_tolerance``.
    stable_window : int
        Number of trials averaged.
    n_trials : int
        Total number of input trials.
    """

    indifference_value: float
    range_in_window: float
    stable: bool
    stable_window: int
    n_trials: int


def estimate_indifference_point(
    *,
    adjusting_amounts: list[float],
    stable_window: int = 6,
    stability_tolerance: float = 2.0,
) -> IndifferencePoint:
    """Estimate the indifference value from a sequence of adjusting
    amounts.

    Parameters
    ----------
    adjusting_amounts : list of float
        Per-trial values of the adjusting reinforcer (the dimension
        being titrated — typically amount, but the same procedure
        applies to delay or probability).
    stable_window : int
        Number of trailing trials to average. Must be ``>= 2``.
    stability_tolerance : float
        Maximum permitted ``max - min`` over the window for the
        estimate to be flagged as ``stable``.
    """
    if stable_window < 2:
        raise ValueError("stable_window must be >= 2")
    if len(adjusting_amounts) < stable_window:
        raise ValueError(
            f"need at least stable_window={stable_window} trials; "
            f"got {len(adjusting_amounts)}"
        )

    window = adjusting_amounts[-stable_window:]
    value = mean(window)
    rng = max(window) - min(window)
    return IndifferencePoint(
        indifference_value=round(value, 6),
        range_in_window=round(rng, 6),
        stable=rng <= stability_tolerance,
        stable_window=stable_window,
        n_trials=len(adjusting_amounts),
    )
