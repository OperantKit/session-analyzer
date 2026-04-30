"""Unit-price and cross-price elasticity analyses.

Behavioral economics complements the exponential demand-curve fit
(Hursh & Silberberg, 2008) with two log-log slope summaries that JEAB
papers routinely report:

- **Unit-price** (Hursh, 1980, 1991): cost normalised by reinforcer
  size. Two operations that share a unit price (e.g., FR50 / 100 µL
  and FR10 / 20 µL both = 0.5 resp/µL) are predicted to support the
  same consumption. The OLS slope of ``log Q`` on ``log P`` estimates
  the *own-price elasticity* α; ``|α| > 1`` is *elastic*, ``|α| < 1``
  is *inelastic*.

- **Cross-price elasticity** (Hursh & Roma, 2013): the OLS slope of
  ``log Q_A`` on ``log P_B`` between two reinforcers. Positive →
  substitutes, negative → complements, near 0 → independent.

Both fits use plain log-log linear regression and assume strictly
positive prices and quantities (zero/negative pairs are dropped).

References
----------
Hursh, S. R. (1980). Economic concepts for the analysis of behavior.
*Journal of the Experimental Analysis of Behavior*, 34(2), 219-238.
https://doi.org/10.1901/jeab.1980.34-219

Hursh, S. R. (1991). Behavioral economics of drug self-administration
and drug abuse policy. *Journal of the Experimental Analysis of
Behavior*, 56(2), 377-393. https://doi.org/10.1901/jeab.1991.56-377

Hursh, S. R., & Roma, P. G. (2013). Behavioral economics and empirical
public policy. *Journal of the Experimental Analysis of Behavior*,
99(1), 98-124. https://doi.org/10.1002/jeab.7

Hursh, S. R., & Silberberg, A. (2008). Economic demand and essential
value. *Psychological Review*, 115(1), 186-198.
https://doi.org/10.1037/0033-295X.115.1.186
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitPriceElasticity:
    """Own-price elasticity from a log-log demand regression.

    Attributes
    ----------
    elasticity : float
        OLS slope ``alpha`` in ``log Q = alpha * log P + c``. Demand
        functions normally yield ``alpha < 0``.
    intercept : float
        ``c``; the log of consumption at unit price 1.
    r_squared : float
    n : int
    is_elastic : bool
        ``|elasticity| > 1`` (consumption is highly responsive).
    is_inelastic : bool
        ``|elasticity| < 1``.
    """

    elasticity: float
    intercept: float
    r_squared: float
    n: int
    is_elastic: bool
    is_inelastic: bool


@dataclass(frozen=True)
class CrossPriceElasticity:
    """Cross-price elasticity from a log-log regression.

    Attributes
    ----------
    elasticity : float
        Slope of ``log Q_A`` on ``log P_B``.
    intercept : float
    r_squared : float
    n : int
    relationship : {"substitutes", "complements", "independent"}
        ``"substitutes"`` if elasticity > 0.05;
        ``"complements"`` if elasticity < -0.05;
        ``"independent"`` otherwise.
    """

    elasticity: float
    intercept: float
    r_squared: float
    n: int
    relationship: str


def compute_unit_prices(prices: list[float], doses: list[float]) -> list[float]:
    """Compute per-condition unit prices ``price / dose``.

    Parameters
    ----------
    prices : list of float
        Programmed cost (responses, schedule completions, ...).
    doses : list of float
        Reinforcer magnitude (volume, mass, count, ...). Must be > 0.
    """
    if len(prices) != len(doses):
        raise ValueError(
            f"prices and doses must have the same length "
            f"({len(prices)} vs {len(doses)})"
        )
    out: list[float] = []
    for p, d in zip(prices, doses, strict=False):
        if d <= 0.0:
            raise ValueError("each dose must be > 0")
        out.append(p / d)
    return out


def _log_log_fit(x: list[float], y: list[float]) -> tuple[float, float, float, int]:
    valid = [(xi, yi) for xi, yi in zip(x, y, strict=False) if xi > 0.0 and yi > 0.0]
    if len(valid) < 2:
        raise ValueError("log-log fit requires at least 2 strictly positive pairs")
    log_x = np.log(np.asarray([v[0] for v in valid], dtype=np.float64))
    log_y = np.log(np.asarray([v[1] for v in valid], dtype=np.float64))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0
    return float(slope), float(intercept), float(r2), len(valid)


def fit_unit_price_elasticity(
    *,
    unit_prices: list[float],
    consumption: list[float],
) -> UnitPriceElasticity:
    """Fit the log-log own-price elasticity.

    Pairs with non-positive price or consumption are dropped.
    """
    if len(unit_prices) != len(consumption):
        raise ValueError(
            f"unit_prices and consumption must have the same length "
            f"({len(unit_prices)} vs {len(consumption)})"
        )
    slope, intercept, r2, n = _log_log_fit(unit_prices, consumption)
    abs_slope = abs(slope)
    return UnitPriceElasticity(
        elasticity=round(slope, 6),
        intercept=round(intercept, 6),
        r_squared=round(r2, 6),
        n=n,
        is_elastic=abs_slope > 1.0,
        is_inelastic=abs_slope < 1.0,
    )


def compute_cross_price_elasticity(
    *,
    prices_other: list[float],
    consumption_target: list[float],
    independence_band: float = 0.05,
) -> CrossPriceElasticity:
    """Fit the cross-price elasticity of reinforcer A on the price of B.

    Parameters
    ----------
    prices_other : list of float
        Price of the *other* reinforcer (B).
    consumption_target : list of float
        Consumption of the *target* reinforcer (A).
    independence_band : float
        Half-width around 0 within which the elasticity is classified
        as ``"independent"`` (default ±0.05).
    """
    if len(prices_other) != len(consumption_target):
        raise ValueError(
            "prices_other and consumption_target must have the same length"
        )
    slope, intercept, r2, n = _log_log_fit(prices_other, consumption_target)
    if abs(slope) <= independence_band:
        relationship = "independent"
    elif slope > 0:
        relationship = "substitutes"
    else:
        relationship = "complements"
    return CrossPriceElasticity(
        elasticity=round(slope, 6),
        intercept=round(intercept, 6),
        r_squared=round(r2, 6),
        n=n,
        relationship=relationship,
    )
