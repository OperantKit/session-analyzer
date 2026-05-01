"""Multi-good cross-elasticity matrix.

Generalizes the single-pair cross-price elasticity (already exposed in
``unit_price.py``) to a multi-good demand system. For each ordered
pair of goods ``(i, j)``, fits the log-log slope

    epsilon_{ij} = d log(Q_i) / d log(P_j)

via simple least squares on the log-log relationship across all
observed conditions where both ``Q_i > 0`` and ``P_j > 0``.

Diagonal entries (``epsilon_{ii}``) are own-price elasticities;
off-diagonal entries express substitution (positive) or
complementarity (negative). The procedure does **not** assume
homogeneity-of-degree-zero or any particular demand-system functional
form; it is a descriptive companion to structural fits.

Reference
---------
Hursh, S. R., & Roma, P. G. (2013). Behavioral economics and empirical
    public policy. *Journal of the Experimental Analysis of Behavior*,
    *99*(1), 98-124. https://doi.org/10.1002/jeab.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CrossElasticityMatrix:
    """Multi-good cross-elasticity matrix.

    Attributes
    ----------
    goods : tuple of str
        Good identifiers in row / column order.
    matrix : tuple[tuple[float, ...], ...]
        ``epsilon_{ij}`` indexed by ``[i][j]``. ``i`` is the consumed
        good (response variable); ``j`` is the priced good
        (regressor). NaN entries indicate that the regression had
        fewer than 2 valid (positive) data points.
    n_conditions : int
        Number of conditions supplied.
    """

    goods: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    n_conditions: int


def compute_cross_elasticity_matrix(
    consumption: Mapping[str, Sequence[float]],
    prices: Mapping[str, Sequence[float]],
) -> CrossElasticityMatrix:
    """Compute the elasticity matrix for a multi-good demand system.

    Parameters
    ----------
    consumption : mapping of str -> sequence of float
        Per-condition consumption ``Q_i`` for each good ``i``.
    prices : mapping of str -> sequence of float
        Per-condition unit price ``P_j`` for each good ``j``.

    Returns
    -------
    CrossElasticityMatrix

    Raises
    ------
    ValueError
        If consumption / prices keys differ, sequences are
        length-mismatched, or fewer than 2 conditions are provided.
    """
    if set(consumption.keys()) != set(prices.keys()):
        raise ValueError(
            "consumption and prices must define the same set of goods "
            f"(got {set(consumption.keys())} vs {set(prices.keys())})"
        )
    goods = sorted(consumption.keys())
    if not goods:
        raise ValueError("consumption / prices must define at least one good")
    n = len(consumption[goods[0]])
    if n < 2:
        raise ValueError(
            f"cross-elasticity requires at least 2 conditions, got {n}"
        )
    for g in goods:
        if len(consumption[g]) != n or len(prices[g]) != n:
            raise ValueError(
                f"all sequences must have length {n}; good {g!r} differs"
            )

    matrix: list[list[float]] = []
    for i, gi in enumerate(goods):
        row: list[float] = []
        q_i = np.array(consumption[gi], dtype=np.float64)
        for gj in goods:
            p_j = np.array(prices[gj], dtype=np.float64)
            mask = (q_i > 0.0) & (p_j > 0.0)
            if mask.sum() < 2:
                row.append(float("nan"))
                continue
            log_q = np.log(q_i[mask])
            log_p = np.log(p_j[mask])
            # Slope of the OLS log_q ~ a + b * log_p.
            sxx = float(np.var(log_p) * mask.sum())
            sxy = float(
                np.sum((log_p - log_p.mean()) * (log_q - log_q.mean()))
            )
            slope = sxy / sxx if sxx > 0.0 else float("nan")
            row.append(float(slope))
        matrix.append(row)

    return CrossElasticityMatrix(
        goods=tuple(goods),
        matrix=tuple(tuple(r) for r in matrix),
        n_conditions=n,
    )
