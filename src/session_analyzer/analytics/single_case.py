"""Non-overlap effect sizes for single-case designs: NAP, PND, Tau, Tau-U.

Single-case (small-N) designs — reversal (ABAB), multiple baseline, and
alternating treatments — compare the observations of a baseline phase
(A, ``m`` sessions) with those of an intervention phase (B, ``n``
sessions). This module computes the non-overlap indices most often
reported for such data. Every index is computed per case; pooling across
cases (meta-analysis) is out of scope.

Notation (``improvement="increase"``; ``"decrease"`` reverses every
inequality):

- ``S_AB`` = sum over all ``m*n`` (A, B) pairs of
  ``+1`` if B > A, ``-1`` if B < A, ``0`` if tied.
- ``S_A`` / ``S_B`` = Kendall's S of the phase against session order
  (``m(m-1)/2`` and ``n(n-1)/2`` pairs).

Indices
-------
NAP (Parker & Vannest, 2009)
    ``(#pairs with B > A + 0.5 * #ties) / (m*n)``. Range [0, 1]; 0.5 means
    no effect. Ceiling: any complete separation scores 1 regardless of the
    size of the shift, and it ignores trend.
PND (Scruggs, Mastropieri & Casto, 1987)
    Proportion of B observations strictly above the A maximum (below the A
    minimum for ``"decrease"``). Returned as a proportion in [0, 1]. No
    known sampling distribution and driven by a single A observation;
    provided for comparison with the literature — prefer NAP.
Tau (non-overlap)
    ``S_AB / (m*n) = 2 * NAP - 1``. Range [-1, 1].
Tau-U (Parker, Vannest, Davis & Sauber, 2011)
    "A vs. B - Trend A": ``(S_AB - S_A) / denominator``. The literature
    uses two denominators (Pustejovsky, 2016):

    - ``method="singlecasees"`` (default): ``m*n``. Matches SingleCaseES
      ``Tau_U()``, the singlecaseresearch.org calculator, and scan
      ``tau_u(method="parker")``. **Can exceed 1** (logical range
      ``±(2n + m - 1) / (2n)``).
    - ``method="parker_original"``: ``m*n + m(m-1)/2`` (sum of the pair
      counts, as in Parker et al., 2011, Table 1). Range [-1, 1].
    - ``method="scan_complete_b"``: scan's default ``tau_u()``
      (``method="complete"``, ``tau_method="b"``): summed pair counts with
      Kendall's tau-b tie correction ``sqrt((n0 - t_x)(n0 - t_y))``.

    :func:`tau_u_table` returns scan's six-row table (pairs / pos / neg /
    ties / S / D / Tau) for any of the three methods. Tau-U has no
    published sampling distribution, so no standard error or confidence
    interval is returned. The baseline-trend correction subtracts a fixed
    amount bounded by the length of phase A; it does not model the slope.

References
----------
Parker, R. I., & Vannest, K. J. (2009). An improved effect size for
single-case research: Nonoverlap of all pairs. *Behavior Therapy, 40*(4),
357-367. https://doi.org/10.1016/j.beth.2008.10.006

Parker, R. I., Vannest, K. J., & Davis, J. L. (2011). Effect size in
single-case research: A review of nine nonoverlap techniques. *Behavior
Modification, 35*(4), 303-322. https://doi.org/10.1177/0145445511399147

Parker, R. I., Vannest, K. J., Davis, J. L., & Sauber, S. B. (2011).
Combining nonoverlap and trend for single-case research: Tau-U. *Behavior
Therapy, 42*(2), 284-299. https://doi.org/10.1016/j.beth.2010.08.006

Pustejovsky, J. E. (2016, November 3). What is Tau-U?
https://jepusto.com/posts/What-is-Tau-U/

Scruggs, T. E., Mastropieri, M. A., & Casto, G. (1987). The quantitative
synthesis of single-subject research: Methodology and validation.
*Remedial and Special Education, 8*(2), 24-33.
https://doi.org/10.1177/074193258700800206
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, groupby
from typing import Literal

Improvement = Literal["increase", "decrease"]
TauUMethod = Literal["singlecasees", "parker_original", "scan_complete_b"]

_IMPROVEMENTS: tuple[str, ...] = ("increase", "decrease")
_TAU_U_METHODS: tuple[str, ...] = ("singlecasees", "parker_original", "scan_complete_b")

ROW_LABELS: tuple[str, ...] = (
    "A vs. B",
    "Trend A",
    "Trend B",
    "A vs. B - Trend A",
    "A vs. B + Trend B",
    "A vs. B + Trend B - Trend A",
)
_TAU_U_ROW = "A vs. B - Trend A"


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleCaseSeries:
    """One case of a single-case design: observations in session order.

    Attributes
    ----------
    values : tuple[float, ...]
        Dependent-variable value of each session, in session order
        (e.g. response rate per session). Missing sessions are dropped
        before construction, not encoded as NaN.
    phases : tuple[str, ...]
        Phase label of each observation (``"A"``, ``"B"``, ``"A1"``, ...).
        Same length as ``values``.
    improvement : ``"increase"`` or ``"decrease"``
        Direction in which a change counts as improvement.
    name : str or None
        Case name (one series per case in a multiple-baseline design).
    """

    values: tuple[float, ...]
    phases: tuple[str, ...]
    improvement: Improvement = "increase"
    name: str | None = None

    def __post_init__(self) -> None:
        values = _as_finite_tuple(self.values, "values")
        phases = tuple(str(p) for p in self.phases)
        if len(values) != len(phases):
            raise ValueError(
                f"values and phases must have the same length "
                f"({len(values)} != {len(phases)})"
            )
        _check_improvement(self.improvement)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "phases", phases)


def select_phases(
    series: SingleCaseSeries,
    a: str | Sequence[str] | None = None,
    b: str | Sequence[str] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the (A, B) observations to compare, each in session order.

    Phases are selected by label, and every observation carrying a
    selected label is used (as in SingleCaseES; scan's ``phases =`` selects
    by position instead). With ``a`` omitted, A is the first label in
    order of appearance; with ``b`` omitted, B is the first other label.
    Several labels pool phases, e.g. ``a=("A1", "A2"), b=("B1", "B2")``
    for a reversal design. A single string is one label, not a sequence
    of one-character labels.
    """
    a_labels, b_labels = _resolve_labels(series, a, b)
    a_values = tuple(v for v, p in zip(series.values, series.phases, strict=True) if p in a_labels)
    b_values = tuple(v for v, p in zip(series.values, series.phases, strict=True) if p in b_labels)
    for labels, values in ((a_labels, a_values), (b_labels, b_values)):
        if not values:
            raise ValueError(f"phase {labels!r} has no observations in the series")
    return a_values, b_values


# --------------------------------------------------------------------------
# Indices
# --------------------------------------------------------------------------


def nap(
    a: Sequence[float], b: Sequence[float], improvement: Improvement = "increase"
) -> float:
    """Nonoverlap of All Pairs (Parker & Vannest, 2009); ties count 0.5."""
    xa, xb = _oriented(a, b, improvement)
    pos, _neg, ties = _between_counts(xa, xb)
    return (pos + 0.5 * ties) / (len(xa) * len(xb))


def pnd(
    a: Sequence[float], b: Sequence[float], improvement: Improvement = "increase"
) -> float:
    """Proportion of B observations beyond the most extreme A observation.

    Strictly above ``max(A)`` for ``"increase"``, strictly below
    ``min(A)`` for ``"decrease"``. Returned as a proportion (multiply by
    100 for the percentage usually reported).
    """
    xa, xb = _oriented(a, b, improvement)
    ceiling = max(xa)
    return sum(1 for y in xb if y > ceiling) / len(xb)


def tau(
    a: Sequence[float], b: Sequence[float], improvement: Improvement = "increase"
) -> float:
    """Non-overlap Tau: ``S_AB / (m*n) = 2 * NAP - 1``."""
    xa, xb = _oriented(a, b, improvement)
    pos, neg, _ties = _between_counts(xa, xb)
    return (pos - neg) / (len(xa) * len(xb))


@dataclass(frozen=True)
class TauURow:
    """One row of :class:`TauUTable` (scan ``tau_u()`` column names)."""

    label: str
    pairs: int
    pos: int
    neg: int
    ties: int
    s: int
    d: float
    tau: float


@dataclass(frozen=True)
class TauUTable:
    """The six Tau-U component rows for one A-B comparison."""

    rows: tuple[TauURow, ...]
    method: TauUMethod
    improvement: Improvement

    def row(self, label: str) -> TauURow:
        for r in self.rows:
            if r.label == label:
                return r
        raise KeyError(f"no row {label!r}; rows are {ROW_LABELS!r}")


def tau_u_table(
    a: Sequence[float],
    b: Sequence[float],
    improvement: Improvement = "increase",
    method: TauUMethod = "singlecasees",
) -> TauUTable:
    """Tau-U component table (A vs. B, trends, and their combinations).

    Row ``"A vs. B - Trend A"`` is what is usually called Tau-U;
    ``"A vs. B + Trend B - Trend A"`` is scan's ``Tau_U(BA)``. See the
    module docstring for the three ``method`` denominators.
    """
    _check_method(method)
    xa, xb = _oriented(a, b, improvement)
    m, n = len(xa), len(xb)

    ab_pos, ab_neg, ab_tie = _between_counts(xa, xb)
    a_pos, a_neg, a_tie = _within_counts(xa)
    b_pos, b_neg, b_tie = _within_counts(xb)
    ab_pairs, a_pairs, b_pairs = m * n, m * (m - 1) // 2, n * (n - 1) // 2

    pairs = [
        ab_pairs,
        a_pairs,
        b_pairs,
        ab_pairs + a_pairs,
        ab_pairs + b_pairs,
        ab_pairs + a_pairs + b_pairs,
    ]
    pos = [ab_pos, a_pos, b_pos, ab_pos + a_neg, ab_pos + b_pos, ab_pos + b_pos + a_neg]
    neg = [ab_neg, a_neg, b_neg, ab_neg + a_pos, ab_neg + b_neg, ab_neg + b_neg + a_pos]
    ties = [ab_tie, a_tie, b_tie, ab_tie + a_tie, ab_tie + b_tie, ab_tie + b_tie + a_tie]
    s = [p - q for p, q in zip(pos, neg, strict=True)]

    if method == "singlecasees":
        pairs[3] = ab_pairs  # revised Tau-U: (S_AB - S_A) / mn
        denominators = [float(p) for p in pairs]
    elif method == "parker_original":
        denominators = [float(p) for p in pairs]
    else:
        denominators = _tau_b_denominators(xa, xb)
        denominators[0] = ab_pairs - ab_tie / 2

    rows = tuple(
        TauURow(
            label=label,
            pairs=pairs[i],
            pos=pos[i],
            neg=neg[i],
            ties=ties[i],
            s=s[i],
            d=denominators[i],
            tau=s[i] / denominators[i] if denominators[i] else math.nan,
        )
        for i, label in enumerate(ROW_LABELS)
    )
    return TauUTable(rows=rows, method=method, improvement=improvement)


def tau_u(
    a: Sequence[float],
    b: Sequence[float],
    improvement: Improvement = "increase",
    method: TauUMethod = "singlecasees",
) -> float:
    """Tau-U = "A vs. B - Trend A" (baseline-trend-corrected non-overlap).

    The default ``method="singlecasees"`` divides by ``m*n`` and **can
    exceed 1**; use ``method="parker_original"`` for the bounded original
    denominator. See the module docstring.
    """
    return tau_u_table(a, b, improvement=improvement, method=method).row(_TAU_U_ROW).tau


# --------------------------------------------------------------------------
# Series-level summary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NonOverlapResult:
    """All non-overlap indices for one A-B comparison of one case.

    ``tau_u`` and ``tau_u_table`` are ``None`` when the baseline-trend
    correction was not applied (see ``trend_correction`` of
    :func:`nonoverlap`).
    """

    name: str | None
    a_labels: tuple[str, ...]
    b_labels: tuple[str, ...]
    n_a: int
    n_b: int
    improvement: Improvement
    nap: float
    nap_rescaled: float
    pnd: float
    tau: float
    tau_u: float | None
    tau_u_table: TauUTable | None
    method: TauUMethod


def nonoverlap(
    series: SingleCaseSeries,
    a: str | Sequence[str] | None = None,
    b: str | Sequence[str] | None = None,
    method: TauUMethod = "singlecasees",
    trend_correction: bool | None = None,
) -> NonOverlapResult:
    """Compute NAP, PND, Tau and Tau-U for one case.

    ``nap_rescaled`` is scan's ``2 * NAP - 1`` (equal to Tau), expressed
    as a proportion rather than scan's percentage.

    ``trend_correction`` controls Tau-U. ``None`` (default) applies it
    only when every selected label occupies one contiguous run of
    sessions; a label that recurs in several runs is treated as
    interleaved (alternating treatments), where a session-order trend
    within a condition has no meaning. A reversal design that reuses the
    same labels (``A, A, B, B, A, A, B, B``) therefore gets no Tau-U by
    default: label the phases ``A1``/``B1``/``A2``/``B2`` or pass
    ``trend_correction=True`` (pooled phases are then concatenated in
    session order, as scan does for combined phases).
    """
    _check_method(method)
    a_labels, b_labels = _resolve_labels(series, a, b)
    xa, xb = select_phases(series, a_labels, b_labels)
    imp = series.improvement
    if trend_correction is None:
        trend_correction = _is_blocked(series.phases, a_labels + b_labels)
    table: TauUTable | None = None
    if trend_correction:
        table = tau_u_table(xa, xb, improvement=imp, method=method)
    nap_value = nap(xa, xb, imp)
    return NonOverlapResult(
        name=series.name,
        a_labels=a_labels,
        b_labels=b_labels,
        n_a=len(xa),
        n_b=len(xb),
        improvement=imp,
        nap=nap_value,
        nap_rescaled=2 * nap_value - 1,
        pnd=pnd(xa, xb, imp),
        tau=tau(xa, xb, imp),
        tau_u=table.row(_TAU_U_ROW).tau if table is not None else None,
        tau_u_table=table,
        method=method,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _check_improvement(improvement: str) -> None:
    if improvement not in _IMPROVEMENTS:
        raise ValueError(f"improvement must be one of {_IMPROVEMENTS!r}, got {improvement!r}")


def _check_method(method: str) -> None:
    if method not in _TAU_U_METHODS:
        raise ValueError(f"method must be one of {_TAU_U_METHODS!r}, got {method!r}")


def _as_finite_tuple(values: Iterable[float], what: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} must be finite numbers (drop missing sessions first)")
    return out


def _oriented(
    a: Sequence[float], b: Sequence[float], improvement: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate and flip signs so that "improvement" is always "larger"."""
    _check_improvement(improvement)
    xa = _as_finite_tuple(a, "phase A values")
    xb = _as_finite_tuple(b, "phase B values")
    if not xa or not xb:
        raise ValueError("phase A and phase B must both be non-empty")
    if improvement == "decrease":
        return tuple(-v for v in xa), tuple(-v for v in xb)
    return xa, xb


def _between_counts(xa: Sequence[float], xb: Sequence[float]) -> tuple[int, int, int]:
    pos = sum(1 for x in xa for y in xb if y > x)
    neg = sum(1 for x in xa for y in xb if y < x)
    return pos, neg, len(xa) * len(xb) - pos - neg


def _within_counts(v: Sequence[float]) -> tuple[int, int, int]:
    """(later > earlier, later < earlier, ties) over all session pairs."""
    pos = neg = tie = 0
    for i, j in combinations(range(len(v)), 2):
        if v[j] > v[i]:
            pos += 1
        elif v[j] < v[i]:
            neg += 1
        else:
            tie += 1
    return pos, neg, tie


def _tie_pairs(v: Sequence[float]) -> float:
    return sum(c * (c - 1) / 2 for c in Counter(v).values())


def _tau_b_denominator(x: Sequence[float], y: Sequence[float]) -> float:
    n0 = len(x) * (len(x) - 1) / 2
    return math.sqrt((n0 - _tie_pairs(x)) * (n0 - _tie_pairs(y)))


def _tau_b_denominators(xa: Sequence[float], xb: Sequence[float]) -> list[float]:
    """Kendall tau-b denominators of scan's six coded comparisons.

    Each row is Kendall's tau between the observations and a coding
    vector whose pair signs reproduce the row's S (scan ``tau_u()``).
    """
    m, n = len(xa), len(xb)
    ab = list(xa) + list(xb)
    falling_a = list(range(m, 0, -1))
    codings: list[tuple[Sequence[float], Sequence[float]]] = [
        (ab, [0] * m + [1] * n),
        (xa, list(range(1, m + 1))),
        (xb, list(range(1, n + 1))),
        (ab, falling_a + [m + 1] * n),
        (ab, [0] * m + list(range(m + 1, m + n + 1))),
        (ab, falling_a + list(range(m + 1, m + n + 1))),
    ]
    return [_tau_b_denominator(x, y) for x, y in codings]


def _resolve_labels(
    series: SingleCaseSeries,
    a: str | Sequence[str] | None,
    b: str | Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    distinct = tuple(dict.fromkeys(series.phases))
    a_labels = _as_labels(a) if a is not None else distinct[:1]
    if b is not None:
        b_labels = _as_labels(b)
    else:
        b_labels = tuple(p for p in distinct if p not in a_labels)[:1]
    if not a_labels or not b_labels:
        raise ValueError("series needs at least two phase labels to compare")
    if set(a_labels) & set(b_labels):
        raise ValueError("phase A and phase B labels must not overlap")
    return a_labels, b_labels


def _as_labels(labels: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, str):
        return (labels,)
    return tuple(labels)


def _is_blocked(phases: Sequence[str], labels: Sequence[str]) -> bool:
    """True when every selected label occupies a single contiguous run."""
    runs = Counter(label for label, _ in groupby(phases))
    return all(runs[label] <= 1 for label in labels)
