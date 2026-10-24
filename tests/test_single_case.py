"""Tests for single-case design non-overlap effect sizes (NAP / PND / Tau / Tau-U).

Expected values are published reference outputs (none was derived from
this implementation), except the property tests in the "Improvement
direction", "Series input" and "Validation" sections, whose small inputs
are checked by hand. Sources:

- SingleCaseES 0.7.4 reference pages (``NAP``, ``PND``, ``Tau``, ``Tau_U``)
  on the Parker, Vannest & Davis (2011) Figure 1 data and the Parker &
  Vannest (2009) NAP example.
- Parker, Vannest, Davis & Sauber (2011), Table 1 (counts and Tau values
  for A = 2,3,5,3 / B = 4,5,5,7,6).
- scan-Book chapter 9 (``tau_u()`` / ``overlap()`` / ``nap()`` / ``pnd()``
  outputs for ``Parker2011`` and ``exampleAB``).
- Pustejovsky (2016), "What is Tau-U?" (original vs revised denominator).
"""

from __future__ import annotations

import math

import pytest

from session_analyzer.analytics.single_case import (
    NonOverlapResult,
    SingleCaseSeries,
    TauUTable,
    nap,
    nonoverlap,
    pnd,
    select_phases,
    tau,
    tau_u,
    tau_u_table,
)

# Parker, Vannest & Davis (2011) Figure 1 / scan ``Parker2011b``.
P2011B_A = (20, 20, 26, 25, 22, 23)
P2011B_B = (28, 25, 24, 27, 30, 30, 29)
# Parker & Vannest (2009) NAP example.
PV2009_A = (4, 3, 4, 3, 4, 7, 5, 2, 3, 2)
PV2009_B = (5, 9, 7, 9, 7, 5, 9, 11, 11, 10, 9)
# Parker et al. (2011) Figure 2 / Table 1 / scan ``Parker2011``.
P2011_A = (2, 3, 5, 3)
P2011_B = (4, 5, 5, 7, 6)
# scan ``exampleAB`` (5 baseline + 15 intervention sessions).
JOHANNA = (54, 53, 56, 58, 52, 61, 62, 71, 66, 64, 78, 70, 74, 82, 77, 86, 68, 80, 86, 87)
KAROLINA = (41, 59, 56, 51, 52, 57, 56, 67, 75, 66, 69, 68, 73, 77, 79, 86, 82, 75, 83, 89)
ANJA = (55, 58, 53, 50, 52, 55, 68, 68, 81, 67, 78, 73, 72, 78, 81, 78, 71, 85, 80, 76)

EXACT = 5e-7  # reference pages print 7 decimals
DISPLAY = 5e-3  # scan-Book prints 2 decimals

AVB = "A vs. B"
TREND_A = "Trend A"
TREND_B = "Trend B"
AVB_MINUS_A = "A vs. B - Trend A"
AVB_PLUS_B = "A vs. B + Trend B"
AVB_PLUS_B_MINUS_A = "A vs. B + Trend B - Trend A"


def _flip(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-v for v in values)


# --------------------------------------------------------------------------
# SingleCaseES 0.7.4 reference values
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.reference(
    module="single_case",
    verdict="published",
    source="Parker, Vannest & Davis (2011)",
)
def test_nap_singlecasees_parker2011b() -> None:
    assert nap(P2011B_A, P2011B_B) == pytest.approx(0.9166667, abs=EXACT)


@pytest.mark.unit
def test_pnd_singlecasees_parker2011b() -> None:
    assert pnd(P2011B_A, P2011B_B) == pytest.approx(0.7142857, abs=EXACT)


@pytest.mark.unit
def test_tau_singlecasees_parker2011b() -> None:
    assert tau(P2011B_A, P2011B_B) == pytest.approx(0.8333333, abs=EXACT)


@pytest.mark.unit
def test_tau_u_default_matches_singlecasees() -> None:
    assert tau_u(P2011B_A, P2011B_B) == pytest.approx(0.7380952, abs=EXACT)


@pytest.mark.unit
def test_tau_u_parker_original_denominator_pustejovsky() -> None:
    got = tau_u(P2011B_A, P2011B_B, method="parker_original")
    assert got == pytest.approx(0.5438596, abs=EXACT)


@pytest.mark.unit
def test_nap_parker_vannest_2009_example() -> None:
    assert nap(PV2009_A, PV2009_B) == pytest.approx(0.9636364, abs=EXACT)


# --------------------------------------------------------------------------
# Pustejovsky (2016): original vs revised denominator
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "original", "revised"),
    [
        ((2, 3, 5, 3), (4, 5, 5, 7, 6), 0.5000000, 0.6500000),
        (P2011B_A, P2011B_B, 0.5438596, 0.7380952),
        ((3, 3, 4, 5), (4, 5, 6, 7, 7), 0.4230769, 0.5500000),
        ((22, 21, 23, 23, 23, 22), (24, 22, 23, 23, 24, 26, 25), 0.4385965, 0.5952381),
    ],
)
def test_tau_u_pustejovsky_table(
    a: tuple[int, ...], b: tuple[int, ...], original: float, revised: float
) -> None:
    assert tau_u(a, b, method="parker_original") == pytest.approx(original, abs=EXACT)
    assert tau_u(a, b, method="singlecasees") == pytest.approx(revised, abs=EXACT)


# --------------------------------------------------------------------------
# Parker et al. (2011) Table 1 and scan ``method = "parker"``
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_parker2011_table1_counts() -> None:
    table = tau_u_table(P2011_A, P2011_B, method="singlecasees")
    avb = table.row(AVB)
    assert (avb.pairs, avb.pos, avb.neg, avb.s) == (20, 17, 1, 16)
    trend_a = table.row(TREND_A)
    assert (trend_a.pairs, trend_a.pos, trend_a.neg, trend_a.s) == (6, 4, 1, 3)
    trend_b = table.row(TREND_B)
    assert (trend_b.pairs, trend_b.pos, trend_b.neg, trend_b.s) == (10, 8, 1, 7)
    plus_b = table.row(AVB_PLUS_B)
    # Table 1 prints #neg = 3 for this column, a misprint: the worked
    # example in the same paper reads "Obtain S = (25 - 2) = 23", and the
    # partition counts are stated to be strictly additive (1 + 1 = 2).
    assert (plus_b.pairs, plus_b.pos, plus_b.neg, plus_b.s) == (30, 25, 2, 23)
    assert plus_b.tau == pytest.approx(23 / 30, abs=EXACT)
    full = table.row(AVB_PLUS_B_MINUS_A)
    assert (full.pairs, full.pos, full.neg, full.s) == (36, 26, 6, 20)


@pytest.mark.unit
def test_parker2011_scan_parker_method() -> None:
    table = tau_u_table(P2011_A, P2011_B, method="singlecasees")
    assert table.row(AVB).tau == pytest.approx(0.80, abs=DISPLAY)
    assert table.row(AVB_MINUS_A).tau == pytest.approx(0.65, abs=DISPLAY)
    assert table.row(AVB_PLUS_B).tau == pytest.approx(0.77, abs=DISPLAY)
    assert table.row(AVB_PLUS_B_MINUS_A).tau == pytest.approx(0.56, abs=DISPLAY)


@pytest.mark.unit
def test_parker2011_original_control_method() -> None:
    # Parker et al. (2011): (16 + 7 - 3) / 36 = .56 and 13 / 26 = .50.
    table = tau_u_table(P2011_A, P2011_B, method="parker_original")
    assert table.row(AVB_PLUS_B_MINUS_A).tau == pytest.approx(20 / 36, abs=EXACT)
    assert table.row(AVB_MINUS_A).tau == pytest.approx(0.5, abs=EXACT)


# --------------------------------------------------------------------------
# scan ``exampleAB``
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_johanna_scan_complete_tau_b_table() -> None:
    a, b = JOHANNA[:5], JOHANNA[5:]
    table = tau_u_table(a, b, method="scan_complete_b")
    want = {
        AVB: (75, 75, 0, 0, 75, 75.00, 1.00),
        TREND_A: (10, 5, 5, 0, 0, 10.00, 0.00),
        TREND_B: (105, 87, 17, 1, 70, 104.50, 0.67),
        AVB_MINUS_A: (85, 80, 5, 0, 75, 126.75, 0.59),
        AVB_PLUS_B: (180, 162, 17, 1, 145, 184.45, 0.79),
        AVB_PLUS_B_MINUS_A: (190, 167, 22, 1, 145, 189.50, 0.77),
    }
    assert [r.label for r in table.rows] == list(want)
    for row in table.rows:
        pairs, pos, neg, ties, s, d, tau_value = want[row.label]
        assert (row.pairs, row.pos, row.neg, row.ties, row.s) == (pairs, pos, neg, ties, s)
        assert row.d == pytest.approx(d, abs=DISPLAY)
        assert row.tau == pytest.approx(tau_value, abs=DISPLAY)


@pytest.mark.unit
def test_johanna_scan_parker_and_tarlow_rows() -> None:
    a, b = JOHANNA[:5], JOHANNA[5:]
    revised = tau_u_table(a, b, method="singlecasees")
    assert revised.row(AVB).tau == pytest.approx(1.00, abs=DISPLAY)
    assert revised.row(AVB_MINUS_A).tau == pytest.approx(1.00, abs=DISPLAY)
    assert revised.row(AVB_PLUS_B).tau == pytest.approx(0.81, abs=DISPLAY)
    assert revised.row(AVB_PLUS_B_MINUS_A).tau == pytest.approx(0.76, abs=DISPLAY)
    # tau-a with the summed-pairs denominator: 75 / 85.
    assert tau_u(a, b, method="parker_original") == pytest.approx(0.88, abs=DISPLAY)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "want_nap", "want_pnd", "want_tau_u_a", "want_tau_u_ba"),
    [
        (JOHANNA, 1.00, 1.00, 0.59, 0.77),
        (KAROLINA, 0.97, 13 / 15, 0.55, 0.78),
        (ANJA, 0.98, 14 / 15, 0.62, 0.64),
    ],
)
def test_example_ab_overlap(
    values: tuple[int, ...],
    want_nap: float,
    want_pnd: float,
    want_tau_u_a: float,
    want_tau_u_ba: float,
) -> None:
    a, b = values[:5], values[5:]
    assert nap(a, b) == pytest.approx(want_nap, abs=DISPLAY)
    assert pnd(a, b) == pytest.approx(want_pnd, abs=DISPLAY)
    table = tau_u_table(a, b, method="scan_complete_b")
    assert table.row(AVB_MINUS_A).tau == pytest.approx(want_tau_u_a, abs=DISPLAY)
    assert table.row(AVB_PLUS_B_MINUS_A).tau == pytest.approx(want_tau_u_ba, abs=DISPLAY)
    assert tau_u(a, b, method="scan_complete_b") == table.row(AVB_MINUS_A).tau


# --------------------------------------------------------------------------
# Improvement direction
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_decrease_mirrors_increase_on_negated_data() -> None:
    a, b = _flip(P2011B_A), _flip(P2011B_B)
    assert nap(a, b, improvement="decrease") == pytest.approx(0.9166667, abs=EXACT)
    assert pnd(a, b, improvement="decrease") == pytest.approx(0.7142857, abs=EXACT)
    assert tau(a, b, improvement="decrease") == pytest.approx(0.8333333, abs=EXACT)
    assert tau_u(a, b, improvement="decrease") == pytest.approx(0.7380952, abs=EXACT)


@pytest.mark.unit
def test_pnd_decrease_uses_baseline_minimum() -> None:
    # Baseline minimum is 3; only 1 and 2 fall strictly below it.
    assert pnd((5, 3, 4), (1, 3, 2, 6), improvement="decrease") == 0.5


@pytest.mark.unit
def test_nap_counts_ties_as_half() -> None:
    assert nap((1,), (1,)) == 0.5
    assert tau((1,), (1,)) == 0.0


@pytest.mark.unit
def test_revised_tau_u_can_exceed_one() -> None:
    # Falling baseline + complete separation: (S_AB - S_A) / mn > 1.
    assert tau_u((5, 4, 3, 2, 1), (10, 11)) > 1.0
    assert tau_u((5, 4, 3, 2, 1), (10, 11), method="parker_original") <= 1.0


# --------------------------------------------------------------------------
# Series input
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_series_nonoverlap_matches_functions() -> None:
    series = SingleCaseSeries(
        values=P2011B_A + P2011B_B,
        phases=("A",) * len(P2011B_A) + ("B",) * len(P2011B_B),
        name="Parker2011b",
    )
    result = nonoverlap(series)
    assert isinstance(result, NonOverlapResult)
    assert result.name == "Parker2011b"
    assert (result.n_a, result.n_b) == (6, 7)
    assert result.nap == pytest.approx(0.9166667, abs=EXACT)
    assert result.nap_rescaled == pytest.approx(2 * result.nap - 1)
    assert result.pnd == pytest.approx(0.7142857, abs=EXACT)
    assert result.tau == pytest.approx(0.8333333, abs=EXACT)
    assert result.tau_u == pytest.approx(0.7380952, abs=EXACT)
    assert result.method == "singlecasees"
    assert isinstance(result.tau_u_table, TauUTable)


@pytest.mark.unit
def test_select_phases_defaults_to_first_two_labels() -> None:
    series = SingleCaseSeries(
        values=(1, 2, 3, 4, 5, 6), phases=("A1", "A1", "B1", "B1", "A2", "B2")
    )
    assert select_phases(series) == ((1.0, 2.0), (3.0, 4.0))
    pooled = select_phases(series, a=("A1", "A2"), b=("B1", "B2"))
    assert pooled == ((1.0, 2.0, 5.0), (3.0, 4.0, 6.0))


@pytest.mark.unit
def test_alternating_treatments_omits_trend_correction() -> None:
    series = SingleCaseSeries(values=(1, 5, 2, 6, 1, 7), phases=("A", "B") * 3)
    result = nonoverlap(series)
    assert result.nap == 1.0
    assert result.tau_u is None
    assert result.tau_u_table is None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_series_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        SingleCaseSeries(values=(1, 2), phases=("A",))


@pytest.mark.unit
def test_rejects_unknown_improvement() -> None:
    with pytest.raises(ValueError, match="improvement"):
        nap((1,), (2,), improvement="up")
    with pytest.raises(ValueError, match="improvement"):
        SingleCaseSeries(values=(1, 2), phases=("A", "B"), improvement="up")


@pytest.mark.unit
def test_rejects_empty_phase_and_nan() -> None:
    with pytest.raises(ValueError, match="empty"):
        nap((), (1,))
    with pytest.raises(ValueError, match="finite"):
        pnd((1, math.nan), (2,))


@pytest.mark.unit
def test_rejects_unknown_tau_u_method() -> None:
    with pytest.raises(ValueError, match="method"):
        tau_u((1, 2), (3, 4), method="tarlow")


@pytest.mark.unit
def test_select_phases_rejects_missing_label() -> None:
    series = SingleCaseSeries(values=(1, 2), phases=("A", "B"))
    with pytest.raises(ValueError, match="no observations"):
        select_phases(series, a=("A",), b=("C",))


@pytest.mark.unit
def test_single_string_label_is_one_label() -> None:
    series = SingleCaseSeries(values=(1, 2, 9, 9, 5, 6), phases=("A", "A", "B", "B", "1", "1"))
    with pytest.raises(ValueError, match="no observations"):
        select_phases(series, a="A1", b="B")
    assert select_phases(series, a="A", b="B") == ((1.0, 2.0), (9.0, 9.0))


@pytest.mark.unit
def test_default_b_skips_labels_already_in_a() -> None:
    series = SingleCaseSeries(values=(1, 2, 3, 4), phases=("A", "B", "B", "C"))
    assert select_phases(series, a=("B",)) == ((2.0, 3.0), (1.0,))


@pytest.mark.unit
def test_reused_reversal_labels_need_explicit_trend_correction() -> None:
    values = (1, 2, 8, 9, 5, 6, 3, 4)
    series = SingleCaseSeries(values=values, phases=("A", "A", "B", "B") * 2)
    assert nonoverlap(series).tau_u is None
    forced = nonoverlap(series, trend_correction=True)
    # Pooled phases are concatenated in session order: A = 1,2,5,6 / B = 8,9,3,4.
    assert forced.tau_u == tau_u((1, 2, 5, 6), (8, 9, 3, 4))
    numbered = SingleCaseSeries(
        values=values, phases=("A1", "A1", "B1", "B1", "A2", "A2", "B2", "B2")
    )
    assert nonoverlap(numbered).tau_u == pytest.approx(tau_u((1, 2), (8, 9)))
