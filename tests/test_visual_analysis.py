"""Tests for the visual-analysis aids of single-case designs.

Level, trend, variability, immediacy and overlap follow the definitions
of the What Works Clearinghouse single-case design standards
(Kratochwill et al., 2010, pp. 17-18). Expected values:

- Trend intercept and slope of ``exampleAB$Johanna`` are scan's published
  ``trend()`` output (scan-Book chapter 8: Linear.A 54.400 / 0.100,
  Linear.B 62.758 / 1.625; measurement times start at 0 in each phase).
- Means, medians, SDs and ranges of the same data, and the small hand
  examples below, were computed with Python's ``statistics`` module or by
  hand, not with this implementation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from session_analyzer.analytics.single_case import SingleCaseSeries
from session_analyzer.analytics.visual_analysis import (
    PhaseContrast,
    PhaseDescriptives,
    VisualAnalysis,
    contrast_phases,
    describe_phases,
    visual_analysis,
)

# scan ``exampleAB`` case "Johanna" (A = 5, B = 15).
JOHANNA_VALUES = (
    54, 53, 56, 58, 52,
    61, 62, 71, 66, 64, 78, 70, 74, 82, 77, 86, 68, 80, 86, 87,
)
JOHANNA = SingleCaseSeries(values=JOHANNA_VALUES, phases=("A",) * 5 + ("B",) * 15, name="Johanna")

# Parker et al. (2011) Figure 2 data (scan ``Parker2011``).
P2011 = SingleCaseSeries(values=(2, 3, 5, 3, 4, 5, 5, 7, 6), phases=("A",) * 4 + ("B",) * 5)


# --------------------------------------------------------------------------
# Phase descriptives
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_trend_matches_scan_trend_output() -> None:
    a, b = describe_phases(JOHANNA)
    assert a.intercept == pytest.approx(54.400, abs=5e-4)
    assert a.slope == pytest.approx(0.100, abs=5e-4)
    assert b.intercept == pytest.approx(62.758, abs=5e-4)
    assert b.slope == pytest.approx(1.625, abs=5e-4)


@pytest.mark.unit
def test_level_and_variability_of_each_phase() -> None:
    a, b = describe_phases(JOHANNA)
    assert isinstance(a, PhaseDescriptives)
    assert (a.label, a.n, a.start, a.stop) == ("A", 5, 0, 5)
    assert (b.label, b.n, b.start, b.stop) == ("B", 15, 5, 20)
    assert a.mean == pytest.approx(54.6)
    assert a.median == pytest.approx(54.0)
    assert a.sd == pytest.approx(2.4083189157584592)
    assert (a.minimum, a.maximum) == (52.0, 58.0)
    assert a.residual_sd == pytest.approx(2.774887385102322)
    assert b.mean == pytest.approx(74.13333333333334)
    assert b.median == pytest.approx(74.0)
    assert b.sd == pytest.approx(8.943207052336001)
    assert (b.minimum, b.maximum) == (61.0, 87.0)
    assert b.residual_sd == pytest.approx(5.40909727527)


@pytest.mark.unit
def test_short_phases_return_nan_where_undefined() -> None:
    series = SingleCaseSeries(values=(1, 2, 3, 5), phases=("A", "B", "B", "C"))
    a, b, c = describe_phases(series)
    assert a.n == 1 and a.mean == 1.0
    assert math.isnan(a.sd) and math.isnan(a.slope) and math.isnan(a.residual_sd)
    assert b.slope == pytest.approx(1.0) and b.intercept == pytest.approx(2.0)
    assert math.isnan(b.residual_sd)  # needs n >= 3
    assert c.label == "C"


@pytest.mark.unit
def test_repeated_labels_are_separate_runs_in_a_reversal() -> None:
    series = SingleCaseSeries(values=(1, 1, 5, 5, 1, 2, 6, 6), phases=tuple("AABBAABB"))
    runs = describe_phases(series)
    assert [(r.label, r.start, r.stop) for r in runs] == [
        ("A", 0, 2), ("B", 2, 4), ("A", 4, 6), ("B", 6, 8),
    ]
    assert runs[2].mean == pytest.approx(1.5)


@pytest.mark.unit
def test_by_label_pools_alternating_treatments() -> None:
    series = SingleCaseSeries(values=(1, 4, 2, 6, 3, 8), phases=tuple("ABABAB"))
    a, b = describe_phases(series, by="label")
    assert (a.label, a.n, a.mean) == ("A", 3, 2.0)
    assert (b.label, b.n, b.mean) == ("B", 3, 6.0)
    # Trend in session order of the pooled points (sessions 0, 2, 4 for A).
    assert a.slope == pytest.approx(0.5)
    assert b.slope == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Adjacent-phase contrasts
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_contrast_level_trend_and_immediacy() -> None:
    (c,) = contrast_phases(JOHANNA)
    assert isinstance(c, PhaseContrast)
    assert (c.from_label, c.to_label) == ("A", "B")
    assert c.level_change == pytest.approx(74.13333333333334 - 54.6)
    assert c.trend_change == pytest.approx(1.625 - 0.1)
    # Mean of B's first three (61, 62, 71) minus mean of A's last three (56, 58, 52).
    assert c.immediacy == pytest.approx(9.333333333333336)
    assert c.immediacy_points == 3
    assert c.overlap == 0.0


@pytest.mark.unit
def test_contrast_overlap_nap_pnd_on_parker_2011() -> None:
    (c,) = contrast_phases(P2011)
    # B points inside A's range [2, 5]: 4, 5, 5.
    assert c.overlap == pytest.approx(0.6)
    # B points above A's maximum 5: 7, 6.
    assert c.pnd == pytest.approx(0.4)
    # 17 of 20 pairs improve, 2 ties: (17 + 0.5 * 2) / 20.
    assert c.nap == pytest.approx(0.9)
    # First three of B (4, 5, 5) minus last three of A (3, 5, 3).
    assert c.immediacy == pytest.approx(1.0)


@pytest.mark.unit
def test_contrast_uses_improvement_direction() -> None:
    series = SingleCaseSeries(
        values=(5, 6, 7, 2, 1, 1), phases=tuple("AAABBB"), improvement="decrease"
    )
    (c,) = contrast_phases(series)
    assert c.nap == 1.0 and c.pnd == 1.0
    assert c.level_change == pytest.approx(4 / 3 - 6.0)


@pytest.mark.unit
def test_immediacy_uses_fewer_points_when_a_phase_is_short() -> None:
    series = SingleCaseSeries(values=(1, 2, 10, 12, 14, 16), phases=tuple("AABBBB"))
    (c,) = contrast_phases(series)
    assert c.immediacy_points == 2
    assert c.immediacy == pytest.approx(11.0 - 1.5)


@pytest.mark.unit
def test_reversal_gives_one_contrast_per_adjacent_pair() -> None:
    labels = ("A1",) * 2 + ("B1",) * 2 + ("A2",) * 2 + ("B2",) * 2
    series = SingleCaseSeries(values=(1, 1, 5, 5, 1, 2, 6, 6), phases=labels)
    pairs = [(c.from_label, c.to_label) for c in contrast_phases(series)]
    assert pairs == [("A1", "B1"), ("B1", "A2"), ("A2", "B2")]


# --------------------------------------------------------------------------
# Combined result
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_visual_analysis_bundles_descriptives_and_contrasts() -> None:
    result = visual_analysis(JOHANNA)
    assert isinstance(result, VisualAnalysis)
    assert result.name == "Johanna"
    assert result.by == "run"
    assert len(result.phases) == 2 and len(result.contrasts) == 1


@pytest.mark.unit
def test_visual_analysis_by_label_has_no_contrasts() -> None:
    series = SingleCaseSeries(values=(1, 4, 2, 6), phases=tuple("ABAB"))
    result = visual_analysis(series, by="label")
    assert result.contrasts == ()


@pytest.mark.unit
def test_invalid_by_raises() -> None:
    with pytest.raises(ValueError, match="by must be"):
        describe_phases(JOHANNA, by="phase")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.integration
def test_plot_writes_png(tmp_path: Path) -> None:
    from session_analyzer.visualizer.single_case import render_single_case_plot

    out = tmp_path / "johanna.png"
    render_single_case_plot(JOHANNA, out)
    assert out.read_bytes()[:8] == PNG_MAGIC


@pytest.mark.integration
def test_plot_figure_has_phase_lines_and_trend_lines() -> None:
    from session_analyzer.visualizer.single_case import build_single_case_figure

    fig = build_single_case_figure(JOHANNA)
    (ax,) = fig.axes
    gids = [line.get_gid() for line in ax.get_lines()]
    assert gids.count("phase-change") == 1
    assert gids.count("level") == 2
    assert gids.count("trend") == 2
    assert gids.count("data") == 2  # one data path per phase, not joined across the line
    # Data paths stay solid whatever the theme's line-style cycle is.
    from session_analyzer.visualizer.single_case import build_single_case_figure as build

    bw = build(JOHANNA, theme="jeab-bw")
    assert {ln.get_linestyle() for ln in bw.axes[0].get_lines() if ln.get_gid() == "data"} == {"-"}
    labels = {t.get_text() for t in ax.texts}
    assert {"A", "B"} <= labels


@pytest.mark.integration
def test_plot_multiple_baseline_stacks_one_panel_per_case(tmp_path: Path) -> None:
    from session_analyzer.visualizer.single_case import build_single_case_figure

    other = SingleCaseSeries(
        values=(3, 4, 3, 4, 3, 9, 9, 8), phases=("A",) * 5 + ("B",) * 3, name="Karolina"
    )
    fig = build_single_case_figure([JOHANNA, other])
    assert len(fig.axes) == 2
    assert "Karolina" in {t.get_text() for t in fig.axes[1].texts}
    assert "Johanna" in {t.get_text() for t in fig.axes[0].texts}


@pytest.mark.integration
def test_plot_alternating_treatments_joins_points_by_label() -> None:
    from session_analyzer.visualizer.single_case import build_single_case_figure

    series = SingleCaseSeries(values=(1, 4, 2, 6, 3, 8), phases=tuple("ABABAB"))
    fig = build_single_case_figure(series, by="label")
    (ax,) = fig.axes
    gids = [line.get_gid() for line in ax.get_lines()]
    assert gids.count("phase-change") == 0
    assert gids.count("data") == 2
