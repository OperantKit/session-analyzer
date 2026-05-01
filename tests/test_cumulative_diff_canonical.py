"""Canonical F&S tick/reset/event-pen coverage for IRT-coded variant."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from session_analyzer.visualizer.cumulative_diff import (  # noqa: E402
    build_irt_coded_figure,
    plot_irt_coded_cumulative_record,
)
from session_analyzer.visualizer.cumulative_record import (  # noqa: E402
    CANONICAL_PIP_ANGLE_DEG,
    CANONICAL_RESET_RESPONSES,
    EventMarkerSpec,
)


from tests.conftest import write_okl_log as _osl  # noqa: E402


@pytest.mark.unit
class TestPipsCanonical:
    def test_reinforcer_drawn_as_diagonal_pip(self) -> None:
        fig, main_ax, _ = build_irt_coded_figure(
            response_points=[(0.1, 1, True), (0.2, 2, False), (0.3, 3, True)],
            reinforcement_points=[(0.2, 2)],
            irt_threshold_sec=5.0,
        )
        pips = [l for l in main_ax.get_lines() if l.get_label() == "_pip"]
        assert len(pips) == 1
        xs, ys = list(pips[0].get_xdata()), list(pips[0].get_ydata())
        p0 = main_ax.transData.transform((xs[0], ys[0]))
        p1 = main_ax.transData.transform((xs[1], ys[1]))
        observed = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        assert observed == pytest.approx(CANONICAL_PIP_ANGLE_DEG, abs=0.5)
        plt.close(fig)


@pytest.mark.unit
class TestResetCanonical:
    def test_default_wraps_at_550(self) -> None:
        pts = [(float(i) * 0.01, i, True) for i in range(1, 601)]
        fig, main_ax, _ = build_irt_coded_figure(
            response_points=pts, reinforcement_points=[], irt_threshold_sec=1.0
        )
        backdrop = next(
            l for l in main_ax.get_lines() if l.get_label() == "_cumulative"
        )
        ys = backdrop.get_ydata()
        assert max(ys) <= CANONICAL_RESET_RESPONSES
        plt.close(fig)

    def test_reset_disabled(self) -> None:
        pts = [(float(i) * 0.01, i, True) for i in range(1, 601)]
        fig, main_ax, _ = build_irt_coded_figure(
            response_points=pts,
            reinforcement_points=[],
            irt_threshold_sec=1.0,
            reset_responses=None,
        )
        backdrop = next(
            l for l in main_ax.get_lines() if l.get_label() == "_cumulative"
        )
        ys = backdrop.get_ydata()
        assert max(ys) == 600
        plt.close(fig)


@pytest.mark.unit
class TestEventPenCanonical:
    def test_event_pen_default_on(self) -> None:
        fig, _, event_ax = build_irt_coded_figure(
            response_points=[(0.1, 1, True)],
            reinforcement_points=[],
            irt_threshold_sec=1.0,
        )
        assert event_ax is not None
        plt.close(fig)

    def test_custom_markers_forwarded(self) -> None:
        fig, _, event_ax = build_irt_coded_figure(
            response_points=[(0.5, 1, True), (1.5, 2, False)],
            reinforcement_points=[],
            irt_threshold_sec=1.0,
            event_markers={"probe": [0.8, 1.2]},
            event_marker_styles={"probe": EventMarkerSpec(label="probe", symbol="x")},
        )
        assert event_ax is not None
        has_line = any(l.get_label() == "probe" for l in event_ax.get_lines())
        has_scatter = len(event_ax.collections) > 0
        assert has_line or has_scatter
        plt.close(fig)


@pytest.mark.unit
class TestPlotWrapperBackCompat:
    def test_existing_api_still_writes_png(self, tmp_path: Path) -> None:
        log = tmp_path / "s.txt"
        _osl(
            log,
            [
                {"timestamp": 0.0, "type": "response"},
                {"timestamp": 1.0, "type": "response"},
                {"timestamp": 11.0, "type": "response"},
                {"timestamp": 11.1, "type": "reinforcer_start"},
            ],
        )
        out = tmp_path / "out.png"
        plot_irt_coded_cumulative_record(log, out, irt_threshold_sec=5.0)
        assert out.exists() and out.stat().st_size > 0
