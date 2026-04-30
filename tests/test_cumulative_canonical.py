"""Canonical Ferster & Skinner cumulative-record tests.

Reference
---------
Lattal, K. A. (2004). Steps and pips in the history of the cumulative recorder.
Journal of the Experimental Analysis of Behavior, 82(3), 329-355.
https://doi.org/10.1901/jeab.2004.82-329

Canonical defaults (Gerbrands C-3/C-4):
- Reinforcement pip: short diagonal tick ~30 deg below horizontal
- Pen reset: every 550 responses (pen returns to baseline)
- Event pen: separate on/off trace below the cumulative record
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from session_analyzer.visualizer.cumulative_record import (  # noqa: E402
    CANONICAL_PIP_ANGLE_DEG,
    CANONICAL_RESET_RESPONSES,
    EventMarkerSpec,
    build_cumulative_record_figure,
    render_cumulative_record,
)


@pytest.mark.unit
class TestCanonicalDefaults:
    def test_canonical_constants(self) -> None:
        assert CANONICAL_PIP_ANGLE_DEG == pytest.approx(-30.0)
        assert CANONICAL_RESET_RESPONSES == 550

    def test_build_figure_uses_canonical_defaults(self) -> None:
        fig, main_ax, event_ax = build_cumulative_record_figure(
            response_times=[float(i) * 0.1 for i in range(1, 11)],
            reinforcement_times=[0.5, 1.0],
        )
        assert fig is not None
        assert main_ax is not None
        # Event pen axis exists by default (canonical F&S)
        assert event_ax is not None
        plt.close(fig)


@pytest.mark.unit
class TestPipIsDiagonal:
    """Reinforcement mark must be a short diagonal pip, not a vertical tick."""

    def test_pip_lines_are_drawn_as_line2d_not_scatter(self) -> None:
        fig, main_ax, _ = build_cumulative_record_figure(
            response_times=[0.1, 0.2, 0.3, 0.4],
            reinforcement_times=[0.25],
        )
        # Pips must be Line2D short segments, not scatter collections with
        # marker="|".
        pip_lines = [
            line for line in main_ax.get_lines() if line.get_label() == "_pip"
        ]
        assert len(pip_lines) == 1, (
            "Expected exactly one pip Line2D labeled '_pip' for one reinforcer."
        )
        xs = pip_lines[0].get_xdata()
        ys = pip_lines[0].get_ydata()
        assert len(xs) == 2 and len(ys) == 2
        # The segment must be slanted (dx != 0 AND dy != 0)
        assert xs[1] != xs[0], "pip must not be vertical"
        assert ys[1] != ys[0], "pip must not be horizontal"
        plt.close(fig)

    def test_pip_angle_param_controls_slope(self) -> None:
        # With +45 deg the pip must slope up-right; with -60 deg it must slope down-right.
        for angle in (45.0, -60.0):
            fig, main_ax, _ = build_cumulative_record_figure(
                response_times=[0.1, 0.2, 0.3],
                reinforcement_times=[0.2],
                pip_angle_deg=angle,
            )
            pip = next(
                line for line in main_ax.get_lines() if line.get_label() == "_pip"
            )
            xs = pip.get_xdata()
            ys = pip.get_ydata()
            # Work in display pixels to avoid data-scale skew.
            p0 = main_ax.transData.transform((xs[0], ys[0]))
            p1 = main_ax.transData.transform((xs[1], ys[1]))
            observed = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            assert observed == pytest.approx(angle, abs=0.5), (
                f"Expected pip angle {angle} deg in display coords, got {observed}"
            )
            plt.close(fig)


@pytest.mark.unit
class TestPenReset:
    def test_default_wraps_at_550(self) -> None:
        # 600 responses: default reset_responses=550 should cause one wrap.
        rts = [float(i) * 0.01 for i in range(1, 601)]
        fig, main_ax, _ = build_cumulative_record_figure(
            response_times=rts, reinforcement_times=[]
        )
        main_line = next(
            line for line in main_ax.get_lines() if line.get_label() == "_cumulative"
        )
        ys = main_line.get_ydata()
        # The displayed cumulative trace must never exceed reset threshold.
        assert max(ys) <= 550
        # And it must have dropped back to 0 at some point (a wrap happened).
        assert min(ys) == 0
        plt.close(fig)

    def test_reset_disabled_with_none(self) -> None:
        rts = [float(i) * 0.01 for i in range(1, 601)]
        fig, main_ax, _ = build_cumulative_record_figure(
            response_times=rts,
            reinforcement_times=[],
            reset_responses=None,
        )
        main_line = next(
            line for line in main_ax.get_lines() if line.get_label() == "_cumulative"
        )
        ys = main_line.get_ydata()
        assert max(ys) == 600
        plt.close(fig)

    def test_custom_reset_threshold(self) -> None:
        rts = [float(i) * 0.01 for i in range(1, 11)]  # 10 responses
        fig, main_ax, _ = build_cumulative_record_figure(
            response_times=rts, reinforcement_times=[], reset_responses=5
        )
        main_line = next(
            line for line in main_ax.get_lines() if line.get_label() == "_cumulative"
        )
        ys = main_line.get_ydata()
        assert max(ys) <= 5
        plt.close(fig)


@pytest.mark.unit
class TestEventPen:
    def test_event_pen_axis_present_by_default(self) -> None:
        fig, _, event_ax = build_cumulative_record_figure(
            response_times=[0.1, 0.2],
            reinforcement_times=[],
        )
        assert event_ax is not None
        plt.close(fig)

    def test_event_pen_disabled(self) -> None:
        fig, _, event_ax = build_cumulative_record_figure(
            response_times=[0.1, 0.2],
            reinforcement_times=[],
            show_event_pen=False,
        )
        assert event_ax is None
        plt.close(fig)

    def test_sd_onoff_step_drawn_on_event_pen(self) -> None:
        # SD onset at t=1.0, SD offset at t=3.0 -> binary step on event pen.
        fig, _, event_ax = build_cumulative_record_figure(
            response_times=[0.5, 1.5, 2.0, 2.5],
            reinforcement_times=[],
            event_markers={"SD": [(1.0, "on"), (3.0, "off")]},
        )
        assert event_ax is not None
        # Expect a Line2D labeled "SD" on the event axis.
        sd_lines = [l for l in event_ax.get_lines() if l.get_label() == "SD"]
        assert len(sd_lines) == 1
        xs = list(sd_lines[0].get_xdata())
        ys = list(sd_lines[0].get_ydata())
        # Must transition 0 -> 1 at t=1.0 and 1 -> 0 at t=3.0.
        idx_on = xs.index(1.0)
        idx_off = xs.index(3.0)
        # With steps-post, ys[i] is the level of the segment starting at xs[i].
        # Crossing "on" at idx_on: level rises; crossing "off" at idx_off: falls.
        assert ys[idx_on] > ys[idx_on - 1]
        assert ys[idx_off] < ys[idx_off - 1]
        plt.close(fig)

    def test_custom_marker_spec(self) -> None:
        spec = EventMarkerSpec(label="timeout", symbol="x", color="#ff0000")
        fig, _, event_ax = build_cumulative_record_figure(
            response_times=[0.5, 1.5],
            reinforcement_times=[],
            event_markers={"timeout": [1.0, 1.2]},
            event_marker_styles={"timeout": spec},
        )
        assert event_ax is not None
        # Custom markers drawn as scatter with the given symbol.
        scatters = event_ax.collections
        assert len(scatters) >= 1
        plt.close(fig)


@pytest.mark.unit
class TestRenderIntegration:
    def test_render_still_produces_png(self) -> None:
        buf = io.BytesIO()
        ok = render_cumulative_record(
            response_times=tuple(float(i) * 0.1 for i in range(1, 11)),
            reinforcement_times=(0.5, 1.0),
            output=buf,
            fmt="png",
        )
        assert ok is True
        data = buf.getvalue()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_with_event_markers(self, tmp_path: Path) -> None:
        out = tmp_path / "canonical.png"
        ok = render_cumulative_record(
            response_times=tuple(float(i) * 0.1 for i in range(1, 11)),
            reinforcement_times=(0.5,),
            output=out,
            event_markers={"SD": [(0.3, "on"), (0.8, "off")]},
        )
        assert ok is True
        assert out.exists() and out.stat().st_size > 0
