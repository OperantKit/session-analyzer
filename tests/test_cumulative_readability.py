"""Readability of the cumulative-record figure.

The legend must tell responses and reinforcement apart, and the event-pen
axis is drawn only when there are events to draw on it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from session_analyzer.visualizer.cumulative_diff import (  # noqa: E402
    build_irt_coded_figure,
)
from session_analyzer.visualizer.cumulative_record import (  # noqa: E402
    build_cumulative_record_figure,
)

RESPONSES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
REINFORCERS = [3.0, 6.0, 9.0]


def _legend_handles(ax: plt.Axes) -> dict[str, object]:
    legend = ax.get_legend()
    assert legend is not None
    return {
        text.get_text(): handle
        for text, handle in zip(
            legend.get_texts(), legend.legend_handles, strict=True
        )
    }


@pytest.mark.unit
class TestLegendDistinguishesSeries:
    def test_legend_entries_use_two_colors(self) -> None:
        fig, main_ax, _ = build_cumulative_record_figure(RESPONSES, REINFORCERS)
        handles = _legend_handles(main_ax)
        colors = {to_hex(h.get_color()) for h in handles.values()}
        assert set(handles) == {"Responses", "Reinforcement"}
        assert len(colors) == 2
        plt.close(fig)

    def test_legend_colors_match_drawn_marks(self) -> None:
        fig, main_ax, _ = build_cumulative_record_figure(RESPONSES, REINFORCERS)
        handles = _legend_handles(main_ax)
        cumulative = next(
            line for line in main_ax.get_lines() if line.get_label() == "_cumulative"
        )
        pips = [line for line in main_ax.get_lines() if line.get_label() == "_pip"]
        assert len(pips) == len(REINFORCERS)
        assert to_hex(handles["Responses"].get_color()) == to_hex(
            cumulative.get_color()
        )
        for pip in pips:
            assert to_hex(pip.get_color()) == to_hex(
                handles["Reinforcement"].get_color()
            )
        plt.close(fig)

    def test_reinforcement_legend_sample_is_not_a_horizontal_line(self) -> None:
        fig, main_ax, _ = build_cumulative_record_figure(RESPONSES, REINFORCERS)
        sample = _legend_handles(main_ax)["Reinforcement"]
        assert sample.get_linestyle() == "None"
        assert sample.get_marker() not in (None, "None", "", " ")
        plt.close(fig)

    @pytest.mark.parametrize(
        "theme",
        [
            "readable",
            "readable-dark",
            "jeab-bw",
            "jeab-bw-marker",
            "nature-color",
            "science-color",
            "preprint-draft",
        ],
    )
    def test_every_preset_keeps_two_colors(self, theme: str) -> None:
        fig, main_ax, _ = build_cumulative_record_figure(
            RESPONSES, REINFORCERS, theme=theme
        )
        colors = {
            to_hex(h.get_color()) for h in _legend_handles(main_ax).values()
        }
        assert len(colors) == 2
        plt.close(fig)


@pytest.mark.unit
class TestNoEmptyEventPen:
    def test_no_event_axis_without_events(self) -> None:
        fig, _, event_ax = build_cumulative_record_figure(RESPONSES, REINFORCERS)
        assert event_ax is None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_empty_event_mapping_draws_no_event_axis(self) -> None:
        fig, _, event_ax = build_cumulative_record_figure(
            RESPONSES, REINFORCERS, event_markers={}
        )
        assert event_ax is None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_event_axis_with_events(self) -> None:
        fig, _, event_ax = build_cumulative_record_figure(
            RESPONSES, REINFORCERS, event_markers={"SD": [(1.0, "on"), (5.0, "off")]}
        )
        assert event_ax is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_irt_no_event_axis_without_events(self) -> None:
        fig, _, event_ax = build_irt_coded_figure(
            response_points=[(t, i + 1, i % 2 == 0) for i, t in enumerate(RESPONSES)],
            reinforcement_points=[(3.0, 3)],
            irt_threshold_sec=1.0,
        )
        assert event_ax is None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_irt_event_axis_with_events(self) -> None:
        fig, _, event_ax = build_irt_coded_figure(
            response_points=[(t, i + 1, i % 2 == 0) for i, t in enumerate(RESPONSES)],
            reinforcement_points=[(3.0, 3)],
            irt_threshold_sec=1.0,
            event_markers={"probe": [2.0, 4.0]},
        )
        assert event_ax is not None
        assert len(fig.axes) == 2
        plt.close(fig)
