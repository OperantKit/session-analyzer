"""Tests for IRT-coded cumulative record plotting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_analyzer.visualizer.cumulative_diff import (
    plot_irt_coded_cumulative_record,
)


from tests.conftest import write_osl_log as _write_osl


@pytest.mark.unit
def test_writes_png_for_valid_log(tmp_path: Path) -> None:
    log = tmp_path / "session.txt"
    # IRTs: (1.0, 10.0, 1.5, 8.0). Threshold 5 s -> first response "long"
    # by convention; then short, long, short, long.
    _write_osl(
        log,
        [
            {"timestamp": 0.0, "type": "response"},
            {"timestamp": 1.0, "type": "response"},
            {"timestamp": 11.0, "type": "response"},
            {"timestamp": 12.5, "type": "response"},
            {"timestamp": 20.5, "type": "response"},
            {"timestamp": 20.6, "type": "reinforcer_start"},
        ],
    )
    out = tmp_path / "plot.png"

    plot_irt_coded_cumulative_record(log, out, irt_threshold_sec=5.0)

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.unit
def test_handles_empty_log_without_error(tmp_path: Path) -> None:
    log = tmp_path / "empty.txt"
    log.write_text("")
    out = tmp_path / "empty.png"

    plot_irt_coded_cumulative_record(log, out, irt_threshold_sec=2.0)

    assert not out.exists()  # nothing plotted


@pytest.mark.unit
def test_handles_log_without_responses(tmp_path: Path) -> None:
    log = tmp_path / "no_responses.txt"
    _write_osl(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 5.0, "type": "reinforcer_start"},
        ],
    )
    out = tmp_path / "nr.png"

    plot_irt_coded_cumulative_record(log, out, irt_threshold_sec=2.0)

    assert not out.exists()


@pytest.mark.unit
def test_raises_on_non_positive_threshold(tmp_path: Path) -> None:
    log = tmp_path / "any.txt"
    with pytest.raises(ValueError, match="irt_threshold_sec"):
        plot_irt_coded_cumulative_record(log, tmp_path / "x.png", irt_threshold_sec=0.0)
    with pytest.raises(ValueError, match="irt_threshold_sec"):
        plot_irt_coded_cumulative_record(log, tmp_path / "x.png", irt_threshold_sec=-1.0)


@pytest.mark.unit
def test_accepts_custom_title(tmp_path: Path) -> None:
    log = tmp_path / "s.txt"
    _write_osl(
        log,
        [
            {"timestamp": 0.0, "type": "response"},
            {"timestamp": 3.0, "type": "response"},
        ],
    )
    out = tmp_path / "p.png"
    plot_irt_coded_cumulative_record(
        log, out, irt_threshold_sec=1.0, title="Custom DRL 1s"
    )
    assert out.exists()
