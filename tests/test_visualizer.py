"""Tests for cumulative-record visualizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_analyzer.visualizer.cumulative_record import plot_cumulative_record


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


@pytest.mark.unit
def test_writes_png_for_log_with_responses(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"
    _write_jsonl(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 0.5, "type": "response"},
            {"timestamp": 1.2, "type": "response"},
            {"timestamp": 1.3, "type": "reinforcer_start"},
            {"timestamp": 2.0, "type": "response"},
        ],
    )
    output = tmp_path / "out.png"

    plot_cumulative_record(log, output)

    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.unit
def test_no_png_written_for_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    output = tmp_path / "out.png"

    plot_cumulative_record(log, output)

    assert not output.exists()


@pytest.mark.unit
def test_no_png_written_when_no_responses(tmp_path: Path) -> None:
    log = tmp_path / "no_resp.jsonl"
    _write_jsonl(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 1.0, "type": "session_end"},
        ],
    )
    output = tmp_path / "out.png"

    plot_cumulative_record(log, output)

    assert not output.exists()


@pytest.mark.unit
def test_skips_blank_lines(tmp_path: Path) -> None:
    log = tmp_path / "blank.jsonl"
    log.write_text(
        '{"timestamp": 0.0, "type": "session_start"}\n'
        "\n"
        '{"timestamp": 0.5, "type": "response"}\n'
        "  \n"
        '{"timestamp": 1.0, "type": "response"}\n'
    )
    output = tmp_path / "out.png"

    plot_cumulative_record(log, output)
    assert output.exists()
