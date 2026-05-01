"""Tests for cumulative-record visualizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_analyzer.visualizer.cumulative_record import plot_cumulative_record


from tests.conftest import write_okl_log as _write_osl


@pytest.mark.unit
def test_writes_png_for_log_with_responses(tmp_path: Path) -> None:
    log = tmp_path / "session.txt"
    _write_osl(
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
    log = tmp_path / "empty.txt"
    log.write_text("")
    output = tmp_path / "out.png"

    plot_cumulative_record(log, output)

    assert not output.exists()


@pytest.mark.unit
def test_no_png_written_when_no_responses(tmp_path: Path) -> None:
    log = tmp_path / "no_resp.txt"
    _write_osl(
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
    log = tmp_path / "blank.txt"
    _write_osl(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 0.5, "type": "response"},
            {"timestamp": 1.0, "type": "response"},
        ],
    )
    # Inject blank lines into the body so the reader has to skip them
    text = log.read_text()
    head, sep, body = text.partition("# ---\n")
    body_lines = body.splitlines()
    body_lines.insert(1, "")
    if len(body_lines) >= 3:
        body_lines.insert(3, "  ")
    log.write_text(head + sep + "\n".join(body_lines) + "\n")
    output = tmp_path / "out.png"

    plot_cumulative_record(log, output)
    assert output.exists()
