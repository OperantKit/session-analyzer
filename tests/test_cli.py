"""Tests for the session-analyzer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_analyzer.cli import main


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


@pytest.mark.unit
def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "session-analyzer" in captured.out
    assert "plot" in captured.out


@pytest.mark.integration
def test_plot_subcommand_writes_png(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"
    _write_jsonl(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 0.4, "type": "response"},
            {"timestamp": 0.9, "type": "response"},
            {"timestamp": 1.0, "type": "reinforcer_start"},
        ],
    )
    output = tmp_path / "cli_out.png"

    rc = main(["plot", str(log), "--output", str(output)])
    assert rc == 0
    assert output.exists()


@pytest.mark.unit
def test_plot_requires_log_path() -> None:
    with pytest.raises(SystemExit):
        main(["plot"])
