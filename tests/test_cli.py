"""Tests for the session-analyzer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_analyzer.cli import main
from tests.conftest import write_okl_log as _write_log


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
    log = tmp_path / "session.txt"
    _write_log(
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


@pytest.mark.integration
def test_diff_plot_subcommand_writes_png(tmp_path: Path) -> None:
    log = tmp_path / "s.txt"
    _write_log(
        log,
        [
            {"timestamp": 0.0, "type": "response"},
            {"timestamp": 0.5, "type": "response"},
            {"timestamp": 5.0, "type": "response"},
            {"timestamp": 5.1, "type": "reinforcer_start"},
        ],
    )
    out = tmp_path / "irt.png"
    rc = main(
        ["diff-plot", str(log), "--output", str(out), "--irt-threshold", "2.0"]
    )
    assert rc == 0
    assert out.exists()


@pytest.mark.integration
def test_latency_subcommand_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "s.txt"
    _write_log(
        log,
        [
            {"timestamp": 0.0, "type": "reinforcer_start"},
            {"timestamp": 1.2, "type": "response"},
            {"timestamp": 10.0, "type": "reinforcer_start"},
            {"timestamp": 11.0, "type": "response"},
            {"timestamp": 11.5, "type": "response"},
        ],
    )
    rc = main(["latency", str(log)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(1.1, abs=1e-6)


@pytest.mark.integration
def test_latency_reports_no_valid_when_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "s.txt"
    _write_log(log, [{"timestamp": 0.0, "type": "response"}])
    rc = main(["latency", str(log)])
    assert rc == 0
    assert "no valid latencies" in capsys.readouterr().out


@pytest.mark.integration
def test_breakpoint_subcommand_detects_terminal_pause(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "s.txt"
    _write_log(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 10.0, "type": "reinforcer_start", "ratio": 1},
            {"timestamp": 30.0, "type": "reinforcer_start", "ratio": 2},
            {"timestamp": 60.0, "type": "reinforcer_start", "ratio": 4},
            {"timestamp": 700.0, "type": "session_end"},
        ],
    )
    rc = main(["breakpoint", str(log), "--pause-threshold", "300"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["breakpoint_ratio"] == 4
    assert out["method"] == "pause_threshold"


@pytest.mark.integration
def test_discounting_subcommand_fits_from_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "disc.csv"
    csv_path.write_text(
        "delay,value\n"
        "0,100\n1,95.24\n7,74.07\n30,40.0\n90,18.18\n180,10.0\n"
    )
    rc = main(["discounting", str(csv_path), "--amount", "100", "--model", "hyperbolic"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["k"] == pytest.approx(0.05, rel=0.05)


@pytest.mark.integration
def test_demand_subcommand_fits_from_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    # Generate from known parameters (q0=50, alpha=0.005, k=3).
    prices = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    q0, alpha, k_val = 50.0, 0.005, 3.0
    rows = ["price,consumption"]
    for p in prices:
        q = q0 * 10 ** (k_val * (math.exp(-alpha * q0 * p) - 1.0))
        rows.append(f"{p},{q}")
    csv_path = tmp_path / "demand.csv"
    csv_path.write_text("\n".join(rows) + "\n")

    rc = main(["demand", str(csv_path), "--k", str(k_val)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["q0"] == pytest.approx(q0, rel=1e-2)
    assert out["alpha"] == pytest.approx(alpha, rel=1e-2)
    assert out["p_max"] > 0.0
    assert out["o_max"] > 0.0


@pytest.mark.unit
def test_unknown_subcommand_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["bogus"])
