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


def _write_single_case_csv(path: Path, rows: list[tuple[str, str, float]]) -> None:
    lines = ["case,phase,value"] + [f"{c},{p},{v}" for c, p, v in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.integration
def test_single_case_subcommand_reports_each_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # scan ``exampleAB`` case "Johanna" and Parker et al. (2011) Figure 2.
    johanna = [54, 53, 56, 58, 52, 61, 62, 71, 66, 64, 78, 70, 74, 82, 77, 86, 68, 80, 86, 87]
    rows = [("Johanna", "A" if i < 5 else "B", v) for i, v in enumerate(johanna)]
    parker = [2, 3, 5, 3, 4, 5, 5, 7, 6]
    rows += [("Parker", "A" if i < 4 else "B", v) for i, v in enumerate(parker)]
    csv_path = tmp_path / "cases.csv"
    _write_single_case_csv(csv_path, rows)
    figure = tmp_path / "cases.png"

    rc = main(["single-case", str(csv_path), "--plot", str(figure)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert [c["name"] for c in out["cases"]] == ["Johanna", "Parker"]
    johanna_out, parker_out = out["cases"]
    phase_a, phase_b = johanna_out["visual_analysis"]["phases"]
    assert phase_a["slope"] == pytest.approx(0.100, abs=5e-4)
    assert phase_b["intercept"] == pytest.approx(62.758, abs=5e-4)
    (contrast,) = parker_out["visual_analysis"]["contrasts"]
    assert contrast["overlap"] == pytest.approx(0.6)
    assert parker_out["nonoverlap"]["nap"] == pytest.approx(0.9)
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.integration
def test_single_case_subcommand_without_case_column_and_nan_as_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "one.csv"
    csv_path.write_text("phase,value\nA,1\nB,2\nB,3\n")

    rc = main(["single-case", str(csv_path), "--improvement", "decrease"])
    assert rc == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw  # strict JSON
    (case,) = json.loads(raw)["cases"]
    assert case["name"] is None
    assert case["visual_analysis"]["improvement"] == "decrease"
    assert case["visual_analysis"]["phases"][0]["sd"] is None


@pytest.mark.integration
def test_single_case_subcommand_alternating_treatments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "atd.csv"
    csv_path.write_text("phase,value\nA,1\nB,4\nA,2\nB,6\nA,3\nB,8\n")

    rc = main(["single-case", str(csv_path), "--by", "label"])
    assert rc == 0
    (case,) = json.loads(capsys.readouterr().out)["cases"]
    assert case["visual_analysis"]["contrasts"] == []
    assert case["nonoverlap"]["nap"] == 1.0
    assert case["nonoverlap"]["tau_u"] is None


@pytest.mark.integration
def test_single_case_subcommand_selects_phases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "abab.csv"
    csv_path.write_text("phase,value\nA1,1\nA1,2\nB1,5\nB1,6\nA2,2\nA2,1\nB2,7\nB2,6\n")

    rc = main(["single-case", str(csv_path), "--a", "A1,A2", "--b", "B1,B2"])
    assert rc == 0
    (case,) = json.loads(capsys.readouterr().out)["cases"]
    assert case["nonoverlap"]["a_labels"] == ["A1", "A2"]
    assert case["nonoverlap"]["n_b"] == 4
    assert len(case["visual_analysis"]["contrasts"]) == 3


@pytest.mark.unit
def test_single_case_subcommand_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("condition,score\nA,1\n")
    with pytest.raises(SystemExit, match="phase"):
        main(["single-case", str(csv_path)])


@pytest.mark.integration
def test_single_case_subcommand_keeps_other_cases_when_one_cannot_be_compared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Multiple baseline: case Y has not entered intervention yet.
    csv_path = tmp_path / "mb.csv"
    csv_path.write_text("case,phase,value\nX,A,1\nX,A,2\nX,B,5\nY,A,1\nY,A,2\n")

    rc = main(["single-case", str(csv_path)])
    assert rc == 0
    x, y = json.loads(capsys.readouterr().out)["cases"]
    assert x["nonoverlap"]["nap"] == 1.0 and x["error"] is None
    assert y["nonoverlap"] is None
    assert "two phase labels" in y["error"]
    assert y["visual_analysis"]["phases"][0]["n"] == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "body", ["phase,value\nA,1\nB,\n", "phase,value\nA,1\nB,nan\n", "phase,value\nA,1\nB\n"]
)
def test_single_case_subcommand_reports_bad_rows(tmp_path: Path, body: str) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(body)
    with pytest.raises(SystemExit, match="line 3"):
        main(["single-case", str(csv_path)])


@pytest.mark.integration
def test_single_case_subcommand_reads_utf8_bom_and_strips_spaces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "excel.csv"
    csv_path.write_bytes("﻿phase, value\n A ,1\nB,2\n".encode())

    rc = main(["single-case", str(csv_path), "--a", "A"])
    assert rc == 0
    (case,) = json.loads(capsys.readouterr().out)["cases"]
    assert [p["label"] for p in case["visual_analysis"]["phases"]] == ["A", "B"]
