"""Theme-integration tests for plot_irt_coded_cumulative_record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_analyzer.visualizer.cumulative_diff import (
    plot_irt_coded_cumulative_record,
)


from tests.conftest import write_okl_log


def _simple_log(tmp_path: Path) -> Path:
    log = tmp_path / "drl.txt"
    write_okl_log(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 1.0, "type": "response"},
            {"timestamp": 2.0, "type": "response"},
            {"timestamp": 9.0, "type": "response"},
            {"timestamp": 9.1, "type": "reinforcer_start"},
            {"timestamp": 10.0, "type": "response"},
        ],
    )
    return log


@pytest.mark.unit
class TestIrtPlotThemed:
    def test_default_theme_png_smoke(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "irt.png"
        plot_irt_coded_cumulative_record(log, out, irt_threshold_sec=5.0)
        assert out.exists() and out.stat().st_size > 0

    def test_named_theme(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "irt.png"
        plot_irt_coded_cumulative_record(
            log, out, irt_threshold_sec=5.0, theme="nature-color"
        )
        assert out.exists()

    def test_svg_output_from_extension(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "irt.svg"
        plot_irt_coded_cumulative_record(
            log, out, irt_threshold_sec=5.0, theme="jeab-bw"
        )
        head = out.read_bytes()[:256]
        assert b"<svg" in head or head.lstrip().startswith(b"<?xml")

    def test_pdf_output(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "irt.pdf"
        plot_irt_coded_cumulative_record(
            log, out, irt_threshold_sec=5.0, theme="preprint-draft"
        )
        assert out.read_bytes()[:4] == b"%PDF"

    def test_unknown_theme_raises(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "irt.png"
        with pytest.raises(KeyError):
            plot_irt_coded_cumulative_record(
                log, out, irt_threshold_sec=5.0, theme="no-such"
            )

    def test_explicit_colors_override_theme(self, tmp_path: Path) -> None:
        """Backward-compat: explicit color params still work alongside theme."""
        log = _simple_log(tmp_path)
        out = tmp_path / "irt.png"
        # Should not raise even though theme chooses other colors.
        plot_irt_coded_cumulative_record(
            log,
            out,
            irt_threshold_sec=5.0,
            theme="jeab-bw",
            short_irt_color="#ff00ff",
            long_irt_color="#00ffff",
        )
        assert out.exists()
