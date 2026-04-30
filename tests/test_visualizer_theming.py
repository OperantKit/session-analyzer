"""Tests for themed cumulative-record rendering (Phase 2)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from session_analyzer.visualizer.cumulative_record import (
    plot_cumulative_record,
    render_cumulative_record,
)
from session_analyzer.visualizer.themes import get_theme


from tests.conftest import write_osl_log as _write_osl


def _simple_log(tmp_path: Path) -> Path:
    log = tmp_path / "session.txt"
    _write_osl(
        log,
        [
            {"timestamp": 0.0, "type": "session_start"},
            {"timestamp": 0.5, "type": "response"},
            {"timestamp": 1.2, "type": "response"},
            {"timestamp": 1.3, "type": "reinforcer_start"},
            {"timestamp": 2.0, "type": "response"},
            {"timestamp": 3.0, "type": "response"},
            {"timestamp": 3.1, "type": "reinforcer_start"},
        ],
    )
    return log


@pytest.mark.unit
class TestPlotCumulativeRecordThemed:
    def test_default_theme_still_writes_png(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "out.png"
        plot_cumulative_record(log, out)
        assert out.exists() and out.stat().st_size > 0

    def test_theme_name_accepted(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "out.png"
        plot_cumulative_record(log, out, theme="jeab-bw")
        assert out.exists()

    def test_theme_spec_accepted(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "out.png"
        plot_cumulative_record(log, out, theme=get_theme("nature-color"))
        assert out.exists()

    def test_unknown_theme_raises(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "out.png"
        with pytest.raises(KeyError):
            plot_cumulative_record(log, out, theme="nonexistent")

    def test_svg_output_by_extension(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "out.svg"
        plot_cumulative_record(log, out, theme="jeab-bw")
        assert out.exists()
        head = out.read_bytes()[:256]
        assert b"<svg" in head or head.lstrip().startswith(b"<?xml")

    def test_pdf_output_by_extension(self, tmp_path: Path) -> None:
        log = _simple_log(tmp_path)
        out = tmp_path / "out.pdf"
        plot_cumulative_record(log, out, theme="nature-color")
        assert out.exists()
        assert out.read_bytes()[:4] == b"%PDF"


@pytest.mark.unit
class TestRenderCumulativeRecord:
    """Direct renderer for in-memory snapshots (used by session-visualizer)."""

    def test_writes_to_bytesio_png(self) -> None:
        buf = io.BytesIO()
        render_cumulative_record(
            response_times=(0.5, 1.2, 2.0, 3.0),
            reinforcement_times=(1.3, 3.1),
            output=buf,
            theme="readable",
            fmt="png",
        )
        data = buf.getvalue()
        assert len(data) > 0
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_writes_to_bytesio_svg(self) -> None:
        buf = io.BytesIO()
        render_cumulative_record(
            response_times=(0.5, 1.2, 2.0),
            reinforcement_times=(1.3,),
            output=buf,
            theme="jeab-bw",
            fmt="svg",
        )
        data = buf.getvalue()
        assert b"<svg" in data[:512] or data.lstrip().startswith(b"<?xml")

    def test_writes_to_path(self, tmp_path: Path) -> None:
        out = tmp_path / "render.png"
        render_cumulative_record(
            response_times=(0.5, 1.0),
            reinforcement_times=(),
            output=out,
            theme="readable",
        )
        assert out.exists()

    def test_empty_series_returns_false(self, tmp_path: Path) -> None:
        """With zero responses we should not produce a figure but not raise."""
        out = tmp_path / "empty.png"
        result = render_cumulative_record(
            response_times=(),
            reinforcement_times=(),
            output=out,
            theme="readable",
        )
        assert result is False
        assert not out.exists()

    def test_non_empty_returns_true(self, tmp_path: Path) -> None:
        out = tmp_path / "nonempty.png"
        result = render_cumulative_record(
            response_times=(0.1, 0.2),
            reinforcement_times=(),
            output=out,
            theme="readable",
        )
        assert result is True

    def test_unknown_format_raises(self) -> None:
        buf = io.BytesIO()
        with pytest.raises(ValueError):
            render_cumulative_record(
                response_times=(0.1,),
                reinforcement_times=(),
                output=buf,
                theme="readable",
                fmt="bmp",
            )

    def test_jeab_bw_png_is_grayscale(self, tmp_path: Path) -> None:
        """The rendered PNG under jeab-bw must have no chromatic pixels."""
        png = tmp_path / "jeab.png"
        render_cumulative_record(
            response_times=tuple(float(i) * 0.5 for i in range(20)),
            reinforcement_times=(2.0, 5.0),
            output=png,
            theme="jeab-bw",
        )
        # Read back with matplotlib's image reader (avoids adding PIL dep).
        import matplotlib.image as mpimg

        arr = mpimg.imread(png)
        # arr shape: (H, W, 3|4). For grayscale PNG savefig still gives RGBA
        # where R==G==B at every pixel.
        if arr.ndim == 3 and arr.shape[2] >= 3:
            diffs = (
                abs(arr[..., 0] - arr[..., 1]).max()
                + abs(arr[..., 1] - arr[..., 2]).max()
            )
            assert diffs < 1e-3, (
                f"jeab-bw produced chromatic pixels (max channel diff={diffs})"
            )
