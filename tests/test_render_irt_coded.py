"""Tests for render_irt_coded_cumulative_record (in-memory entrypoint).

This is the counterpart to render_cumulative_record — an array/file-like
renderer suitable for HTTP endpoint handlers that already have a live
snapshot and should not round-trip through an OKL v1 log.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from session_analyzer.visualizer.cumulative_diff import (
    render_irt_coded_cumulative_record,
)


@pytest.mark.unit
class TestRenderIrtCoded:
    def test_writes_png_bytes(self) -> None:
        buf = io.BytesIO()
        ok = render_irt_coded_cumulative_record(
            response_times=(0.0, 1.0, 6.0, 7.0, 13.0),
            reinforcement_times=(6.5,),
            output=buf,
            irt_threshold_sec=5.0,
            theme="readable",
            fmt="png",
            show_event_pen=False,
        )
        assert ok is True
        data = buf.getvalue()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_writes_svg(self) -> None:
        buf = io.BytesIO()
        render_irt_coded_cumulative_record(
            response_times=(0.0, 1.0, 6.0, 10.0),
            reinforcement_times=(),
            output=buf,
            irt_threshold_sec=4.0,
            theme="jeab-bw",
            fmt="svg",
            show_event_pen=False,
        )
        head = buf.getvalue()[:512]
        assert b"<svg" in head or head.lstrip().startswith(b"<?xml")

    def test_writes_pdf(self) -> None:
        buf = io.BytesIO()
        render_irt_coded_cumulative_record(
            response_times=(0.0, 1.0, 6.0, 10.0),
            reinforcement_times=(6.5,),
            output=buf,
            irt_threshold_sec=5.0,
            theme="nature-color",
            fmt="pdf",
            show_event_pen=False,
        )
        assert buf.getvalue()[:4] == b"%PDF"

    def test_writes_to_path(self, tmp_path: Path) -> None:
        out = tmp_path / "irt.png"
        render_irt_coded_cumulative_record(
            response_times=(0.0, 1.0, 6.0, 7.0),
            reinforcement_times=(),
            output=out,
            irt_threshold_sec=5.0,
            theme="readable",
            show_event_pen=False,
        )
        assert out.exists()

    def test_empty_response_times_returns_false(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        ok = render_irt_coded_cumulative_record(
            response_times=(),
            reinforcement_times=(),
            output=out,
            irt_threshold_sec=5.0,
            theme="readable",
            show_event_pen=False,
        )
        assert ok is False
        assert not out.exists()

    def test_non_positive_threshold_raises(self) -> None:
        buf = io.BytesIO()
        with pytest.raises(ValueError, match="irt_threshold_sec"):
            render_irt_coded_cumulative_record(
                response_times=(0.0, 1.0),
                reinforcement_times=(),
                output=buf,
                irt_threshold_sec=0.0,
                theme="readable",
                fmt="png",
            )

    def test_unknown_theme_raises(self) -> None:
        buf = io.BytesIO()
        with pytest.raises(KeyError):
            render_irt_coded_cumulative_record(
                response_times=(0.0, 1.0),
                reinforcement_times=(),
                output=buf,
                irt_threshold_sec=5.0,
                theme="no-such",
                fmt="png",
            )

    def test_unknown_fmt_raises(self) -> None:
        buf = io.BytesIO()
        with pytest.raises(ValueError):
            render_irt_coded_cumulative_record(
                response_times=(0.0, 1.0),
                reinforcement_times=(),
                output=buf,
                irt_threshold_sec=5.0,
                theme="readable",
                fmt="bmp",
            )

    def test_irt_classification_matches_threshold(self, tmp_path: Path) -> None:
        """First response is always classified 'long'; subsequent classified
        against the threshold. The function should not raise and should
        produce a figure for mixed populations."""
        out = tmp_path / "mixed.png"
        render_irt_coded_cumulative_record(
            response_times=(0.0, 0.5, 1.0, 10.0, 10.5, 20.0),
            reinforcement_times=(10.0, 20.0),
            output=out,
            irt_threshold_sec=5.0,
            theme="readable",
            show_event_pen=False,
        )
        assert out.exists()
