"""Tests for response-rate timeline visualizer."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from session_analyzer.visualizer.response_rate import (
    compute_windowed_rate,
    render_response_rate_timeline,
)


@pytest.mark.unit
class TestComputeWindowedRate:
    def test_constant_rate_over_interval(self) -> None:
        """A response every 1 s gives 1 resp/s = 60 rpm in steady state."""
        response_times = [float(i) for i in range(0, 60)]
        times, rates = compute_windowed_rate(
            response_times, window_sec=10.0, step_sec=1.0
        )
        # After a full window has accumulated, rate should be ~60 rpm.
        steady = [r for t, r in zip(times, rates, strict=True) if t >= 15.0]
        assert len(steady) > 0
        for r in steady:
            assert 55.0 <= r <= 65.0

    def test_empty_response_times(self) -> None:
        times, rates = compute_windowed_rate([], window_sec=10.0, step_sec=1.0)
        assert times == []
        assert rates == []

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_windowed_rate([0.0], window_sec=0.0, step_sec=1.0)

    def test_invalid_step_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_windowed_rate([0.0], window_sec=10.0, step_sec=0.0)

    def test_step_larger_than_data_produces_one_bin(self) -> None:
        times, rates = compute_windowed_rate(
            [0.1, 0.5, 0.9], window_sec=1.0, step_sec=10.0
        )
        # Exactly one bin in this degenerate case.
        assert len(times) == 1
        assert all(r >= 0.0 for r in rates)
        # Bin must be pinned inside observed time span, not projected forward.
        assert times[0] <= 0.9 + 1e-9


@pytest.mark.unit
class TestRenderResponseRateTimeline:
    def test_writes_png(self, tmp_path: Path) -> None:
        out = tmp_path / "rr.png"
        ok = render_response_rate_timeline(
            response_times=[float(i) * 0.5 for i in range(40)],
            output=out,
            window_sec=5.0,
            step_sec=1.0,
            theme="readable",
        )
        assert ok is True
        assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_writes_svg_to_bytesio(self) -> None:
        buf = io.BytesIO()
        render_response_rate_timeline(
            response_times=[float(i) * 0.5 for i in range(20)],
            output=buf,
            window_sec=5.0,
            step_sec=1.0,
            theme="jeab-bw",
            fmt="svg",
        )
        data = buf.getvalue()
        assert b"<svg" in data[:512] or data.lstrip().startswith(b"<?xml")

    def test_writes_pdf(self) -> None:
        buf = io.BytesIO()
        render_response_rate_timeline(
            response_times=[float(i) * 0.5 for i in range(20)],
            output=buf,
            window_sec=5.0,
            step_sec=1.0,
            theme="nature-color",
            fmt="pdf",
        )
        assert buf.getvalue()[:4] == b"%PDF"

    def test_empty_returns_false(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        ok = render_response_rate_timeline(
            response_times=[],
            output=out,
            window_sec=5.0,
            step_sec=1.0,
        )
        assert ok is False
        assert not out.exists()

    def test_unknown_theme_raises(self) -> None:
        buf = io.BytesIO()
        with pytest.raises(KeyError):
            render_response_rate_timeline(
                response_times=[0.0, 1.0],
                output=buf,
                window_sec=5.0,
                step_sec=1.0,
                theme="no-such",
                fmt="png",
            )

    def test_unknown_fmt_raises(self) -> None:
        buf = io.BytesIO()
        with pytest.raises(ValueError):
            render_response_rate_timeline(
                response_times=[0.0, 1.0],
                output=buf,
                window_sec=5.0,
                step_sec=1.0,
                fmt="bmp",
            )

    def test_reinforcement_overlay_optional(self, tmp_path: Path) -> None:
        out = tmp_path / "with_reinf.png"
        render_response_rate_timeline(
            response_times=[float(i) * 0.5 for i in range(40)],
            output=out,
            window_sec=5.0,
            step_sec=1.0,
            theme="readable",
            reinforcement_times=[5.0, 10.0, 15.0],
        )
        assert out.exists()
