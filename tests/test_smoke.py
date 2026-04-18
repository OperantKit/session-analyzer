"""Smoke tests verifying public imports."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_import_top_level() -> None:
    import session_analyzer

    assert hasattr(session_analyzer, "plot_cumulative_record")


@pytest.mark.unit
def test_import_analytics() -> None:
    from session_analyzer.analytics import (
        BoutMetrics,
        MatchingParameters,
        analyze_bouts,
        estimate_matching_parameters,
    )

    assert callable(analyze_bouts)
    assert callable(estimate_matching_parameters)
    assert BoutMetrics is not None
    assert MatchingParameters is not None


@pytest.mark.unit
def test_import_cli() -> None:
    from session_analyzer.cli import main

    assert callable(main)
