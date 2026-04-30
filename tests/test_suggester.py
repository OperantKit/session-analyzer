"""Tests for the DSL-driven analysis suggester (stub).

The suggester inspects a parsed contingency-dsl experiment spec and
returns a list of recommended live analyses — e.g. a Concurrent schedule
should surface the Generalized Matching Law panel, a Progressive Ratio
should surface breakpoint / demand curve, a single FR/VI should not.

This session lands only the stable surface and hand-written fixtures;
the full DSL parse integration lands in a follow-up. The stub is
deliberately data-driven from a small dict spec so downstream consumers
can wire it up now.
"""

from __future__ import annotations

import pytest

from session_analyzer.suggester import AnalysisSuggestion, suggest_from_spec


@pytest.mark.unit
def test_suggest_concurrent_surfaces_matching_law() -> None:
    spec = {"schedule": {"type": "Concurrent", "components": ["VI30", "VI60"]}}
    out = suggest_from_spec(spec)
    names = {s.name for s in out}
    assert "matching_law" in names


@pytest.mark.unit
def test_suggest_progressive_ratio_surfaces_breakpoint() -> None:
    spec = {"schedule": {"type": "PR", "step": 2}}
    out = suggest_from_spec(spec)
    names = {s.name for s in out}
    assert "breakpoint" in names
    assert "demand_curve" in names


@pytest.mark.unit
def test_suggest_simple_fr_does_not_recommend_matching() -> None:
    spec = {"schedule": {"type": "FR", "n": 10}}
    out = suggest_from_spec(spec)
    names = {s.name for s in out}
    assert "matching_law" not in names
    # A simple ratio schedule should at minimum offer cumulative record.
    assert "cumulative_record" in names


@pytest.mark.unit
def test_suggestion_is_frozen_dataclass() -> None:
    s = AnalysisSuggestion(name="x", reason="y", tier="light")
    with pytest.raises(Exception):
        s.name = "z"  # type: ignore[misc]


@pytest.mark.unit
def test_suggest_tags_heavy_vs_light_tier() -> None:
    """Light fits can run on the periodic ticker; heavy fits require session-analyzer.

    Asserts specific tier assignments so a regression that swapped them
    (e.g. marking GML slope as heavy) would fail the test.
    """
    concurrent = {s.name: s for s in suggest_from_spec({"schedule": {"type": "Concurrent"}})}
    assert concurrent["matching_law"].tier == "light"  # log-log OLS

    pr = {s.name: s for s in suggest_from_spec({"schedule": {"type": "PR", "step": 2}})}
    assert pr["breakpoint"].tier == "light"
    assert pr["demand_curve"].tier == "heavy"  # Hursh-Silberberg non-linear
