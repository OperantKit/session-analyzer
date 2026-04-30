"""Tests for the contingency-dsl AST adapter.

``suggest_from_ast`` takes a resolved ``contingency-dsl`` Program AST
(matching ``schema/operant/ast-resolved.schema.json``) and returns the
same ``AnalysisSuggestion`` list that ``suggest_from_spec`` produces for
its minimal dict shape — but driven by the real schedule structure.

AST fragments here follow the resolved schema literally (Phase 2/3):

    Atomic:   {"type":"Atomic","dist":"F","domain":"R","value":10}
    Compound: {"type":"Compound","combinator":"Conc","components":[...]}
    Modifier: {"type":"Modifier","modifier":"PR","pr_step":"linear",...}

The tests focus on *mapping semantics*, not parser correctness — a
pre-parsed dict is supplied directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from session_analyzer.suggester import (
    AnalysisSuggestion,
    suggest_from_ast,
    suggestions_to_json,
)


def _program(schedule: dict[str, Any]) -> dict[str, Any]:
    """Wrap a schedule fragment in a minimal Program envelope."""
    return {
        "type": "Program",
        "param_decls": [],
        "bindings": [],
        "schedule": schedule,
    }


# ---- Atomic schedules ------------------------------------------------------


@pytest.mark.unit
def test_atomic_fr_gets_cumulative_record_only() -> None:
    ast = _program({"type": "Atomic", "dist": "F", "domain": "R", "value": 10})
    names = {s.name for s in suggest_from_ast(ast)}
    assert names == {"cumulative_record"}


@pytest.mark.unit
def test_atomic_fi_surfaces_post_reinforcement_pause() -> None:
    ast = _program({"type": "Atomic", "dist": "F", "domain": "I", "value": 30})
    names = {s.name for s in suggest_from_ast(ast)}
    assert "post_reinforcement_pause" in names


@pytest.mark.unit
def test_variable_interval_surfaces_irt_distribution() -> None:
    ast = _program({"type": "Atomic", "dist": "V", "domain": "I", "value": 60})
    names = {s.name for s in suggest_from_ast(ast)}
    assert "irt_distribution" in names


# ---- Compound: Concurrent --------------------------------------------------


@pytest.mark.unit
def test_concurrent_surfaces_matching_law() -> None:
    ast = _program(
        {
            "type": "Compound",
            "combinator": "Conc",
            "components": [
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 60},
            ],
        }
    )
    out = {s.name: s for s in suggest_from_ast(ast)}
    assert "matching_law" in out
    assert out["matching_law"].tier == "light"


@pytest.mark.unit
def test_concurrent_with_punishment_surfaces_delta_log_ratio() -> None:
    """Compound(Conc) with PUNISH params suggests a de Villiers-style analysis."""
    ast = _program(
        {
            "type": "Compound",
            "combinator": "Conc",
            "components": [
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 60},
            ],
            "params": {"PUNISH": {"component": [{"component": 2, "schedule": {
                "type": "Atomic", "dist": "F", "domain": "R", "value": 1,
            }}]}},
        }
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "matching_law" in names
    assert "punishment_sensitivity" in names


# ---- Compound: Mult/Mix/Chain ---------------------------------------------


@pytest.mark.unit
def test_multiple_schedule_surfaces_component_records() -> None:
    ast = _program(
        {
            "type": "Compound",
            "combinator": "Mult",
            "components": [
                {"type": "Atomic", "dist": "F", "domain": "R", "value": 10},
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
            ],
        }
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "component_cumulative_records" in names
    # Matching law not appropriate for Mult — components are not concurrent.
    assert "matching_law" not in names


# ---- Modifier: PR, DRL, DRH -----------------------------------------------


@pytest.mark.unit
def test_progressive_ratio_surfaces_breakpoint_and_demand() -> None:
    ast = _program(
        {
            "type": "Modifier",
            "modifier": "PR",
            "pr_step": "linear",
            "pr_start": 1,
            "pr_increment": 1,
        }
    )
    out = {s.name: s for s in suggest_from_ast(ast)}
    assert "breakpoint" in out
    assert out["breakpoint"].tier == "light"
    assert out["demand_curve"].tier == "heavy"


@pytest.mark.unit
def test_drl_surfaces_irt_distribution() -> None:
    ast = _program(
        {"type": "Modifier", "modifier": "DRL", "value": 10, "time_unit": "s"}
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "irt_distribution" in names


@pytest.mark.unit
def test_drh_surfaces_rate_histogram() -> None:
    ast = _program(
        {"type": "Modifier", "modifier": "DRH", "value": 5, "time_unit": "s"}
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "rate_histogram" in names


# ---- AversiveSchedule ------------------------------------------------------


@pytest.mark.unit
def test_sidman_avoidance_surfaces_avoidance_rate() -> None:
    ast = _program(
        {
            "type": "AversiveSchedule",
            "kind": "Sidman",
            "params": {"shock_shock": 5, "response_shock": 30},
        }
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "avoidance_rate" in names


@pytest.mark.unit
def test_escape_surfaces_escape_latency() -> None:
    ast = _program(
        {
            "type": "AversiveSchedule",
            "kind": "Escape",
            "params": {"onset": 5},
        }
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "escape_latency" in names


# ---- Recursion: nested schedules merge suggestions -------------------------


@pytest.mark.unit
def test_mult_with_pr_component_inherits_breakpoint() -> None:
    """A Mult of (FR, PR) should surface both component-records and breakpoint."""
    ast = _program(
        {
            "type": "Compound",
            "combinator": "Mult",
            "components": [
                {"type": "Atomic", "dist": "F", "domain": "R", "value": 10},
                {
                    "type": "Modifier",
                    "modifier": "PR",
                    "pr_step": "linear",
                    "pr_start": 1,
                    "pr_increment": 1,
                },
            ],
        }
    )
    names = {s.name for s in suggest_from_ast(ast)}
    assert "component_cumulative_records" in names
    assert "breakpoint" in names


@pytest.mark.unit
def test_suggestions_are_deduplicated() -> None:
    """Two VI components must not produce duplicate irt_distribution entries."""
    ast = _program(
        {
            "type": "Compound",
            "combinator": "Conc",
            "components": [
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 60},
            ],
        }
    )
    out = suggest_from_ast(ast)
    assert len(out) == len({s.name for s in out})


# ---- JSON serialization for frontend consumption --------------------------


@pytest.mark.unit
def test_suggestions_to_json_is_plain_dict_list() -> None:
    """Frontend contract: the JSON payload is a list of flat dicts.

    Each entry has exactly ``name``, ``reason``, ``tier``. No nested
    objects, no None fields, no extra keys. The dashboard can consume
    this shape directly with JSON.parse — no custom deserializer needed.
    """
    suggestions = [
        AnalysisSuggestion(name="cumulative_record", reason="baseline", tier="light"),
        AnalysisSuggestion(name="matching_law", reason="Conc schedule", tier="light"),
    ]
    payload = suggestions_to_json(suggestions)

    assert isinstance(payload, list)
    assert payload == [
        {"name": "cumulative_record", "reason": "baseline", "tier": "light"},
        {"name": "matching_law", "reason": "Conc schedule", "tier": "light"},
    ]


@pytest.mark.unit
def test_suggest_from_ast_rejects_invalid_program_shape() -> None:
    with pytest.raises(ValueError):
        suggest_from_ast({"not": "a program"})


@pytest.mark.unit
def test_mult_wrapping_conc_does_not_surface_matching_law() -> None:
    """Mult(Conc(...), ...) must not leak matching_law.

    The matching law assumes one shared time base across concurrent
    operanda. Mult separates components by discriminative stimuli, so
    the response/reinforcement ratios are not commensurable across the
    Mult boundary. The suppression is context-driven: a top-level Conc
    still gets matching_law, but a Conc nested inside Mult does not.
    """
    nested = _program(
        {
            "type": "Compound",
            "combinator": "Mult",
            "components": [
                {
                    "type": "Compound",
                    "combinator": "Conc",
                    "components": [
                        {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
                        {"type": "Atomic", "dist": "V", "domain": "I", "value": 60},
                    ],
                },
                {"type": "Atomic", "dist": "F", "domain": "R", "value": 10},
            ],
        }
    )
    names = {s.name for s in suggest_from_ast(nested)}
    assert "matching_law" not in names
    assert "component_cumulative_records" in names


@pytest.mark.unit
def test_malformed_nested_ast_returns_empty_not_attributeerror() -> None:
    """A non-dict component must not crash the walker."""
    ast = _program(
        {
            "type": "Compound",
            "combinator": "Conc",
            "components": [
                "not-a-dict",  # schema-invalid but must not raise
                {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
            ],
        }
    )
    # Walker should silently skip non-Mapping entries and still produce
    # suggestions for the well-formed sibling.
    out = suggest_from_ast(ast)
    names = {s.name for s in out}
    assert "matching_law" in names  # Conc still recognized at top level


@pytest.mark.unit
def test_suggest_from_ast_accepts_bare_schedule_expression() -> None:
    """Permit passing a schedule expression directly, without Program envelope.

    This makes it trivial for the frontend to pass a subtree for
    what-if analysis (e.g. a single phase from a phase-sequence).
    """
    names = {
        s.name
        for s in suggest_from_ast(
            {"type": "Atomic", "dist": "F", "domain": "R", "value": 10}
        )
    }
    assert "cumulative_record" in names
