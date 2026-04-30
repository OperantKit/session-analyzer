"""DSL-driven analysis suggester.

Given a ``contingency-dsl`` experiment spec, return the list of
:class:`AnalysisSuggestion` entries that a live dashboard or a written
report should surface. The goal is to avoid showing the same five
panels for every experiment and instead display only what the schedule
structure justifies (e.g. generalized-matching slope only for ``Conc``,
breakpoint only for Progressive Ratio).

This package is **pure stdlib** and has no runtime dependencies. It
sits beside ``session-analyzer`` in ``apps/analysis/`` because it is
meta-analysis: it does not compute any statistic itself, it only
recommends which statistics a downstream tool (live dashboard, report
generator, session-analyzer driver) should run.

Two input forms are supported:

1. **Resolved AST** — the canonical form produced by ``contingency-dsl``
   after Phase 2/3 (see ``schema/operant/ast-resolved.schema.json``).
   Use :func:`suggest_from_ast` for this path. Both a full ``Program``
   envelope and a bare ``ScheduleExpr`` subtree are accepted.
2. **Minimal dict** — a hand-written shape convenient for tests and
   frontend prototyping (``{"schedule": {"type": "FR", "n": 10}}``).
   Use :func:`suggest_from_spec` for this path.

The tier field distinguishes:

- ``"light"`` — suitable for in-process periodic ticks (moving windows,
  log-log OLS). Can run inside a live visualizer on a short cadence.
- ``"heavy"`` — defer to ``session-analyzer``. Non-linear least
  squares, EM / Bayesian inference, bootstrap CIs.

### Frontend contract

:func:`suggestions_to_json` serializes a suggestion list into a plain
list of flat JSON objects ``{name, reason, tier}`` that can be consumed
from TypeScript without any custom deserializer. See
``frontend-types/suggestions.ts`` for the matching TS types.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "AnalysisSuggestion",
    "Tier",
    "suggest_from_ast",
    "suggest_from_spec",
    "suggestions_to_json",
]

Tier = Literal["light", "heavy"]


@dataclass(frozen=True)
class AnalysisSuggestion:
    """A single recommended analysis panel.

    Attributes
    ----------
    name:
        Stable identifier consumed by the dashboard (e.g. ``matching_law``,
        ``breakpoint``, ``cumulative_record``). Frontend code keys panel
        components on this value.
    reason:
        Human-readable justification referencing the DSL structure that
        triggered the suggestion. Shown in the panel's info tooltip.
    tier:
        ``"light"`` (ok on an in-process periodic tick) or
        ``"heavy"`` (defer to ``session-analyzer``).
    """

    name: str
    reason: str
    tier: Tier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggestions_to_json(suggestions: Iterable[AnalysisSuggestion]) -> list[dict[str, str]]:
    """Serialize suggestions into a list of flat JSON-ready dicts.

    The output shape is::

        [{"name": "...", "reason": "...", "tier": "light" | "heavy"}, ...]

    This is the contract consumed by ``operantkit-frontend`` and any
    other client that reads a ``/suggest`` endpoint. No field is
    optional.
    """
    return [asdict(s) for s in suggestions]


def suggest_from_spec(spec: Mapping[str, Any]) -> list[AnalysisSuggestion]:
    """Suggest analyses from a minimal hand-written spec dict.

    Accepted input::

        {"schedule": {"type": "Concurrent", "components": [...]}}
        {"schedule": {"type": "PR", "step": 2}}
        {"schedule": {"type": "FR", "n": 10}}

    This form is retained for tests and lightweight prototyping. For
    production use, prefer :func:`suggest_from_ast` which consumes the
    full resolved contingency-dsl AST.
    """
    sched = spec.get("schedule")
    if not isinstance(sched, Mapping):
        return _dedup([_baseline_cumulative()])

    stype = sched.get("type")
    out = [_baseline_cumulative()]

    if stype == "Concurrent":
        out.append(_matching_law())
    elif stype == "PR":
        out.extend(_progressive_ratio())
    return _dedup(out)


def suggest_from_ast(ast: Mapping[str, Any]) -> list[AnalysisSuggestion]:
    """Suggest analyses from a resolved contingency-dsl AST.

    ``ast`` may be either:

    - a full ``Program`` node (``{"type":"Program", "schedule": ..., ...}``), or
    - a bare ``ScheduleExpr`` subtree (``Atomic``, ``Compound``, ``Modifier``,
      ``Special``, ``SecondOrder``, ``AversiveSchedule``).

    Accepting both lets the frontend pass a single phase of a
    phase-sequence for what-if inspection without reconstructing a
    full Program envelope.
    """
    if not isinstance(ast, Mapping) or "type" not in ast:
        raise ValueError("suggest_from_ast requires a dict with a 'type' field")

    if ast["type"] == "Program":
        schedule = ast.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("Program AST is missing a 'schedule' subtree")
    else:
        schedule = ast

    out: list[AnalysisSuggestion] = [_baseline_cumulative()]
    _walk(schedule, out, ctx=frozenset())
    return _dedup(out)


# ---------------------------------------------------------------------------
# Internal walker
# ---------------------------------------------------------------------------


def _walk(
    node: Any,
    out: list[AnalysisSuggestion],
    ctx: frozenset[str],
) -> None:
    """Recursively inspect a ScheduleExpr and append suggestions in-place.

    ``ctx`` carries markers set by ancestor nodes to suppress
    semantically inappropriate suggestions. For example, a ``Conc``
    nested inside ``Mult`` should not surface ``matching_law``:
    matching-law analyses are valid across concurrent operanda sharing a
    single time base, not across Mult components separated by
    discriminative stimuli.

    Non-Mapping inputs are silently ignored rather than raising, so a
    schema-invalid payload produces an empty suggestion list instead of
    an AttributeError at a surprising call site.
    """
    if not isinstance(node, Mapping):
        return
    ntype = node.get("type")
    if ntype == "Atomic":
        _from_atomic(node, out)
    elif ntype == "Compound":
        _from_compound(node, out, ctx=ctx)
    elif ntype == "Modifier":
        _from_modifier(node, out)
    elif ntype == "SecondOrder":
        # Walk both the outer and the inner unit to collect all relevant
        # per-subschedule suggestions.
        for key in ("overall", "unit"):
            _walk(node.get(key), out, ctx=ctx)
    elif ntype == "AversiveSchedule":
        _from_aversive(node, out)
    # Special (EXT / CRF) contributes nothing beyond the baseline.


def _from_atomic(node: Mapping[str, Any], out: list[AnalysisSuggestion]) -> None:
    dist = node.get("dist")
    domain = node.get("domain")
    if domain == "I":
        out.append(
            AnalysisSuggestion(
                name="post_reinforcement_pause",
                reason="Interval schedule: post-reinforcement pause is a canonical descriptor.",
                tier="light",
            )
        )
    if dist == "V":
        out.append(
            AnalysisSuggestion(
                name="irt_distribution",
                reason="Variable schedule: IRT distribution reveals the local rate structure.",
                tier="light",
            )
        )


def _from_compound(
    node: Mapping[str, Any],
    out: list[AnalysisSuggestion],
    ctx: frozenset[str],
) -> None:
    combinator = node.get("combinator")
    components = node.get("components") or []
    params = node.get("params") or {}

    child_ctx = ctx
    if combinator == "Conc":
        # Suppress matching_law when nested inside a Mult/Mix/Chain/Tand:
        # matching-law analyses assume one shared time base, not
        # discriminative-stimulus-separated components.
        if "separated_components" not in ctx:
            out.append(
                AnalysisSuggestion(
                    name="matching_law",
                    reason=(
                        "Concurrent schedule: the generalized matching law "
                        "(Baum, 1974) — log response ratio vs log reinforcement "
                        "ratio — is the canonical summary."
                    ),
                    tier="light",
                )
            )
            if "PUNISH" in params:
                out.append(
                    AnalysisSuggestion(
                        name="punishment_sensitivity",
                        reason=(
                            "Concurrent schedule with PUNISH term: contrast "
                            "response ratio against punishment ratio "
                            "(de Villiers, 1980)."
                        ),
                        tier="light",
                    )
                )
    elif combinator in {"Mult", "Mix", "Chain", "Tand"}:
        out.append(
            AnalysisSuggestion(
                name="component_cumulative_records",
                reason=(
                    f"{combinator} schedule: separate cumulative records "
                    "per component expose component-by-component control."
                ),
                tier="light",
            )
        )
        child_ctx = ctx | {"separated_components"}
    # Alt / Conj / Overlay / Interpolate fall through to component walking.

    for sub in components:
        _walk(sub, out, ctx=child_ctx)


def _from_modifier(node: Mapping[str, Any], out: list[AnalysisSuggestion]) -> None:
    modifier = node.get("modifier")
    if modifier == "PR":
        out.extend(_progressive_ratio())
    elif modifier == "DRL":
        out.append(
            AnalysisSuggestion(
                name="irt_distribution",
                reason=(
                    "DRL: the IRT distribution is the dependent variable "
                    "— mode should exceed the IRT criterion."
                ),
                tier="light",
            )
        )
    elif modifier == "DRH":
        out.append(
            AnalysisSuggestion(
                name="rate_histogram",
                reason="DRH: local response-rate histogram shows whether the rate criterion is met.",
                tier="light",
            )
        )


def _from_aversive(node: Mapping[str, Any], out: list[AnalysisSuggestion]) -> None:
    kind = node.get("kind")
    if kind in {"Sidman", "DiscrimAv"}:
        out.append(
            AnalysisSuggestion(
                name="avoidance_rate",
                reason=(
                    f"{kind} avoidance: response rate and shock frequency "
                    "are the primary outcomes."
                ),
                tier="light",
            )
        )
    elif kind == "Escape":
        out.append(
            AnalysisSuggestion(
                name="escape_latency",
                reason="Escape schedule: latency from onset to response is the dependent measure.",
                tier="light",
            )
        )


# ---------------------------------------------------------------------------
# Common fragments
# ---------------------------------------------------------------------------


def _baseline_cumulative() -> AnalysisSuggestion:
    return AnalysisSuggestion(
        name="cumulative_record",
        reason="Every operant session benefits from a cumulative record.",
        tier="light",
    )


def _matching_law() -> AnalysisSuggestion:
    return AnalysisSuggestion(
        name="matching_law",
        reason=(
            "Concurrent schedule detected. The generalized matching law "
            "(log response ratio vs. log reinforcement ratio) is the "
            "canonical summary."
        ),
        tier="light",
    )


def _progressive_ratio() -> list[AnalysisSuggestion]:
    return [
        AnalysisSuggestion(
            name="breakpoint",
            reason="Progressive Ratio: the breakpoint is the primary outcome.",
            tier="light",
        ),
        AnalysisSuggestion(
            name="demand_curve",
            reason=(
                "Progressive Ratio: Hursh-Silberberg demand curve "
                "(non-linear; defer to session-analyzer)."
            ),
            tier="heavy",
        ),
    ]


def _dedup(items: list[AnalysisSuggestion]) -> list[AnalysisSuggestion]:
    """Drop duplicate names while preserving first-occurrence order."""
    seen: set[str] = set()
    out: list[AnalysisSuggestion] = []
    for s in items:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out
