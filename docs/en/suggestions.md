# DSL-driven analysis suggestions

[日本語版](../ja/suggestions.md)

`session_analyzer.suggester` maps a resolved `contingency-dsl` AST to
the set of analyses a dashboard or a report should surface. It is the
authoritative mapping from *experiment design* to *which statistics are
worth computing* — so `session-visualizer`, `operantkit-frontend`, and
report generators all derive their panel lists from the same source.

The module is **pure stdlib**: `from session_analyzer.suggester import
suggest_from_ast` does not require `scipy` or `matplotlib` to be
importable at call time (though `session-analyzer` itself declares
them).

## Python

```python
from session_analyzer.suggester import suggest_from_ast, suggestions_to_json

ast = {
    "type": "Program",
    "param_decls": [],
    "bindings": [],
    "schedule": {
        "type": "Compound",
        "combinator": "Conc",
        "components": [
            {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
            {"type": "Atomic", "dist": "V", "domain": "I", "value": 60},
        ],
    },
}
[(s["name"], s["tier"]) for s in suggestions_to_json(suggest_from_ast(ast))]
# [('cumulative_record', 'light'), ('matching_law', 'light'),
#  ('post_reinforcement_pause', 'light'), ('irt_distribution', 'light')]
```

Each suggestion also carries a `reason` string. Both a full `Program`
envelope and a bare `ScheduleExpr` subtree are accepted — the frontend
can pass a single phase of a phase-sequence for what-if inspection
without rebuilding a Program.

## HTTP contract

`session-visualizer` exposes the suggester at `POST /suggest` when
`session-analyzer` is installed:

```
POST /suggest
Content-Type: application/json

<contingency-dsl resolved AST — Program or ScheduleExpr subtree>
```

Response:

```
{"suggestions": [{"name": "...", "reason": "...", "tier": "light" | "heavy"}, ...]}
```

Malformed input returns HTTP 200 with `{"suggestions": [], "error": "..."}`
so the frontend can render a graceful empty state.

## TypeScript types

Hand-written TS mirror at
[`frontend-types/suggestions.ts`](../../frontend-types/suggestions.ts). Import
from `operantkit-frontend` without any code-generation step:

```ts
import type { AnalysisSuggestion, SuggestResponse } from
  "session-analyzer/frontend-types/suggestions";

const { suggestions }: SuggestResponse = await fetch(`${VIZ}/suggest`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(ast),
}).then(r => r.json());
```

## Mapping

| DSL node | Suggested panels | Tier |
|---|---|---|
| *(all sessions)* | `cumulative_record` | light |
| `Atomic dist=V` (VR/VI) | `irt_distribution` | light |
| `Atomic domain=I` (FI/VI) | `post_reinforcement_pause` | light |
| `Compound combinator=Conc` | `matching_law` | light |
| `Compound combinator=Conc` + `PUNISH` | `punishment_sensitivity` | light |
| `Compound combinator=Mult/Mix/Chain/Tand` | `component_cumulative_records` | light |
| `Modifier modifier=PR` | `breakpoint`, `demand_curve` | light / heavy |
| `Modifier modifier=DRL` | `irt_distribution` | light |
| `Modifier modifier=DRH` | `rate_histogram` | light |
| `AversiveSchedule kind=Sidman/DiscrimAv` | `avoidance_rate` | light |
| `AversiveSchedule kind=Escape` | `escape_latency` | light |

Context-sensitive suppression: a `Conc` nested inside
`Mult/Mix/Chain/Tand` does **not** surface `matching_law` — the matching
law assumes one shared time base, not discriminative-stimulus-separated
components. Nested schedules are walked recursively; duplicate names are
collapsed.

## Tier semantics

- **light** — safe to run on an in-process periodic tick (moving
  windows, log-log OLS).
- **heavy** — defer to this package's non-linear fits (demand curve
  non-linear least squares, EM bout decomposition, bootstrap CIs).
