// Mirror of session_analyzer.suggester for operantkit-frontend.
//
// Keep in sync with src/session_analyzer/suggester.py. This file is
// hand-written (not generated) because the JSON contract is intentionally
// tiny and stable — generating TS from the Python dataclass would pull in
// a cross-language build step for no benefit.
//
// The suggester lives in session-analyzer because that package owns the
// canonical set of analyses. The HTTP contract is served by whichever
// tool exposes it (session-visualizer's `/suggest` endpoint, report
// generators, notebooks); the TS types below are transport-neutral.
//
// Usage in operantkit-frontend:
//
//     import type { AnalysisSuggestion, SuggestResponse } from
//       "session-analyzer/frontend-types/suggestions";
//
//     const res = await fetch(`${VIZ_ORIGIN}/suggest`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(ast),
//     });
//     const { suggestions }: SuggestResponse = await res.json();

export type Tier = "light" | "heavy";

/** One recommended dashboard panel. Frontend keys panels on `name`. */
export interface AnalysisSuggestion {
  /** Stable identifier — e.g. "matching_law", "breakpoint", "cumulative_record". */
  name: string;
  /** Human-readable justification tied to the DSL structure. Shown in tooltip. */
  reason: string;
  /**
   * Execution tier:
   * - "light": safe to run on the visualizer's in-process periodic tick.
   * - "heavy": requires hand-off to session-analyzer.
   */
  tier: Tier;
}

/** Response shape of POST /suggest. */
export interface SuggestResponse {
  suggestions: AnalysisSuggestion[];
  /** Populated when the posted AST was malformed; `suggestions` is then empty. */
  error?: string;
}

/**
 * Known analysis names. Frontend uses this for exhaustive switch checks
 * over AnalysisName.
 *
 * Sync policy: this list must be kept in parity with the `name=` strings
 * emitted by `session_visualizer.suggester`. Additions on the Python side
 * do not break TS consumers at runtime — unknown names simply fall into
 * `string` — but they weaken the static exhaustiveness guarantee. When
 * editing suggester.py, mirror the new name here.
 */
export const ANALYSIS_NAMES = [
  "cumulative_record",
  "post_reinforcement_pause",
  "irt_distribution",
  "rate_histogram",
  "matching_law",
  "punishment_sensitivity",
  "component_cumulative_records",
  "breakpoint",
  "demand_curve",
  "avoidance_rate",
  "escape_latency",
] as const;

export type AnalysisName = (typeof ANALYSIS_NAMES)[number];
