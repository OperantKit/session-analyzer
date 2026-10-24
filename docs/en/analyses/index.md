# Analysis catalog

A one-stop reference for every analysis implemented in
`session_analyzer.analytics`. Each entry lists **theme** (what it
measures), the **equation** where relevant, an **example output** shaped
exactly as the package returns it, and the **canonical reference**.

Analyses are split across eight category files. The `suggester.name`
column referenced inside each file is the stable identifier returned
by [`session_analyzer.suggester`](../../../src/session_analyzer/suggester.py)
and consumed by `operantkit-frontend`.

Embedded figures are rendered with the `jeab-bw` theme (JEAB / JABA
monochrome, 3.3 × 2.5 in, 300 dpi Helvetica) via
[`docs/generate_figures.py`](../../generate_figures.py). Regenerate
with `.venv/bin/python docs/generate_figures.py`. The data is
synthetic parameter-recovery caricature — not from any real
experiment.

[日本語版](../../ja/analyses/index.md)

---

## Categories

1. [Response rate & local pattern](rate-and-pattern.md) — cumulative
   record, PRP, IRT bout decomposition, FR run length, latency,
   log-survivor.
2. [Choice & matching](choice-and-matching.md) — generalized matching
   law, single-alternative hyperbola, choice allocation, concurrent-
   chains prediction, time-allocation matching.
3. [Value & economics](value-and-economics.md) — progressive-ratio
   breakpoint, exponential demand, delay discounting, unit-price /
   cross-price elasticity, adjusting indifference point.
4. [Temporal](temporal.md) — peak procedure (binned + MLE),
   within-session dynamics.
5. [Stimulus control](stimulus-control.md) — generalization gradients
   (Gaussian / exponential), discrimination ratio, peak shift, signal
   detection theory (d', c, β, ROC).
6. [Persistence & extinction](persistence-and-extinction.md) —
   behavioral momentum, Kaplan-Meier survival, resurgence (3-phase),
   renewal (ABA / ABC / AAB).
7. [Pavlovian](pavlovian.md) — tracking index (sign- vs goal-tracking),
   observing response, conditioned-reinforcement strength.
8. [Variability](variability.md) — U-value, Lag-N variability.

---

## Tiering

The `tier` field in `AnalysisSuggestion` reflects CPU budget, not
epistemic status:

- **light** — fit cost is O(ms) for typical session sizes: closed-form
  summaries, log-log OLS, Kaplan-Meier, binned peak analysis. Safe to
  run on the `session-visualizer` periodic tick (10 s / 1 min).
- **heavy** — non-linear least squares (demand, discounting,
  generalization), EM mixture (bout), three-state MLE (peak). Minutes
  at the tail; defer to post-hoc runs of `session-analyzer`.

---

## See also

- [`suggester.py`](../../../src/session_analyzer/suggester.py) — maps
  `contingency-dsl` AST → subset of the analyses above.
- [Modules and their sources](../modules.md) — every module in
  `session_analyzer.analytics` and `session_analyzer.visualizer`, the
  import paths; argument signatures are in each module's docstrings.
