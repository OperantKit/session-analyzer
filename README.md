# session-analyzer

:jp: [日本語版 README](README.ja.md)

Cumulative-record visualization, statistical analysis, and behavioral model fitting for OperantKit session logs.

Reads OKL v1 session logs (typically produced by `session-recorder`) and produces:

- Cumulative response records with reinforcement tick marks
- IRT-coded cumulative records for DRL / DRO visualization
- Bout-decomposed response rate metrics (Shull, Gaynor, & Grimes, 2001)
- Generalized Matching Law (GML) parameter fits (Baum, 1974)
- Single-alternative hyperbola fits (Herrnstein, 1970)
- Choice allocation analysis for concurrent schedules
- Response latency summaries
- Progressive-ratio breakpoint detection (Richardson & Roberts, 1996). The breakpoint is the last completed ratio; some MED-PC programmes store the ratio in progress in their breakpoint variable, which is one step higher. `extract_step_ratios(..., timeout_sec=...)` skips responses during a post-reinforcement timeout.
- Delay-discounting fits (Mazur, 1987; Rachlin, 2006)
- Exponential demand-curve fits with P_max / O_max (Hursh & Silberberg, 2008)
- Resistance-to-change / behavioral momentum estimation (Nevin, 1992)
- Peak-procedure analysis — binned and three-state MLE (Roberts, 1981; Church, Meck, & Gibbon, 1994)
- Stimulus generalization gradients — Gaussian and exponential (Guttman & Kalish, 1956; Shepard, 1987)
- Post-reinforcement pause (PRP) summaries and quarter-life index (Felton & Lyon, 1966; Herrnstein & Morse, 1957)
- FR run-length decomposition (Schlinger, Derenne, & Baron, 2008)
- Kaplan-Meier survival / extinction curves (Kaplan & Meier, 1958)
- Concurrent-chains hyperbolic-decay prediction (Mazur, 2001)
- Pavlovian conditioned-approach tracking index (Flagel et al., 2007; Meyer et al., 2012)
- Operant-variability U-value (Page & Neuringer, 1985) and Lag-N pass rate (Neuringer, 2002)
- Log-survivor IRT analysis with bilinear bout fit (Shull, 1991)
- Signal detection theory (d', c, β, A', B'') and ROC / AUC (Green & Swets, 1966; Hautus, 1995)
- Resurgence (3-phase target / alternative paradigm; Shahan & Sweeney, 2011)
- Renewal classification (ABA / ABC / AAB; Bouton, 2004)
- Time-allocation matching (Baum & Rachlin, 1969)
- Within-session response-rate dynamics (McSweeney, Roll, & Cannon, 1994)
- Discrimination ratio and peak shift (Hanson, 1959; Annau & Kamin, 1961)
- Adjusting indifference point (Mazur, 1987)
- Unit-price and cross-price elasticity (Hursh, 1980; Hursh & Roma, 2013)
- Observing response and conditioned-reinforcement strength (Wyckoff, 1952; Dinsmoor, 1983)
- Rescorla-Wagner associative-strength simulation and fitting (Rescorla & Wagner, 1972)
- Tabular TD(0) reinforcement-learning trace (Sutton, 1988; Schultz, Dayan, & Montague, 1997)
- Scalar-timing summary and scalar-property test (Gibbon, 1977)
- Killeen quantitative law of effect / MPR (Killeen, 1994)
- Stay-switch decomposition and COD compliance for concurrent schedules (Stubbs & Pliskoff, 1969)
- Extinction-burst detection (Lerman & Iwata, 1995)
- Schneider two-state break-and-run segmentation (Schneider, 1969)
- Psychometric-function fitting (logit / probit; Wichmann & Hill, 2001)
- EZ-diffusion model parameter recovery (Wagenmakers, van der Maas, & Grasman, 2007)
- IRT autocorrelation and periodogram spectrum
- N-gram block / conditional entropy and Markov-order estimation (Hopson, Burt, & Hopson, 2002)
- Multi-good cross-elasticity matrix
- Hierarchical empirical-Bayes Generalized Matching Law fit (Gelman & Hill, 2007)
- Gaussian Hidden Markov Model segmentation (Rabiner, 1989)

## Install

```bash
pip install session-analyzer
```

For development:

```
mise install                                # Python 3.11
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

## 30-second example

Write a short FR 5 session with [session-recorder](https://github.com/OperantKit/session-recorder), read it back, measure post-reinforcement pauses, draw the cumulative record, and fit the generalized matching law:

```python
import math
from pathlib import Path

from experiment_core import ReinforcerStartEvent, ResponseEvent
from session_recorder import OklSink, iter_records
from session_analyzer import plot_cumulative_record
from session_analyzer.analytics import analyze_prps, compute_prps, estimate_matching_parameters

# A short FR 5 log: a pause, a run of five responses 0.5 s apart, a reinforcer.
sink = OklSink("fr5.okl")
sink.write_header(session_name="fr5", clock_type="VirtualClock", session_start=0.0)
t, n = 0.0, 0
for pause in (4.0, 6.0, 8.0):
    t += pause
    for k in range(5):
        t += 0.5 if k else 0.0
        n += 1
        sink.emit(ResponseEvent(id=n, timestamp=t))
    sink.emit(ReinforcerStartEvent(id=n // 5, timestamp=t))

records = list(iter_records("fr5.okl"))
responses = [r.timestamp for r in records if r.type == "response"]
reinforcers = [r.timestamp for r in records if r.type == "reinforcer_start"]
print(compute_prps(reinforcers, responses))  # [6.0, 8.0]  post-reinforcement pauses
print(analyze_prps(compute_prps(reinforcers, responses)).mean)  # 7.0

plot_cumulative_record(Path("fr5.okl"), Path("fr5.png"))  # cumulative record with reinforcer pips

# Generalized matching law over four concurrent VI VI conditions
# (illustrative ratios generated with a = 0.8, log b = 0.1):
r_ratios = [0.25, 0.5, 2.0, 4.0]
b_ratios = [math.exp(0.8 * math.log(r) + 0.1) for r in r_ratios]
fit = estimate_matching_parameters(b_ratios, r_ratios)
print(round(fit.sensitivity, 3), round(fit.bias, 3), round(fit.r_squared, 3))  # 0.8 0.1 1.0
```

`fr5.png` is a cumulative record (step pen, reinforcer pips) read from the OKL v1 file. In a real study the log comes from a session run with [session-runner](https://github.com/OperantKit/session-runner) or [operantkit-cli](https://github.com/OperantKit/operantkit-cli), and the matching ratios come from several such logs. The same analyses are available from the command line (see CLI below).

## Why not a 50-line hand-written analysis script?

Short scripts usually get a few things subtly wrong; this package fixes them in one tested place:

- **Published models with their equations and citations.** 43 analyses listed above (44 analysis modules), from the generalized matching law (Baum, 1974) and Herrnstein's hyperbola to demand curves, delay discounting, peak-procedure MLE, signal detection, and Rescorla-Wagner fits. Every analysis module except the latency summary cites its source in the module docstring.
- **Input checks that fail loudly.** For example, `compute_prps` raises on unsorted times and `estimate_matching_parameters` raises on mismatched lengths or fewer than two positive ratio pairs instead of returning a fit of nothing.
- **Reads the session log directly.** Parsing of OKL v1 is delegated to session-recorder (`iter_records`), so analyses see the same events that were recorded, including the phase markers that the renewal analysis uses.
- **Cumulative records drawn the conventional way.** A stepping pen with reset, reinforcer pips at a fixed angle, and an event pen, plus an IRT-coded variant for DRL / DRO.
- **Design-aware suggestions.** `session_analyzer.suggester` maps a contingency-dsl AST to the analyses worth computing for that design (for example, matching for `Conc(VI, VI)`).
- **Tested.** 560 tests (`pytest`), plus the examples in this README.

## Checked against published values

46 analytics modules: 6 match published values, 5 match an independent reference implementation, 1 matches the source's formulas, 7 have published values whose text was not retrieved and 27 have no published values; those 34 are checked by closed forms, recovery of known parameters and hand computation. [How each module is checked](docs/en/reference-values.md); `tests/test_reference_value_coverage.py` fails when a module has no test pinned to its source.

| Module | Verdict | Source | Reference implementation | What the test checks |
|---|---|---|---|---|
| `ddm` | `published` | Wagenmakers, van der Maas & Grasman (2007) | EZ-diffusion (R, appendix) | appendix output v 0.09993853, a 0.1399702, Ter 0.30003 |
| `demand` | `published` | Hursh & Silberberg (2008); Kaplan et al. (2019) | beezdemand (R) | beezdemand README output: Q0 10.158665 (`fit_space="log10"`), Pmaxd 8.453799, Omaxd 22.63159 |
| `pr_schedule` | `published` | Richardson & Roberts (1996) | — | ratio series steps 1-10 equal the published series (secondary protocol); 11-20 unchecked |
| `signal_detection` | `published` | Stanislaw & Todorov (1999) | — | Figure 1: d′ 2.00, c −0.50, ln β −1.00, A′ .89, B″ −0.55 |
| `single_case` | `published` | Parker, Vannest & Davis (2011) | SingleCaseES (R), scan (R) | NAP, PND, Tau-U equal SingleCaseES / scan published output |
| `visual_analysis` | `published` | Kratochwill et al. (2010) | scan (R) | trend equals scan's `trend()` output (scan-Book ch. 8) |
| `discounting` | `reference-implementation` | Myerson & Green (1995) | discountingtools (R) | recovers k from data made with the discountingtools generator |
| `hierarchical_gml` | `reference-implementation` | Gelman & Hill (2007) | statsmodels `MixedLM` | estimates, log-likelihood and fixed-effect SEs equal statsmodels (ML) |
| `hmm_segment` | `reference-implementation` | Rabiner (1989) | hmmlearn `GaussianHMM` | estimates, log-likelihood and Viterbi path equal hmmlearn |
| `psychometric` | `reference-implementation` | Wichmann & Hill (2001) | statsmodels binomial GLM | μ, σ and log-likelihood equal statsmodels (γ = 0, lapse = 0) |
| `survival` | `reference-implementation` | Kaplan & Meier (1958) | lifelines `KaplanMeierFitter` | Freireich 6-MP arm: S(t) and median 23 equal lifelines |
| `tracking` | `definition` | Meyer et al. (2012) | — | formulas and ±0.5 cut-offs of Table 1 (no numeric example) |
| `generalization` | `not-retrieved` | Guttman & Kalish (1956) | — | paywalled; equals an independent profiled least-squares fit |
| `killeen_law` | `not-retrieved` | Killeen (1994) | — | fitted-parameter table paywalled; β and γ identifiable |
| `matching` | `not-retrieved` | Baum (1974) | `numpy.polyfit` | Table 1 not retrieved; fit equals ordinary least squares, single-observation case by hand |
| `momentum` | `not-retrieved` | Nevin (1992) | `numpy.linalg.lstsq` | published values not retrieved; recovers a known m |
| `peak` | `not-retrieved` | Church, Meck & Gibbon (1994) | — | paywalled; recovers known low-high-low trials |
| `schneider_segment` | `not-retrieved` | Schneider (1969) | — | breakpoint table not retrieved; recovers an exact two-line record |
| `time_matching` | `not-retrieved` | Baum & Rachlin (1969) | `numpy.polyfit` | data table not retrieved; equals least squares and `matching` |
| `bout` | `properties` | Shull, Gaynor & Grimes (2001) | — | recovers a known two-component mixture (estimator differs from the original log-survivor method) |
| `breakpoint` | `properties` | Richardson & Roberts (1996) | — | last completed ratio on the published ratio series, threshold included |
| `choice` | `properties` | Herrnstein (1961) | — | proportions and thresholds by hand |
| `concurrent_chains` | `properties` | Mazur (2001) | — | V = A/(1 + kD) closed form |
| `cross_elasticity_matrix` | `properties` | Hursh & Roma (2013) | `numpy.polyfit` | recovers partial elasticities; agrees with `unit_price` |
| `extinction_burst` | `properties` | Lerman & Iwata (1995) | — | threshold and window boundary by hand |
| `indifference` | `properties` | Mazur (1987) | — | stability window by hand, titration converges |
| `irt_spectrum` | `properties` | Wei (2006) | `numpy.correlate`, `scipy.signal.periodogram` | equals NumPy / SciPy standard functions |
| `lag_variability` | `properties` | Page & Neuringer (1985) | — | Lag-N pass rate by hand |
| `latency` | `properties` | none cited | Python `statistics` | pairing and cut-off by hand; summary equals `statistics` |
| `laws_of_covariance` | `properties` | Podlesnik & Baum (2024) | — | closed form (the point prediction is this module's own) |
| `log_survivor` | `properties` | Shull (1991) | — | Poisson IRTs give a line with slope −rate |
| `matching_induction` | `properties` | Baum & Grace (2020) | — | induction term is tanh(ln r / 2); parameters recovered (the form is this module's own) |
| `ngram_entropy` | `properties` | Towse & Neil (1998) | `scipy.stats.entropy` | de Bruijn sequence gives H_n = n bits |
| `observing` | `properties` | Dinsmoor (1983) | — | rates and ratios by hand |
| `prp` | `properties` | Fry, Kelleher & Cook (1960) | — | quarter-life equals 0.25^(1/p) for power cumulative records |
| `renewal` | `properties` | Bouton (2004) | — | all context designs and ratios by hand |
| `rescorla_wagner` | `properties` | Rescorla & Wagner (1972) | — | V_n = λ(1 − (1 − αβ)^n) |
| `resurgence` | `properties` | Epstein (1985) | — | windowed means, differences and ratios by hand |
| `run_length` | `properties` | Schlinger, Derenne & Baron (2008) | — | pause and run by hand |
| `scalar_timing` | `properties` | Gibbon (1977) | — | coefficient of variation by hand |
| `stay_switch` | `properties` | Stubbs & Pliskoff (1969) | — | changeovers and COD compliance by hand |
| `stimulus_control` | `properties` | Hanson (1959) | — | suppression ratio and peak shift by hand |
| `td_learning` | `properties` | Sutton (1988) | — | converges to the discounted return (closed form) |
| `unit_price` | `properties` | Hursh (1980) | `numpy.polyfit` | unit-price equivalence, elasticity recovery, equals least squares |
| `variability` | `properties` | Page & Neuringer (1985) | `scipy.stats.entropy` | U = 1 / 0 / 2/3 closed forms |
| `within_session` | `properties` | McSweeney, Roll & Cannon (1994) | — | recovers a known inverted U |


## Analysis catalog

For an at-a-glance reference of every analysis — theme, equation,
example output, canonical citation — see
[`docs/en/analyses/index.md`](docs/en/analyses/index.md)
([日本語](docs/ja/analyses/index.md)). The catalog is split across
eight category files (rate-and-pattern / choice-and-matching /
value-and-economics / temporal / stimulus-control /
persistence-and-extinction / pavlovian / variability).

## Public API

```python
from session_analyzer import plot_cumulative_record
from session_analyzer.visualizer import plot_irt_coded_cumulative_record
from session_analyzer.analytics import (
    analyze_bouts, BoutMetrics,
    compute_log_survivor, fit_bilinear_log_survivor,
    LogSurvivorCurve, LogSurvivorBilinearFit,
    estimate_matching_parameters, MatchingParameters,
    fit_single_alternative_matching, SingleAlternativeMatchingParameters,
    estimate_time_matching_parameters, TimeMatchingParameters,
    compute_choice_allocation, classify_choice_mode,
    compute_latencies, analyze_latencies, LatencyMetrics,
    detect_breakpoint, BreakpointResult,
    fit_hyperbolic_discounting, fit_hyperboloid_discounting,
    fit_exponential_demand, DemandParameters,
    fit_unit_price_elasticity, compute_cross_price_elasticity,
    compute_unit_prices, UnitPriceElasticity, CrossPriceElasticity,
    estimate_indifference_point, IndifferencePoint,
    estimate_resistance_to_change, MomentumParameters,
    analyze_peak_trial_binned, analyze_peak_trial_mle,
    PeakTrialBinnedMetrics, PeakTrialMLEMetrics,
    analyze_within_session, WithinSessionDynamics,
    fit_gaussian_gradient, fit_exponential_gradient,
    GaussianGradientParameters, ExponentialGradientParameters,
    compute_discrimination_ratio, detect_peak_shift,
    DiscriminationRatio, PeakShiftResult,
    compute_sdt, compute_roc, SignalDetectionMetrics, ROCCurve,
    compute_prps, analyze_prps, compute_quarter_life, PRPMetrics,
    analyze_fr_runs, RunMetrics,
    kaplan_meier, SurvivalCurve,
    analyze_resurgence, ResurgenceMetrics,
    analyze_renewal, classify_renewal_design, PhaseSummary, RenewalMetrics,
    predict_concurrent_chains_choice, ConcurrentChainsPrediction,
    compute_tracking_index, TrackingIndex, TrackingType,
    compute_observing_metrics, compute_conditioned_reinforcement_strength,
    ObservingMetrics, ConditionedReinforcementStrength,
    compute_u_value, VariabilityMetrics,
    analyze_lag_variability, compute_lag_pass_rate, LagVariabilityMetrics,
    SingleCaseSeries, nonoverlap, NonOverlapResult, nap, pnd, tau, tau_u, tau_u_table,
    visual_analysis, describe_phases, contrast_phases,
    VisualAnalysis, PhaseDescriptives, PhaseContrast,
)
```

## CLI

```
session-analyzer plot path/to/session.txt --output cumulative_record.png
session-analyzer diff-plot path/to/session.txt --irt-threshold 5.0 -o drl.png
session-analyzer latency path/to/session.txt --reference-type reinforcer_start
session-analyzer breakpoint path/to/session.txt --pause-threshold 300
session-analyzer discounting delays_values.csv --amount 100 --model hyperbolic
session-analyzer demand prices_consumption.csv --k 3.0
session-analyzer single-case cases.csv --plot cases.png
```

All analytic subcommands emit JSON to stdout. CSV inputs for
``discounting`` require columns ``delay,value``; ``demand`` requires
``price,consumption``; ``single-case`` requires ``phase,value`` (rows in
session order) and optionally ``case`` (one series per case). Its options:
``--improvement increase|decrease``, ``--by run|label``, ``--a`` / ``--b``
(comma-separated phase labels for NAP / PND / Tau-U), ``--method`` (Tau-U
denominator), ``--plot`` and ``--theme``. Undefined values (for example the
slope of a one-session phase) are ``null``. A case whose phases cannot be
compared (for example a multiple-baseline tier still in baseline) gets
``"nonoverlap": null`` and an ``"error"`` message; the other cases are still
reported. NAP / PND / Tau-U compare ``--a`` with ``--b`` (by default the first
two labels), so with three or more alternating conditions run once per pair.

## Single-case designs (reversal, multiple baseline, alternating treatments)

One value per session, each labelled with its phase. `visual_analysis`
returns, for each phase, the quantities visual inspection rests on —
level (mean, median), trend (least-squares slope and intercept; session
numbers start at 0 in each phase, as in scan `trend()`) and variability
(SD about the trend line, SD, range) — and, for each pair of adjacent
phases, the change in level and trend, immediacy (first three points of
the later phase minus last three of the earlier one) and overlap
(proportion of the later phase inside the earlier phase's range), with
NAP and PND alongside. The definitions follow the What Works
Clearinghouse single-case design standards (Kratochwill et al., 2010).
`nonoverlap` adds NAP, PND, Tau and Tau-U (three published denominators).

```python
from session_analyzer.analytics import SingleCaseSeries, nonoverlap, visual_analysis
from session_analyzer.visualizer import render_single_case_plot

case = SingleCaseSeries(
    values=(54, 53, 56, 58, 52, 61, 62, 71, 66, 64, 78, 70, 74, 82, 77),
    phases=("A",) * 5 + ("B",) * 10,
    name="Case 1",
)
va = visual_analysis(case)
print([(p.label, round(p.mean, 1), round(p.slope, 2)) for p in va.phases])
print(round(va.contrasts[0].immediacy, 2), va.contrasts[0].overlap)
print(round(nonoverlap(case).nap, 3))
render_single_case_plot(case, "case1.png")
```

Phases are contiguous runs, so a reversal labelled `A, A, B, B, A, A, B, B`
yields four phases and three contrasts. For alternating treatments pass
`by="label"` (each condition pooled, no contrasts). Several series are
stacked as panels sharing the session axis (multiple baseline):

![Multiple-baseline graph with phase-change lines, level (dotted) and trend (dashed) lines](docs/assets/figures/single_case.png)

The figure is drawn from [`docs/examples/single_case_multiple_baseline.csv`](docs/examples/single_case_multiple_baseline.csv)
(synthetic data). Contrasts use the series' `improvement` direction, so in
a reversal the withdrawal contrast (B to A) is expected to show low NAP /
PND. These are aids to visual inspection; deciding whether
an effect was demonstrated is left to the analyst.

## DSL-driven analysis suggestions (for `operantkit-frontend`)

`session_analyzer.suggester` maps a resolved `contingency-dsl` AST to
the set of analyses a dashboard or a report should surface. It is the
authoritative mapping from *experiment design* to *which statistics are
worth computing* — so `session-visualizer`, `operantkit-frontend`, and
report generators all derive their panel lists from the same source.

The module is **pure stdlib**: `from session_analyzer.suggester import
suggest_from_ast` does not require `scipy` or `matplotlib` to be
importable at call time (though `session-analyzer` itself declares
them).

### Python

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
suggestions_to_json(suggest_from_ast(ast))
# [
#   {"name": "cumulative_record", "reason": "...", "tier": "light"},
#   {"name": "matching_law",      "reason": "...", "tier": "light"},
# ]
```

Both a full `Program` envelope and a bare `ScheduleExpr` subtree are
accepted — the frontend can pass a single phase of a phase-sequence for
what-if inspection without rebuilding a Program.

### HTTP contract

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

### TypeScript types

Hand-written TS mirror at
[`frontend-types/suggestions.ts`](frontend-types/suggestions.ts). Import
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

### Mapping

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

### Tier semantics

- **light** — safe to run on an in-process periodic tick (moving
  windows, log-log OLS).
- **heavy** — defer to this package's non-linear fits (demand curve
  non-linear least squares, EM bout decomposition, bootstrap CIs).

## Dependencies

`session-analyzer` depends on `session-recorder` (`pip install session-analyzer`
installs it), which owns the OKL v1 parser. It does not depend on
`contingency-py` or `session-runner`; the coupling to the session side is the
OKL v1 log schema (see "Log format" below).

## Log format

`session-analyzer` consumes OKL v1 (OperantKitLog v1) — a plain
UTF-8 text file with the `.txt` extension produced by
`session-recorder`. The full wire format (magic line, codebook header,
TAB-separated body) is defined in
`session_recorder.format`; this package reads it via
`session_recorder.iter_records` and only depends on each parsed
record exposing `.timestamp` (float, seconds since session start,
monotonic), `.type` (str), and `.args` (dict).

Of the canonical event types, `session-analyzer` consumes `"response"`
and `"reinforcer_start"`. Phase boundaries (`"phase_enter"` /
`"phase_exit"`) are read by analyses that need them (e.g., renewal).
Unknown types are ignored.

## References

- Annau, Z., & Kamin, L. J. (1961). The conditioned emotional response as a function of intensity of the US. *Journal of Comparative and Physiological Psychology*, 54(4), 428-432. https://doi.org/10.1037/h0042199
- Baum, W. M. (1974). On two types of deviation from the matching law: bias and undermatching. *Journal of the Experimental Analysis of Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Baum, W. M., & Rachlin, H. C. (1969). Choice as time allocation. *Journal of the Experimental Analysis of Behavior*, 12(6), 861-874. https://doi.org/10.1901/jeab.1969.12-861
- Bouton, M. E. (2004). Context and behavioral processes in extinction. *Learning & Memory*, 11(5), 485-494. https://doi.org/10.1101/lm.78804
- Dinsmoor, J. A. (1983). Observing and conditioned reinforcement. *Behavioral and Brain Sciences*, 6(4), 693-704. https://doi.org/10.1017/S0140525X00017969
- Epstein, R. (1985). Extinction-induced resurgence: Preliminary investigations and possible applications. *The Psychological Record*, 35(2), 143-153. https://doi.org/10.1007/BF03394920
- Green, D. M., & Swets, J. A. (1966). *Signal detection theory and psychophysics*. Wiley.
- Hanson, H. M. (1959). Effects of discrimination training on stimulus generalization. *Journal of Experimental Psychology*, 58(5), 321-334. https://doi.org/10.1037/h0042606
- Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d'. *Behavior Research Methods, Instruments, & Computers*, 27(1), 46-51. https://doi.org/10.3758/BF03203619
- Hursh, S. R. (1980). Economic concepts for the analysis of behavior. *Journal of the Experimental Analysis of Behavior*, 34(2), 219-238. https://doi.org/10.1901/jeab.1980.34-219
- Hursh, S. R., & Roma, P. G. (2013). Behavioral economics and empirical public policy. *Journal of the Experimental Analysis of Behavior*, 99(1), 98-124. https://doi.org/10.1002/jeab.7
- McSweeney, F. K., Roll, J. M., & Cannon, C. B. (1994). The generality of within-session patterns of responding. *Animal Learning & Behavior*, 22(3), 252-266. https://doi.org/10.3758/BF03209833
- Neuringer, A. (2002). Operant variability: Evidence, functions, and theory. *Psychonomic Bulletin & Review*, 9(4), 672-705. https://doi.org/10.3758/BF03196324
- Shahan, T. A., & Sweeney, M. M. (2011). A model of resurgence based on behavioral momentum theory. *Journal of the Experimental Analysis of Behavior*, 95(1), 91-108. https://doi.org/10.1901/jeab.2011.95-91
- Shull, R. L. (1991). Mathematical description of operant behavior: An introduction. In I. H. Iversen & K. A. Lattal (Eds.), *Experimental analysis of behavior* (Vol. 2, pp. 243-282). Elsevier.
- Wyckoff, L. B. (1952). The role of observing responses in discrimination learning. Part I. *Psychological Review*, 59(6), 431-442. https://doi.org/10.1037/h0053932
- Church, R. M., Meck, W. H., & Gibbon, J. (1994). Application of scalar timing theory to individual trials. *Journal of Experimental Psychology: Animal Behavior Processes*, 20(2), 135-155. https://doi.org/10.1037/0097-7403.20.2.135
- Felton, M., & Lyon, D. O. (1966). The post-reinforcement pause. *Journal of the Experimental Analysis of Behavior*, 9(2), 131-134. https://doi.org/10.1901/jeab.1966.9-131
- Flagel, S. B., Watson, S. J., Robinson, T. E., & Akil, H. (2007). Individual differences in the propensity to approach signals vs goals promote different adaptations in the dopamine system of rats. *Psychopharmacology*, 191(3), 599-607. https://doi.org/10.1007/s00213-006-0535-8
- Guttman, N., & Kalish, H. I. (1956). Discriminability and stimulus generalization. *Journal of Experimental Psychology*, 51(1), 79-88. https://doi.org/10.1037/h0046219
- Herrnstein, R. J. (1970). On the law of effect. *Journal of the Experimental Analysis of Behavior*, 13(2), 243-266. https://doi.org/10.1901/jeab.1970.13-243
- Herrnstein, R. J., & Morse, W. H. (1957). A conjunctive schedule of reinforcement. *Journal of the Experimental Analysis of Behavior*, 1(1), 15-24. https://doi.org/10.1901/jeab.1958.1-15
- Hursh, S. R., & Silberberg, A. (2008). Economic demand and essential value. *Psychological Review*, 115(1), 186-198. https://doi.org/10.1037/0033-295X.115.1.186
- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481. https://doi.org/10.1080/01621459.1958.10501452
- Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H. Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5* (pp. 55-73). Lawrence Erlbaum Associates.
- Mazur, J. E. (2001). Hyperbolic value addition and general models of animal choice. *Psychological Review*, 108(1), 96-112. https://doi.org/10.1037/0033-295X.108.1.96
- Meyer, P. J., Lovic, V., Saunders, B. T., Yager, L. M., Flagel, S. B., Morrow, J. D., & Robinson, T. E. (2012). Quantifying individual variation in the propensity to attribute incentive salience to reward cues. *PLoS ONE*, 7(6), e38987. https://doi.org/10.1371/journal.pone.0038987
- Nevin, J. A. (1992). An integrative model for the study of behavioral momentum. *Journal of the Experimental Analysis of Behavior*, 57(3), 301-316. https://doi.org/10.1901/jeab.1992.57-301
- Page, S., & Neuringer, A. (1985). Variability is an operant. *Journal of Experimental Psychology: Animal Behavior Processes*, 11(3), 429-452. https://doi.org/10.1037/0097-7403.11.3.429
- Rachlin, H. (2006). Notes on discounting. *Journal of the Experimental Analysis of Behavior*, 85(3), 425-435. https://doi.org/10.1901/jeab.2006.85-05
- Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio schedules in drug self-administration studies in rats: a method to evaluate reinforcing efficacy. *Journal of Neuroscience Methods*, 66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
- Roberts, S. (1981). Isolation of an internal clock. *Journal of Experimental Psychology: Animal Behavior Processes*, 7(3), 242-268. https://doi.org/10.1037/0097-7403.7.3.242
- Schlinger, H. D., Derenne, A., & Baron, A. (2008). What 50 years of research tell us about pausing under ratio schedules of reinforcement. *The Behavior Analyst*, 31(1), 39-60. https://doi.org/10.1007/BF03392160
- Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. *Science*, 237(4820), 1317-1323. https://doi.org/10.1126/science.3629243
- Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate viewed as engagement bouts: effects of relative reinforcement and schedule type. *Journal of the Experimental Analysis of Behavior*, 75(3), 247-274. https://doi.org/10.1901/jeab.2001.75-247
- Kratochwill, T. R., Hitchcock, J. H., Horner, R. H., Levin, J. R., Odom, S. L., Rindskopf, D. M., & Shadish, W. R. (2010). *Single-case designs technical documentation*. What Works Clearinghouse. https://ies.ed.gov/ncee/wwc/Docs/ReferenceResources/wwc_scd.pdf
- Parker, R. I., & Vannest, K. J. (2009). An improved effect size for single-case research: Nonoverlap of all pairs. *Behavior Therapy*, 40(4), 357-367. https://doi.org/10.1016/j.beth.2008.10.006
- Parker, R. I., Vannest, K. J., Davis, J. L., & Sauber, S. B. (2011). Combining nonoverlap and trend for single-case research: Tau-U. *Behavior Therapy*, 42(2), 284-299. https://doi.org/10.1016/j.beth.2010.08.006
