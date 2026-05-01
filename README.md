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
- Progressive-ratio breakpoint detection (Richardson & Roberts, 1996)
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

```
mise install                                # Python 3.11
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

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
```

All analytic subcommands emit JSON to stdout. CSV inputs for
``discounting`` require columns ``delay,value``; ``demand`` requires
``price,consumption``.

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

## Local dependency notes

This package is distributed as a standalone analyzer. It does not declare
`session-recorder` or `contingency-py` as install-time dependencies — both are
sister packages in the same monorepo and are intended to be installed
side-by-side in a development environment when end-to-end pipelines are
exercised. The runtime contract with `session-recorder` is the OKL v1
log schema (see "Log format" below); coupling is kept at the data layer
rather than the import layer.

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
