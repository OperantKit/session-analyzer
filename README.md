# session-analyzer

:jp: [日本語版 README](README.ja.md)

session-analyzer draws cumulative records and computes behavior-analytic measures from session data: post-reinforcement pauses, the generalized matching law, demand curves, delay discounting, single-case effect sizes, and more. It is for researchers and students who analyze operant sessions in Python. In [OperantKit](https://github.com/OperantKit) it reads the session logs that [session-recorder](https://github.com/OperantKit/session-recorder) writes, and its functions also take plain lists of times or values.

- **Published models in one tested place.** 46 analysis modules, each naming the paper its model comes from.
- **Inputs are checked.** Unsorted times, mismatched lengths, or too few data points raise an error instead of returning a meaningless fit.
- **Cumulative records drawn the conventional way.** A stepping pen with reset and reinforcer pips, plus an IRT-coded variant for DRL / DRO.

Notes for AI coding agents are in [AGENTS.md](AGENTS.md).

<p align="center"><img src="docs/assets/figures/cumulative_record.png" width="360" alt="Cumulative record: a stepping line with short diagonal pips at each reinforcer"></p>

## Installation

Python 3.11 or newer. session-analyzer is not on PyPI yet, so `pip install session-analyzer` does not work today. Install it and the OperantKit packages it needs from GitHub without dependency resolution, then add the three third-party libraries it uses:

```bash
G=git+https://github.com/OperantKit
pip install --no-deps $G/contingency-dsl-py $G/contingency-py $G/experiment-core \
  $G/session-recorder $G/session-analyzer
pip install numpy scipy matplotlib
```

`pip check` reports version mismatches between these development builds; they do not affect the examples below.

## Example

Write a short FR 5 session with session-recorder, measure the post-reinforcement pauses (PRP), draw the cumulative record, and fit the generalized matching law:

```python
import math
from pathlib import Path

from experiment_core import ReinforcerStartEvent, ResponseEvent
from session_recorder import OklSink, iter_records
from session_analyzer import plot_cumulative_record
from session_analyzer.analytics import analyze_prps, compute_prps, estimate_matching_parameters

# A short FR 5 log: a pause, a run of five responses 0.5 s apart, a reinforcer.
sink = OklSink("fr5.txt")
sink.write_header(session_name="fr5", clock_type="VirtualClock", session_start=0.0)
t, n = 0.0, 0
for pause in (4.0, 6.0, 8.0):
    t += pause
    for k in range(5):
        t += 0.5 if k else 0.0
        n += 1
        sink.emit(ResponseEvent(id=n, timestamp=t))
    sink.emit(ReinforcerStartEvent(id=n // 5, timestamp=t))

records = list(iter_records("fr5.txt"))
responses = [r.timestamp for r in records if r.type == "response"]
reinforcers = [r.timestamp for r in records if r.type == "reinforcer_start"]
print(compute_prps(reinforcers, responses))  # [6.0, 8.0]
print(analyze_prps(compute_prps(reinforcers, responses)).mean)  # 7.0

plot_cumulative_record(Path("fr5.txt"), Path("fr5.png"))

# Generalized matching law over four concurrent VI VI conditions
# (illustrative ratios generated with a = 0.8, log b = 0.1):
r_ratios = [0.25, 0.5, 2.0, 4.0]
b_ratios = [math.exp(0.8 * math.log(r) + 0.1) for r in r_ratios]
fit = estimate_matching_parameters(b_ratios, r_ratios)
print(round(fit.sensitivity, 3), round(fit.bias, 3), round(fit.r_squared, 3))  # 0.8 0.1 1.0
```

The pauses after the first and second reinforcers are 6 s and 8 s, and the generalized matching law fit (Baum, 1974) recovers the sensitivity (a = 0.8) and bias (log b = 0.1) the ratios were made with. In a real study the log comes from a session run with [session-runner](https://github.com/OperantKit/session-runner) or [operantkit-cli](https://github.com/OperantKit/operantkit-cli), and the matching ratios come from several such logs.

## What is inside

- **Analyses** in `session_analyzer.analytics`: response rate and local pattern (PRP, bouts, IRT log-survivor, FR runs), choice and matching (generalized matching law, time allocation, changeover-delay compliance, concurrent chains), value and economics (progressive-ratio breakpoint, demand, delay discounting, unit price), timing (peak procedure, scalar timing), stimulus control (generalization gradients, signal detection), persistence and extinction (behavioral momentum, resurgence, renewal, extinction burst), Pavlovian measures (sign- and goal-tracking, Rescorla-Wagner), variability, and single-case designs. [Every module and its source](docs/en/modules.md); [equations and example output](docs/en/analyses/index.md) for the main analyses.
- **Plots** in `session_analyzer.visualizer`: `plot_cumulative_record`, `plot_irt_coded_cumulative_record`, and `render_single_case_plot`.
- **Analysis suggestions** in `session_analyzer.suggester`: given a [contingency-dsl](https://github.com/OperantKit/contingency-dsl) expression, the analyses worth computing for that design (for example the matching law for `Conc(VI, VI)`). It is the authoritative mapping from experiment design to analyses for [session-visualizer](https://github.com/OperantKit/session-visualizer) and [operantkit-frontend](https://github.com/OperantKit/operantkit-frontend); see [DSL-driven analysis suggestions](docs/en/suggestions.md) for the mapping, the HTTP contract, and the TypeScript types.

### Reading session logs

Logs are OKL v1 files written by session-recorder (the format is specified in [OperantKitLog](https://github.com/OperantKit/OperantKitLog)). session-analyzer reads them through `session_recorder.iter_records` and uses each record's `.timestamp` (seconds since session start), `.type`, and `.args`. Of the canonical event types it uses `response` and `reinforcer_start`; analyses that need phases (for example renewal) also read `phase_enter` / `phase_exit`. Other types are ignored.

### Command line

```
session-analyzer plot session.txt --output cumulative_record.png
session-analyzer diff-plot session.txt --irt-threshold 5.0 -o drl.png
session-analyzer latency session.txt --reference-type reinforcer_start
session-analyzer breakpoint session.txt --pause-threshold 300
session-analyzer discounting delays_values.csv --amount 100 --model hyperbolic
session-analyzer demand prices_consumption.csv --k 3.0
session-analyzer single-case cases.csv --plot cases.png
```

The analysis subcommands print JSON. CSV inputs need the columns `delay,value` (`discounting`), `price,consumption` (`demand`), or `phase,value` with an optional `case` column (`single-case`, rows in session order). `session-analyzer <subcommand> --help` lists the options.

### Single-case designs

For reversal, multiple-baseline, and alternating-treatments designs, give one value per session labelled with its phase. `visual_analysis` returns what visual inspection rests on: for each phase the level (mean, median), trend (least-squares slope; session numbers start at 0 in each phase), and variability; for each pair of adjacent phases the change in level and trend, immediacy (first three points of the later phase minus the last three of the earlier one), and overlap. The definitions follow the What Works Clearinghouse single-case design standards (Kratochwill et al., 2010). `nonoverlap` adds NAP, PND, Tau, and Tau-U.

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
# [('A', 54.6, 0.1), ('B', 70.5, 1.97)]
print(round(va.contrasts[0].immediacy, 2), va.contrasts[0].overlap)  # 9.33 0.0
print(round(nonoverlap(case).nap, 3))  # 1.0
render_single_case_plot(case, "case1.png")
```

Phases are contiguous runs, so a reversal labelled `A, A, B, B, A, A, B, B` gives four phases and three contrasts. For alternating treatments pass `by="label"`. Several series are drawn as stacked panels sharing the session axis (multiple baseline), as in this figure drawn from synthetic data in [`docs/examples/single_case_multiple_baseline.csv`](docs/examples/single_case_multiple_baseline.csv):

![Multiple-baseline graph with phase-change lines, level (dotted) and trend (dashed) lines](docs/assets/figures/single_case.png)

These are aids to visual inspection; deciding whether an effect was demonstrated is left to the analyst.

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


## Limitations

- Only the single-case non-overlap measures (`single_case`) are tested against published reference values. The other analyses cite their sources but are not yet checked against the numbers reported in the original papers; check the one you use against the original before you report it.
- Logs must be OKL v1. Data from other systems have to be converted first (session-recorder converts MED-PC data files for the programs it has a protocol for).

## Development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

Every `python` block in this README and `README.ja.md` is run by `tests/test_readme_examples.py`.

## References

- Baum, W. M. (1974). On two types of deviation from the matching law: bias and undermatching. *Journal of the Experimental Analysis of Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Kratochwill, T. R., Hitchcock, J. H., Horner, R. H., Levin, J. R., Odom, S. L., Rindskopf, D. M., & Shadish, W. R. (2010). *Single-case designs technical documentation*. What Works Clearinghouse. https://ies.ed.gov/ncee/wwc/Docs/ReferenceResources/wwc_scd.pdf

The references for each analysis are in [docs/en/modules.md](docs/en/modules.md).

## License

MIT. See [LICENSE](LICENSE).
