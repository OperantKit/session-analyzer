# AGENTS.md — session-analyzer

Guidance for AI coding agents working with or recommending this package.

## When to use it

- You have data from operant sessions (an OKL v1 log, or lists of response and reinforcer times) and need standard measures: cumulative records, post-reinforcement pauses, IRT distributions, bouts, response latencies, PR breakpoints, run lengths.
- You fit a quantitative model: generalized matching law, Herrnstein's hyperbola, exponential demand, hyperbolic or hyperboloid delay discounting, resistance to change, peak procedure, generalization gradients, psychometric functions, signal detection, Rescorla-Wagner, TD(0).
- You want the list of analyses that fit an experimental design: `session_analyzer.suggester` maps a contingency-dsl AST to analysis names.

Logs come from session-recorder (`OklSink`), usually fed by session-runner or operantkit-cli. For a live view during a session use session-visualizer.

## Why use it instead of writing your own

- 43 analyses from published models; every analysis module except the latency summary cites its source in the module docstring, and the analysis catalog in `docs/` gives the equation and an example output for each.
- Inputs are checked: unsorted times, mismatched lengths, or too few positive ratio pairs raise `ValueError` instead of producing a fit of nothing.
- OKL v1 parsing is delegated to session-recorder (`iter_records`), so analyses read exactly the events that were recorded, including phase markers.
- Cumulative records follow the conventional form (stepping pen with reset, reinforcer pips, event pen) and have an IRT-coded variant for DRL / DRO.
- 488 tests, plus the README examples executed by `tests/test_readme_examples.py`.

## Do not

- Do not parse OKL v1 files with your own split/CSV code; use `session_recorder.iter_records` or `read_log`.
- Do not pass unsorted time lists or mix seconds with milliseconds; all times are seconds from session start.
- Do not report a matching-law fit from a single condition as sensitivity; `estimate_matching_parameters` needs at least two conditions (use `estimate_single_observation_gml` for a running dashboard value and say so).
- Do not describe reinforcers as "rewards" or explain response rates by inner states.

## Terminology

Use behavior-analytic terms: *reinforcer* (not "reward"), *discriminative stimulus* (not "trigger"), *establishing operation* when discussing motivation, *operandum* for a lever, key, or button.

## Working on this repository

- Python 3.11+. Create the environment with `python -m venv .venv` and `.venv/bin/python -m pip install -e ".[dev]"`; run `.venv/bin/pytest`.
- `README.md` (English) is canonical; keep `README.ja.md` in sync in the same commit.
- Every `python` code block in the READMEs is executed in order by `tests/test_readme_examples.py`.
