# session-analyzer

Cumulative-record visualization, statistical analysis, and behavioral model fitting for OperantKit session logs.

Reads JSONL session logs (typically produced by `session-recorder`) and produces:

- Cumulative response records with reinforcement tick marks
- Bout-decomposed response rate metrics (Shull, Gaynor, & Grimes, 2001)
- Generalized Matching Law (GML) parameter fits (Baum, 1974)

## Install

```
mise install                                # Python 3.11
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

## Public API

```python
from session_analyzer import plot_cumulative_record
from session_analyzer.analytics import (
    analyze_bouts, BoutMetrics,
    estimate_matching_parameters, MatchingParameters,
)
```

## CLI

```
session-analyzer plot path/to/session.jsonl --output cumulative_record.png
```

## Local dependency notes

This package is distributed as a standalone analyzer. It does not declare
`session-recorder` or `contingency-py` as install-time dependencies — both are
sister packages in the same monorepo and are intended to be installed
side-by-side in a development environment when end-to-end pipelines are
exercised. The runtime contract with `session-recorder` is the JSONL log
schema (see "Log format" below); coupling is kept at the data layer rather
than the import layer.

## Log format

Each line of the JSONL session log is a JSON object with at least:

- `timestamp` (float) — seconds since session start (monotonic)
- `type` (str) — event kind. `session-analyzer` consumes `"response"` and
  `"reinforcer_start"`. Other event types are ignored.

## References

- Baum, W. M. (1974). On two types of deviation from the matching law: bias and undermatching. *Journal of the Experimental Analysis of Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate viewed as engagement bouts: effects of relative reinforcement and schedule type. *Journal of the Experimental Analysis of Behavior*, 75(3), 247-274. https://doi.org/10.1901/jeab.2001.75-247

## License

MIT. See [LICENSE](LICENSE).
