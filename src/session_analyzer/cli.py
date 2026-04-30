"""Session-analyzer CLI.

Subcommands:

- ``plot``        — cumulative record from an OKL v1 session log
- ``diff-plot``   — IRT-coded cumulative record for DRL/DRO
- ``latency``     — response-latency summary from an OKL v1 log
- ``breakpoint``  — PR breakpoint detection from an OKL v1 log
- ``discounting`` — fit Mazur/Rachlin discounting from a CSV
- ``demand``      — fit exponential demand curve from a CSV

Implemented with ``argparse`` to keep install-time dependencies minimal
(no Typer / Click).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from session_recorder import ParsedRecord, iter_records


def _to_dict(record: ParsedRecord) -> dict[str, Any]:
    """Flatten a :class:`ParsedRecord` into the dict shape expected by
    legacy callers (``{"type", "timestamp", **fields}``)."""
    return {"type": record.type, "timestamp": record.timestamp, **record.args}


def _read_log(log_path: Path) -> list[dict[str, Any]]:
    """Read every body record as a flat dict.

    Delegates to :func:`session_recorder.iter_records` so the format
    contract is owned by ``session-recorder``. Subcommands that
    branch on ``record["type"] == "<custom>"`` (e.g. PR completion
    types defined by the experiment, not the core event union) keep
    working unchanged because flat dicts are returned.
    """
    return [_to_dict(r) for r in iter_records(log_path)]


def _read_two_column_csv(
    csv_path: Path, col_a: str, col_b: str
) -> tuple[list[float], list[float]]:
    a: list[float] = []
    b: list[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or col_a not in reader.fieldnames or col_b not in reader.fieldnames:
            raise SystemExit(
                f"CSV must contain columns '{col_a}' and '{col_b}' "
                f"(found: {reader.fieldnames})"
            )
        for row in reader:
            a.append(float(row[col_a]))
            b.append(float(row[col_b]))
    return a, b


def _emit_json(obj: Any) -> str:
    if is_dataclass(obj):
        return json.dumps(asdict(obj), indent=2, default=str)
    return json.dumps(obj, indent=2, default=str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session-analyzer",
        description="Visualize and analyze session-recorder OKL v1 logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_p = subparsers.add_parser(
        "plot",
        help="Plot a cumulative record from a session log.",
    )
    plot_p.add_argument("log_path", type=Path)
    plot_p.add_argument(
        "--output", "-o", type=Path, default=Path("cumulative_record.png")
    )

    diff_p = subparsers.add_parser(
        "diff-plot",
        help="Plot an IRT-coded cumulative record (DRL / DRO).",
    )
    diff_p.add_argument("log_path", type=Path)
    diff_p.add_argument("--output", "-o", type=Path, default=Path("irt_coded.png"))
    diff_p.add_argument(
        "--irt-threshold",
        type=float,
        required=True,
        help="IRT threshold in seconds; responses below are flagged as short.",
    )

    lat_p = subparsers.add_parser(
        "latency",
        help="Compute response-latency summary from a session log.",
    )
    lat_p.add_argument("log_path", type=Path)
    lat_p.add_argument(
        "--reference-type",
        default="reinforcer_start",
        help="Event type used as latency reference (default: reinforcer_start).",
    )
    lat_p.add_argument(
        "--response-type",
        default="response",
        help="Event type treated as a response (default: response).",
    )
    lat_p.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="Optional upper cutoff in seconds.",
    )

    bp_p = subparsers.add_parser(
        "breakpoint",
        help="Detect progressive-ratio breakpoint from a session log.",
    )
    bp_p.add_argument("log_path", type=Path)
    bp_p.add_argument(
        "--pause-threshold",
        type=float,
        default=300.0,
        help="Pause threshold in seconds (default 300; Richardson & Roberts, 1996).",
    )
    bp_p.add_argument(
        "--completion-type",
        default="reinforcer_start",
        help="Event type that marks a completed ratio (default: reinforcer_start).",
    )
    bp_p.add_argument(
        "--ratio-field",
        default="ratio",
        help="JSON field on the completion event carrying the ratio value (default: ratio).",
    )

    disc_p = subparsers.add_parser(
        "discounting",
        help="Fit a delay-discounting model from a CSV of (delay, value).",
    )
    disc_p.add_argument("csv_path", type=Path)
    disc_p.add_argument("--amount", type=float, required=True)
    disc_p.add_argument(
        "--model",
        choices=["hyperbolic", "hyperboloid"],
        default="hyperbolic",
    )

    dem_p = subparsers.add_parser(
        "demand",
        help="Fit an exponential demand curve from a CSV of (price, consumption).",
    )
    dem_p.add_argument("csv_path", type=Path)
    dem_p.add_argument(
        "--k",
        type=float,
        default=None,
        help="Fix the k span parameter; omit to estimate jointly.",
    )

    return parser


def _cmd_plot(args: argparse.Namespace) -> int:
    from session_analyzer.visualizer.cumulative_record import plot_cumulative_record

    plot_cumulative_record(args.log_path, args.output)
    return 0


def _cmd_diff_plot(args: argparse.Namespace) -> int:
    from session_analyzer.visualizer.cumulative_diff import (
        plot_irt_coded_cumulative_record,
    )

    plot_irt_coded_cumulative_record(
        args.log_path,
        args.output,
        irt_threshold_sec=args.irt_threshold,
    )
    return 0


def _cmd_latency(args: argparse.Namespace) -> int:
    from session_analyzer.analytics.latency import (
        analyze_latencies,
        compute_latencies,
    )

    events = _read_log(args.log_path)
    ref_times = sorted(
        float(e["timestamp"]) for e in events if e.get("type") == args.reference_type
    )
    resp_times = sorted(
        float(e["timestamp"]) for e in events if e.get("type") == args.response_type
    )
    latencies = compute_latencies(
        ref_times, resp_times, max_latency_sec=args.max_latency
    )
    metrics = analyze_latencies(latencies)
    if metrics is None:
        print(json.dumps({"n": 0, "message": "no valid latencies"}, indent=2))
        return 0
    print(_emit_json(metrics))
    return 0


def _cmd_breakpoint(args: argparse.Namespace) -> int:
    from session_analyzer.analytics.breakpoint import detect_breakpoint

    events = _read_log(args.log_path)
    if not events:
        print(json.dumps({"message": "empty log"}, indent=2))
        return 0

    start = float(events[0]["timestamp"])
    end = float(events[-1]["timestamp"])

    completions: list[tuple[int, float]] = []
    for i, e in enumerate(events):
        if e.get("type") != args.completion_type:
            continue
        ratio_val = e.get(args.ratio_field, i + 1)
        try:
            ratio_int = int(ratio_val)
        except (TypeError, ValueError):
            ratio_int = i + 1
        completions.append((ratio_int, float(e["timestamp"]) - start))

    result = detect_breakpoint(
        completions,
        session_end_time=end - start,
        pause_threshold_sec=args.pause_threshold,
    )
    print(_emit_json(result))
    return 0


def _cmd_discounting(args: argparse.Namespace) -> int:
    from session_analyzer.analytics.discounting import (
        fit_hyperbolic_discounting,
        fit_hyperboloid_discounting,
    )

    delays, values = _read_two_column_csv(args.csv_path, "delay", "value")
    if args.model == "hyperbolic":
        result = fit_hyperbolic_discounting(delays, values, amount=args.amount)
    else:
        result = fit_hyperboloid_discounting(delays, values, amount=args.amount)
    print(_emit_json(result))
    return 0


def _cmd_demand(args: argparse.Namespace) -> int:
    from session_analyzer.analytics.demand import fit_exponential_demand

    prices, consumption = _read_two_column_csv(args.csv_path, "price", "consumption")
    result = fit_exponential_demand(prices, consumption, k=args.k)
    print(_emit_json(result))
    return 0


_DISPATCH = {
    "plot": _cmd_plot,
    "diff-plot": _cmd_diff_plot,
    "latency": _cmd_latency,
    "breakpoint": _cmd_breakpoint,
    "discounting": _cmd_discounting,
    "demand": _cmd_demand,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
