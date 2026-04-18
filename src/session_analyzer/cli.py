"""Session analyzer CLI: ``plot`` command and ``--help`` entry point.

Implemented with ``argparse`` to keep install-time dependencies minimal
(no Typer / Click).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session-analyzer",
        description="Visualize and analyze session-recorder JSONL logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_p = subparsers.add_parser(
        "plot",
        help="Plot a cumulative record from a session log.",
    )
    plot_p.add_argument(
        "log_path",
        type=Path,
        help="Path to a JSONL session log produced by session-recorder.",
    )
    plot_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("cumulative_record.png"),
        help="Destination PNG path (default: cumulative_record.png).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "plot":
        # Lazy import so loading --help does not pull matplotlib / numpy.
        from session_analyzer.visualizer.cumulative_record import plot_cumulative_record

        plot_cumulative_record(args.log_path, args.output)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2  # unreachable; argparse.error raises SystemExit


if __name__ == "__main__":
    raise SystemExit(main())
