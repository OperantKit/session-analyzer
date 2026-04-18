"""Cumulative-record plotting from session-recorder JSONL logs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

# Use a non-interactive backend by default. Callers that want an interactive
# backend can override before importing this module.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def plot_cumulative_record(log_path: Path, output_path: Path) -> None:
    """Plot a cumulative response record from a JSONL session log.

    Reads ``log_path`` line-by-line, extracts ``"response"`` and
    ``"reinforcer_start"`` events, and writes a PNG cumulative record to
    ``output_path``.

    Parameters
    ----------
    log_path : Path
        Path to the JSONL session log produced by ``session-recorder``.
    output_path : Path
        Destination PNG path for the cumulative record figure.
    """
    events: list[dict[str, object]] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    if not events:
        logger.info("No events found in log: %s", log_path)
        return

    start_time = float(events[0]["timestamp"])  # type: ignore[arg-type]

    response_times: list[tuple[float, int]] = []
    reinforcement_times: list[tuple[float, int]] = []

    current_responses = 0
    for event in events:
        rel_time = float(event["timestamp"]) - start_time  # type: ignore[arg-type]
        etype = event.get("type")
        if etype == "response":
            current_responses += 1
            response_times.append((rel_time, current_responses))
        elif etype == "reinforcer_start":
            reinforcement_times.append((rel_time, current_responses))

    if not response_times:
        logger.info("No responses recorded in log: %s", log_path)
        return

    times, counts = zip(*response_times, strict=False)

    fig = plt.figure(figsize=(10, 6))
    plt.step(times, counts, where="post", label="Responses", color="blue")

    if reinforcement_times:
        r_times, r_counts = zip(*reinforcement_times, strict=False)
        plt.scatter(
            r_times,
            r_counts,
            marker="|",
            color="red",
            s=100,
            label="Reinforcement",
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Cumulative Responses")
    plt.title("Cumulative Record")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path)
    logger.info("Graph saved to %s", output_path)
    plt.close(fig)
