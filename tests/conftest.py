"""Shared pytest configuration for session-analyzer.

Forces the matplotlib ``Agg`` backend so visualizer tests run in
headless environments (CI, ssh sessions without DISPLAY).

Also exposes :func:`write_osl_log`, a helper for tests that need to
construct OKL v1 fixtures with arbitrary event dicts (including types
not in the canonical codebook). Pre-OKL tests used to write raw JSONL
directly; the helper lets them stay readable while satisfying the
self-describing-codebook requirement. The codebook emitted by this
helper is **derived from the events themselves** (every observed field
is marked optional), so tests that omit canonical fields like
``response.id`` still parse cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

from session_recorder.format import (
    HEADER_TERMINATOR,
    MAGIC,
    Field,
    encode_str,
)

matplotlib.use("Agg")


def write_osl_log(
    path: Path,
    events: list[dict[str, Any]],
    *,
    session_name: str = "test",
    clock_type: str = "TestClock",
    session_start: float = 0.0,
) -> None:
    """Write a minimal OKL v1 file from a list of dict events.

    Codebook is derived from the events: each observed field per type
    becomes an optional column. Field types are inferred from the first
    non-None value seen on that field. Empty bodies still produce a
    valid header (with no event types declared).
    """
    field_order: dict[str, list[str]] = {}
    field_types: dict[str, dict[str, str]] = {}
    for ev in events:
        t = ev["type"]
        for k, v in ev.items():
            if k in ("type", "timestamp"):
                continue
            field_order.setdefault(t, [])
            field_types.setdefault(t, {})
            if k not in field_order[t]:
                field_order[t].append(k)
            if k not in field_types[t] and v is not None:
                field_types[t][k] = _infer_field_type(v)
        # Make sure every type appears in the codebook even if it has no fields
        field_order.setdefault(t, [])
        field_types.setdefault(t, {})

    codebook: dict[str, tuple[Field, ...]] = {}
    for t, fields in field_order.items():
        codebook[t] = tuple(
            Field(name=name, ty=field_types[t].get(name, "str"), optional=True)
            for name in fields
        )

    lines = [MAGIC]
    lines.append(f'# session_name = "{session_name}"')
    lines.append(f'# clock_type = "{clock_type}"')
    lines.append(f"# session_start = {session_start}")
    lines.append("# events:")
    for type_name, fs in codebook.items():
        rendered = " ".join(f"{f.name}:{f.ty}{'?' if f.optional else ''}" for f in fs)
        lines.append(f"#   {type_name:<18}: {rendered}")
    lines.append(HEADER_TERMINATOR)
    for ev in events:
        ts = ev["timestamp"]
        t = ev["type"]
        cols: list[str] = [repr(float(ts)), t]
        for f in codebook[t]:
            value = ev.get(f.name)
            if value is None:
                cols.append("-")
            elif f.ty == "int":
                cols.append(str(int(value)))
            elif f.ty == "float":
                cols.append(repr(float(value)))
            elif f.ty == "bool":
                cols.append("true" if value else "false")
            else:
                cols.append(encode_str(str(value)))
        lines.append("\t".join(cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_field_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"
