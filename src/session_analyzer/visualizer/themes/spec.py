"""Declarative, rendering-engine-independent theme specification.

A ``ThemeSpec`` captures everything needed to consistently style a figure
across rendering backends (matplotlib for static export, Plotly.js / recharts
for browser live dashboards). Adapters translate a ``ThemeSpec`` into the
native options of each engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GridMode = Literal["none", "minor", "major"]
"""Grid density. ``none`` suppresses grid entirely (JEAB house style)."""


@dataclass(frozen=True)
class ThemeSpec:
    """Declarative figure theme.

    All dimensions are in inches / points so that a single spec works for
    both on-screen (96 dpi) and publication (300–600 dpi) output.
    """

    name: str
    description: str

    # ---- Typography ----
    font_family: str
    font_size_pt: float

    # ---- Colors (hex strings ``#rrggbb``) ----
    background: str
    foreground: str
    palette: tuple[str, ...]
    """Discrete categorical palette, cycled over series."""

    # ---- Line / marker style (used for B&W where color is unavailable) ----
    line_style_cycle: tuple[str, ...]
    """matplotlib linestyle strings: ``-``, ``--``, ``-.``, ``:``."""
    marker_cycle: tuple[str, ...]
    line_width_pt: float

    # ---- Figure geometry ----
    figure_width_in: float
    figure_height_in: float
    dpi: int

    # ---- Axes ----
    grid: GridMode
    spine_width_pt: float
    legend_frame: bool

    # ---- Meta ----
    color_blind_safe: bool = False
    intended_use: str = ""
    """Short human-readable note on where this theme is appropriate."""

    tags: tuple[str, ...] = field(default_factory=tuple)
