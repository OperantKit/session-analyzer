"""Figure theme system for session-analyzer visualizations.

Provides a rendering-engine-independent :class:`ThemeSpec` plus built-in
presets covering screen, monochrome paper (JEAB / JABA), color paper
(Nature / Science), and preprint-draft use cases.
"""

from __future__ import annotations

from .presets import THEMES, get_theme, list_themes
from .spec import GridMode, ThemeSpec

__all__ = [
    "THEMES",
    "GridMode",
    "ThemeSpec",
    "get_theme",
    "list_themes",
]
