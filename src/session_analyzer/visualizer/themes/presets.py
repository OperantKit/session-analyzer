"""Built-in theme presets."""

from __future__ import annotations

import types
from collections.abc import Mapping

from .spec import ThemeSpec

OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)
"""Color-vision-deficiency-safe 8-color qualitative palette (Okabe & Ito, 2008)."""


READABLE_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)


READABLE_DARK_PALETTE: tuple[str, ...] = (
    "#4c9be8",
    "#ffa447",
    "#5fd55f",
    "#ff6b6b",
    "#c08be8",
    "#d18b6b",
)


JEAB_GRAYSCALE: tuple[str, ...] = (
    "#000000",
    "#4d4d4d",
    "#808080",
    "#b3b3b3",
)


BW_LINE_STYLES: tuple[str, ...] = ("-", "--", "-.", ":")
COLOR_LINE_STYLES: tuple[str, ...] = ("-",)

OPEN_CLOSED_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v")


_PRESETS: dict[str, ThemeSpec] = {
    "readable": ThemeSpec(
        name="readable",
        description="High-contrast default for live monitoring and demos.",
        font_family="DejaVu Sans",
        font_size_pt=11.0,
        background="#ffffff",
        foreground="#202020",
        palette=READABLE_PALETTE,
        line_style_cycle=COLOR_LINE_STYLES,
        marker_cycle=("o",),
        line_width_pt=2.0,
        figure_width_in=7.0,
        figure_height_in=4.5,
        dpi=96,
        grid="major",
        spine_width_pt=1.0,
        legend_frame=True,
        color_blind_safe=False,
        intended_use="Live dashboards, screen viewing, demos.",
        tags=("screen", "live"),
    ),
    "readable-dark": ThemeSpec(
        name="readable-dark",
        description="Dark variant of ``readable`` for long sessions / OLED.",
        font_family="DejaVu Sans",
        font_size_pt=11.0,
        background="#111111",
        foreground="#eaeaea",
        palette=READABLE_DARK_PALETTE,
        line_style_cycle=COLOR_LINE_STYLES,
        marker_cycle=("o",),
        line_width_pt=2.0,
        figure_width_in=7.0,
        figure_height_in=4.5,
        dpi=96,
        grid="major",
        spine_width_pt=1.0,
        legend_frame=False,
        color_blind_safe=False,
        intended_use="Long-running monitoring, OLED displays.",
        tags=("screen", "live", "dark"),
    ),
    "jeab-bw": ThemeSpec(
        name="jeab-bw",
        description="JEAB / JABA / Behavioural Processes monochrome house style.",
        font_family="Helvetica",
        font_size_pt=8.0,
        background="#ffffff",
        foreground="#000000",
        palette=JEAB_GRAYSCALE,
        line_style_cycle=BW_LINE_STYLES,
        marker_cycle=("o", "s", "^", "D"),
        line_width_pt=1.0,
        figure_width_in=3.3,
        figure_height_in=2.5,
        dpi=300,
        grid="none",
        spine_width_pt=0.75,
        legend_frame=False,
        color_blind_safe=True,
        intended_use="Direct paste into JEAB / JABA / Beh. Processes manuscripts.",
        tags=("paper", "monochrome", "jeab"),
    ),
    "jeab-bw-marker": ThemeSpec(
        name="jeab-bw-marker",
        description="JEAB monochrome with emphasized open/closed markers.",
        font_family="Helvetica",
        font_size_pt=8.0,
        background="#ffffff",
        foreground="#000000",
        palette=JEAB_GRAYSCALE,
        line_style_cycle=BW_LINE_STYLES,
        marker_cycle=("o", "s", "^", "D", "v"),
        line_width_pt=1.0,
        figure_width_in=3.3,
        figure_height_in=2.5,
        dpi=300,
        grid="none",
        spine_width_pt=0.75,
        legend_frame=False,
        color_blind_safe=True,
        intended_use="Single-subject figures with phase-distinguishing markers.",
        tags=("paper", "monochrome", "jeab", "marker"),
    ),
    "nature-color": ThemeSpec(
        name="nature-color",
        description="Nature / Nat Neurosci / Cell / eLife color-safe style.",
        font_family="Helvetica",
        font_size_pt=7.0,
        background="#ffffff",
        foreground="#000000",
        palette=OKABE_ITO,
        line_style_cycle=COLOR_LINE_STYLES,
        marker_cycle=OPEN_CLOSED_MARKERS,
        line_width_pt=0.75,
        figure_width_in=3.5,  # ~89 mm single column
        figure_height_in=2.6,
        dpi=600,
        grid="none",
        spine_width_pt=0.5,
        legend_frame=False,
        color_blind_safe=True,
        intended_use="Direct paste into Nature-family color figures.",
        tags=("paper", "color", "nature"),
    ),
    "science-color": ThemeSpec(
        name="science-color",
        description="Science / PNAS color style (slightly heavier spines).",
        font_family="Helvetica",
        font_size_pt=7.0,
        background="#ffffff",
        foreground="#000000",
        palette=OKABE_ITO,
        line_style_cycle=COLOR_LINE_STYLES,
        marker_cycle=OPEN_CLOSED_MARKERS,
        line_width_pt=1.0,
        figure_width_in=3.5,
        figure_height_in=2.6,
        dpi=600,
        grid="none",
        spine_width_pt=0.75,
        legend_frame=False,
        color_blind_safe=True,
        intended_use="Direct paste into Science / PNAS color figures.",
        tags=("paper", "color", "science"),
    ),
    "preprint-draft": ThemeSpec(
        name="preprint-draft",
        description="Draft-quality color style for bioRxiv / psyArXiv.",
        font_family="DejaVu Sans",
        font_size_pt=9.0,
        background="#ffffff",
        foreground="#202020",
        palette=OKABE_ITO,
        line_style_cycle=COLOR_LINE_STYLES,
        marker_cycle=OPEN_CLOSED_MARKERS,
        line_width_pt=1.25,
        figure_width_in=5.0,
        figure_height_in=3.5,
        dpi=150,
        grid="minor",
        spine_width_pt=0.75,
        legend_frame=True,
        color_blind_safe=True,
        intended_use="Quick-share draft figures for preprint servers.",
        tags=("paper", "color", "draft"),
    ),
}


THEMES: Mapping[str, ThemeSpec] = types.MappingProxyType(_PRESETS)
"""Read-only registry of built-in presets.

Use :func:`get_theme` for lookup and :func:`register_theme` to add
user-defined presets. The underlying mapping is read-only so that a
typo like ``THEMES["jeab"] = ...`` raises instead of silently mutating
the shared registry.
"""


def get_theme(name: str) -> ThemeSpec:
    """Look up a theme by name. Raises ``KeyError`` if unknown."""
    if name not in _PRESETS:
        raise KeyError(f"Unknown theme: {name!r}. Available: {sorted(_PRESETS)}")
    return _PRESETS[name]


def list_themes() -> list[str]:
    """Return the list of registered theme names, sorted."""
    return sorted(_PRESETS.keys())


def register_theme(spec: ThemeSpec, *, overwrite: bool = False) -> None:
    """Register a user-defined theme so it becomes visible to :func:`get_theme`.

    Parameters
    ----------
    spec:
        The theme to register. ``spec.name`` is used as the lookup key.
    overwrite:
        If ``False`` (default), attempting to register a name that already
        exists raises :class:`ValueError`. Set to ``True`` to replace.
    """
    if not overwrite and spec.name in _PRESETS:
        raise ValueError(
            f"Theme {spec.name!r} already registered; pass overwrite=True to replace."
        )
    _PRESETS[spec.name] = spec
