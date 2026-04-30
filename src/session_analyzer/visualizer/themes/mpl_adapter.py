"""Adapter: :class:`ThemeSpec` -> matplotlib ``rcParams`` and Axes styling."""

from __future__ import annotations

from typing import Any

from cycler import Cycler, cycler

from .spec import ThemeSpec


def _build_prop_cycle(theme: ThemeSpec) -> Cycler:
    """Compose a combined color + linestyle prop cycle from a theme.

    If ``line_style_cycle`` has the same length as ``palette``, the two
    axes are zipped so that each palette slot gets a distinct linestyle
    (used for B&W themes where color alone cannot distinguish series).
    Otherwise only the color axis is cycled.
    """
    color_cycle = cycler(color=list(theme.palette))
    if len(theme.line_style_cycle) == len(theme.palette):
        return color_cycle + cycler(linestyle=list(theme.line_style_cycle))
    return color_cycle


def theme_to_rcparams(theme: ThemeSpec) -> dict[str, Any]:
    """Build a matplotlib ``rcParams``-compatible dict from a theme.

    The returned mapping is intended to be passed to
    ``matplotlib.rcParams.update(...)`` or used inside
    ``matplotlib.rc_context(...)``.
    """
    sans_serif_fallback = ["DejaVu Sans", "Arial", "Liberation Sans"]
    serif_fallback = ["DejaVu Serif", "Times New Roman", "Liberation Serif"]
    if theme.font_family.lower() in {"times", "times new roman", "serif"}:
        generic = "serif"
        font_list = [theme.font_family, *serif_fallback]
    else:
        generic = "sans-serif"
        font_list = [theme.font_family, *sans_serif_fallback]

    return {
        "figure.figsize": (theme.figure_width_in, theme.figure_height_in),
        "figure.dpi": theme.dpi,
        "figure.facecolor": theme.background,
        "figure.edgecolor": theme.background,
        "savefig.dpi": theme.dpi,
        "savefig.facecolor": theme.background,
        "savefig.edgecolor": theme.background,
        "font.family": generic,
        f"font.{generic}": font_list,
        "font.size": theme.font_size_pt,
        "axes.facecolor": theme.background,
        "axes.edgecolor": theme.foreground,
        "axes.labelcolor": theme.foreground,
        "axes.linewidth": theme.spine_width_pt,
        "axes.grid": theme.grid != "none",
        "axes.prop_cycle": _build_prop_cycle(theme),
        "xtick.color": theme.foreground,
        "ytick.color": theme.foreground,
        "text.color": theme.foreground,
        "lines.linewidth": theme.line_width_pt,
        "legend.frameon": theme.legend_frame,
        "legend.edgecolor": theme.foreground,
        "legend.labelcolor": theme.foreground,
    }


def apply_theme_to_axes(ax: Any, theme: ThemeSpec) -> None:
    """Apply theme styling directly to an existing matplotlib ``Axes``.

    Useful when ``rc_context`` is not convenient (e.g., styling a single
    subplot within a multi-panel figure already laid out by the caller).
    """
    ax.set_facecolor(theme.background)
    for spine in ax.spines.values():
        spine.set_linewidth(theme.spine_width_pt)
        spine.set_edgecolor(theme.foreground)
    ax.tick_params(colors=theme.foreground, width=theme.spine_width_pt)
    ax.xaxis.label.set_color(theme.foreground)
    ax.yaxis.label.set_color(theme.foreground)
    if ax.get_title():
        ax.title.set_color(theme.foreground)

    if theme.grid == "none":
        ax.grid(False)
    elif theme.grid == "minor":
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)
        ax.grid(True, which="major", linewidth=0.5, alpha=0.5)
    else:  # "major"
        ax.grid(True, which="major", linewidth=0.5, alpha=0.5)

    ax.set_prop_cycle(_build_prop_cycle(theme))
