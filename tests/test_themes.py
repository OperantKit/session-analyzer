"""Tests for visualization theme system (ThemeSpec + presets + mpl adapter)."""

from __future__ import annotations

import pytest

from session_analyzer.visualizer.themes import (
    THEMES,
    ThemeSpec,
    get_theme,
    list_themes,
)
from session_analyzer.visualizer.themes.mpl_adapter import (
    apply_theme_to_axes,
    theme_to_rcparams,
)

EXPECTED_THEME_IDS = {
    "readable",
    "readable-dark",
    "jeab-bw",
    "jeab-bw-marker",
    "nature-color",
    "science-color",
    "preprint-draft",
}

MONOCHROME_THEMES = {"jeab-bw", "jeab-bw-marker"}


@pytest.mark.unit
class TestThemeRegistry:
    def test_all_expected_presets_registered(self) -> None:
        assert set(list_themes()) == EXPECTED_THEME_IDS

    def test_themes_mapping_is_complete(self) -> None:
        assert set(THEMES.keys()) == EXPECTED_THEME_IDS

    def test_get_theme_returns_spec(self) -> None:
        theme = get_theme("readable")
        assert isinstance(theme, ThemeSpec)
        assert theme.name == "readable"

    def test_get_theme_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            get_theme("nonexistent")

    def test_themespec_is_frozen(self) -> None:
        theme = get_theme("readable")
        with pytest.raises(Exception):
            theme.name = "mutated"  # type: ignore[misc]


@pytest.mark.unit
class TestThemeSpecInvariants:
    @pytest.mark.parametrize("theme_id", sorted(EXPECTED_THEME_IDS))
    def test_palette_non_empty(self, theme_id: str) -> None:
        theme = get_theme(theme_id)
        assert len(theme.palette) >= 1

    @pytest.mark.parametrize("theme_id", sorted(EXPECTED_THEME_IDS))
    def test_dpi_is_reasonable(self, theme_id: str) -> None:
        theme = get_theme(theme_id)
        assert 72 <= theme.dpi <= 600

    @pytest.mark.parametrize("theme_id", sorted(EXPECTED_THEME_IDS))
    def test_figure_width_positive(self, theme_id: str) -> None:
        theme = get_theme(theme_id)
        assert theme.figure_width_in > 0
        assert theme.figure_height_in > 0

    @pytest.mark.parametrize("theme_id", sorted(EXPECTED_THEME_IDS))
    def test_font_size_reasonable(self, theme_id: str) -> None:
        theme = get_theme(theme_id)
        assert 5.0 <= theme.font_size_pt <= 20.0

    @pytest.mark.parametrize("theme_id", sorted(EXPECTED_THEME_IDS))
    def test_line_styles_non_empty(self, theme_id: str) -> None:
        theme = get_theme(theme_id)
        assert len(theme.line_style_cycle) >= 1

    @pytest.mark.parametrize("theme_id", sorted(MONOCHROME_THEMES))
    def test_monochrome_themes_are_grayscale(self, theme_id: str) -> None:
        """JEAB B&W presets must contain only grayscale colors."""
        theme = get_theme(theme_id)
        for color in theme.palette:
            assert _is_grayscale_hex(color), (
                f"{theme_id} palette contains non-gray color: {color}"
            )
        assert _is_grayscale_hex(theme.foreground)
        assert _is_grayscale_hex(theme.background)

    @pytest.mark.parametrize("theme_id", sorted(MONOCHROME_THEMES))
    def test_monochrome_themes_have_multiple_line_styles(self, theme_id: str) -> None:
        """B&W themes must distinguish series by line style, not color."""
        theme = get_theme(theme_id)
        assert len(theme.line_style_cycle) >= 3

    def test_readable_dark_has_dark_background(self) -> None:
        theme = get_theme("readable-dark")
        r, g, b = _hex_to_rgb(theme.background)
        assert (r + g + b) / 3 < 64  # dark

    def test_readable_has_light_background(self) -> None:
        theme = get_theme("readable")
        r, g, b = _hex_to_rgb(theme.background)
        assert (r + g + b) / 3 > 224  # near-white

    def test_nature_color_is_colorblind_safe(self) -> None:
        theme = get_theme("nature-color")
        assert theme.color_blind_safe is True

    def test_jeab_bw_dpi_is_publication_grade(self) -> None:
        theme = get_theme("jeab-bw")
        assert theme.dpi >= 300

    def test_readable_dpi_is_screen_grade(self) -> None:
        theme = get_theme("readable")
        assert theme.dpi <= 150


@pytest.mark.unit
class TestMplAdapter:
    def test_rcparams_contains_core_keys(self) -> None:
        theme = get_theme("readable")
        rc = theme_to_rcparams(theme)
        assert "figure.figsize" in rc
        assert "figure.dpi" in rc
        assert "savefig.dpi" in rc
        assert "font.size" in rc
        assert "axes.prop_cycle" in rc

    def test_rcparams_figsize_matches_spec(self) -> None:
        theme = get_theme("jeab-bw")
        rc = theme_to_rcparams(theme)
        w, h = rc["figure.figsize"]
        assert w == pytest.approx(theme.figure_width_in)
        assert h == pytest.approx(theme.figure_height_in)

    def test_rcparams_dpi_matches_spec(self) -> None:
        theme = get_theme("nature-color")
        rc = theme_to_rcparams(theme)
        assert rc["savefig.dpi"] == theme.dpi

    def test_rcparams_prop_cycle_contains_palette(self) -> None:
        theme = get_theme("nature-color")
        rc = theme_to_rcparams(theme)
        cycle = rc["axes.prop_cycle"]
        colors = list(cycle.by_key()["color"])
        assert colors == list(theme.palette)

    def test_apply_theme_to_axes_sets_background(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theme = get_theme("readable-dark")
        fig, ax = plt.subplots()
        try:
            apply_theme_to_axes(ax, theme)
            # Facecolor should match theme background (RGBA tuple).
            fc = ax.get_facecolor()
            expected = _hex_to_rgb(theme.background)
            assert int(fc[0] * 255) == expected[0]
            assert int(fc[1] * 255) == expected[1]
            assert int(fc[2] * 255) == expected[2]
        finally:
            plt.close(fig)

    def test_apply_theme_grid_off_for_jeab(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theme = get_theme("jeab-bw")
        assert theme.grid == "none"
        fig, ax = plt.subplots()
        try:
            apply_theme_to_axes(ax, theme)
            # After apply, no gridlines should be visible on either axis.
            for line in ax.xaxis.get_gridlines():
                assert not line.get_visible()
            for line in ax.yaxis.get_gridlines():
                assert not line.get_visible()
        finally:
            plt.close(fig)

    def test_apply_theme_sets_linestyle_cycle_for_bw(self) -> None:
        """B&W themes must cycle linestyles, not just colors."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theme = get_theme("jeab-bw")
        fig, ax = plt.subplots()
        try:
            apply_theme_to_axes(ax, theme)
            lines = [ax.plot([0, 1], [i, i + 1])[0] for i in range(len(theme.palette))]
            styles = {line.get_linestyle() for line in lines}
            assert len(styles) >= 2, (
                f"B&W theme should cycle linestyles, got only {styles}"
            )
        finally:
            plt.close(fig)


@pytest.mark.unit
class TestRegistryImmutability:
    def test_themes_mapping_is_readonly(self) -> None:
        from session_analyzer.visualizer.themes import THEMES

        with pytest.raises(TypeError):
            THEMES["hacked"] = get_theme("readable")  # type: ignore[index]

    def test_register_theme_adds_to_registry(self) -> None:
        from session_analyzer.visualizer.themes import ThemeSpec, list_themes
        from session_analyzer.visualizer.themes.presets import register_theme

        spec = ThemeSpec(
            name="_test_user_theme",
            description="test",
            font_family="DejaVu Sans",
            font_size_pt=10.0,
            background="#ffffff",
            foreground="#000000",
            palette=("#123456",),
            line_style_cycle=("-",),
            marker_cycle=("o",),
            line_width_pt=1.0,
            figure_width_in=4.0,
            figure_height_in=3.0,
            dpi=100,
            grid="none",
            spine_width_pt=1.0,
            legend_frame=True,
        )
        try:
            register_theme(spec)
            assert "_test_user_theme" in list_themes()
            assert get_theme("_test_user_theme") is spec
        finally:
            from session_analyzer.visualizer.themes.presets import _PRESETS

            _PRESETS.pop("_test_user_theme", None)

    def test_register_theme_refuses_duplicate(self) -> None:
        from session_analyzer.visualizer.themes.presets import register_theme

        existing = get_theme("readable")
        with pytest.raises(ValueError, match="already registered"):
            register_theme(existing)


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _is_grayscale_hex(hex_str: str) -> bool:
    r, g, b = _hex_to_rgb(hex_str)
    return r == g == b
