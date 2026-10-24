"""Every analytics module has a test pinned to its source.

A test carries ``@pytest.mark.reference(module=..., verdict=..., source=...)``
when it checks an analytics module against a published value, a reference
implementation, the original definition, or (when none of those is available)
closed forms and hand computations. This file reads the marks from the test
sources and fails when an analytics module has none.

Verdicts:

- ``published``: equals values printed in the source or in a reference package's
  published output.
- ``reference-implementation``: equals an independent package run on the same data.
- ``definition``: equals the formulas of the source (no numeric example exists).
- ``properties``: closed forms, recovery of known parameters, hand computation.
- ``not-retrieved``: the source has values but its text was not retrieved;
  checked by properties meanwhile.

``properties`` and ``not-retrieved`` modules are listed in ``PROPERTIES_ONLY``
with the reason, so that a new module cannot silently join them. The coverage
table in README.md, README.ja.md and docs/{en,ja}/reference-values.md must
list every module with the same verdict.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYTICS = ROOT / "src" / "session_analyzer" / "analytics"
TESTS = ROOT / "tests"

VERDICTS = {"published", "reference-implementation", "definition", "properties", "not-retrieved"}
WEAK = {"properties", "not-retrieved"}

PROPERTIES_ONLY: dict[str, str] = {
    "latency": "no source defines a reference dataset; the measure is a pairing rule",
    "breakpoint": "the breakpoint is a definition whose value depends on the procedure",
    "prp": "Felton & Lyon (1966) and Fry et al. (1960) give no reusable table in the abstracts",
    "bout": "Shull et al. (2001) estimate from log-survivor plots, a different estimator",
    "matching": "Baum (1974) Table 1 is in a scanned PMC PDF not retrieved",
    "schneider_segment": "Schneider (1969) breakpoint table is in a scanned PMC PDF not retrieved",
    "choice": "proportions and thresholds are definitions",
    "variability": "Page & Neuringer (1985) is paywalled; U has closed forms",
    "ngram_entropy": "entropy of a known sequence has a closed form",
    "resurgence": "the module reports descriptive statistics",
    "momentum": "Nevin (1992) values are in a scanned PMC PDF not retrieved",
    "rescorla_wagner": "the model has a closed-form solution",
    "peak": "Church, Meck & Gibbon (1994) is paywalled",
    "scalar_timing": "the coefficient of variation depends on the experiment",
    "log_survivor": "a Poisson process has a closed-form log-survivor line",
    "run_length": "pause and run are definitions",
    "unit_price": "Hursh & Roma (2013) is paywalled; the fit is ordinary least squares",
    "cross_elasticity_matrix": "no published matrix; each entry is an ordinary least-squares slope",
    "indifference": "the stability window is a definition",
    "concurrent_chains": "the hyperbolic value has a closed form",
    "time_matching": "Baum & Rachlin (1969) table is in a scanned PMC PDF not retrieved",
    "matching_induction": "the three-parameter form is this module's own",
    "stay_switch": "changeover counts are definitions",
    "killeen_law": "Killeen (1994) fitted values are paywalled",
    "within_session": "McSweeney, Roll & Cannon (1994) is paywalled",
    "lag_variability": "the Lag-N criterion is a definition",
    "laws_of_covariance": "the point prediction is this module's own",
    "extinction_burst": "Lerman & Iwata (1995) define no window or ratio",
    "generalization": "Guttman & Kalish (1956) is paywalled",
    "stimulus_control": "Hanson (1959) and Annau & Kamin (1961) are paywalled",
    "irt_spectrum": "no published spectrum; agrees with NumPy / SciPy",
    "td_learning": "the fixed point has a closed form",
    "renewal": "the module reports descriptive statistics",
    "observing": "the module reports rates and ratios",
}

TABLES = [
    ROOT / "README.md",
    ROOT / "README.ja.md",
    ROOT / "docs" / "en" / "reference-values.md",
    ROOT / "docs" / "ja" / "reference-values.md",
]


def _analytics_modules() -> set[str]:
    return {p.stem for p in ANALYTICS.glob("*.py") if p.stem != "__init__"}


def _reference_marks() -> dict[str, list[tuple[str, str, str]]]:
    """Map module name to [(verdict, source, "file::test"), ...] read from the test sources."""
    found: dict[str, list[tuple[str, str, str]]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call):
                    continue
                if ast.unparse(deco.func) != "pytest.mark.reference":
                    continue
                kw = {k.arg: ast.literal_eval(k.value) for k in deco.keywords}
                found.setdefault(kw.get("module", ""), []).append(
                    (kw.get("verdict", ""), kw.get("source", ""), f"{path.name}::{node.name}")
                )
    return found


@pytest.mark.unit
def test_every_analytics_module_has_a_reference_test() -> None:
    marks = _reference_marks()
    missing = sorted(_analytics_modules() - set(marks))
    assert not missing, f"analytics modules without a @pytest.mark.reference test: {missing}"


@pytest.mark.unit
def test_reference_marks_name_existing_modules_with_known_verdicts() -> None:
    modules = _analytics_modules()
    for module, entries in _reference_marks().items():
        assert module in modules, f"mark names unknown module {module!r}: {entries}"
        for verdict, source, where in entries:
            assert verdict in VERDICTS, f"{where}: unknown verdict {verdict!r}"
            assert source.strip(), f"{where}: empty source"


@pytest.mark.unit
def test_properties_only_modules_are_allowlisted_with_a_reason() -> None:
    marks = _reference_marks()
    weak = {m for m, entries in marks.items() if all(v in WEAK for v, _, _ in entries)}
    assert weak == set(PROPERTIES_ONLY), (
        f"not allowlisted: {sorted(weak - set(PROPERTIES_ONLY))}; "
        f"stale allowlist entries: {sorted(set(PROPERTIES_ONLY) - weak)}"
    )
    assert all(reason.strip() for reason in PROPERTIES_ONLY.values())


@pytest.mark.unit
@pytest.mark.parametrize("table", TABLES, ids=lambda p: str(p.relative_to(ROOT)))
def test_coverage_table_lists_every_module_with_its_verdict(table: Path) -> None:
    text = table.read_text(encoding="utf-8")
    best = {m: entries[0][0] for m, entries in _reference_marks().items()}
    for module in sorted(_analytics_modules()):
        row = re.search(rf"^\| `{module}` \|.*$", text, re.M)
        assert row, f"{table.name}: no row for {module}"
        assert f"`{best.get(module)}`" in row.group(0), f"{table.name}: verdict of {module} differs"
