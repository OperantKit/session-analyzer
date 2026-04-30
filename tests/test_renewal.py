"""Tests for renewal analysis (ABA / ABC / AAB)."""

from __future__ import annotations

import pytest

from session_analyzer.analytics.renewal import (
    PhaseSummary,
    RenewalMetrics,
    analyze_renewal,
    classify_renewal_design,
)


def _phase(label: str, ctx: str | None, responses: int, duration: float) -> PhaseSummary:
    return PhaseSummary(
        label=label, context=ctx, response_count=responses, duration_sec=duration
    )


@pytest.mark.unit
def test_classify_aba() -> None:
    assert classify_renewal_design(("A", "B", "A")) == "ABA"


@pytest.mark.unit
def test_classify_abc() -> None:
    assert classify_renewal_design(("A", "B", "C")) == "ABC"


@pytest.mark.unit
def test_classify_aab() -> None:
    assert classify_renewal_design(("A", "A", "B")) == "AAB"


@pytest.mark.unit
def test_classify_aaa_no_renewal() -> None:
    assert classify_renewal_design(("A", "A", "A")) == "AAA"


@pytest.mark.unit
def test_classify_unknown_pattern() -> None:
    # ABB: train in A, extinguish & test in B  — no canonical name
    assert classify_renewal_design(("A", "B", "B")) == "OTHER"


@pytest.mark.unit
def test_renewal_aba_classic() -> None:
    phases = [
        _phase("Train", "A", responses=600, duration=600.0),  # 1.0 r/s
        _phase("Extinct", "B", responses=60, duration=600.0),  # 0.1 r/s
        _phase("Test", "A", responses=300, duration=600.0),  # 0.5 r/s — renewed
    ]
    metrics = analyze_renewal(phases)
    assert isinstance(metrics, RenewalMetrics)
    assert metrics.design == "ABA"
    assert metrics.train_rate == pytest.approx(1.0)
    assert metrics.extinction_terminal_rate == pytest.approx(0.1)
    assert metrics.test_rate == pytest.approx(0.5)
    assert metrics.renewal_magnitude == pytest.approx(0.4)
    assert metrics.renewal_ratio == pytest.approx(0.5)
    assert metrics.renewed is True
    assert metrics.contexts == ("A", "B", "A")


@pytest.mark.unit
def test_renewal_aba_no_renewal() -> None:
    phases = [
        _phase("Train", "A", responses=600, duration=600.0),
        _phase("Extinct", "B", responses=60, duration=600.0),
        _phase("Test", "A", responses=30, duration=600.0),  # 0.05 r/s
    ]
    metrics = analyze_renewal(phases)
    # test_rate (0.05) < extinction (0.1) -> negative magnitude, not renewed
    assert metrics.renewal_magnitude < 0.0
    assert metrics.renewed is False


@pytest.mark.unit
def test_renewal_aab() -> None:
    phases = [
        _phase("Train", "A", responses=600, duration=600.0),
        _phase("Extinct", "A", responses=60, duration=600.0),
        _phase("Test", "B", responses=240, duration=600.0),
    ]
    metrics = analyze_renewal(phases)
    assert metrics.design == "AAB"
    assert metrics.renewed is True


@pytest.mark.unit
def test_renewal_requires_three_phases() -> None:
    phases = [
        _phase("Train", "A", responses=100, duration=100.0),
        _phase("Test", "B", responses=50, duration=100.0),
    ]
    with pytest.raises(ValueError, match="three phases"):
        analyze_renewal(phases)


@pytest.mark.unit
def test_renewal_validates_durations() -> None:
    phases = [
        _phase("Train", "A", responses=100, duration=0.0),
        _phase("Extinct", "B", responses=10, duration=100.0),
        _phase("Test", "A", responses=50, duration=100.0),
    ]
    with pytest.raises(ValueError, match="duration"):
        analyze_renewal(phases)


@pytest.mark.unit
def test_renewal_handles_missing_context_names() -> None:
    """Phases without explicit context names get fallback labels."""
    phases = [
        _phase("Train", None, responses=600, duration=600.0),
        _phase("Extinct", None, responses=60, duration=600.0),
        _phase("Test", None, responses=300, duration=600.0),
    ]
    metrics = analyze_renewal(phases)
    # All None -> all considered same context -> AAA
    assert metrics.design == "AAA"
