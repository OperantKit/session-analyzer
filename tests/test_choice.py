"""Tests for choice allocation analysis."""

import pytest

from session_analyzer.analytics.choice import (
    ChoiceMode,
    classify_choice_mode,
    compute_choice_allocation,
    detect_exclusive_choice,
)


class TestComputeChoiceAllocation:
    def test_basic_allocation(self) -> None:
        alloc = compute_choice_allocation([80, 20])
        assert alloc.proportions == (0.8, 0.2)
        assert alloc.counts == (80, 20)
        assert alloc.total == 100

    def test_zero_responses_uniform(self) -> None:
        alloc = compute_choice_allocation([0, 0, 0])
        assert len(alloc.proportions) == 3
        assert alloc.proportions == pytest.approx((1 / 3, 1 / 3, 1 / 3))
        assert alloc.total == 0

    def test_three_operanda(self) -> None:
        alloc = compute_choice_allocation([50, 30, 20])
        assert alloc.proportions == (0.5, 0.3, 0.2)
        assert alloc.total == 100

    def test_single_operandum_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            compute_choice_allocation([100])

    def test_proportions_sum_to_one(self) -> None:
        alloc = compute_choice_allocation([7, 13, 3])
        assert sum(alloc.proportions) == pytest.approx(1.0)


class TestDetectExclusiveChoice:
    def test_exclusive_detected(self) -> None:
        alloc = compute_choice_allocation([96, 4])
        result = detect_exclusive_choice(alloc)
        assert result.is_exclusive is True
        assert result.dominant_operandum == 0
        assert result.dominant_proportion == 0.96
        assert result.threshold == 0.95

    def test_not_exclusive(self) -> None:
        alloc = compute_choice_allocation([70, 30])
        result = detect_exclusive_choice(alloc)
        assert result.is_exclusive is False
        assert result.dominant_operandum is None
        assert result.dominant_proportion is None

    def test_exactly_at_threshold(self) -> None:
        alloc = compute_choice_allocation([95, 5])
        result = detect_exclusive_choice(alloc)
        assert result.is_exclusive is True
        assert result.dominant_operandum == 0

    def test_custom_threshold(self) -> None:
        alloc = compute_choice_allocation([90, 10])
        result_strict = detect_exclusive_choice(alloc, threshold=0.95)
        result_loose = detect_exclusive_choice(alloc, threshold=0.90)
        assert result_strict.is_exclusive is False
        assert result_loose.is_exclusive is True

    def test_second_operandum_dominant(self) -> None:
        alloc = compute_choice_allocation([3, 97])
        result = detect_exclusive_choice(alloc)
        assert result.is_exclusive is True
        assert result.dominant_operandum == 1

    def test_invalid_threshold_raises(self) -> None:
        alloc = compute_choice_allocation([50, 50])
        with pytest.raises(ValueError, match="threshold"):
            detect_exclusive_choice(alloc, threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            detect_exclusive_choice(alloc, threshold=1.5)


class TestClassifyChoiceMode:
    def test_exclusive(self) -> None:
        alloc = compute_choice_allocation([98, 2])
        assert classify_choice_mode(alloc) == ChoiceMode.EXCLUSIVE

    def test_matching(self) -> None:
        alloc = compute_choice_allocation([52, 48])
        assert classify_choice_mode(alloc) == ChoiceMode.MATCHING

    def test_intermediate(self) -> None:
        alloc = compute_choice_allocation([75, 25])
        mode = classify_choice_mode(alloc)
        assert mode == ChoiceMode.INTERMEDIATE

    def test_three_operanda_matching(self) -> None:
        alloc = compute_choice_allocation([35, 33, 32])
        assert classify_choice_mode(alloc) == ChoiceMode.MATCHING

    def test_custom_thresholds(self) -> None:
        alloc = compute_choice_allocation([85, 15])
        # Default thresholds: not exclusive, not matching
        assert classify_choice_mode(alloc) == ChoiceMode.INTERMEDIATE
        # Loose exclusive threshold
        assert (
            classify_choice_mode(alloc, exclusive_threshold=0.80)
            == ChoiceMode.EXCLUSIVE
        )

    def test_enum_values(self) -> None:
        assert ChoiceMode.EXCLUSIVE.value == "exclusive"
        assert ChoiceMode.MATCHING.value == "matching"
        assert ChoiceMode.INTERMEDIATE.value == "intermediate"


# ---------------------------------------------------------------------------
# Hand computations and threshold boundaries
# ---------------------------------------------------------------------------
#
# No published value: Herrnstein (1961) and Herrnstein & Loveland (1975)
# define relative response rate B1 / (B1 + B2); the 0.95 exclusive-choice
# threshold and the 0.10 matching spread are this module's conventions
# (module docstring). The checks are hand computations at the boundaries,
# which the docstrings define as inclusive (">=", "<=").


@pytest.mark.unit
def test_allocation_matches_hand_computation_for_three_operanda() -> None:
    alloc = compute_choice_allocation([30, 50, 20])
    assert alloc.proportions == pytest.approx((0.3, 0.5, 0.2), rel=1e-12)
    assert alloc.total == 100


@pytest.mark.unit
def test_exclusive_threshold_is_inclusive_at_exactly_095() -> None:
    result = detect_exclusive_choice(compute_choice_allocation([95, 5]))
    assert result.is_exclusive
    assert result.dominant_operandum == 0


@pytest.mark.unit
def test_matching_tolerance_is_inclusive_at_exactly_010() -> None:
    """55 / 45 has a spread of exactly 10 points; ``<=`` must classify it as
    matching even though 0.55 - 0.45 in floating point exceeds 0.1."""
    assert classify_choice_mode(compute_choice_allocation([55, 45])) is ChoiceMode.MATCHING
    assert classify_choice_mode(compute_choice_allocation([56, 44])) is ChoiceMode.INTERMEDIATE
