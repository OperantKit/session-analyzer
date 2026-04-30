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
