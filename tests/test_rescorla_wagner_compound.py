"""Tests for the compound-CS Rescorla-Wagner simulation.

The single-CS functions in the same module are the one-stimulus case of the
compound simulation; the agreement tests below check that the two give the
same trace.
"""

from __future__ import annotations

import random

import pytest

from session_analyzer.analytics import (
    CompoundRescorlaWagner,
    CompoundTrial,
    simulate_compound_rescorla_wagner,
)
from session_analyzer.analytics.rescorla_wagner import simulate_rescorla_wagner

ALPHA = 0.3
BETA = 0.5
LAMBDA = 1.0


def _single_cs(us_present: list[bool]) -> list[CompoundTrial]:
    return [
        CompoundTrial(stimuli=frozenset({"tone"}), us="shock" if p else None, stage="acquisition")
        for p in us_present
    ]


def _compound_trace(us_present: list[bool]) -> list[float]:
    model = CompoundRescorlaWagner({"tone": ALPHA}, beta=BETA, lambda_by_us={"shock": LAMBDA})
    trace = simulate_compound_rescorla_wagner(_single_cs(us_present), model)
    return [r.v_after["tone"] for r in trace.records]


# ---------------------------------------------------------------------------
# Single CS: the compound simulation reduces to the single-CS one
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_cs_matches_closed_form_acquisition() -> None:
    got = _compound_trace([True] * 20)
    ab = ALPHA * BETA
    want = [LAMBDA * (1 - (1 - ab) ** n) for n in range(1, 21)]
    assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.unit
def test_single_cs_matches_simulate_rescorla_wagner() -> None:
    us = [True] * 12 + [False] * 8
    want = simulate_rescorla_wagner(us, alpha_beta=ALPHA * BETA, lambda_=LAMBDA)
    assert _compound_trace(us) == pytest.approx(want, abs=1e-12)


@pytest.mark.unit
def test_single_cs_function_is_the_one_stimulus_case_exactly() -> None:
    rng = random.Random(3)
    for _ in range(20):
        us = [rng.random() < 0.6 for _ in range(rng.randint(1, 40))]
        ab = rng.uniform(0.01, 1.0)
        lam = rng.uniform(0.1, 2.0)
        v0 = rng.uniform(0.0, 1.0)
        model = CompoundRescorlaWagner({"cs": ab}, lambda_by_us={"us": lam}, v0=v0)
        trials = [CompoundTrial(frozenset({"cs"}), "us" if p else None) for p in us]
        compound = [
            r.v_after["cs"] for r in simulate_compound_rescorla_wagner(trials, model).records
        ]
        assert simulate_rescorla_wagner(us, alpha_beta=ab, lambda_=lam, initial_v=v0) == compound


# ---------------------------------------------------------------------------
# Compound stimuli
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compound_uses_summed_error() -> None:
    trial = CompoundTrial(stimuli=frozenset({"a", "b"}), us="shock")
    model = CompoundRescorlaWagner({}, alpha_default=0.5, lambda_by_us={"shock": 1.0})
    assert simulate_compound_rescorla_wagner([trial], model).records[0].v_after == {
        "a": 0.5,
        "b": 0.5,
    }
    # The summed strength after trial 1 is 1.0, so the error on trial 2 is 0.
    after2 = simulate_compound_rescorla_wagner([trial, trial], model).records[1].v_after
    assert after2 == pytest.approx({"a": 0.5, "b": 0.5})
    assert simulate_compound_rescorla_wagner([trial, trial], model).records[1].sum_v_before == 1.0


@pytest.mark.unit
def test_absent_stimuli_are_not_updated() -> None:
    trials = [
        CompoundTrial(frozenset({"a"}), "us"),
        CompoundTrial(frozenset({"b"}), "us"),
    ]
    model = CompoundRescorlaWagner({}, alpha_default=0.5, lambda_by_us={"us": 1.0})
    records = simulate_compound_rescorla_wagner(trials, model).records
    assert records[0].v_after == {"a": 0.5, "b": 0.0}
    assert records[1].v_before == {"a": 0.5, "b": 0.0}
    assert records[1].v_after == {"a": 0.5, "b": 0.5}


@pytest.mark.unit
def test_blocking_leaves_the_added_cs_weak() -> None:
    # Kamin-type blocking: A+ pretraining, then AX+.
    a = CompoundTrial(frozenset({"A"}), "us", stage="pretraining")
    ax = CompoundTrial(frozenset({"A", "X"}), "us", stage="compound")
    model = CompoundRescorlaWagner({}, alpha_default=0.3, lambda_by_us={"us": 1.0})
    trace = simulate_compound_rescorla_wagner([a] * 40 + [ax] * 10, model)
    assert trace.v_at_stage_start("compound")["A"] > 0.99
    assert trace.final()["X"] < 0.01


@pytest.mark.unit
def test_nonreinforced_beta() -> None:
    model = CompoundRescorlaWagner(
        {"cs": 0.5}, lambda_by_us={"us": 1.0}, beta_nonreinforced=0.2, v0=1.0
    )
    trace = simulate_compound_rescorla_wagner([CompoundTrial(frozenset({"cs"}))], model)
    assert trace.final()["cs"] == pytest.approx(1.0 - 0.5 * 0.2)


@pytest.mark.unit
def test_unknown_us_label_has_zero_lambda() -> None:
    trial = CompoundTrial(frozenset({"light"}), us="tone")
    model = CompoundRescorlaWagner({}, lambda_by_us={"shock": 1.0})
    assert simulate_compound_rescorla_wagner([trial], model).records[0].v_after["light"] == 0.0


@pytest.mark.unit
def test_context_is_one_more_stimulus() -> None:
    trial = CompoundTrial(frozenset({"box"}), us="shock")
    model = CompoundRescorlaWagner({}, alpha_default=0.4, lambda_by_us={"shock": 1.0})
    assert simulate_compound_rescorla_wagner([trial], model).records[0].v_after[
        "box"
    ] == pytest.approx(0.4)


@pytest.mark.unit
def test_two_us_labels_keep_separate_systems() -> None:
    tone = frozenset({"tone"})
    trials = [CompoundTrial(tone, "shock", stage="phase1") for _ in range(30)] + [
        CompoundTrial(tone, "food", stage="phase2") for _ in range(30)
    ]
    model = CompoundRescorlaWagner({}, lambda_by_us={"shock": 1.0, "food": 1.0})
    trace = simulate_compound_rescorla_wagner(trials, model)
    assert trace.systems == ("shock", "food")
    before = trace.v_at_stage_start("phase2", us="shock")["tone"]
    assert trace.final(us="shock")["tone"] < before
    assert trace.final(us="food")["tone"] > 0.9


@pytest.mark.unit
def test_primary_system_is_first_used_label() -> None:
    model = CompoundRescorlaWagner({}, lambda_by_us={"shock": 1.0, "food": 1.0})
    trace = simulate_compound_rescorla_wagner([CompoundTrial(frozenset({"cs"}), "food")], model)
    assert trace.systems == ("food",)
    assert trace.records[0].primary == "food"


@pytest.mark.unit
def test_no_reinforced_trial_uses_default_system() -> None:
    model = CompoundRescorlaWagner({}, lambda_by_us={"us": 1.0}, v0=0.5)
    trace = simulate_compound_rescorla_wagner([CompoundTrial(frozenset({"cs"}))], model)
    assert trace.systems == ("",)
    assert trace.final()["cs"] == pytest.approx(0.5 * 0.7)


@pytest.mark.unit
def test_empty_trials_give_empty_trace() -> None:
    trace = simulate_compound_rescorla_wagner([], CompoundRescorlaWagner({}, lambda_by_us={}))
    assert trace.records == ()
    assert trace.final() == {}
    with pytest.raises(KeyError, match="stage"):
        trace.v_at_stage_start("missing")


@pytest.mark.unit
def test_model_rejects_out_of_range_rates() -> None:
    with pytest.raises(ValueError, match="alpha"):
        CompoundRescorlaWagner({"cs": 0.0}, lambda_by_us={})
    with pytest.raises(ValueError, match="alpha"):
        CompoundRescorlaWagner({}, lambda_by_us={}, alpha_default=1.5)
    with pytest.raises(ValueError, match="beta"):
        CompoundRescorlaWagner({}, lambda_by_us={}, beta=0.0)
    with pytest.raises(ValueError, match="beta_nonreinforced"):
        CompoundRescorlaWagner({}, lambda_by_us={}, beta_nonreinforced=2.0)


@pytest.mark.unit
def test_model_mappings_are_read_only() -> None:
    alphas = {"cs": 0.3}
    model = CompoundRescorlaWagner(alphas, lambda_by_us={"us": 1.0})
    alphas["cs"] = 0.9
    assert model.alpha("cs") == 0.3
    with pytest.raises(TypeError):
        model.lambda_by_us["us"] = 2.0  # type: ignore[index]


@pytest.mark.unit
def test_trial_rejects_a_plain_string_as_stimuli() -> None:
    with pytest.raises(TypeError, match="stimuli"):
        CompoundTrial("tone")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixed values from the compound executor this function replaces
# ---------------------------------------------------------------------------
#
# The expected values below were produced by the compound-CS executor of the
# respondent simulator on the same trials, so that its users can move to this
# function without a change in the numbers.


@pytest.mark.unit
def test_unknown_us_label_is_a_trial_without_us() -> None:
    model = CompoundRescorlaWagner({}, lambda_by_us={"shock": 1.0}, v0=1.0)
    trace = simulate_compound_rescorla_wagner([CompoundTrial(frozenset({"light"}), "tone")], model)
    assert trace.final() == pytest.approx({"light": 0.7})


@pytest.mark.unit
def test_other_systems_use_nonreinforced_beta_and_start_at_v0() -> None:
    model = CompoundRescorlaWagner(
        {},
        alpha_default=0.5,
        lambda_by_us={"shock": 1.0, "food": 1.0},
        beta_nonreinforced=0.2,
        v0=1.0,
    )
    trials = [CompoundTrial(frozenset({"cs"}), "shock"), CompoundTrial(frozenset({"cs"}), "food")]
    record = simulate_compound_rescorla_wagner(trials, model).records[0]
    assert dict(record.v_after_by_us["shock"]) == pytest.approx({"cs": 1.0})
    assert dict(record.v_after_by_us["food"]) == pytest.approx({"cs": 0.9})


@pytest.mark.unit
def test_primary_follows_the_order_of_lambda_by_us() -> None:
    model = CompoundRescorlaWagner({}, lambda_by_us={"shock": 1.0, "food": 0.5})
    trials = [CompoundTrial(frozenset({"cs"}), "food"), CompoundTrial(frozenset({"cs"}), "shock")]
    trace = simulate_compound_rescorla_wagner(trials, model)
    assert trace.systems == ("shock", "food")
    assert trace.records[0].primary == "shock"
    assert trace.final() == pytest.approx({"cs": 0.3})
    assert trace.final(us="food") == pytest.approx({"cs": 0.105})


@pytest.mark.unit
def test_sum_v_before_counts_presented_stimuli_only() -> None:
    model = CompoundRescorlaWagner({}, alpha_default=0.5, lambda_by_us={"us": 1.0})
    trials = [CompoundTrial(frozenset({"a", "b"}), "us"), CompoundTrial(frozenset({"a"}), "us")]
    record = simulate_compound_rescorla_wagner(trials, model).records[1]
    assert record.v_before == {"a": 0.5, "b": 0.5}
    assert record.sum_v_before == 0.5


@pytest.mark.unit
@pytest.mark.reference(
    module="rescorla_wagner",
    verdict="properties",
    source="Rescorla & Wagner (1972)",
)
def test_overshadowing_shares_lambda_by_salience() -> None:
    model = CompoundRescorlaWagner({"L": 0.5, "T": 0.1}, lambda_by_us={"us": 1.0})
    trace = simulate_compound_rescorla_wagner(
        [CompoundTrial(frozenset({"L", "T"}), "us")] * 30, model
    )
    final = trace.final()
    # At the asymptote V_L + V_T = lambda and V_L / V_T = alpha_L / alpha_T.
    assert final["L"] == pytest.approx(0.8333333333323725, abs=1e-15)
    assert final["T"] == pytest.approx(0.16666666666647456, abs=1e-15)


@pytest.mark.unit
def test_conditioned_inhibitor_gets_negative_strength() -> None:
    a = CompoundTrial(frozenset({"A"}), "us", stage="p1")
    a2 = CompoundTrial(frozenset({"A"}), "us", stage="p2")
    ax = CompoundTrial(frozenset({"A", "X"}), None, stage="p2")
    model = CompoundRescorlaWagner({}, alpha_default=0.3, lambda_by_us={"us": 1.0})
    trace = simulate_compound_rescorla_wagner([a] * 30 + [a2, ax] * 30, model)
    assert trace.final()["A"] == pytest.approx(0.992359657854852, abs=1e-15)
    assert trace.final()["X"] == pytest.approx(-0.990396717521415, abs=1e-15)
