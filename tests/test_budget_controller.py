"""R1: the controller spends a hard budget and says what it did.

The contracts that matter are: the cap is never exceeded, priority is transparent and
deterministic, refusals are recorded rather than hidden, and — most important — the whole
mechanism vanishes when it is disabled, leaving B0_CLEAN behaviour untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aic2026.budget import (
    ACTION_DENSE_TEMPORAL_ZOOM,
    ACTION_OFFICIAL_GRID_REFINE,
    ACTION_QA_VLM_CALL,
    ACTION_UNIT_COST,
    BudgetAction,
    BudgetLedger,
    EventUncertainty,
    allocate,
    channel_disagreement,
    kis_uncertainty,
    margin_uncertainty,
    prioritize,
    split_budget_by_uncertainty,
    support_concentration,
    temporal_ambiguity,
    trake_event_uncertainty,
)
from aic2026.config import AdaptiveBudgetConfig, ConfigError, app_config_from_dict
from tests.release_support import build_engine


def candidate(video_id: str, score: float, timestamp: float = 0.0):
    return SimpleNamespace(video_id=video_id, score=score, timestamp=timestamp, keyframe_id=f"{video_id}/k")


# ------------------------------------------------------------------- uncertainty


def test_a_runaway_winner_is_certain() -> None:
    assert margin_uncertainty([1.0, 0.1]) < 0.2


def test_a_photo_finish_is_uncertain() -> None:
    assert margin_uncertainty([1.0, 0.99]) > 0.9


def test_one_candidate_alone_is_settled_not_unknown() -> None:
    assert margin_uncertainty([1.0]) == 0.0
    assert margin_uncertainty([]) == 0.0


def test_channel_disagreement_needs_two_channels() -> None:
    assert channel_disagreement([["a", "b"]]) == 0.0
    assert channel_disagreement([]) == 0.0


def test_identical_channel_heads_agree_completely() -> None:
    assert channel_disagreement([["a", "b", "c"], ["a", "b", "c"]]) == 0.0


def test_disjoint_channel_heads_disagree_completely() -> None:
    assert channel_disagreement([["a", "b"], ["c", "d"]]) == 1.0


def test_support_concentration_spans_one_video_to_all_distinct() -> None:
    assert support_concentration(["V", "V", "V"]) == 0.0
    assert support_concentration(["A", "B", "C"]) == 1.0


def test_temporal_ambiguity_counts_separated_regions() -> None:
    assert temporal_ambiguity([1.0, 1.1, 1.2], window_s=2.0) == 0.0
    assert temporal_ambiguity([0.0, 10.0, 20.0], window_s=2.0) == 1.0


def test_uncertainty_is_bounded_and_never_called_a_probability() -> None:
    signals = kis_uncertainty([candidate("A", 1.0), candidate("B", 0.99, 30.0)])
    assert 0.0 <= signals.uncertainty <= 1.0
    payload = signals.to_dict()
    assert "probability" not in str(payload).lower()
    assert "NOT more likely to be wrong" in payload["note"]


def test_every_uncertainty_component_is_logged() -> None:
    payload = kis_uncertainty([candidate("A", 1.0), candidate("A", 0.5)]).to_dict()
    for key in ("score_margin", "channel_disagreement", "support_concentration", "temporal_ambiguity"):
        assert key in payload


# ---------------------------------------------------------------------- actions


def test_priority_rewards_utility_uncertainty_and_cheapness() -> None:
    cheap = BudgetAction(name=ACTION_OFFICIAL_GRID_REFINE, rank=1, uncertainty=0.5)
    dear = BudgetAction(name=ACTION_QA_VLM_CALL, rank=1, uncertainty=0.5)
    assert cheap.priority > dear.priority

    high_rank = BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, rank=1, uncertainty=0.5)
    low_rank = BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, rank=80, uncertainty=0.5)
    assert high_rank.priority > low_rank.priority

    certain = BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, rank=1, uncertainty=0.0)
    assert certain.priority == 0.0


def test_expected_gain_is_named_a_proxy_everywhere() -> None:
    payload = BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM).to_dict()
    assert "expected_gain_proxy" in payload
    assert "expected_accuracy" not in payload


def test_prioritize_is_deterministic() -> None:
    actions = [
        BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, target="b", rank=1, uncertainty=0.5),
        BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, target="a", rank=1, uncertainty=0.5),
    ]
    assert [a.target for a in prioritize(actions)] == ["a", "b"]
    assert prioritize(actions) == prioritize(list(reversed(actions)))


# ----------------------------------------------------------------------- ledger


def test_budget_cap_is_never_exceeded() -> None:
    ledger = BudgetLedger(max_cost_units=10.0)
    action = BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, units=1)  # 4.0 each
    assert ledger.try_spend(action) and ledger.try_spend(action)
    assert ledger.spent == pytest.approx(8.0)
    assert not ledger.try_spend(BudgetAction(name=ACTION_QA_VLM_CALL))  # 200.0
    assert ledger.spent == pytest.approx(8.0)
    assert ledger.spent <= ledger.max_cost_units


def test_a_refused_action_is_recorded_not_hidden() -> None:
    ledger = BudgetLedger(max_cost_units=1.0)
    ledger.try_spend(BudgetAction(name=ACTION_QA_VLM_CALL))
    payload = ledger.to_dict()
    assert payload["actions_accepted"] == 0
    assert payload["actions_rejected"] == 1
    assert payload["rejected"][0]["reason"] == "over_budget"
    assert payload["stop_reason"] == "budget_exhausted"


def test_allocation_buys_in_priority_order_until_it_cannot() -> None:
    ledger = BudgetLedger(max_cost_units=9.0)
    actions = [
        BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, target="low", rank=90, uncertainty=1.0),
        BudgetAction(name=ACTION_DENSE_TEMPORAL_ZOOM, target="high", rank=1, uncertainty=1.0),
    ]
    taken = allocate(actions, ledger)
    assert [item.target for item in taken] == ["high", "low"]
    assert ledger.spent <= 9.0


def test_zero_budget_buys_nothing() -> None:
    ledger = BudgetLedger(max_cost_units=0.0)
    assert allocate([BudgetAction(name=ACTION_OFFICIAL_GRID_REFINE)], ledger) == []
    assert ledger.spent == 0.0


# -------------------------------------------------------------- budget splitting


def test_event_budget_sums_to_the_total_cap() -> None:
    split = split_budget_by_uncertainty({0: 0.1, 1: 0.7, 2: 0.2}, 30, maximum=24)
    assert sum(split.values()) == 30
    assert set(split) == {0, 1, 2}


def test_the_weakest_event_gets_the_most() -> None:
    split = split_budget_by_uncertainty({0: 0.05, 1: 0.9, 2: 0.05}, 30, maximum=24)
    assert split[1] == max(split.values())
    assert split[1] > split[0] and split[1] > split[2]


def test_no_event_may_exceed_its_cap() -> None:
    split = split_budget_by_uncertainty({0: 0.99, 1: 0.01}, 40, maximum=24)
    assert split[0] <= 24
    assert sum(split.values()) == 40


def test_no_signal_splits_evenly_and_deterministically() -> None:
    split = split_budget_by_uncertainty({0: 0.0, 1: 0.0, 2: 0.0}, 9)
    assert sum(split.values()) == 9
    assert max(split.values()) - min(split.values()) <= 1
    assert split == split_budget_by_uncertainty({0: 0.0, 1: 0.0, 2: 0.0}, 9)


def test_trake_event_uncertainty_flags_the_thin_event() -> None:
    by_event = {
        0: [candidate("A", 1.0), candidate("B", 0.4), candidate("C", 0.3)],
        1: [candidate("A", 1.0)],
    }
    signals = {item.event_index: item for item in trake_event_uncertainty(by_event, expanded=[1])}
    assert signals[1].uncertainty > signals[0].uncertainty
    assert signals[1].required_expansion is True
    assert signals[0].video_coverage == 3


def test_event_uncertainty_is_bounded() -> None:
    worst = EventUncertainty(event_index=0, video_coverage=0, score_margin=1.0, required_expansion=True)
    assert 0.0 <= worst.uncertainty <= 1.0


# ------------------------------------------------------------------ configuration


def test_adaptive_budget_is_off_by_default() -> None:
    assert AdaptiveBudgetConfig().enabled is False


def test_nested_yaml_shape_is_accepted() -> None:
    config = app_config_from_dict(
        {
            "aic2026": {
                "adaptive_budget": {
                    "enabled": True,
                    "max_cost_units": 100,
                    "official_grid": {"enabled": False, "neighbors": 2},
                    "progressive_video": {"stage_frames": [4, 4], "stop_margin": 0.2},
                    "trake_weakest_event": {"event_frame_cap": 12},
                }
            }
        }
    )
    budget = config.adaptive_budget
    assert budget.enabled is True
    assert budget.official_grid_enabled is False and budget.official_grid_neighbors == 2
    assert budget.progressive_stage_frames == (4, 4)
    assert budget.progressive_stop_margin == 0.2
    assert budget.trake_event_frame_cap == 12


@pytest.mark.parametrize(
    "override",
    [
        {"max_cost_units": 0},
        {"max_cost_units": -1},
        {"official_grid": {"neighbors": 0}},
        {"progressive_video": {"stage_frames": []}},
        {"progressive_video": {"stage_frames": [0]}},
        {"progressive_video": {"stop_margin": 1.5}},
        {"trake_weakest_event": {"event_frame_cap": 0}},
    ],
)
def test_unbounded_or_meaningless_budgets_are_rejected(override) -> None:
    with pytest.raises(ConfigError):
        app_config_from_dict({"aic2026": {"adaptive_budget": override}})


def test_every_budget_knob_is_consumed_by_code() -> None:
    """No dead knobs: each field is read somewhere in the engine or its modules."""
    import dataclasses
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "aic2026").glob("*.py")
        if path.name != "config.py"
    )
    for field in dataclasses.fields(AdaptiveBudgetConfig):
        assert field.name in sources, f"adaptive_budget.{field.name} is never read"


# ------------------------------------------------------- disabled == B0_CLEAN


def test_disabled_controller_leaves_no_trace_in_kis(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    diagnostics = engine.search_kis_detailed("a", top_k=10).diagnostics()
    assert diagnostics["adaptive_budget"] == {"enabled": False}


def test_disabled_controller_leaves_no_trace_in_trake(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    diagnostics = engine.search_trake_detailed(["a", "b"], max_results=5).diagnostics
    assert diagnostics["adaptive_budget"] == {"enabled": False}


def test_enabling_the_controller_does_not_change_kis_rows(tmp_path) -> None:
    """R1's shipped stages are evidence-only, so the submitted rows must not move."""
    baseline, _, _ = build_engine(tmp_path / "off")
    adaptive, _, _ = build_engine(
        tmp_path / "on", adaptive_budget={"enabled": True, "cutoff_aware": {"enabled": False}}
    )
    assert [p.row() for p in baseline.search_kis("a", top_k=20)] == [
        p.row() for p in adaptive.search_kis("a", top_k=20)
    ]


def test_enabled_controller_reports_its_spending(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path, adaptive_budget={"enabled": True})
    budget = engine.search_kis_detailed("a", top_k=10).diagnostics()["adaptive_budget"]
    assert budget["enabled"] is True
    assert 0.0 <= budget["uncertainty"]["uncertainty"] <= 1.0
    assert budget["actions"]["spent"] <= budget["actions"]["max_cost_units"]


def test_enabled_controller_respects_a_tiny_budget(tmp_path) -> None:
    engine, _, _ = build_engine(
        tmp_path, adaptive_budget={"enabled": True, "max_cost_units": 0.001}
    )
    budget = engine.search_kis_detailed("a", top_k=10).diagnostics()["adaptive_budget"]
    assert budget["actions"]["spent"] <= 0.001
    assert budget["official_grid"] is None
    assert budget["actions"]["actions_rejected"] >= 1
