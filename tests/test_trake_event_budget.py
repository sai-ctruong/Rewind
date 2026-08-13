"""R1: give TRAKE's optional budget to the structurally weakest event.

The organizer awards zero for the wrong video, so finding a complete video hypothesis
comes first and R1 does not touch it. What is allocated here is the *optional* refinement
budget after such a hypothesis exists — and every Phase 7/8 invariant must survive it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aic2026.budget import EventUncertainty, split_budget_by_uncertainty, trake_event_uncertainty
from tests.release_support import build_engine


def candidate(video_id: str, score: float):
    return SimpleNamespace(video_id=video_id, score=score, keyframe_id=f"{video_id}/k{score}")


# ------------------------------------------------------------- event uncertainty


def test_an_event_reaching_fewer_videos_is_more_uncertain() -> None:
    by_event = {
        0: [candidate("A", 1.0), candidate("B", 0.6), candidate("C", 0.5), candidate("D", 0.4)],
        1: [candidate("A", 1.0)],
    }
    signals = {item.event_index: item.uncertainty for item in trake_event_uncertainty(by_event)}
    assert signals[1] > signals[0]


def test_a_photo_finish_event_is_more_uncertain_than_a_clear_one() -> None:
    by_event = {
        0: [candidate("A", 1.0), candidate("B", 0.1)],
        1: [candidate("A", 1.0), candidate("B", 0.99)],
    }
    signals = {item.event_index: item.uncertainty for item in trake_event_uncertainty(by_event)}
    assert signals[1] > signals[0]


def test_an_event_that_needed_expansion_is_flagged() -> None:
    by_event = {0: [candidate("A", 1.0)], 1: [candidate("A", 1.0)]}
    signals = {item.event_index: item for item in trake_event_uncertainty(by_event, expanded=[1])}
    assert signals[1].required_expansion is True
    assert signals[0].required_expansion is False
    assert signals[1].uncertainty > signals[0].uncertainty


def test_uncertainty_payload_shows_its_inputs() -> None:
    payload = EventUncertainty(event_index=2, video_coverage=3, candidate_count=9).to_dict()
    assert payload["event_index"] == 2
    assert payload["video_coverage"] == 3
    assert 0.0 <= payload["uncertainty"] <= 1.0


# ---------------------------------------------------------------- the allocation


def test_the_allocation_sums_to_the_cap() -> None:
    split = split_budget_by_uncertainty({0: 0.2, 1: 0.5, 2: 0.3}, 24, maximum=24)
    assert sum(split.values()) == 24


def test_no_event_can_consume_everything() -> None:
    split = split_budget_by_uncertainty({0: 1.0, 1: 0.0, 2: 0.0}, 30, maximum=10)
    assert split[0] <= 10
    assert sum(split.values()) == 30


def test_the_allocation_is_deterministic() -> None:
    weights = {0: 0.31, 1: 0.33, 2: 0.36}
    assert split_budget_by_uncertainty(weights, 25, maximum=20) == split_budget_by_uncertainty(
        weights, 25, maximum=20
    )


def test_every_event_can_receive_a_floor() -> None:
    split = split_budget_by_uncertainty({0: 1.0, 1: 0.0}, 10, minimum=2, maximum=10)
    assert split[1] >= 2
    assert sum(split.values()) == 10


# -------------------------------------------------------------------- in engine


def test_disabled_budget_reports_nothing(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    diagnostics = engine.search_trake_detailed(["a", "b"], max_results=5).diagnostics
    assert diagnostics["adaptive_budget"] == {"enabled": False}


def test_enabled_budget_allocates_per_event(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path, adaptive_budget={"enabled": True})
    diagnostics = engine.search_trake_detailed(["a", "b", "c"], max_results=10).diagnostics
    budget = diagnostics["adaptive_budget"]
    assert budget["enabled"] is True and budget["trake_weakest_event"] is True
    allocation = budget["frame_budget_by_event"]
    assert set(allocation) == {0, 1, 2}
    assert sum(allocation.values()) == budget["frame_budget_total"]
    assert all(value <= budget["event_frame_cap"] for value in allocation.values())


def test_the_weakest_event_is_named_and_gets_the_most(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path, adaptive_budget={"enabled": True})
    budget = engine.search_trake_detailed(["a", "b", "c"], max_results=10).diagnostics[
        "adaptive_budget"
    ]
    weakest = budget["weakest_event_index"]
    assert weakest in {0, 1, 2}
    allocation = budget["frame_budget_by_event"]
    assert allocation[weakest] == max(allocation.values())


def test_allocation_never_claims_an_improvement(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path, adaptive_budget={"enabled": True})
    budget = engine.search_trake_detailed(["a", "b"], max_results=5).diagnostics["adaptive_budget"]
    assert "not, without" in budget["note"]
    assert "improve" in budget["note"]  # only as the thing it refuses to claim


# ------------------------------------------------------ Phase 7/8 invariants hold


def test_every_event_still_appears_in_every_row(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path, adaptive_budget={"enabled": True})
    outcome = engine.search_trake_detailed(["a", "b", "c"], max_results=20)
    for prediction in outcome.predictions:
        row = prediction.row()
        assert len(row) == 1 + 3  # video_id + one frame per event
    structural = outcome.structural_summary()
    assert structural["malformed_prediction_count"] == 0
    assert structural["wrong_event_count_prediction_count"] == 0
    assert structural["cross_video_step_count"] == 0


def test_enabling_the_budget_does_not_change_the_sequences(tmp_path) -> None:
    """R1's TRAKE stage allocates; it does not re-rank."""
    baseline, _, _ = build_engine(tmp_path / "off")
    adaptive, _, _ = build_engine(tmp_path / "on", adaptive_budget={"enabled": True})
    left = [p.row() for p in baseline.search_trake_detailed(["a", "b"], max_results=20).predictions]
    right = [p.row() for p in adaptive.search_trake_detailed(["a", "b"], max_results=20).predictions]
    assert left == right


def test_a_disabled_weakest_event_stage_still_reports_itself(tmp_path) -> None:
    engine, _, _ = build_engine(
        tmp_path, adaptive_budget={"enabled": True, "trake_weakest_event": {"enabled": False}}
    )
    budget = engine.search_trake_detailed(["a", "b"], max_results=5).diagnostics["adaptive_budget"]
    assert budget == {"enabled": True, "trake_weakest_event": False}
