"""Event-local visual refinement of a TRAKE sequence, on tiny synthetic MP4s.

Each event must be scored against its OWN text, the work must stay inside a hard frame
budget, the refined view must not display an impossible event order, and the submitted
frames must not change. No accuracy is claimed anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.frame_provider import FrameProvider
from aic2026.frame_scorer import ScorerStatus
from aic2026.local_refinement import LocalRefinementFrame
from aic2026.trake import (
    STEP_ALIGNED,
    AlignmentConfig,
    EventCandidate,
    TrakeAlignedStep,
    TrakeAlignment,
    TrakeEvent,
    to_complete_prediction,
)
from aic2026.trake_refinement import (
    EVENT_REFINED,
    SEQUENCE_REFINED,
    SEQUENCE_UNAVAILABLE,
    FrameBudget,
    TrakeSequenceRefiner,
    apply_refinement,
    local_ordered_refinement,
)
from tests.refinement_support import recover_frame_idx, write_synthetic_video

FPS = 10.0
FRAMES = 31


class EventAwareScorer:
    """Prefers a different frame per event text, so per-event queries are observable.

    `preferences` maps event text to the frame index that text should like best. If a
    single shared query were used for the whole sequence, every event would return the
    same preferred frame and the tests below would fail.
    """

    def __init__(self, preferences: dict[str, int]):
        self.preferences = dict(preferences)
        self.queries: list[str] = []
        self.batch_sizes: list[int] = []

    def prepare_query(self, query: str):
        self.queries.append(str(query))
        return str(query)

    def score_frames(self, prepared_query, frames):
        self.batch_sizes.append(len(frames))
        target = self.preferences.get(str(prepared_query))
        scores = []
        for image in frames:
            index = recover_frame_idx(image)
            scores.append(1.0 if target is None else 1.0 - 0.01 * abs(index - target))
        return scores

    def status(self, *, initialize: bool = False) -> ScorerStatus:
        return ScorerStatus(
            backend="event_aware_fake",
            model_name="fake",
            device="cpu",
            state="ready",
            available=True,
        )


class BrokenScorer:
    def prepare_query(self, query: str):
        raise RuntimeError("scorer unavailable")

    def score_frames(self, prepared_query, frames):  # pragma: no cover - never reached
        raise AssertionError("must not be called")

    def status(self, *, initialize: bool = False) -> ScorerStatus:
        return ScorerStatus(backend="broken", model_name="x", device="cpu", state="unavailable")


def candidate(event: int, video: str, frame: int, time: float, score: float = 0.9):
    return EventCandidate(event, video, f"{video}/kf_{frame:06d}", str(frame), time, score)


def sequence(video: str, coarse: list[int], texts: list[str]):
    steps = tuple(
        TrakeAlignedStep(
            event_index=index,
            event_text=texts[index],
            video_id=video,
            status=STEP_ALIGNED,
            candidate=candidate(index, video, frame, frame / FPS),
        )
        for index, frame in enumerate(coarse)
    )
    alignment = TrakeAlignment(
        video_id=video,
        events=tuple(TrakeEvent(i, texts[i]) for i in range(len(coarse))),
        steps=steps,
        score=1.0,
    )
    prediction = to_complete_prediction(alignment, AlignmentConfig(min_gap_s=0.0))
    assert prediction is not None
    return prediction


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    write_synthetic_video(tmp_path / "video" / "V1.mp4", frames=FRAMES, fps=FPS)
    return tmp_path


def refiner_for(root: Path, scorer, **overrides) -> TrakeSequenceRefiner:
    settings = dict(
        refinement_enabled=True,
        refinement_frames_per_event=5,
        refinement_fine_fps=5.0,
        refinement_window_s=0.5,
        refinement_max_events_per_alignment=4,
        refinement_max_frames_per_query=64,
        min_gap_s=0.0,
    )
    settings.update(overrides)
    return TrakeSequenceRefiner(
        AlignmentConfig(**settings), frame_provider=FrameProvider(root), scorer=scorer
    )


TEXTS = ["a person appears", "the person moves", "the person leaves"]


# ------------------------------------------------------------- event queries


def test_each_event_is_scored_against_its_own_text(root: Path) -> None:
    # Event 0 prefers an earlier frame, event 1 its coarse frame, event 2 a later one.
    scorer = EventAwareScorer({TEXTS[0]: 8, TEXTS[1]: 15, TEXTS[2]: 25})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    outcome = refiner_for(root, scorer).refine(prediction, FrameBudget(limit=64))
    assert outcome.status == SEQUENCE_REFINED
    # One prepared query per event, each the event's own text -- never the sentence.
    assert scorer.queries == TEXTS
    chosen = {item.event_index: item.best_visual_frame_idx for item in outcome.events}
    assert chosen[0] < 10, "event 0 should move earlier"
    assert chosen[2] > 22, "event 2 should move later"


def test_frames_are_scored_in_one_batch_per_event(root: Path) -> None:
    scorer = EventAwareScorer({text: 15 for text in TEXTS})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    refiner_for(root, scorer).refine(prediction, FrameBudget(limit=64))
    # Three events, three batched calls, never one call per frame.
    assert len(scorer.batch_sizes) == 3
    assert all(size > 1 for size in scorer.batch_sizes)


def test_refine_window_s_changes_the_sample_window(root: Path) -> None:
    prediction = sequence("V1", [15, 20, 25], TEXTS)
    narrow = refiner_for(root, EventAwareScorer({}), refinement_window_s=0.2)
    wide = TrakeSequenceRefiner(
        narrow.config, frame_provider=FrameProvider(root), scorer=EventAwareScorer({}),
        window_s=1.5,
    )
    assert narrow.window_s == 0.2
    assert wide.window_s == 1.5
    narrow_out = narrow.refine(prediction, FrameBudget(limit=64))
    wide_out = wide.refine(prediction, FrameBudget(limit=64))
    narrow_frames = max(item.frames_sampled for item in narrow_out.events)
    wide_frames = max(item.frames_sampled for item in wide_out.events)
    assert wide_frames > narrow_frames, "a wider window must sample more frames"


# ------------------------------------------------------------------- budgets


def test_the_per_query_frame_budget_is_hard(root: Path) -> None:
    scorer = EventAwareScorer({})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    budget = FrameBudget(limit=6)
    outcome = refiner_for(root, scorer).refine(prediction, budget)
    assert budget.used <= 6
    assert outcome.frames_decoded <= 6


def test_only_the_configured_number_of_events_is_refined(root: Path) -> None:
    scorer = EventAwareScorer({})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    outcome = refiner_for(root, scorer, refinement_max_events_per_alignment=2).refine(
        prediction, FrameBudget(limit=64)
    )
    refined = [item for item in outcome.events if item.status == EVENT_REFINED]
    assert len(refined) <= 2
    assert outcome.events[2].status == "not_selected"


def test_frames_per_event_is_respected(root: Path) -> None:
    scorer = EventAwareScorer({})
    prediction = sequence("V1", [15, 20, 25], TEXTS)
    outcome = refiner_for(
        root, scorer, refinement_frames_per_event=3, refinement_window_s=2.0
    ).refine(prediction, FrameBudget(limit=64))
    assert all(item.frames_sampled <= 3 for item in outcome.events)


# ---------------------------------------------------------------- fallbacks


def test_a_missing_mp4_preserves_the_coarse_sequence(tmp_path: Path) -> None:
    scorer = EventAwareScorer({})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    outcome = refiner_for(tmp_path, scorer).refine(prediction, FrameBudget(limit=64))
    assert outcome.applied is False
    assert outcome.final_sequence_score == pytest.approx(outcome.coarse_alignment_score)
    unchanged = apply_refinement(prediction, outcome)
    assert unchanged.frame_ids == prediction.frame_ids


def test_an_unavailable_scorer_preserves_the_coarse_sequence(root: Path) -> None:
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    outcome = refiner_for(root, BrokenScorer()).refine(prediction, FrameBudget(limit=64))
    assert outcome.applied is False
    assert apply_refinement(prediction, outcome).frame_ids == prediction.frame_ids


def test_no_scorer_at_all_is_reported_not_crashed(root: Path) -> None:
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    refiner = TrakeSequenceRefiner(
        AlignmentConfig(refinement_enabled=True), frame_provider=FrameProvider(root), scorer=None
    )
    outcome = refiner.refine(prediction, FrameBudget(limit=64))
    assert outcome.status == SEQUENCE_UNAVAILABLE
    assert outcome.warnings


# ---------------------------------------------------------------- ordering


def test_local_ordered_refinement_picks_the_best_ordered_path() -> None:
    per_event = [
        (
            LocalRefinementFrame(frame_idx=10, timestamp=1.0, score=0.9),
            LocalRefinementFrame(frame_idx=20, timestamp=2.0, score=0.1),
        ),
        (
            LocalRefinementFrame(frame_idx=30, timestamp=0.5, score=1.0),  # too early
            LocalRefinementFrame(frame_idx=40, timestamp=3.0, score=0.8),
        ),
    ]
    picks = local_ordered_refinement(per_event)
    assert picks == (0, 1)


def test_local_ordered_refinement_returns_none_when_nothing_is_ordered() -> None:
    per_event = [
        (LocalRefinementFrame(frame_idx=10, timestamp=9.0, score=1.0),),
        (LocalRefinementFrame(frame_idx=20, timestamp=1.0, score=1.0),),
    ]
    assert local_ordered_refinement(per_event) is None
    assert local_ordered_refinement([]) is None
    assert local_ordered_refinement([()]) is None


def test_a_reversed_independent_choice_is_resolved_into_an_ordered_one(root: Path) -> None:
    # Event 0 would independently prefer a LATE frame and event 1 an EARLY one, which
    # reversed would display an impossible reading of the sequence.
    scorer = EventAwareScorer({TEXTS[0]: 20, TEXTS[1]: 8})
    prediction = sequence("V1", [12, 16, 22], TEXTS)
    outcome = refiner_for(root, scorer, refinement_window_s=1.5).refine(
        prediction, FrameBudget(limit=64)
    )
    assert outcome.status == SEQUENCE_REFINED
    assert outcome.order_violation_detected is True
    assert outcome.order_violation_resolved is True
    times = [
        item.best_visual_timestamp
        for item in outcome.events
        if item.best_visual_timestamp is not None
    ]
    assert times == sorted(times), "the displayed refined order must stay monotonic"


def test_refined_visual_frames_are_reported_but_never_submitted(root: Path) -> None:
    scorer = EventAwareScorer({TEXTS[0]: 8, TEXTS[1]: 15, TEXTS[2]: 25})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    outcome = refiner_for(root, scorer).refine(prediction, FrameBudget(limit=64))
    updated = apply_refinement(prediction, outcome)
    # The row is byte-identical to the coarse one.
    assert updated.frame_ids == prediction.frame_ids == ("10", "15", "22")
    assert len(updated.frame_ids) == updated.event_count
    moved = [
        step for step in updated.steps
        if step.visual_frame_idx is not None
        and str(step.visual_frame_idx) != step.submission_frame_idx
    ]
    assert moved, "the fixture is meant to move at least one visual frame"
    for step in updated.steps:
        assert step.submission_frame_idx == step.coarse_official_frame_idx


def test_score_decomposition_after_refinement(root: Path) -> None:
    scorer = EventAwareScorer({TEXTS[0]: 8, TEXTS[1]: 15, TEXTS[2]: 25})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    config_alpha = 0.10
    outcome = refiner_for(root, scorer, refinement_rerank_alpha=config_alpha).refine(
        prediction, FrameBudget(limit=64)
    )
    updated = apply_refinement(prediction, outcome)
    payload = updated.to_dict()
    assert payload["coarse_alignment_score"] == pytest.approx(prediction.score)
    assert payload["final_sequence_score"] == pytest.approx(
        payload["coarse_alignment_score"]
        + config_alpha * max(-1.0, min(1.0, outcome.visual_gain_aggregate)),
        abs=1e-6,
    )
    assert payload["refinement_status"] == SEQUENCE_REFINED
    # Every component stays visible.
    assert set(payload) >= {
        "coarse_alignment_score", "visual_gain_aggregate", "final_sequence_score"
    }


def test_refinement_payload_has_no_filesystem_paths(root: Path) -> None:
    scorer = EventAwareScorer({})
    prediction = sequence("V1", [10, 15, 22], TEXTS)
    outcome = refiner_for(root, scorer).refine(prediction, FrameBudget(limit=64))
    assert str(root) not in str(outcome.to_dict())
