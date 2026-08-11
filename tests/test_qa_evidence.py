"""Evidence selection, visual sources, budgets, and the Q&A refinement budget.

The pre-Phase-6 selector seeded its choice with the first and last frames of the window,
so asking for one or two frames returned window boundaries instead of the evidence the
retriever actually liked. These tests pin the replacement behaviour down.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.engine import AICCompetitionEngine
from aic2026.qa import (
    ANSWER_STATUS_VISUAL_UNAVAILABLE,
    QAEvidenceBundle,
    QAEvidenceFrame,
    QAFrameHypothesis,
    select_diverse_evidence,
    select_evidence_frames,
    select_temporally_diverse,
)
from aic2026.text_encoder import HashingTextEncoder
from ingestion.schemas import KeyframeRecord
from tests.qa_support import (
    FakeVisualQAAnswerer,
    ScriptedQAAnswerer,
    colour_of_jpeg,
    make_qa_config,
    make_qa_root,
)


def frame(timestamp: float, score: float, *, video_id: str = "V", available: bool = True):
    return QAEvidenceFrame(
        video_id=video_id,
        frame_idx=int(timestamp * 10),
        timestamp=timestamp,
        source="keyframe_jpeg",
        keyframe_id=f"{video_id}/kf_{int(timestamp * 10):06d}",
        retrieval_score=score,
        image_available=available,
    )


def build(tmp_path: Path, answerer, **qa):
    root = make_qa_root(tmp_path / "data")
    config = make_qa_config(root, tmp_path / "cache", tmp_path / "frames", **qa)
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        qa_answerer=answerer,
    )
    return engine


# ----------------------------------------------------------- selection policy


def test_one_frame_evidence_picks_the_strongest_not_a_boundary() -> None:
    pool = [frame(0.0, 0.1), frame(5.0, 0.9), frame(10.0, 0.2)]
    selected = select_evidence_frames(pool, count=1, diversity_s=1.0)
    assert len(selected) == 1
    assert selected[0].timestamp == 5.0
    assert selected[0].role == "primary"


def test_two_frame_evidence_is_strongest_plus_diverse_context() -> None:
    pool = [frame(5.0, 0.9), frame(5.2, 0.85), frame(9.0, 0.6), frame(0.0, 0.1)]
    selected = select_evidence_frames(pool, count=2, diversity_s=1.0)
    assert [item.timestamp for item in selected] == [5.0, 9.0]
    # The 5.2s frame is nearly as strong but describes the same instant.
    assert 5.2 not in {item.timestamp for item in selected}
    assert [item.role for item in selected] == ["primary", "after"]


def test_three_frame_evidence_straddles_the_event() -> None:
    pool = [frame(5.0, 0.9), frame(9.0, 0.7), frame(1.0, 0.6), frame(5.1, 0.8)]
    selected = select_evidence_frames(pool, count=3, diversity_s=1.0)
    times = [item.timestamp for item in selected]
    assert times == [1.0, 5.0, 9.0]
    assert [item.role for item in selected] == ["before", "primary", "after"]


def test_evidence_is_returned_in_timestamp_order() -> None:
    pool = [frame(9.0, 0.5), frame(1.0, 0.9), frame(5.0, 0.7)]
    selected = select_evidence_frames(pool, count=3, diversity_s=0.5)
    assert [item.timestamp for item in selected] == [1.0, 5.0, 9.0]


def test_evidence_budget_is_never_exceeded_and_never_under_filled() -> None:
    pool = [frame(float(i), 1.0 - 0.01 * i) for i in range(20)]
    assert len(select_evidence_frames(pool, count=4, diversity_s=1.0)) == 4
    # Diversity is relaxed rather than returning fewer frames than requested.
    crowded = [frame(5.0 + 0.01 * i, 1.0 - 0.01 * i) for i in range(6)]
    assert len(select_evidence_frames(crowded, count=4, diversity_s=5.0)) == 4
    assert select_evidence_frames([], count=3, diversity_s=1.0) == ()


def test_selection_is_deterministic() -> None:
    pool = [frame(float(i), 0.5) for i in range(8)]
    first = select_evidence_frames(pool, count=3, diversity_s=1.0)
    second = select_evidence_frames(pool, count=3, diversity_s=1.0)
    assert [f.evidence_id for f in first] == [f.evidence_id for f in second]


def test_temporal_diversity_helper_spreads_frame_hypotheses() -> None:
    frames = [
        QAFrameHypothesis(f"V/{i}", "V", i, 10.0 + 0.1 * i, 0.9 - 0.01 * i) for i in range(5)
    ] + [QAFrameHypothesis("V/far", "V", 400, 40.0, 0.5)]
    chosen = select_temporally_diverse(frames, count=2, diversity_s=2.0)
    assert [item.timestamp for item in chosen] == [10.0, 40.0]


def test_legacy_record_selector_prefers_the_query_relevant_frame() -> None:
    records = [
        KeyframeRecord(str(i), "V", float(i), np.ones(2), objects=["car"] if i == 5 else [])
        for i in range(12)
    ]
    selected = select_diverse_evidence(records, 5.0, count=5, query="car")
    assert len(selected) == 5
    assert [item.record.timestamp for item in selected] == sorted(
        item.record.timestamp for item in selected
    )
    assert any(item.record.id == "5" for item in selected)
    # The strongest frame is chosen for being strongest, not for being first or last.
    assert next(item for item in selected if item.record.id == "5").relevance > 0


def test_bundle_refuses_evidence_from_another_video() -> None:
    with pytest.raises(ValueError, match="never cross videos"):
        QAEvidenceBundle(
            video_id="A",
            question="q",
            frames=(frame(1.0, 0.5, video_id="A"), frame(2.0, 0.5, video_id="B")),
        )


# ------------------------------------------------------------- visual sources


def test_jpeg_and_mp4_backed_videos_both_supply_evidence(tmp_path: Path) -> None:
    backend = FakeVisualQAAnswerer()
    engine = build(tmp_path, backend)
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    by_video = {item["video_id"]: item for item in info["hypotheses"]}
    sources = {
        video: {item["source"] for item in payload["evidence"]}
        for video, payload in by_video.items()
    }
    # L21_V001 has BTC JPEGs; L21_V002 has none and must decode its MP4.
    assert "keyframe_jpeg" in sources["L21_V001"]
    assert "video_decode" in sources["L21_V002"]
    assert all(payload["visual_evidence_loaded"] > 0 for payload in by_video.values())


def test_evidence_images_are_the_right_video_pixels(tmp_path: Path) -> None:
    engine = build(tmp_path, FakeVisualQAAnswerer())
    hypotheses = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)[1]["hypotheses"]
    for payload in hypotheses:
        expected = {"L21_V001": "red", "L21_V002": "blue"}[payload["video_id"]]
        assert payload["normalized_answer"] == expected


def test_visual_unavailable_is_explicit(tmp_path: Path) -> None:
    engine = build(tmp_path, FakeVisualQAAnswerer())
    # Simulate a dataset with no readable pixels at all.
    engine._qa_image_bytes = lambda frame: None
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    statuses = {item["answer_status"] for item in info["hypotheses"]}
    assert statuses == {ANSWER_STATUS_VISUAL_UNAVAILABLE}
    assert info["diagnostics"]["visual_unavailable"] == len(info["hypotheses"])
    assert info["diagnostics"]["frame_decode_failures"] > 0


def test_a_non_visual_backend_never_triggers_a_decode(tmp_path: Path) -> None:
    engine = build(tmp_path, ScriptedQAAnswerer({}, default="grey", visual=False))
    calls: list[str] = []
    original = engine._qa_image_bytes
    engine._qa_image_bytes = lambda frame: (calls.append(frame.evidence_id), original(frame))[1]
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert calls == [], "a text-only backend must not pay for image decoding"
    # Availability is still reported honestly, from a cheap check.
    assert any(item["visual_available"] for item in info["hypotheses"])
    assert all(item["visual_evidence_loaded"] == 0 for item in info["hypotheses"])


def test_evidence_frame_count_bounds_the_backend_call(tmp_path: Path) -> None:
    backend = FakeVisualQAAnswerer()
    engine = build(tmp_path, backend, evidence_frame_count=2)
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert all(len(item["evidence"]) <= 2 for item in info["hypotheses"])
    assert max(backend.image_counts) <= 2
    assert info["diagnostics"]["evidence_frames_used"] <= 2 * len(info["hypotheses"])


# --------------------------------------------------------- refinement budget


def test_qa_refinement_uses_its_own_small_budget(tmp_path: Path) -> None:
    engine = build(
        tmp_path,
        FakeVisualQAAnswerer(),
        use_local_refinement=True,
        refinement_candidate_budget=1,
        refinement_max_frames=4,
    )
    refiner = engine._qa_refiner()
    # NOT Phase 5's KIS budget of 5 regions x 32 frames.
    assert refiner.config.candidate_budget == 1
    assert refiner.config.max_frames == 4
    assert refiner.config.mode == "always"
    assert engine.refinement_config.candidate_budget >= 1
    assert refiner.frame_provider is engine.frame_provider


def test_refinement_is_off_by_default_and_costs_nothing(tmp_path: Path) -> None:
    engine = build(tmp_path, FakeVisualQAAnswerer())
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert info["diagnostics"]["local_refinement_calls"] == 0
    assert info["diagnostics"]["refinement_ms"] == 0.0


def test_refinement_can_add_evidence_without_changing_the_submission_frame(
    tmp_path: Path,
) -> None:
    from aic2026.frame_scorer import ScorerStatus

    class ColourScorer:
        """Prefers frames later in the window, so the best frame is not the coarse one."""

        def prepare_query(self, query):
            return query

        def score_frames(self, prepared, frames):
            return [0.1 * index for index in range(len(frames))]

        def status(self, *, initialize: bool = False):
            return ScorerStatus(
                backend="fake", model_name="fake", device="cpu", state="ready", available=True
            )

    root = make_qa_root(tmp_path / "data")
    config = make_qa_config(
        root,
        tmp_path / "cache",
        tmp_path / "frames",
        use_local_refinement=True,
        refinement_candidate_budget=1,
        refinement_max_frames=4,
    )
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        qa_answerer=FakeVisualQAAnswerer(),
        frame_scorer=ColourScorer(),
    )
    predictions, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert info["diagnostics"]["local_refinement_calls"] == len(info["hypotheses"])
    refined = [item for item in info["hypotheses"] if item.get("refinement")]
    assert refined, "bounded refinement should have produced local evidence"
    assert any(
        entry["source"] == "local_refinement"
        for item in refined
        for entry in item["evidence"]
    )
    # The refined frame is evidence only: the submitted frame is still official.
    official = {"5", "15", "25"}
    assert all(prediction.frame_id in official for prediction in predictions)
