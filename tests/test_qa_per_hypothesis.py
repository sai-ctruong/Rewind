"""The Phase 6 regression: one answer per video hypothesis, never copied across videos.

Before Phase 6, `answer_qa` answered once for the globally top candidate and attached
that answer to every prediction row. The fixture here makes such a copy impossible to
miss: one video is red, the other is blue, and the fake backend answers from pixels.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.engine import AICCompetitionEngine
from aic2026.qa import (
    ANSWER_STATUS_ABSTAINED,
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_BACKEND_FAILED,
    QAFrameHypothesis,
    group_hypotheses_by_video,
)
from aic2026.text_encoder import HashingTextEncoder
from tests.qa_support import (
    BrokenQAAnswerer,
    FakeVisualQAAnswerer,
    ScriptedQAAnswerer,
    make_qa_config,
    make_qa_root,
)


def build(tmp_path: Path, answerer, **qa):
    root = make_qa_root(tmp_path / "data")
    config = make_qa_config(root, tmp_path / "cache", tmp_path / "frames", **qa)
    engine, load = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        qa_answerer=answerer,
    )
    return engine, load


# ------------------------------------------------------------- answer isolation


def test_two_videos_produce_their_own_answers(tmp_path: Path) -> None:
    backend = FakeVisualQAAnswerer()
    engine, _ = build(tmp_path, backend)
    predictions, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)

    assert len(info["hypotheses"]) >= 2, "several video hypotheses must be answered"
    by_video = {item["video_id"]: item["normalized_answer"] for item in info["hypotheses"]}
    assert by_video["L21_V001"] == "red"
    assert by_video["L21_V002"] == "blue"
    # The backend was invoked once per video, not once in total.
    assert sorted(backend.calls) == ["L21_V001", "L21_V002"]
    assert predictions


def test_an_answer_is_never_copied_to_another_video(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    predictions, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)
    expected = {"L21_V001": "red", "L21_V002": "blue"}
    for prediction in predictions:
        assert prediction.answer == expected[prediction.video_id], (
            f"{prediction.video_id} carries {prediction.answer!r}, which belongs to "
            "another video"
        )
        assert prediction.qa["answer_video_id"] == prediction.video_id
    assert {p.answer for p in predictions} == {"red", "blue"}


def test_every_evidence_frame_belongs_to_its_prediction_video(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    predictions, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)
    assert predictions
    for prediction in predictions:
        evidence = prediction.qa["evidence"]
        assert evidence
        assert {item["video_id"] for item in evidence} == {prediction.video_id}


def test_cross_video_answer_copy_count_is_zero(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)
    assert info["diagnostics"]["cross_video_answer_copy_count"] == 0


def test_answer_without_matching_evidence_video_count_is_zero(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)
    assert info["diagnostics"]["answer_without_matching_evidence_video_count"] == 0
    assert info["diagnostics"]["distinct_answer_videos"] >= 2


def test_a_backend_answering_the_wrong_video_is_rejected(tmp_path: Path) -> None:
    class Liar(ScriptedQAAnswerer):
        def answer(self, question, evidence, *, expected_answer_type=None):
            result = super().answer(question, evidence, expected_answer_type=expected_answer_type)
            return type(result)(**{**result.to_dict(), "video_id": "SOMEWHERE_ELSE"})

    engine, _ = build(tmp_path, Liar({"L21_V001": "red", "L21_V002": "blue"}))
    with pytest.raises(RuntimeError, match="when asked about"):
        engine.answer_qa("a vehicle", "What colour is it?", top_k=10)


# -------------------------------------------------------------------- grouping


def test_candidates_group_by_video_with_bounded_budgets() -> None:
    class Candidate:
        def __init__(self, keyframe_id, video_id, timestamp, score):
            self.keyframe_id = keyframe_id
            self.video_id = video_id
            self.timestamp = timestamp
            self.score = score

    candidates = [
        Candidate("A/1", "A", 0.0, 0.90),
        Candidate("A/2", "A", 5.0, 0.80),
        Candidate("A/3", "A", 9.0, 0.70),
        Candidate("B/1", "B", 1.0, 0.85),
        Candidate("C/1", "C", 2.0, 0.60),
        Candidate("D/1", "D", 3.0, 0.10),
    ]
    hypotheses = group_hypotheses_by_video(
        candidates,
        top_video_hypotheses=3,
        frame_hypotheses_per_video=2,
        diversity_s=1.0,
        support_bonus=0.05,
    )
    assert [item.video_id for item in hypotheses] == ["A", "B", "C"]
    assert all(len(item.frames) <= 2 for item in hypotheses)
    assert hypotheses[0].support_count == 2
    assert hypotheses[0].retrieval_score > hypotheses[0].best_candidate_score
    assert [item.rank for item in hypotheses] == [1, 2, 3]


def test_support_bonus_cannot_invert_a_clear_retrieval_lead() -> None:
    class Candidate:
        def __init__(self, keyframe_id, video_id, timestamp, score):
            self.keyframe_id, self.video_id = keyframe_id, video_id
            self.timestamp, self.score = timestamp, score

    candidates = [Candidate("A/1", "A", 0.0, 0.90)] + [
        Candidate(f"B/{i}", "B", float(i), 0.50) for i in range(8)
    ]
    hypotheses = group_hypotheses_by_video(
        candidates, top_video_hypotheses=2, frame_hypotheses_per_video=1,
        diversity_s=0.5, support_bonus=0.05,
    )
    assert hypotheses[0].video_id == "A"


def test_frame_hypotheses_are_temporally_diverse() -> None:
    class Candidate:
        def __init__(self, keyframe_id, video_id, timestamp, score):
            self.keyframe_id, self.video_id = keyframe_id, video_id
            self.timestamp, self.score = timestamp, score

    # Four near-identical adjacent keyframes plus one far away.
    candidates = [
        Candidate("A/1", "A", 10.0, 0.90),
        Candidate("A/2", "A", 10.1, 0.89),
        Candidate("A/3", "A", 10.2, 0.88),
        Candidate("A/4", "A", 10.3, 0.87),
        Candidate("A/5", "A", 40.0, 0.60),
    ]
    hypotheses = group_hypotheses_by_video(
        candidates, top_video_hypotheses=1, frame_hypotheses_per_video=2,
        diversity_s=2.0, support_bonus=0.0,
    )
    chosen = [frame.timestamp for frame in hypotheses[0].frames]
    assert chosen == [10.0, 40.0], "adjacent duplicates must not consume the budget"


def test_top_video_hypotheses_is_enforced_end_to_end(tmp_path: Path) -> None:
    backend = FakeVisualQAAnswerer()
    engine, _ = build(tmp_path, backend, top_video_hypotheses=1)
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)
    assert len(info["hypotheses"]) == 1
    assert len(set(backend.calls)) == 1


# ------------------------------------------------------------------- failures


def test_a_backend_failure_is_confined_to_one_hypothesis(tmp_path: Path) -> None:
    backend = FakeVisualQAAnswerer(fail_for=["L21_V002"])
    engine, _ = build(tmp_path, backend)
    predictions, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=50)
    by_video = {item["video_id"]: item for item in info["hypotheses"]}
    assert by_video["L21_V001"]["answer_status"] == ANSWER_STATUS_ANSWERED
    assert by_video["L21_V001"]["normalized_answer"] == "red"
    assert by_video["L21_V002"]["answer_status"] == ANSWER_STATUS_BACKEND_FAILED
    assert info["diagnostics"]["backend_failures"] == 1
    # The failure must not become a confident answer, and must not borrow V001's.
    failed = [p for p in predictions if p.video_id == "L21_V002"]
    assert failed and all(p.answer == "unknown" for p in failed)


def test_a_total_backend_failure_never_fabricates_an_answer(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, BrokenQAAnswerer())
    predictions, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert info["diagnostics"]["backend_failures"] == len(info["hypotheses"])
    assert all(p.answer == "unknown" for p in predictions)
    assert all(p.qa["answer_status"] == ANSWER_STATUS_BACKEND_FAILED for p in predictions)
    assert all("backend failed" in (p.qa["warning"] or "").lower() for p in predictions)


def test_empty_answers_abstain_rather_than_look_confident(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, ScriptedQAAnswerer({}, default=""))
    predictions, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert info["diagnostics"]["abstentions"] == len(info["hypotheses"])
    assert all(p.qa["answer_status"] == ANSWER_STATUS_ABSTAINED for p in predictions)
    assert all(p.answer == "unknown" for p in predictions)


def test_abstention_can_be_disabled(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, ScriptedQAAnswerer({}, default=""), abstain_enabled=False)
    _, info = engine.answer_qa("a vehicle", "What colour is it?", top_k=20)
    assert info["diagnostics"]["abstentions"] == 0


# ------------------------------------------------------------ output and order


def test_duplicate_prediction_rows_are_removed(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    predictions, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=100)
    keys = [(p.video_id, p.frame_id, p.answer) for p in predictions]
    assert len(keys) == len(set(keys))


def test_max_answers_is_capped_and_ordering_is_deterministic(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    first, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=3)
    second, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=3)
    assert len(first) <= 3
    assert [(p.video_id, p.frame_id, p.answer) for p in first] == [
        (p.video_id, p.frame_id, p.answer) for p in second
    ]
    everything, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=100)
    assert len(everything) <= 100


def test_score_decomposition_is_visible(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    predictions, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=10)
    breakdown = predictions[0].score_breakdown
    assert set(breakdown) == {
        "video_retrieval", "frame_retrieval", "answer_reliability", "qa_score"
    }
    assert 0.0 <= breakdown["answer_reliability"] <= 1.0


def test_reliability_only_nudges_the_retrieval_order(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer(), answer_reliability_weight=0.2)
    predictions, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=100)
    for prediction in predictions:
        parts = prediction.score_breakdown
        # The multiplier is bounded by +/- weight/2 = 10%.
        ratio = parts["qa_score"] / max(parts["frame_retrieval"], 1e-9)
        assert 0.9 <= ratio <= 1.1


def test_submission_frame_stays_the_official_mapped_frame(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    predictions, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=100)
    official = {"5", "15", "25"}
    for prediction in predictions:
        assert prediction.frame_id in official
        assert prediction.qa["submission_frame_idx"] == int(prediction.frame_id)
        assert prediction.qa["coarse_official_frame_idx"] == int(prediction.frame_id)
        assert prediction.row()[:2] == [prediction.video_id, prediction.frame_id]


def test_prediction_rows_carry_video_frame_answer(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    predictions, _ = engine.answer_qa("a vehicle", "What colour is it?", top_k=10)
    for prediction in predictions:
        row = prediction.row()
        assert len(row) == 3
        assert row[2] == prediction.answer


def test_no_candidates_returns_an_honest_empty_result(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeVisualQAAnswerer())
    engine.search_candidates = lambda *args, **kwargs: []
    predictions, info = engine.answer_qa("nothing", "What colour is it?", top_k=10)
    assert predictions == []
    assert info["answer_normalized"] == "unknown"
    assert info["diagnostics"]["cross_video_answer_copy_count"] == 0
