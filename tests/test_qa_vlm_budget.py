"""Q&A compute escalation: a hard VLM call budget, and no fabrication when it runs out.

A VLM call is the single most expensive action in this system. Calling one per candidate
is how a Q&A query becomes unaffordable, so the number of calls and the number of images
per call are both capped. Running out of budget produces an explicit non-answer, never a
guess — and with no visual backend at all, the cost trace must show zero calls.
"""
from __future__ import annotations

import pytest

from aic2026.qa import ANSWER_STATUS_BUDGET_EXHAUSTED
from aic2026.submission_validation import (
    NON_SUBMITTABLE_QA_STATUSES,
    submission_rows_for,
    validate_submission,
)
from aic2026.engine import AICCompetitionEngine
from aic2026.text_encoder import HashingTextEncoder
from tests.qa_support import FakeVisualQAAnswerer, make_qa_config, make_qa_root
from tests.release_support import build_engine


def build_qa(tmp_path, **qa):
    """A tiny multi-video Q&A engine with an injected VISUAL backend."""
    root = make_qa_root(tmp_path / "data")
    config = make_qa_config(root, tmp_path / "cache", tmp_path / "frames", **qa)
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        qa_answerer=FakeVisualQAAnswerer(),
    )
    return engine


# ------------------------------------------------------------------- the budget


def test_the_call_budget_is_configurable_and_bounded() -> None:
    from aic2026.config import ConfigError, app_config_from_dict

    config = app_config_from_dict({"aic2026": {"qa": {"max_vlm_calls_per_query": 3}}})
    assert config.qa.max_vlm_calls_per_query == 3
    with pytest.raises(ConfigError, match="unbounded VLM budget"):
        app_config_from_dict({"aic2026": {"qa": {"max_vlm_calls_per_query": 0}}})


def test_images_per_call_is_capped() -> None:
    from aic2026.config import ConfigError, app_config_from_dict

    with pytest.raises(ConfigError, match="max_visual_frames_per_call"):
        app_config_from_dict({"aic2026": {"qa": {"max_visual_frames_per_call": 0}}})


def test_calls_never_exceed_the_cap(tmp_path) -> None:
    engine = build_qa(tmp_path, top_video_hypotheses=4, max_vlm_calls_per_query=2)
    _, info = engine.answer_qa("a", "what colour?", top_k=20)
    budget = info["diagnostics"]["vlm_budget"]
    assert budget["max_vlm_calls_per_query"] == 2
    assert budget["vlm_calls_used"] <= 2
    assert info["diagnostics"]["cost"]["qa"]["vlm_calls"] <= 2


def test_hypotheses_beyond_the_budget_are_reported_not_dropped(tmp_path) -> None:
    engine = build_qa(tmp_path, top_video_hypotheses=4, max_vlm_calls_per_query=1)
    predictions, info = engine.answer_qa("a", "what colour?", top_k=20)
    budget = info["diagnostics"]["vlm_budget"]
    if budget["hypotheses_skipped_for_budget"]:
        statuses = {p.qa.get("answer_status") for p in predictions if p.qa}
        assert ANSWER_STATUS_BUDGET_EXHAUSTED in statuses


def test_a_budget_exhausted_row_is_never_submittable(tmp_path) -> None:
    assert ANSWER_STATUS_BUDGET_EXHAUSTED in NON_SUBMITTABLE_QA_STATUSES
    engine = build_qa(tmp_path, top_video_hypotheses=4, max_vlm_calls_per_query=1)
    predictions, info = engine.answer_qa("a", "what colour?", top_k=20)
    if not info["diagnostics"]["vlm_budget"]["hypotheses_skipped_for_budget"]:
        pytest.skip("this fixture produced fewer hypotheses than the budget")
    rows = submission_rows_for("qa", predictions)
    result = validate_submission("qa", rows)
    codes = {issue.code for issue in result.errors}
    assert not result.valid
    assert "QA_NON_SUBMITTABLE_STATUS" in codes


def test_a_budget_exhausted_row_carries_no_invented_answer(tmp_path) -> None:
    engine = build_qa(tmp_path, top_video_hypotheses=4, max_vlm_calls_per_query=1)
    predictions, _ = engine.answer_qa("a", "what colour?", top_k=20)
    for prediction in predictions:
        if prediction.qa and prediction.qa.get("answer_status") == ANSWER_STATUS_BUDGET_EXHAUSTED:
            assert not (prediction.answer or "").strip() or prediction.answer == "unknown"


def test_images_per_call_limits_the_evidence_sent(tmp_path) -> None:
    engine = build_qa(tmp_path, evidence_frame_count=8, max_visual_frames_per_call=2)
    _, info = engine.answer_qa("a", "what colour?", top_k=20)
    assert info["diagnostics"]["vlm_budget"]["max_visual_frames_per_call"] == 2
    for hypothesis in info["diagnostics"].get("hypotheses", []) or []:
        assert len(hypothesis.get("evidence", [])) <= 2


# ------------------------------------------------------- no backend, no calls


def test_a_non_visual_backend_spends_nothing(tmp_path) -> None:
    """The mock reasons over text. That is not a VLM call and is not counted as one."""
    engine, _, _ = build_engine(tmp_path)
    _, info = engine.answer_qa("a", "what colour?", top_k=5)
    budget = info["diagnostics"]["vlm_budget"]
    assert budget["backend_visual_capable"] is False
    assert budget["vlm_calls_used"] == 0
    assert info["diagnostics"]["cost"]["qa"]["vlm_calls"] == 0
    assert info["diagnostics"]["cost"]["qa"]["vlm_images"] == 0


def test_a_non_visual_backend_skips_nobody_for_budget(tmp_path) -> None:
    """The budget must not truncate a path that costs nothing to walk.

    Whether the mock then *answers* or abstains is its own business — that is a Phase 6
    behaviour, not a budget decision, and this test deliberately does not assert it.
    """
    engine, _, _ = build_engine(tmp_path)
    _, info = engine.answer_qa("a", "what colour?", top_k=5)
    diagnostics = info["diagnostics"]
    assert diagnostics["vlm_budget"]["hypotheses_skipped_for_budget"] == 0
    assert diagnostics["retrieved_video_hypotheses"] >= 1


def test_no_visual_backend_produces_no_exportable_answer(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    predictions, _ = engine.answer_qa("a", "what colour?", top_k=5)
    result = validate_submission("qa", submission_rows_for("qa", predictions))
    assert not result.valid
