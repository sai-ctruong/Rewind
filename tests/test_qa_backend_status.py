"""Backend capability reporting, runtime-state isolation, and the HTTP surface.

The rule these tests defend: a backend must never be described as more capable than it
is. The mock reasons over text and must report `visual_capable=False`; an unconfigured
local VLM must report `not_available` rather than looking production-ready; and no test
here downloads a model, contacts an API, or needs a key.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import aic2026.engine as engine_module
import ui.app as appmod
from aic2026.config import ConfigError, app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.qa import (
    BACKEND_STATE_NOT_AVAILABLE,
    BACKEND_STATE_NOT_LOADED,
    BACKEND_STATE_READY,
    ApiVqaAnswerer,
    LocalVlmQAAnswerer,
    MockTextQAAnswerer,
    QABackendUnavailable,
    QAEvidenceBundle,
    QAEvidenceFrame,
    VisualQAAnswerer,
    answer_reliability_score,
    build_qa_answerer,
)
from aic2026.text_encoder import HashingTextEncoder
from tests.qa_support import (
    FakeLocalVlm,
    FakeVisualQAAnswerer,
    ScriptedQAAnswerer,
    make_qa_config,
    make_qa_root,
)


def bundle(video_id: str = "V", *, with_image: bool = True) -> QAEvidenceBundle:
    return QAEvidenceBundle(
        video_id=video_id,
        question="What colour is it?",
        frames=(
            QAEvidenceFrame(
                video_id=video_id,
                frame_idx=10,
                timestamp=1.0,
                source="keyframe_jpeg",
                keyframe_id=f"{video_id}/kf_000001",
                role="primary",
                text="a red motorcycle",
                image_available=with_image,
                image_bytes=b"jpeg-bytes" if with_image else None,
            ),
        ),
    )


# ------------------------------------------------------------------ mock backend


def test_mock_backend_reports_itself_as_non_visual() -> None:
    status = MockTextQAAnswerer().status()
    assert status.backend_type == "mock"
    assert status.visual_capable is False
    assert status.production_ready is False
    assert status.state == BACKEND_STATE_READY
    assert "non-visual" in (status.warning or "").lower()


def test_mock_answers_are_labelled_as_non_visual() -> None:
    result = MockTextQAAnswerer().answer("What is happening?", bundle("A"))
    assert result.video_id == "A"
    assert result.visual is False
    assert "not visual q&a" in (result.warning or "").lower()


def test_mock_satisfies_the_backend_protocol() -> None:
    assert isinstance(MockTextQAAnswerer(), VisualQAAnswerer)
    assert isinstance(FakeVisualQAAnswerer(), VisualQAAnswerer)
    assert isinstance(LocalVlmQAAnswerer(), VisualQAAnswerer)
    assert isinstance(ApiVqaAnswerer(), VisualQAAnswerer)


def test_fake_visual_backend_reports_visual_true() -> None:
    status = FakeVisualQAAnswerer().status()
    assert status.visual_capable is True
    assert status.supports_multi_image is True


# ------------------------------------------------------------- local VLM backend


def test_unconfigured_local_vlm_is_explicitly_unavailable() -> None:
    backend = LocalVlmQAAnswerer()
    status = backend.status()
    assert status.state == BACKEND_STATE_NOT_AVAILABLE
    assert status.production_ready is False
    assert status.available is False
    assert "never downloaded automatically" in (status.fallback_reason or "")
    with pytest.raises(QABackendUnavailable, match="No local VLM"):
        backend.answer("q", bundle())


def test_an_injected_local_vlm_answers_and_reports_ready() -> None:
    model = FakeLocalVlm(reply="two")
    backend = LocalVlmQAAnswerer(model, model_name="fake-vlm", device="cpu")
    status = backend.status()
    assert status.state == BACKEND_STATE_READY
    assert status.visual_capable is True
    assert status.production_ready is True
    result = backend.answer("How many?", bundle("A"), expected_answer_type="number")
    assert result.video_id == "A"
    assert result.visual is True
    assert result.normalized_answer == "2"
    assert model.image_counts == [1]
    assert "single integer" in model.prompts[0]


def test_a_single_image_local_vlm_uses_the_strongest_frame() -> None:
    model = FakeLocalVlm(reply="red")
    backend = LocalVlmQAAnswerer(model, supports_multi_image=False)
    frames = tuple(
        QAEvidenceFrame(
            video_id="A",
            frame_idx=index,
            timestamp=float(index),
            source="keyframe_jpeg",
            keyframe_id=f"A/kf_{index}",
            role="primary" if index == 1 else "after",
            image_available=True,
            image_bytes=f"frame-{index}".encode(),
        )
        for index in range(3)
    )
    backend.answer("q", QAEvidenceBundle(video_id="A", question="q", frames=frames))
    assert model.image_counts == [1]
    assert backend.status().supports_multi_image is False


def test_local_vlm_without_pixels_refuses_rather_than_guessing() -> None:
    backend = LocalVlmQAAnswerer(FakeLocalVlm())
    with pytest.raises(QABackendUnavailable, match="No visual evidence"):
        backend.answer("q", bundle(with_image=False))


# -------------------------------------------------------------- API backend


def test_api_backend_absence_is_explicit_and_never_logs_a_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = ApiVqaAnswerer()
    status = backend.status()
    assert status.state == BACKEND_STATE_NOT_AVAILABLE
    assert status.visual_capable is True
    assert status.fallback_reason == "ANTHROPIC_API_KEY is not set."
    # Only the variable NAME appears anywhere in the report.
    assert "sk-" not in str(status.to_dict())


def test_api_backend_with_a_key_present_is_not_loaded_until_used(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    status = ApiVqaAnswerer().status()
    assert status.state == BACKEND_STATE_NOT_LOADED
    assert status.production_ready is True
    assert "not-a-real-key" not in str(status.to_dict())


def test_api_backend_uses_an_injected_client_without_network() -> None:
    class FakeBlock:
        type = "text"
        text = "red"

    class FakeMessage:
        content = [FakeBlock()]

    class FakeMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeMessage()

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = FakeClient()
    backend = ApiVqaAnswerer(client=client, max_images=4)
    assert backend.status().state == BACKEND_STATE_READY
    result = backend.answer("What colour?", bundle("A"), expected_answer_type="color")
    assert result.video_id == "A"
    assert result.normalized_answer == "red"
    assert result.visual is True
    assert len(client.messages.calls) == 1


# ------------------------------------------------------------ backend selection


def test_auto_selection_downloads_nothing_and_falls_back_visibly(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = build_qa_answerer("auto")
    assert isinstance(backend, MockTextQAAnswerer)
    assert backend.status().visual_capable is False


def test_auto_selection_prefers_an_injected_local_model(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = build_qa_answerer("auto", local_model=FakeLocalVlm())
    assert isinstance(backend, LocalVlmQAAnswerer)
    assert backend.status().visual_capable is True


def test_explicit_backend_types_are_honoured() -> None:
    assert isinstance(build_qa_answerer("mock"), MockTextQAAnswerer)
    assert isinstance(build_qa_answerer("local_vlm"), LocalVlmQAAnswerer)
    assert isinstance(build_qa_answerer("api"), ApiVqaAnswerer)
    with pytest.raises(ValueError, match="Unsupported qa.backend.type"):
        build_qa_answerer("telepathy")


def test_no_model_or_api_is_touched_when_building_backends(monkeypatch) -> None:
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a unit test attempted a network/model load")

    monkeypatch.setattr("aic2026.qa.ApiVqaAnswerer._load", explode)
    for kind in ("auto", "mock", "local_vlm", "api"):
        build_qa_answerer(kind).status()


# --------------------------------------------------------------- reliability


def test_reliability_is_a_transparent_bounded_heuristic() -> None:
    visual = FakeVisualQAAnswerer().status()
    mock = MockTextQAAnswerer().status()
    strong = answer_reliability_score(
        backend=visual, evidence_count=4, visual_evidence_count=4,
        answer="3", expected_answer_type="number", retrieval_margin=1.0,
    )
    weak = answer_reliability_score(
        backend=mock, evidence_count=1, visual_evidence_count=0,
        answer="", expected_answer_type="number", retrieval_margin=0.0,
    )
    assert 0.0 <= weak < strong <= 1.0
    # A non-visual backend can never earn the visual term.
    assert answer_reliability_score(
        backend=mock, evidence_count=4, visual_evidence_count=4,
        answer="3", expected_answer_type="number", retrieval_margin=1.0,
    ) < strong
    # An answer of the wrong shape does not earn the type term.
    assert answer_reliability_score(
        backend=visual, evidence_count=4, visual_evidence_count=4,
        answer="maybe three", expected_answer_type="number", retrieval_margin=1.0,
    ) < strong


# ------------------------------------------------------------------- config


def test_qa_backend_config_is_validated() -> None:
    def config(**qa):
        return app_config_from_dict({"aic2026": {"qa": qa}})

    assert config(backend={"type": "mock"}).qa.backend_type == "mock"
    with pytest.raises(ConfigError, match="qa.backend.type"):
        config(backend={"type": "telepathy"})
    with pytest.raises(ConfigError, match="qa.backend.device"):
        config(backend={"device": "tpu"})
    with pytest.raises(ConfigError, match="qa.evidence_frame_count must be <="):
        config(evidence_frame_count=999)
    with pytest.raises(ConfigError, match="qa.abstain_threshold"):
        config(abstain_threshold=2.0)
    with pytest.raises(ConfigError, match="qa.refinement_max_frames must be <="):
        config(refinement_max_frames=500)
    with pytest.raises(ConfigError, match="answerer_batch_size was removed"):
        config(answerer_batch_size=1)


def test_legacy_answer_confidence_threshold_is_translated() -> None:
    config = app_config_from_dict({"aic2026": {"qa": {"answer_confidence_threshold": 0.6}}})
    assert config.qa.abstain_threshold == 0.6
    assert config.qa.answer_confidence_threshold == 0.6


def test_shipped_qa_defaults() -> None:
    from aic2026.config import load_app_config

    config = load_app_config("configs/settings.yaml")
    assert config.qa.enabled is True
    assert config.qa.backend_type == "auto"
    assert config.qa.use_local_refinement is False
    assert config.qa.refinement_candidate_budget == 1
    assert config.qa.refinement_max_frames == 12


# ----------------------------------------------------- runtime state and HTTP


def build_engine(tmp_path: Path, answerer, root_name: str = "data"):
    root = make_qa_root(tmp_path / root_name)
    config = make_qa_config(root, tmp_path / f"cache_{root_name}", tmp_path / "frames")
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / f"cache_{root_name}",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        qa_answerer=answerer,
    )
    return engine, config


def test_engine_qa_status_never_loads_a_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    engine, _ = build_engine(tmp_path, None)
    status = engine.qa_status()
    assert status["backend_type"] == "mock"
    assert status["visual_capable"] is False
    assert status["backend_state"] == BACKEND_STATE_READY
    assert status["use_local_refinement"] is False


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """An app whose engines get a deterministic fake VISUAL backend."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        engine_module, "build_qa_answerer", lambda *a, **k: FakeVisualQAAnswerer()
    )
    root_a = make_qa_root(tmp_path / "root_a")
    root_b = make_qa_root(
        tmp_path / "root_b", videos=("L21_V001", "L21_V003"), jpeg_videos=("L21_V001",)
    )
    config = make_qa_config(root_a, tmp_path / "cache_a", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    return app.test_client(), root_a, root_b


def test_health_exposes_qa_backend_without_loading_a_model(client) -> None:
    http, root_a, _ = client
    assert http.get("/api/health").get_json()["qa"] is None  # no engine yet
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    qa = http.get("/api/health").get_json()["qa"]
    assert qa["backend_type"] == "fake_visual"
    assert qa["visual_capable"] is True
    assert qa["supports_multi_image"] is True
    assert set(qa["backend"]) >= {
        "backend_type", "state", "visual_capable", "supports_multi_image",
        "production_ready", "model_name", "device", "warning",
    }


def test_vqa_response_separates_hypotheses_and_uses_logical_urls(client) -> None:
    http, root_a, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    body = http.post(
        "/api/video/vqa",
        json={"question": "What colour is it?", "event": "a vehicle", "topk": 20},
    ).get_json()
    assert len(body["hypotheses"]) >= 2
    answers = {item["video_id"]: item["normalized_answer"] for item in body["hypotheses"]}
    assert answers["L21_V001"] == "red" and answers["L21_V002"] == "blue"
    assert body["diagnostics"]["cross_video_answer_copy_count"] == 0
    assert body["diagnostics"]["answer_without_matching_evidence_video_count"] == 0
    for item in body["hypotheses"]:
        for evidence in item["evidence"]:
            assert evidence["video_id"] == item["video_id"]
            if evidence["image"]:
                assert evidence["image"].startswith("/api/")
                assert f"generation={body['generation']}" in evidence["image"]
                assert http.get(evidence["image"]).status_code == 200
    # No filesystem path is ever returned.
    assert str(root_a) not in str(body)


def test_expected_answer_type_flows_from_the_request(client) -> None:
    http, root_a, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    body = http.post(
        "/api/video/vqa",
        json={"question": "How many frames?", "event": "a vehicle",
              "expected_answer_type": "number", "topk": 10},
    ).get_json()
    assert body["expected_answer_type"] == "number"
    assert all(row[2].isdigit() for row in body["predictions"])
    bad = http.post(
        "/api/video/vqa",
        json={"question": "q", "event": "e", "expected_answer_type": "sonnet"},
    )
    assert bad.status_code == 400
    assert bad.get_json()["error_code"] == "INVALID_ANSWER_TYPE"


def test_evidence_retrieval_mode_flows_from_the_request(client) -> None:
    http, root_a, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    body = http.post(
        "/api/video/vqa",
        json={
            "question": "what color is it?",
            "event": "a vehicle",
            "retrieval_query_mode": "evidence",
            "topk": 10,
        },
    ).get_json()
    assert body["retrieval_query_mode"] == "evidence"
    assert body["ground_query"]
    assert "color" in body["ground_query"]


def test_vqa_http_caps_answer_text_at_100_characters(tmp_path: Path, monkeypatch) -> None:
    long_answer = " ".join(["metadata_title"] * 40)
    monkeypatch.setattr(
        engine_module,
        "build_qa_answerer",
        lambda *a, **k: ScriptedQAAnswerer(
            {"L21_V001": long_answer, "L21_V002": long_answer}
        ),
    )
    root = make_qa_root(tmp_path / "root_long")
    config = make_qa_config(root, tmp_path / "cache_long", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    http = app.test_client()
    assert http.post("/api/video/index_folder", json={"path": str(root)}).status_code == 200

    body = http.post(
        "/api/video/vqa",
        json={"question": "what is shown?", "event": "a vehicle", "topk": 20},
    ).get_json()

    assert len(body["answer"]) <= 100
    assert len(body["answer_normalized"]) <= 100
    assert all(len(item["normalized_answer"]) <= 100 for item in body["hypotheses"])
    assert all(len(row[2]) <= 100 for row in body["predictions"])
    assert all(
        len(row["answer"]["current_value"]) <= 100
        for row in body["result_batch"]["rows"]
        if row.get("answer")
    )


def test_qa_uses_one_runtime_generation_snapshot(client) -> None:
    http, root_a, root_b = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    first = http.post(
        "/api/video/vqa", json={"question": "What colour is it?", "event": "a vehicle"}
    ).get_json()
    assert http.post("/api/video/index_folder", json={"path": str(root_b)}).status_code == 200
    second = http.post(
        "/api/video/vqa", json={"question": "What colour is it?", "event": "a vehicle"}
    ).get_json()
    assert second["generation"] == first["generation"] + 1
    # Root B holds V001 and V003 (green); V002 must be gone, and no evidence may
    # survive from the previous generation.
    videos = {item["video_id"] for item in second["hypotheses"]}
    assert "L21_V002" not in videos
    assert "L21_V003" in videos
    answers = {item["video_id"]: item["normalized_answer"] for item in second["hypotheses"]}
    assert answers["L21_V003"] == "green"
    for item in second["hypotheses"]:
        for evidence in item["evidence"]:
            if evidence["image"]:
                assert f"generation={second['generation']}" in evidence["image"]


def test_a_stale_generation_cannot_fetch_qa_evidence(client) -> None:
    http, root_a, root_b = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    body = http.post(
        "/api/video/vqa", json={"question": "What colour is it?", "event": "a vehicle"}
    ).get_json()
    url = next(
        evidence["image"]
        for item in body["hypotheses"]
        for evidence in item["evidence"]
        if evidence["image"]
    )
    assert http.post("/api/video/index_folder", json={"path": str(root_b)}).status_code == 200
    assert http.get(url).status_code == 409


def test_kis_refinement_still_works_alongside_qa(client) -> None:
    http, root_a, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    body = http.post("/api/video/search", json={"query": "a vehicle", "topk": 5}).get_json()
    assert body["count"] >= 1
    # Refinement is disabled in this fixture's config, which must be reported honestly.
    assert body["refinement"]["mode"] == "disabled"
    assert body["diagnostics"]["refinement_triggered"] is False
