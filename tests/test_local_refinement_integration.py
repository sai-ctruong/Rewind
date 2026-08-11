"""End-to-end local refinement: engine, ranking, runtime state, and the HTTP surface.

The dataset here is synthetic but structurally real: official-style map CSVs, CLIP
feature arrays, BTC keyframe JPEGs for one video and none for the other, and genuine
MP4s whose pixels encode each frame's index. That is what makes it possible to assert
that refinement actually decoded the right frames of the right dataset, offline.
"""
from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import aic2026.engine as engine_module
import ui.app as appmod
from aic2026.config import app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.local_refinement import MODE_ALWAYS, MODE_DISABLED, MODE_UNCERTAINTY
from aic2026.runtime_state import build_runtime_state
from aic2026.text_encoder import HashingTextEncoder
from tests.refinement_support import FakeFrameScorer, frame_value, write_synthetic_video

FPS = 10.0
FRAME_IDS = (5, 15, 25)
JPEG_VIDEO = "L21_V001"
MP4_ONLY_VIDEO = "L21_V002"


def make_root(root: Path, *, jpeg_red: int = 200) -> Path:
    """A miniature but structurally official AIC root with real MP4s."""
    for video_id in (JPEG_VIDEO, MP4_ONLY_VIDEO):
        (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
        (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate(FRAME_IDS, start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        features = np.eye(len(FRAME_IDS), 2, dtype=np.float32)
        features[2] = np.array([0.6, 0.8], dtype=np.float32)  # unit norm
        np.save(root / "clip-features-32" / f"{video_id}.npy", features)
        write_synthetic_video(root / "video" / f"{video_id}.mp4", frames=31, fps=FPS)
    # Only the first video has BTC keyframe JPEGs; the second must fall back to its MP4.
    keyframe_dir = root / "keyframes" / JPEG_VIDEO
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    for ordinal in range(1, len(FRAME_IDS) + 1):
        Image.new("RGB", (16, 16), (jpeg_red, 0, 0)).save(keyframe_dir / f"{ordinal:03d}.jpg")
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path, **refinement):
    settings = {
        "mode": MODE_ALWAYS,
        "candidate_budget": 4,
        "window_before_s": 0.5,
        "window_after_s": 0.5,
        "fine_fps": 5.0,
        "max_frames": 4,
    }
    settings.update(refinement)
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir),
                    "frame_cache_dir": str(frame_cache_dir),
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
                "ranking": {"min_frame_gap": 0, "final_top_k": 100},
                "refinement": settings,
            }
        }
    )


def build_engine(tmp_path: Path, *, target_frame_idx: int = 17, **refinement):
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames", **refinement)
    scorer = FakeFrameScorer(target_frame_idx=target_frame_idx)
    engine, load = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        frame_scorer=scorer,
    )
    return engine, load, scorer, config


# ------------------------------------------------------------------- ranking


def test_coarse_score_is_preserved_alongside_the_refined_one(tmp_path: Path) -> None:
    engine, _, _, _ = build_engine(tmp_path)
    outcome = engine.search_kis_detailed("a person", top_k=10)
    refined = [p for p in outcome.predictions if p.refinement and p.refinement["applied"]]
    assert refined, "the always mode must refine something in this fixture"
    for prediction in refined:
        breakdown = prediction.score_breakdown
        assert "coarse_fused" in breakdown and "refined" in breakdown
        assert breakdown["fused"] == pytest.approx(breakdown["coarse_fused"])
        assert prediction.refinement["coarse_score"] == pytest.approx(breakdown["coarse_fused"])


def test_refined_score_components_are_visible(tmp_path: Path) -> None:
    engine, _, _, config = build_engine(tmp_path)
    outcome = engine.search_kis_detailed("a person", top_k=10)
    item = next(p for p in outcome.predictions if p.refinement and p.refinement["applied"])
    payload = item.refinement
    alpha = config.refinement.rerank_alpha
    assert payload["refined_score"] == pytest.approx(
        payload["coarse_score"] + alpha * payload["score_gain"], abs=1e-6
    )
    assert item.score_breakdown["visual_gain"] == pytest.approx(payload["score_gain"])


def test_reranking_is_deterministic(tmp_path: Path) -> None:
    engine, _, _, _ = build_engine(tmp_path)
    first = engine.search_kis("a person walking", top_k=20)
    second = engine.search_kis("a person walking", top_k=20)
    assert [(p.video_id, p.frame_id, round(p.score, 9)) for p in first] == [
        (p.video_id, p.frame_id, round(p.score, 9)) for p in second
    ]


def test_refinement_does_not_drop_untouched_candidates(tmp_path: Path) -> None:
    engine, _, _, _ = build_engine(tmp_path, candidate_budget=1)
    without = engine.search_kis("a person", top_k=100, refine=False)
    with_refinement = engine.search_kis("a person", top_k=100)
    assert {(p.video_id, p.frame_id) for p in without} == {
        (p.video_id, p.frame_id) for p in with_refinement
    }
    assert len(with_refinement) == len(without)


def test_top_100_cap_still_holds_with_refinement(tmp_path: Path) -> None:
    engine, _, _, _ = build_engine(tmp_path)
    predictions = engine.search_kis("a person", top_k=100)
    assert 0 < len(predictions) <= 100
    keys = [(p.video_id, p.frame_id) for p in predictions]
    assert len(keys) == len(set(keys)), "the official rows must stay unique"


# -------------------------------------------------------------------- engine


def test_search_kis_calls_refinement_when_enabled(tmp_path: Path) -> None:
    engine, _, scorer, _ = build_engine(tmp_path)
    outcome = engine.search_kis_detailed("a person", top_k=10)
    assert outcome.refinement is not None
    assert outcome.refinement.decision.triggered is True
    assert outcome.diagnostics()["candidates_refined"] >= 1
    assert scorer.score_calls == 1


def test_search_kis_does_not_refine_when_disabled_by_config(tmp_path: Path) -> None:
    engine, _, scorer, _ = build_engine(tmp_path, mode=MODE_DISABLED)
    outcome = engine.search_kis_detailed("a person", top_k=10)
    assert outcome.refinement is None
    assert scorer.prepare_calls == 0 and scorer.score_calls == 0
    assert all(p.refinement is None for p in outcome.predictions)
    assert engine.refinement_status()["mode"] == MODE_DISABLED


def test_search_kis_does_not_refine_when_disabled_per_request(tmp_path: Path) -> None:
    engine, _, scorer, _ = build_engine(tmp_path)
    outcome = engine.search_kis_detailed("a person", top_k=10, refine=False)
    assert outcome.refinement is None
    assert scorer.score_calls == 0
    assert outcome.predictions, "turning refinement off must still return coarse results"


def test_query_preparation_happens_once_per_search(tmp_path: Path) -> None:
    engine, _, scorer, _ = build_engine(tmp_path)
    engine.search_kis("a person", top_k=10)
    assert scorer.prepare_calls == 1
    engine.search_kis("a person", top_k=10)
    assert scorer.prepare_calls == 2, "each search prepares its own query exactly once"


def test_search_still_returns_results_when_the_scorer_is_broken(tmp_path: Path) -> None:
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames")
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        frame_scorer=FakeFrameScorer(17, fail=True),
    )
    outcome = engine.search_kis_detailed("a person", top_k=10)
    assert outcome.predictions, "a scorer failure must never fail the search"
    assert outcome.refinement is not None
    assert outcome.refinement.applied is False
    assert outcome.diagnostics()["scorer_failures"] >= 1
    # Every candidate keeps its coarse official frame.
    assert all(p.frame_id in {str(i) for i in FRAME_IDS} for p in outcome.predictions)


def test_frame_ids_stay_official_under_preserve_coarse(tmp_path: Path) -> None:
    engine, _, _, _ = build_engine(tmp_path, target_frame_idx=17)
    outcome = engine.search_kis_detailed("a person", top_k=10)
    moved = [
        p for p in outcome.predictions
        if p.refinement and p.refinement.get("best_is_coarse_frame") is False
    ]
    assert moved, "the fixture is meant to move the visual frame off the coarse one"
    for prediction in moved:
        assert prediction.frame_id == str(prediction.refinement["coarse_official_frame_idx"])
        assert prediction.refinement["submission_frame_idx"] == int(prediction.frame_id)
        assert prediction.refinement["best_visual_frame_idx"] != int(prediction.frame_id)


def test_engine_owns_a_refiner_bound_to_its_own_frame_provider(tmp_path: Path) -> None:
    engine, _, _, _ = build_engine(tmp_path)
    assert engine.local_refiner.frame_provider is engine.frame_provider
    assert engine.local_refiner.scorer is engine.frame_scorer


# ------------------------------------------------------------- runtime state


def test_runtime_state_adopts_the_engine_frame_provider(tmp_path: Path) -> None:
    engine, load, _, config = build_engine(tmp_path)
    state = build_runtime_state(
        app_config=config,
        config_path="<test>",
        generation=2,
        data_root=config.dataset.root,
        cache_dir=config.dataset.cache_dir,
        engine=engine,
        load=load,
    )
    assert state.frame_provider is engine.frame_provider
    assert state.engine.local_refiner.frame_provider is state.frame_provider
    state.verify_engine_identity()


def test_a_refiner_from_another_generation_is_rejected(tmp_path: Path) -> None:
    from aic2026.frame_provider import FrameProvider
    from aic2026.runtime_state import RuntimeStateError

    engine, load, _, config = build_engine(tmp_path)
    state = build_runtime_state(
        app_config=config,
        config_path="<test>",
        generation=2,
        data_root=config.dataset.root,
        cache_dir=config.dataset.cache_dir,
        engine=engine,
        load=load,
    )
    # Simulate the Phase 0 style of bug: a provider from a different generation.
    stale = dataclass_replace(state, frame_provider=FrameProvider(config.dataset.root))
    with pytest.raises(RuntimeStateError, match="local refiner"):
        stale.verify_engine_identity()


# ---------------------------------------------------------------- HTTP layer


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """An app whose engines get a deterministic fake scorer instead of CLIP."""
    monkeypatch.setattr(
        engine_module,
        "build_frame_scorer",
        lambda *args, **kwargs: FakeFrameScorer(target_frame_idx=17),
    )
    root_a = make_root(tmp_path / "root_a", jpeg_red=250)
    root_b = make_root(tmp_path / "root_b", jpeg_red=40)
    config = make_config(root_a, tmp_path / "cache_a", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    return app.test_client(), root_a, root_b, tmp_path


def test_health_reports_refinement_without_loading_a_model(client) -> None:
    http, root_a, _, _ = client
    health = http.get("/api/health").get_json()
    refinement = health["refinement"]
    assert refinement["enabled"] is True
    assert refinement["mode"] == MODE_ALWAYS
    assert refinement["frame_output_policy"] == "preserve_coarse"
    # No engine yet, so no scorer: the point is that asking never triggers a load.
    assert "scorer" not in refinement or refinement["scorer"]["state"] != "ready"

    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    after = http.get("/api/health").get_json()["refinement"]
    assert after["scorer"]["available"] is True
    assert after["mode"] == MODE_ALWAYS
    assert after["scorer_model_name"] == "openai/clip-vit-base-patch32"


def test_search_json_separates_coarse_refined_and_submission_frames(client) -> None:
    http, root_a, _, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    body = http.post("/api/video/search", json={"query": "a person", "topk": 10}).get_json()
    assert body["refinement"]["decision"]["triggered"] is True
    assert body["diagnostics"]["candidates_refined"] >= 1
    moved = [
        item for item in body["results"]
        if item["refinement"] and item["refinement"].get("best_is_coarse_frame") is False
    ]
    assert moved
    item = moved[0]
    assert item["submission_frame_id"] == item["frame_id"]
    assert item["refined_frame_id"] != int(item["frame_id"])
    assert item["refined_image"].startswith("/api/video/decoded_frame/")
    assert "generation=" in item["refined_image"]
    # No filesystem path is ever exposed.
    assert not any(str(root_a) in str(value) for value in item.values())
    assert http.get(item["refined_image"]).status_code == 200


def test_refinement_can_be_disabled_per_request_for_comparison(client) -> None:
    http, root_a, _, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    on = http.post("/api/video/search", json={"query": "a person", "topk": 10}).get_json()
    off = http.post(
        "/api/video/search", json={"query": "a person", "topk": 10, "refine": False}
    ).get_json()
    assert on["diagnostics"]["refinement_triggered"] is True
    assert off["diagnostics"]["refinement_triggered"] is False
    assert off["refinement"]["decision"] is None
    assert all(item["refinement"] is None for item in off["results"])
    assert off["count"] == on["count"]


def test_a_root_switch_repoints_the_refiner_and_the_decoded_frame_route(client) -> None:
    http, root_a, root_b, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    first = http.get("/api/health").get_json()["runtime"]
    manager = None
    assert http.post("/api/video/index_folder", json={"path": str(root_b)}).status_code == 200
    second = http.get("/api/health").get_json()
    assert second["runtime"]["generation"] == first["generation"] + 1
    assert Path(second["runtime"]["data_root"]).name == "root_b"

    # Refinement now runs against root B: the refiner, the provider and the routes all
    # moved together, which is exactly what Phase 4 made structurally impossible to skip.
    from ui.app import STATE_EXTENSION_KEY

    manager = http.application.extensions[STATE_EXTENSION_KEY]
    state = manager.get_state()
    assert state.engine.local_refiner.frame_provider is state.frame_provider
    resolved = state.frame_provider.video_path(JPEG_VIDEO)
    assert resolved is not None and Path(resolved).parents[1].name == "root_b"

    stale = http.get(
        f"/api/video/decoded_frame/{JPEG_VIDEO}/12?generation={first['generation']}"
    )
    assert stale.status_code == 409
    assert stale.get_json()["error_code"] == "STALE_RESULT_GENERATION"


def test_decoded_frame_route_refuses_out_of_scope_ids(client) -> None:
    http, root_a, _, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    assert http.get("/api/video/decoded_frame/L99_V404/3").status_code == 404
    assert http.get("/api/video/decoded_frame/..%2F..%2Fetc/3").status_code == 404


def test_decoded_frame_route_returns_the_requested_frame(client) -> None:
    import cv2

    http, root_a, _, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    def blue_of(frame_idx: int) -> float:
        response = http.get(f"/api/video/decoded_frame/{MP4_ONLY_VIDEO}/{frame_idx}")
        assert response.status_code == 200
        assert response.headers["X-Frame-Role"] == "refined_visual_frame"
        assert response.headers["X-Frame-Id"] == str(frame_idx)
        decoded = cv2.imdecode(np.frombuffer(response.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return float(decoded[:, :, 0].mean())

    # Two lossy stages (mp4v, then JPEG) shift absolute values, so identity is asserted
    # on the ramp: distinct frames must come back in the order they were written.
    early, middle, late = blue_of(5), blue_of(12), blue_of(25)
    assert early < middle < late
    assert middle - early == pytest.approx(frame_value(12) - frame_value(5), abs=8)
    assert late - middle == pytest.approx(frame_value(25) - frame_value(12), abs=8)


# -------------------------------------------------------------- regressions


def test_keyframe_jpeg_and_mp4_fallback_both_still_work(client) -> None:
    http, root_a, _, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    from ui.app import STATE_EXTENSION_KEY

    state = http.application.extensions[STATE_EXTENSION_KEY].get_state()
    jpeg_id = next(
        rid for rid, raw in state.engine.entry.raws.items()
        if raw.video_id == JPEG_VIDEO and raw.image_path
    )
    mp4_id = next(
        rid for rid, raw in state.engine.entry.raws.items()
        if raw.video_id == MP4_ONLY_VIDEO and not raw.image_path
    )
    jpeg = http.get(f"/api/video/frame/{jpeg_id}")
    assert jpeg.status_code == 200 and jpeg.headers["X-Frame-Source"] == "keyframe_jpeg"
    mp4 = http.get(f"/api/video/frame/{mp4_id}")
    assert mp4.status_code == 200 and mp4.headers["X-Frame-Source"] == "video_decode"


def test_cache_and_scope_are_unchanged_by_phase_5(tmp_path: Path) -> None:
    engine, load, _, config = build_engine(tmp_path)
    assert load.cache_valid is True and load.cache_legacy is False and load.cache_stale is False
    assert load.stats is not None
    assert load.stats.scope_mode == "patterns"
    assert load.stats.scope_include_patterns == ("*",)
    assert load.stats.videos == 2
    # Rebuilding from the same cache must be a hit, not a rebuild.
    again, reload_result = AICCompetitionEngine.from_data_root(
        config.dataset.root,
        cache_dir=config.dataset.cache_dir,
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        frame_scorer=FakeFrameScorer(17),
    )
    assert reload_result.cache_hit is True and reload_result.cache_valid is True
    assert again.entry.num_indexed == engine.entry.num_indexed


def test_uncertainty_is_the_shipped_default_mode() -> None:
    from aic2026.config import load_app_config

    config = load_app_config("configs/settings.yaml")
    assert config.refinement.enabled is True
    assert config.refinement.mode == MODE_UNCERTAINTY
    assert config.refinement.frame_output_policy == "preserve_coarse"
    assert config.refinement.scorer_type == "clip"
    assert config.refinement.scorer_required is False
