"""Fusion over channel evidence, engine integration, and cache implications.

Channel scores live in unrelated spaces — a CLIP cosine, a BM25 sum, a detector
confidence, a metadata overlap ratio — so they are normalized before they meet. These
tests check that fusion consumes the normalized evidence, that provenance survives into
KIS / Q&A / TRAKE, and that the cache treats build-time and query-time channel settings
differently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ui.app as appmod
from aic2026.cache_manifest import cache_fingerprint
from aic2026.config import app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.retrieval_channels import (
    CHANNEL_ASR,
    CHANNEL_BM25,
    CHANNEL_CAPTION,
    CHANNEL_CLIP,
    CHANNEL_METADATA,
    CHANNEL_OBJECTS,
    CHANNEL_OCR,
    CHANNEL_SCHEMA_VERSION,
)
from aic2026.text_encoder import HashingTextEncoder

FPS = 10.0
FRAME_IDS = (5, 15, 25, 35)
VIDEOS = ("L21_V001", "L21_V002", "L21_V003")


def make_root(root: Path) -> Path:
    import json

    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    (root / "media-info").mkdir(parents=True, exist_ok=True)
    for position, video_id in enumerate(VIDEOS):
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate(FRAME_IDS, start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        angle = 0.2 + 0.9 * position
        np.save(
            root / "clip-features-32" / f"{video_id}.npy",
            np.array(
                [[np.cos(angle + 0.01 * i), np.sin(angle + 0.01 * i)] for i in range(len(FRAME_IDS))],
                dtype=np.float32,
            ),
        )
        folder = root / "keyframes" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        for ordinal in range(1, len(FRAME_IDS) + 1):
            Image.new("RGB", (16, 16), (25 * position, 12, 6)).save(folder / f"{ordinal:03d}.jpg")
        objects = root / "objects" / video_id
        objects.mkdir(parents=True, exist_ok=True)
        label = ["Motorcycle", "Car", "Chair"][position]
        for ordinal in range(1, len(FRAME_IDS) + 1):
            (objects / f"{ordinal:03d}.json").write_text(
                json.dumps(
                    {
                        "detection_class_entities": [label, "Person"],
                        "detection_scores": ["0.92", "0.77"],
                        "detection_boxes": [[0, 0, 1, 1], [0, 0, 1, 1]],
                    }
                ),
                encoding="utf-8",
            )
        (root / "media-info" / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "title": ["ban tin giao thong", "phong su xe hoi", "chuong trinh noi that"][position],
                    "description": "mo ta video",
                    "keywords": ["tin tuc", label.lower()],
                }
            ),
            encoding="utf-8",
        )
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path, **overrides):
    payload = {
        "dataset": {
            "root": str(root),
            "cache_dir": str(cache_dir),
            "frame_cache_dir": str(frame_cache_dir),
            "load_objects": True,
            "include_media_text": True,
            "validation": {"expected_feature_dim": 2},
        },
        "encoder": {"feature_dim": 2},
        "ranking": {"min_frame_gap": 0},
        "refinement": {"mode": "disabled"},
        "qa": {"backend": {"type": "mock"}, "top_video_hypotheses": 3},
        "trake": {"min_gap_s": 0.0, "per_event_top_k": 6, "top_video_hypotheses": 5},
        "retrieval_channels": {"channels": {"clip": {"top_k": 2}}},
    }
    for key, value in overrides.items():
        payload.setdefault(key, {})
        payload[key] = {**payload.get(key, {}), **value} if isinstance(value, dict) else value
    return app_config_from_dict({"aic2026": payload})


@pytest.fixture()
def engine(tmp_path: Path):
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames")
    built, _ = AICCompetitionEngine.from_data_root(
        root, cache_dir=tmp_path / "cache", app_config=config, text_encoder=HashingTextEncoder(2)
    )
    return built


# ------------------------------------------------------------------- fusion


def test_fusion_consumes_normalized_channel_evidence(engine) -> None:
    result = engine.search_candidates_detailed("motorcycle", top_k=50)
    assert result.candidates
    for candidate in result.candidates:
        parts = candidate.score_breakdown
        assert set(parts) >= {"dense", "object", "metadata", "bm25", "fused"}
        # Every component arrives already on a comparable [0, 1] scale.
        for name in ("dense", "object", "metadata", "bm25"):
            assert 0.0 <= parts[name] <= 1.0, (name, parts[name])


def test_absent_channels_contribute_zero_not_fabricated_scores(engine) -> None:
    result = engine.search_candidates_detailed("motorcycle", top_k=50)
    status = engine.channel_status()
    # OCR/ASR/caption have no data in this fixture and must say so.
    for name in (CHANNEL_OCR, CHANNEL_ASR, CHANNEL_CAPTION):
        assert status[name]["available"] is False
        assert status[name]["reason"] == "no_populated_source_data"
    for candidate in result.union.candidates:
        assert CHANNEL_OCR not in candidate.channels
        assert candidate.normalized_score(CHANNEL_OCR) == 0.0


def test_channel_provenance_is_visible_on_every_pooled_candidate(engine) -> None:
    result = engine.search_candidates_detailed("motorcycle", top_k=50)
    for candidate in result.union.candidates:
        assert candidate.channels
        payload = candidate.to_dict()
        assert set(payload["ranks"]) == set(candidate.channels)
        assert set(payload["normalized_scores"]) == set(candidate.channels)


def test_final_ordering_is_deterministic(engine) -> None:
    first = engine.search_candidates("motorcycle", top_k=30)
    second = engine.search_candidates("motorcycle", top_k=30)
    assert [(c.keyframe_id, round(c.score, 9)) for c in first] == [
        (c.keyframe_id, round(c.score, 9)) for c in second
    ]


def test_coarse_diagnostics_report_timings_and_coverage(engine) -> None:
    result = engine.search_candidates_detailed("motorcycle", top_k=30)
    diagnostics = result.diagnostics
    assert diagnostics["channel_search_ms"] >= 0
    assert diagnostics["fusion_ms"] >= 0
    assert diagnostics["total_coarse_ms"] >= 0
    assert diagnostics["candidate_union_size"] >= len(result.candidates)
    assert diagnostics["channel_schema_version"] == CHANNEL_SCHEMA_VERSION
    assert diagnostics["query"]["original"] == "motorcycle"


def test_coarse_retrieval_stays_fast(engine) -> None:
    import time

    engine.search_candidates("motorcycle", top_k=30)  # warm the channel indices
    started = time.perf_counter()
    for _ in range(5):
        engine.search_candidates("motorcycle on the road", top_k=30)
    elapsed = (time.perf_counter() - started) / 5 * 1000.0
    # Channels must not turn millisecond coarse retrieval into multi-second work.
    assert elapsed < 500.0, f"coarse retrieval took {elapsed:.1f} ms per query"


# ------------------------------------------------------- downstream consumers


def test_kis_uses_the_shared_multi_channel_pool(engine) -> None:
    outcome = engine.search_kis_detailed("motorcycle", top_k=20)
    assert outcome.predictions
    assert outcome.coarse is not None
    channels = outcome.diagnostics()["channels"]["channels"]
    assert channels[CHANNEL_OBJECTS]["searched"] is True
    assert channels[CHANNEL_METADATA]["searched"] is True
    seen = {
        channel
        for prediction in outcome.predictions
        for channel in prediction.evidence.get("channels", [])
    }
    assert CHANNEL_OBJECTS in seen or CHANNEL_METADATA in seen


def test_qa_uses_the_pool_and_keeps_video_isolation(engine) -> None:
    predictions, info = engine.answer_qa("a vehicle", "What is it?", top_k=20)
    assert info["diagnostics"]["cross_video_answer_copy_count"] == 0
    assert info["diagnostics"]["answer_without_matching_evidence_video_count"] == 0
    for prediction in predictions:
        assert prediction.qa["answer_video_id"] == prediction.video_id


def test_trake_expansion_deepens_the_channels(engine) -> None:
    depths: list[float] = []
    original = engine.search_candidates_detailed

    def spy(query, **kwargs):
        depths.append(float(kwargs.get("depth_scale", 1.0)))
        return original(query, **kwargs)

    engine.search_candidates_detailed = spy
    outcome = engine.search_trake_detailed(
        ["a vehicle appears", "the vehicle moves", "the vehicle leaves"], max_results=20
    )
    assert depths, "TRAKE did not retrieve anything"
    if outcome.diagnostics["candidate_expansion_triggered"]:
        # A deeper request must scale the channel depths, not silently reuse the old ones.
        assert max(depths) > 1.0
    for prediction in outcome.predictions:
        assert len(prediction.event_frame_ids) == 3
    assert outcome.structural_summary()["cross_video_step_count"] == 0


# ---------------------------------------------------------------------- cache


def test_build_time_channel_sources_change_the_fingerprint(tmp_path: Path) -> None:
    root = make_root(tmp_path / "data")
    with_objects = make_config(root, tmp_path / "c1", tmp_path / "frames")
    without = app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(tmp_path / "c2"),
                    "frame_cache_dir": str(tmp_path / "frames"),
                    "load_objects": False,
                    "include_media_text": False,
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
            }
        }
    )
    assert cache_fingerprint(with_objects) != cache_fingerprint(without)


def test_query_time_channel_settings_do_not_force_a_rebuild(tmp_path: Path) -> None:
    root = make_root(tmp_path / "data")
    shallow = make_config(root, tmp_path / "cache", tmp_path / "frames")
    deep = make_config(
        root,
        tmp_path / "cache",
        tmp_path / "frames",
        retrieval_channels={"channels": {"clip": {"top_k": 999}, "objects": {"enabled": False}}},
    )
    assert shallow.retrieval_channels.clip_top_k != deep.retrieval_channels.clip_top_k
    # Changing how deep a channel searches is a QUERY-time decision: the built data is
    # identical, so the cache must remain valid.
    assert cache_fingerprint(shallow) == cache_fingerprint(deep)


def test_the_manifest_records_the_channel_schema(tmp_path: Path) -> None:
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames")
    _, load = AICCompetitionEngine.from_data_root(
        root, cache_dir=tmp_path / "cache", app_config=config, text_encoder=HashingTextEncoder(2)
    )
    payload = load.cache_manifest.to_dict()
    assert payload["channel_schema_version"] == CHANNEL_SCHEMA_VERSION
    assert payload["load_objects"] is True
    assert payload["include_media_text"] is True


# ----------------------------------------------------------------------- HTTP


def test_health_reports_every_channel(tmp_path: Path) -> None:
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    http = app.test_client()
    assert http.get("/api/health").get_json()["retrieval_channels"] is None
    assert http.post("/api/video/index_folder", json={"path": str(root)}).status_code == 200
    channels = http.get("/api/health").get_json()["retrieval_channels"]
    assert set(channels) == {
        CHANNEL_CLIP, CHANNEL_BM25, CHANNEL_OBJECTS, CHANNEL_METADATA,
        CHANNEL_OCR, CHANNEL_ASR, CHANNEL_CAPTION,
    }
    assert channels[CHANNEL_OBJECTS]["available"] is True
    assert channels[CHANNEL_OBJECTS]["records"] > 0
    assert channels[CHANNEL_OCR]["available"] is False
    assert channels[CHANNEL_OCR]["reason"] == "no_populated_source_data"
    assert channels[CHANNEL_METADATA]["scope"] == "video"
