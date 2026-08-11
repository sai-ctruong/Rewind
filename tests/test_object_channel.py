"""Objects as an INDEPENDENT candidate generator.

This file holds the critical Phase 9 regression: a frame whose only strong signal is its
detector labels must be able to enter the candidate pool on its own. Before Phase 9 the
pool was CLIP union BM25 and object scores only rescored what was already inside it, so
such a frame could never appear at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic2026.config import app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.query_normalization import normalize_query
from aic2026.retrieval_channels import CHANNEL_OBJECTS, ObjectChannel
from aic2026.text_encoder import HashingTextEncoder
from tests.test_retrieval_channels import make_entry, raw

FPS = 10.0


# ------------------------------------------------------------------ unit level


def test_an_object_only_frame_is_retrievable() -> None:
    entry = make_entry(
        {
            "A/1": raw(
                "A/1", "A", 0.0, frame_idx=1,
                detections=[{"label": "motorcycle", "confidence": 0.95}],
            ),
            "B/1": raw(
                "B/1", "B", 0.0, frame_idx=2,
                detections=[{"label": "chair", "confidence": 0.95}],
            ),
        }
    )
    result = ObjectChannel(entry).search(normalize_query("motorcycle"), top_k=10)
    assert [item.keyframe_id for item in result.candidates] == ["A/1"]
    assert result.candidates[0].evidence == ("motorcycle",)


def test_a_vietnamese_query_reaches_english_detector_labels() -> None:
    entry = make_entry(
        {
            "A/1": raw(
                "A/1", "A", 0.0, frame_idx=1,
                detections=[{"label": "motorcycle", "confidence": 0.9}],
            )
        }
    )
    result = ObjectChannel(entry).search(normalize_query("xe máy trên đường"), top_k=10)
    assert [item.keyframe_id for item in result.candidates] == ["A/1"]


def test_confidence_orders_frames_deterministically() -> None:
    entry = make_entry(
        {
            "LOW/1": raw(
                "LOW/1", "LOW", 0.0, frame_idx=1,
                detections=[{"label": "car", "confidence": 0.40}],
            ),
            "HIGH/1": raw(
                "HIGH/1", "HIGH", 0.0, frame_idx=2,
                detections=[{"label": "car", "confidence": 0.95}],
            ),
        }
    )
    first = ObjectChannel(entry).search(normalize_query("car"), top_k=10)
    second = ObjectChannel(entry).search(normalize_query("car"), top_k=10)
    order = [item.keyframe_id for item in first.candidates]
    assert order == ["HIGH/1", "LOW/1"]
    assert order == [item.keyframe_id for item in second.candidates]


def test_the_confidence_threshold_excludes_weak_detections() -> None:
    entry = make_entry(
        {
            "A/1": raw(
                "A/1", "A", 0.0, frame_idx=1,
                detections=[{"label": "car", "confidence": 0.10}],
            )
        }
    )
    assert ObjectChannel(entry, confidence_threshold=0.5).search(
        normalize_query("car"), top_k=10
    ).candidates == ()
    assert ObjectChannel(entry, confidence_threshold=0.05).search(
        normalize_query("car"), top_k=10
    ).candidates


def test_term_coverage_beats_a_single_confident_label() -> None:
    entry = make_entry(
        {
            "ONE/1": raw(
                "ONE/1", "ONE", 0.0, frame_idx=1,
                detections=[{"label": "person", "confidence": 1.0}],
            ),
            "BOTH/1": raw(
                "BOTH/1", "BOTH", 0.0, frame_idx=2,
                detections=[
                    {"label": "person", "confidence": 0.7},
                    {"label": "motorcycle", "confidence": 0.7},
                ],
            ),
        }
    )
    result = ObjectChannel(entry).search(normalize_query("person motorcycle"), top_k=10)
    assert result.candidates[0].keyframe_id == "BOTH/1"


def test_duplicate_labels_on_one_frame_are_collapsed() -> None:
    entry = make_entry(
        {
            "A/1": raw(
                "A/1", "A", 0.0, frame_idx=1,
                detections=[
                    {"label": "car", "confidence": 0.6},
                    {"label": "car", "confidence": 0.9},
                    {"label": "Cars", "confidence": 0.5},
                ],
            )
        }
    )
    result = ObjectChannel(entry).search(normalize_query("car"), top_k=10)
    assert len(result.candidates) == 1
    # The strongest detection of the label wins; they are not summed into a fake boost.
    assert result.candidates[0].raw_score == pytest.approx(0.9)


def test_plain_labels_without_detections_still_work() -> None:
    entry = make_entry({"A/1": raw("A/1", "A", 0.0, frame_idx=1, objects=["bicycle"])})
    result = ObjectChannel(entry).search(normalize_query("bicycle"), top_k=10)
    assert [item.keyframe_id for item in result.candidates] == ["A/1"]


def test_a_negated_object_term_is_not_positively_boosted() -> None:
    entry = make_entry(
        {
            "CAR/1": raw(
                "CAR/1", "CAR", 0.0, frame_idx=1,
                detections=[{"label": "car", "confidence": 0.95}],
            ),
            "PERSON/1": raw(
                "PERSON/1", "PERSON", 0.0, frame_idx=2,
                detections=[{"label": "person", "confidence": 0.95}],
            ),
        }
    )
    channel = ObjectChannel(entry)
    positive = channel.search(normalize_query("xe hơi và người"), top_k=10)
    assert {item.keyframe_id for item in positive.candidates} == {"CAR/1", "PERSON/1"}
    # "no car, but a person" must not retrieve the car frame through the object channel.
    negated = channel.search(normalize_query("không có xe hơi nhưng có người"), top_k=10)
    retrieved = {item.keyframe_id for item in negated.candidates}
    assert "CAR/1" not in retrieved
    assert "PERSON/1" in retrieved


def test_no_object_data_reports_unavailable() -> None:
    entry = make_entry({"A/1": raw("A/1", "A", 0.0, frame_idx=1)})
    channel = ObjectChannel(entry)
    status = channel.status()
    assert status.available is False
    assert status.record_count == 0
    assert channel.search(normalize_query("car"), top_k=10).candidates == ()


def test_a_query_with_no_object_terms_returns_nothing_gracefully() -> None:
    entry = make_entry(
        {"A/1": raw("A/1", "A", 0.0, frame_idx=1, detections=[{"label": "car", "confidence": 0.9}])}
    )
    result = ObjectChannel(entry).search(normalize_query("zzzz"), top_k=10)
    assert result.candidates == ()
    assert result.status.available is True


def test_the_index_is_built_once_not_per_query() -> None:
    entry = make_entry(
        {
            f"A/{i}": raw(
                f"A/{i}", "A", float(i), frame_idx=i,
                detections=[{"label": "car", "confidence": 0.9}],
            )
            for i in range(50)
        }
    )
    channel = ObjectChannel(entry)
    postings = channel._postings
    channel.search(normalize_query("car"), top_k=10)
    channel.search(normalize_query("car"), top_k=10)
    # The same object, never rebuilt; queries only read the postings.
    assert channel._postings is postings


# ------------------------------------------------------------- engine level


def make_root(root: Path, *, with_objects: bool) -> Path:
    """Two videos. `OBJ` is deliberately weak for CLIP but carries motorcycle labels."""
    import json

    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    for position, video_id in enumerate(("L21_V001", "L21_V002")):
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate((5, 15, 25), start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        angle = 0.2 + 1.2 * position
        features = np.array(
            [[np.cos(angle + 0.01 * i), np.sin(angle + 0.01 * i)] for i in range(3)],
            dtype=np.float32,
        )
        np.save(root / "clip-features-32" / f"{video_id}.npy", features)
        folder = root / "keyframes" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        for ordinal in range(1, 4):
            Image.new("RGB", (16, 16), (20 * position, 10, 5)).save(folder / f"{ordinal:03d}.jpg")
        if with_objects and video_id == "L21_V002":
            objects = root / "objects" / video_id
            objects.mkdir(parents=True, exist_ok=True)
            for ordinal in range(1, 4):
                (objects / f"{ordinal:03d}.json").write_text(
                    json.dumps(
                        {
                            "detection_class_entities": ["Motorcycle", "Person"],
                            "detection_scores": ["0.94", "0.81"],
                            "detection_boxes": [[0, 0, 1, 1], [0, 0, 1, 1]],
                        }
                    ),
                    encoding="utf-8",
                )
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path, **channels):
    settings = {"channels": {"metadata": {"enabled": False}}}
    settings.update(channels)
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir),
                    "frame_cache_dir": str(frame_cache_dir),
                    "load_objects": True,
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
                "ranking": {"min_frame_gap": 0},
                "refinement": {"mode": "disabled"},
                "qa": {"backend": {"type": "mock"}},
                "retrieval_channels": settings,
            }
        }
    )


def build(tmp_path: Path, *, objects_enabled: bool):
    root = make_root(tmp_path / "data", with_objects=True)
    cache = tmp_path / f"cache_{objects_enabled}"
    config = make_config(
        root,
        cache,
        tmp_path / "frames",
        channels={
            "objects": {"enabled": objects_enabled},
            "metadata": {"enabled": False},
            # A deliberately shallow dense pool, so most frames are reachable ONLY if
            # another channel can propose them. This is the pre-Phase-9 blind spot.
            "clip": {"top_k": 1},
            "bm25": {"enabled": False},
        },
    )
    engine, _ = AICCompetitionEngine.from_data_root(
        root, cache_dir=cache, app_config=config, text_encoder=HashingTextEncoder(2)
    )
    return engine


def test_objects_can_introduce_a_candidate_the_dense_pool_would_miss(tmp_path: Path) -> None:
    """The critical Phase 9 proof, at engine level."""
    with_objects = build(tmp_path / "on", objects_enabled=True)
    without = build(tmp_path / "off", objects_enabled=False)

    on = with_objects.search_candidates_detailed("motorcycle", top_k=100)
    off = without.search_candidates_detailed("motorcycle", top_k=100)

    on_ids = {candidate.keyframe_id for candidate in on.candidates}
    off_ids = {candidate.keyframe_id for candidate in off.candidates}
    introduced = on_ids - off_ids
    assert introduced, "the object channel introduced nothing new"
    # Every newly introduced candidate is one the object channel found.
    pooled = on.union.by_id()
    for keyframe_id in introduced:
        assert CHANNEL_OBJECTS in pooled[keyframe_id].channels
    assert on.diagnostics["channels"][CHANNEL_OBJECTS]["unique_candidates_introduced"] >= 1


def test_a_disabled_object_channel_cannot_introduce_it(tmp_path: Path) -> None:
    engine = build(tmp_path, objects_enabled=False)
    result = engine.search_candidates_detailed("motorcycle", top_k=100)
    assert all(
        CHANNEL_OBJECTS not in candidate.channels for candidate in result.union.candidates
    )
    assert result.diagnostics["channels"][CHANNEL_OBJECTS]["searched"] is False


def test_object_provenance_reaches_the_prediction(tmp_path: Path) -> None:
    engine = build(tmp_path, objects_enabled=True)
    predictions = engine.search_kis("motorcycle", top_k=50)
    assert predictions
    channels = {
        channel
        for prediction in predictions
        for channel in prediction.evidence.get("channels", [])
    }
    assert CHANNEL_OBJECTS in channels


def test_object_candidates_survive_downstream_fusion(tmp_path: Path) -> None:
    engine = build(tmp_path, objects_enabled=True)
    result = engine.search_candidates_detailed("motorcycle", top_k=100)
    object_only = [
        item.keyframe_id
        for item in result.union.candidates
        if item.introduced_only_by() == CHANNEL_OBJECTS
    ]
    assert object_only
    survivors = {candidate.keyframe_id for candidate in result.candidates}
    assert set(object_only) & survivors, "an object-only candidate was dropped by fusion"
