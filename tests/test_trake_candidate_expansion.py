"""Adaptive candidate expansion: deepen only the events that block completeness.

The Phase 7 real smoke discarded 65 of 77 video hypotheses, and 59 of the missing
positions had no candidate at all for that event at the initial depth. These tests fix
the expansion policy: selective, bounded, and incapable of inventing a candidate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic2026.config import ConfigError, app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.text_encoder import HashingTextEncoder

FPS = 10.0
FRAME_IDS = tuple(range(5, 305, 10))  # 30 keyframes per video
VIDEOS = ("L21_V001", "L21_V002", "L21_V003", "L21_V004")


def make_root(root: Path) -> Path:
    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    for position, video_id in enumerate(VIDEOS):
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate(FRAME_IDS, start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        # Spread each video around its own direction so deeper retrieval genuinely
        # reaches videos a shallow top-k never returned.
        base = 0.15 + 0.45 * position
        features = np.array(
            [
                [np.cos(base + 0.02 * i), np.sin(base + 0.02 * i)]
                for i in range(len(FRAME_IDS))
            ],
            dtype=np.float32,
        )
        np.save(root / "clip-features-32" / f"{video_id}.npy", features)
        folder = root / "keyframes" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        for ordinal in range(1, len(FRAME_IDS) + 1):
            Image.new("RGB", (16, 16), (30 * position, 20, 10)).save(
                folder / f"{ordinal:03d}.jpg"
            )
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path, **trake):
    settings = {
        "min_gap_s": 0.0,
        "top_video_hypotheses": 10,
        "per_event_top_k": 4,
        "candidate_depth_expansion": [40, 120],
        "candidate_depth_max": 200,
        "target_complete_video_hypotheses": 3,
    }
    settings.update(trake)
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
                "refinement": {"mode": "disabled"},
                "qa": {"backend": {"type": "mock"}},
                "trake": settings,
            }
        }
    )


def build(tmp_path: Path, **trake):
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames", **trake)
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
    )
    return engine


EVENTS = ["first event happens", "second event happens", "third event happens"]


# ------------------------------------------------------------------ triggering


def test_expansion_is_reported_and_bounded(tmp_path: Path) -> None:
    engine = build(tmp_path)
    outcome = engine.search_trake_detailed(EVENTS, max_results=50)
    diagnostics = outcome.diagnostics
    assert set(diagnostics) >= {
        "candidate_expansion_triggered",
        "events_expanded",
        "depth_before",
        "depth_after",
        "new_candidates_added",
        "initial_candidate_counts",
        "expanded_candidate_counts",
        "new_complete_video_hypotheses",
        "complete_alignments_before_expansion",
        "complete_alignments_after_expansion",
    }
    # Never deeper than the configured ceiling.
    assert all(value <= 200 for value in diagnostics["depth_after"].values())
    assert diagnostics["new_candidates_added"] >= 0


def test_expansion_does_not_trigger_when_coverage_is_already_enough(tmp_path: Path) -> None:
    # The initial depth already reaches every candidate, so the target of 1 complete
    # video is met before any stage runs and nothing is re-retrieved.
    engine = build(
        tmp_path, target_complete_video_hypotheses=1, per_event_top_k=200, candidate_depth_max=200
    )
    outcome = engine.search_trake_detailed(EVENTS, max_results=20)
    assert outcome.diagnostics["videos_with_full_event_coverage"] >= 1
    assert outcome.diagnostics["candidate_expansion_triggered"] is False
    assert outcome.diagnostics["events_expanded"] == []
    assert outcome.diagnostics["depth_before"] == outcome.diagnostics["depth_after"]


def test_no_expansion_stages_configured_means_no_expansion(tmp_path: Path) -> None:
    engine = build(tmp_path, candidate_depth_expansion=[])
    outcome = engine.search_trake_detailed(EVENTS, max_results=20)
    assert outcome.diagnostics["candidate_expansion_triggered"] is False
    assert outcome.diagnostics["depth_after"] == outcome.diagnostics["depth_before"]


def test_expansion_can_complete_previously_incomplete_videos(tmp_path: Path) -> None:
    shallow = build(tmp_path / "shallow", candidate_depth_expansion=[])
    deep = build(tmp_path / "deep", candidate_depth_expansion=[40, 120])
    without = shallow.search_trake_detailed(EVENTS, max_results=50)
    with_expansion = deep.search_trake_detailed(EVENTS, max_results=50)
    # Structural coverage only: more videos can cover every event at greater depth.
    assert (
        with_expansion.diagnostics["videos_with_full_event_coverage"]
        >= without.diagnostics["videos_with_full_event_coverage"]
    )
    assert len(with_expansion.predictions) >= len(without.predictions)
    assert with_expansion.diagnostics["complete_alignments_after_expansion"] >= (
        with_expansion.diagnostics["complete_alignments_before_expansion"]
    )


def test_expansion_never_exceeds_the_hard_depth_ceiling(tmp_path: Path) -> None:
    engine = build(
        tmp_path,
        candidate_depth_expansion=[1000, 5000],
        candidate_depth_max=25,
        target_complete_video_hypotheses=99,
    )
    outcome = engine.search_trake_detailed(EVENTS, max_results=50)
    assert all(value <= 25 for value in outcome.diagnostics["depth_after"].values())


def test_expansion_never_fabricates_a_candidate(tmp_path: Path) -> None:
    engine = build(tmp_path, target_complete_video_hypotheses=99)
    outcome = engine.search_trake_detailed(EVENTS, max_results=50)
    indexed = {raw.video_id for raw in engine.entry.raws.values()}
    official = {str(raw.frame_idx) for raw in engine.entry.raws.values()}
    for prediction in outcome.trake_predictions:
        assert prediction.video_id in indexed
        for step in prediction.steps:
            # Every submitted frame is a real mapped frame of a real indexed video.
            assert step.submission_frame_idx in official
            assert step.candidate is not None
            assert step.candidate.keyframe_id in engine.entry.raws


def test_expansion_keeps_every_structural_invariant(tmp_path: Path) -> None:
    engine = build(tmp_path, target_complete_video_hypotheses=99)
    outcome = engine.search_trake_detailed(EVENTS, max_results=50)
    summary = outcome.structural_summary()
    assert summary["malformed_prediction_count"] == 0
    assert summary["wrong_event_count_prediction_count"] == 0
    assert summary["cross_video_step_count"] == 0
    assert outcome.diagnostics["unordered_submission_sequence_count"] == 0
    for prediction in outcome.predictions:
        assert len(prediction.event_frame_ids) == len(EVENTS)


def test_no_complete_sequence_explains_itself(tmp_path: Path) -> None:
    engine = build(tmp_path)
    engine.search_candidates = lambda *args, **kwargs: []
    outcome = engine.search_trake_detailed(EVENTS, max_results=20)
    assert outcome.predictions == []
    diagnostics = outcome.diagnostics
    assert diagnostics["returned_complete_predictions"] == 0
    # The reason is available rather than implied.
    assert "missing_without_candidates" in diagnostics
    assert "missing_with_rejected_candidates" in diagnostics
    assert diagnostics["videos_with_full_event_coverage"] == 0


def test_expansion_uses_the_existing_retrieval_path_only(tmp_path: Path) -> None:
    engine = build(tmp_path, target_complete_video_hypotheses=99)
    depths: list[int] = []
    original = engine.search_candidates

    def spy(query, **kwargs):
        depths.append(int(kwargs.get("top_k", 0)))
        return original(query, **kwargs)

    engine.search_candidates = spy
    engine.search_trake_detailed(EVENTS, max_results=20)
    # Only ever the same fusion retriever, at increasing depths.
    assert depths
    assert max(depths) <= 200
    assert min(depths) == 4


# --------------------------------------------------------------------- config


def test_expansion_config_is_validated() -> None:
    def build_config(**trake):
        return app_config_from_dict({"aic2026": {"trake": trake}})

    assert build_config(candidate_depth_expansion=[50, 100]).trake.candidate_depth_expansion == (50, 100)
    with pytest.raises(ConfigError, match="must be ascending"):
        build_config(candidate_depth_expansion=[300, 100])
    with pytest.raises(ConfigError, match="candidate_depth_expansion entries must be > 0"):
        build_config(candidate_depth_expansion=[0])
    with pytest.raises(ConfigError, match="candidate_depth_max must be >="):
        build_config(per_event_top_k=500, candidate_depth_max=100)
    with pytest.raises(ConfigError, match="target_complete_video_hypotheses"):
        build_config(target_complete_video_hypotheses=0)
