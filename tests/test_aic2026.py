from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic2026.dataset import AICDatasetLoader, official_frame_id
from aic2026.engine import AICCompetitionEngine
from aic2026.metrics import (
    FrameRange,
    KISGroundTruth,
    QAGroundTruth,
    RankedAnswer,
    TRAKEGroundTruth,
    evaluate_query,
    final_score_from_r_scores,
    kis_r_score,
    qa_r_score,
    trake_r_score,
    write_submission,
)


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        low = text.lower()
        if "red" in low or "do" in low:
            return np.array([1.0, 0.0], dtype=np.float32)
        if "blue" in low or "xanh" in low:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([0.7, 0.7], dtype=np.float32)


def make_aic_root(root: Path) -> Path:
    for rel in ("clip-features-32", "map-keyframes", "keyframes/L01_V001", "objects/L01_V001", "media-info", "video"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "map-keyframes" / "L01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,100\n2,1.0,30.0,130\n3,2.0,30.0,160\n",
        encoding="utf-8",
    )
    np.save(root / "clip-features-32" / "L01_V001.npy", np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float16))
    for i in (1, 2, 3):
        Image.new("RGB", (8, 8), (255 if i != 2 else 0, 0, 255 if i == 2 else 0)).save(
            root / "keyframes" / "L01_V001" / f"{i:03d}.jpg"
        )
        (root / "objects" / "L01_V001" / f"{i:03d}.json").write_text(json.dumps({
            "detection_class_entities": ["red object" if i != 2 else "blue object"],
            "detection_scores": ["0.9"],
        }), encoding="utf-8")
    (root / "media-info" / "L01_V001.json").write_text(json.dumps({
        "title": "tiny video",
        "keywords": ["test"],
    }), encoding="utf-8")
    return root


def test_loader_uses_official_frame_idx(tmp_path) -> None:
    root = make_aic_root(tmp_path / "data")
    entry, stats = AICDatasetLoader(root).build_entry()
    assert stats.frames == 3 and stats.feature_dim == 2
    first = entry.raws["L01_V001/100"]
    assert first.frame_idx == 100
    assert official_frame_id(entry, first.id) == "100"
    assert first.image_path and first.objects == ["red object"]


def test_engine_search_and_trake_return_official_rows(tmp_path) -> None:
    root = make_aic_root(tmp_path / "data")
    entry, _ = AICDatasetLoader(root).build_entry()
    engine = AICCompetitionEngine(entry, text_encoder=TinyTextEncoder(), query_templates=("{q}",), bm25_weight=0.0)

    kis = engine.search_kis("red", top_k=2)
    assert kis[0].row()[0] == "L01_V001"
    assert kis[0].row()[1] in {"100", "160"}

    trake, matches = engine.search_trake(["red", "blue"], per_event_k=3, max_results=5, refine_window_s=0.1)
    assert trake
    assert trake[0].row()[0] == "L01_V001"
    assert trake[0].event_frame_ids == ["100", "130"]
    assert matches[0].steps[0].timestamp < matches[0].steps[1].timestamp


def test_official_metrics_match_aic_examples() -> None:
    gt = KISGroundTruth("V", (FrameRange(500, 510),))
    assert kis_r_score(RankedAnswer("V", ("505",)), gt) == 1.0
    assert kis_r_score(RankedAnswer("V", ("600",)), gt) == 0.0
    assert kis_r_score(RankedAnswer("X", ("505",)), gt) == 0.0

    qa_gt = QAGroundTruth("V", (FrameRange(800, 900),), ("mau xanh", "xanh"))
    assert qa_r_score(RankedAnswer("V", ("888",), "mau xanh"), qa_gt) == 1.0
    assert qa_r_score(RankedAnswer("V", ("888",), "mau trang"), qa_gt) == 0.0

    trake_gt = TRAKEGroundTruth("V", (FrameRange(95, 105), FrameRange(145, 155), FrameRange(195, 205), FrameRange(245, 255)))
    assert trake_r_score(RankedAnswer("V", ("101", "156", "203", "251")), trake_gt) == pytest.approx(0.75)


def test_final_score_is_mean_of_best_r_at_fixed_cutoffs() -> None:
    report = final_score_from_r_scores([0.5, 0.1, 0.8, 0.6])
    assert report["R@1"] == 0.5
    assert report["R@5"] == 0.8
    assert report["R@20"] == 0.8
    assert report["Final Score"] == pytest.approx(0.74)


def test_evaluate_query_and_write_submission(tmp_path) -> None:
    gt = KISGroundTruth("V", (FrameRange(10, 20),))
    report = evaluate_query("kis", [RankedAnswer("X", ("15",)), RankedAnswer("V", ("12",))], gt)
    assert report["R@1"] == 0.0 and report["R@5"] == 1.0
    out = write_submission([["V", "12"], ["V", "13"]], tmp_path / "sub.csv")
    assert out.read_text(encoding="utf-8").splitlines() == ["V,12", "V,13"]