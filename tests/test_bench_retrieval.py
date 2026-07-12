"""Test harness benchmark (evaluation/bench_retrieval) — logic đo/chấm, KHÔNG model thật.

Dùng lại video màu tổng hợp + mock encoder ở test_video_engine để kiểm:
  - measure_embed_throughput trả số liệu hợp lệ và ĐO ĐƯỢC nhánh lô khi có embed_batch.
  - evaluate_labeled tính đúng hit@k / recall@k / MRR trên bộ nhãn theo cửa sổ thời gian.
"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from evaluation.bench_retrieval import (  # noqa: E402
    Label, evaluate_labeled, measure_embed_throughput, relevant_ids, sweep_configs,
)
from retrieval.video_engine import VideoSearchEngine  # noqa: E402
from tests.test_video_engine import (  # noqa: E402
    BatchMockEncoder, ColorMockEncoder, _make_video,
)


@pytest.fixture()
def color_entry(tmp_path):
    video = tmp_path / "scenes.mp4"
    # đỏ ~[0,1)s, xanh lá ~[1,2)s, xanh dương ~[2,3)s
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return engine, entry


# ------------------------------- throughput ----------------------------------
def test_measure_throughput_reports_batched(color_entry) -> None:
    _, entry = color_entry
    raws = list(entry.raws.values())
    enc = BatchMockEncoder()
    r = measure_embed_throughput(enc, raws, batch_size=8, repeat=2)
    assert r["n_frames"] == len(raws)
    assert r["sequential_fps"] > 0
    # có embed_batch -> phải đo được nhánh lô + speedup.
    assert "batched_fps" in r and r["batched_fps"] > 0
    assert "speedup" in r


def test_measure_throughput_encoder_without_batch(color_entry) -> None:
    _, entry = color_entry
    enc = ColorMockEncoder()          # không có embed_batch
    r = measure_embed_throughput(enc, list(entry.raws.values()))
    assert r["sequential_fps"] > 0
    assert "batched_fps" not in r     # không bịa nhánh lô khi encoder không hỗ trợ


# -------------------------------- accuracy -----------------------------------
def test_relevant_ids_by_time_window(color_entry) -> None:
    _, entry = color_entry
    rel = relevant_ids(entry, Label("màu đỏ", entry.video_id, (0.0, 1.0)))
    assert rel and all(entry.raws[i].timestamp < 1.0 for i in rel)


def test_evaluate_labeled_perfect_on_color(color_entry) -> None:
    engine, entry = color_entry
    labels = [
        Label("màu đỏ", entry.video_id, (0.0, 1.0)),
        Label("green", entry.video_id, (1.0, 2.0)),
        Label("xanh dương", entry.video_id, (2.0, 3.0)),
    ]
    report = evaluate_labeled(engine, entry, labels, ks=(1, 5), top_k=5)
    agg = report["aggregate"]
    # Mock encoder xếp đúng cảnh top-1 -> hit@1 = MRR = 1.0.
    assert agg["n_queries"] == 3
    assert agg["hit@1"] == pytest.approx(1.0)
    assert agg["mrr"] == pytest.approx(1.0)
    assert len(report["per_query"]) == 3


def test_evaluate_labeled_penalizes_wrong(color_entry) -> None:
    engine, entry = color_entry
    # Nhãn CỐ Ý sai cửa sổ (đỏ nhưng gán vào vùng xanh dương) -> hit@1 phải là 0.
    bad = [Label("màu đỏ", entry.video_id, (2.0, 3.0))]
    agg = evaluate_labeled(engine, entry, bad, ks=(1,), top_k=5)["aggregate"]
    assert agg["hit@1"] == pytest.approx(0.0)


# --------------------------------- sweep -------------------------------------
def test_sweep_configs_compares_configs(tmp_path) -> None:
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    labels = [
        Label("màu đỏ", "scenes", (0.0, 1.0)),
        Label("xanh dương", "scenes", (2.0, 3.0)),
    ]

    def make(every):
        def factory():
            eng = VideoSearchEngine(sample_every_s=every, max_frames=50, enable_ocr=False)
            eng.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
            return eng
        return factory

    rows = sweep_configs({"every_0.2": make(0.2), "every_0.5": make(0.5)},
                         [video], labels, tmp_path / "f", ks=(1,), top_k=5)
    assert len(rows) == 2
    for r in rows:
        assert "hit@1" in r and "index_seconds" in r and r["num_indexed"] >= 1
    assert {r["config"] for r in rows} == {"every_0.2", "every_0.5"}
