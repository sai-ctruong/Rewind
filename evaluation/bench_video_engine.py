"""Benchmark độ chính xác tìm-kiếm-video: engine CŨ vs engine MỚI (Mục 11.3).

So sánh trên video tổng hợp CÓ GROUND-TRUTH (3 cảnh CAT/DOG/CAR với nhãn chữ to,
mỗi cảnh chiếm 1 khoảng thời gian biết trước) — đo Top-1 accuracy:
  - OLD : 1 encoder (SigLIP1 multilingual), không prompt ensemble, lấy mẫu 1.0s.
  - NEW : ensemble SigLIP2 + SigLIP1-multilingual, prompt ensemble, 0.5s + dedup.

Query cả tiếng Anh lẫn tiếng Việt. Kết quả lưu evaluation/benchmarks/ (gitignore) để
tham chiếu khi tinh chỉnh tham số [PROVISIONAL].

Chạy:  python -m evaluation.bench_video_engine
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from retrieval.video_engine import DEFAULT_ENCODERS, VideoSearchEngine

BENCH_DIR = Path("evaluation/benchmarks")

# (nhãn vẽ lên cảnh, khoảng thời gian giây, các query EN + VI)
SCENES = [
    ("CAT", ["a photo of a cat", "một con mèo"]),
    ("DOG", ["a photo of a dog", "một con chó"]),
    ("CAR", ["a photo of a car", "một chiếc xe hơi"]),
]
SECONDS_PER_SCENE = 2.0
FPS = 10


def make_ground_truth_video(path: Path) -> list[tuple[str, float, float, list[str]]]:
    """Video 3 cảnh nhãn chữ; trả [(label, t_start, t_end, queries)]."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (224, 224))
    assert vw.isOpened()
    bgs = [(30, 30, 30), (200, 200, 200), (60, 20, 120)]
    gt = []
    t = 0.0
    for (label, queries), bg in zip(SCENES, bgs):
        frame = np.zeros((224, 224, 3), np.uint8)
        frame[:] = bg
        cv2.putText(frame, label, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (0, 255, 255), 6)
        for _ in range(int(FPS * SECONDS_PER_SCENE)):
            vw.write(frame)
        gt.append((label, t, t + SECONDS_PER_SCENE, queries))
        t += SECONDS_PER_SCENE
    vw.release()
    return gt


def run_config(name: str, engine: VideoSearchEngine, video: Path, out: Path, gt) -> dict:
    t0 = time.perf_counter()
    entry = engine.index_video(video, out / name)
    t_index = time.perf_counter() - t0

    correct, total, details = 0, 0, []
    t0 = time.perf_counter()
    for label, lo, hi, queries in gt:
        for q in queries:
            res = engine.search(entry, q, top_k=1)
            hit = bool(res) and lo <= res[0].timestamp < hi
            correct += int(hit)
            total += 1
            details.append({"query": q, "expect": label,
                            "top1_t": res[0].timestamp if res else None, "hit": hit})
    t_search = (time.perf_counter() - t0) / max(total, 1)
    return {"config": name, "top1_accuracy": correct / total, "n_queries": total,
            "frames_indexed": entry.num_indexed, "frames_sampled": entry.num_sampled,
            "index_seconds": round(t_index, 1),
            "search_seconds_per_query": round(t_search, 2), "details": details}


def main() -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    video = BENCH_DIR / "gt_scenes.mp4"
    gt = make_ground_truth_video(video)

    configs = {
        # Cấu hình CŨ: đúng như bản trước nâng cấp.
        "old_single_siglip1": VideoSearchEngine(
            encoder_names=[DEFAULT_ENCODERS[1]], sample_every_s=1.0,
            query_templates=["{q}"],
        ),
        # Cấu hình MỚI: ensemble + prompt ensemble + 0.5s + dedup.
        "new_ensemble": VideoSearchEngine(),
    }
    report = []
    for name, engine in configs.items():
        print(f"== {name} ==", flush=True)
        r = run_config(name, engine, video, BENCH_DIR / "bench_frames", gt)
        print(f"   Top-1 = {r['top1_accuracy']:.2f} ({r['n_queries']} query) | "
              f"index {r['index_seconds']}s | {r['search_seconds_per_query']}s/query",
              flush=True)
        report.append(r)

    out = BENCH_DIR / "video_engine_benchmark.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Đã lưu] {out}")


if __name__ == "__main__":
    main()
