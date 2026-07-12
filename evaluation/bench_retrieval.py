"""Benchmark truy xuất video (CLAUDE.md Mục 11.3 — ĐO THẬT, không đoán tham số).

Gồm 2 phần, tách thành các HÀM THUẦN để test offline bằng mock (mock-first):

  1. THROUGHPUT EMBEDDING (`measure_embed_throughput`) — lượng hoá lợi ích của A1
     (batch embedding): đo frame/giây khi embed LẺ (`.embed`) vs THEO LÔ (`.embed_batch`)
     trên cùng encoder, in tỉ lệ tăng tốc. Đây là con số blueprint yêu cầu đo sau A1.

  2. ĐỘ CHÍNH XÁC TRÊN BỘ NHÃN (`evaluate_labeled`) — với mỗi (query → cửa sổ thời
     gian đúng), chạy engine.search rồi tính Recall@K / MRR bằng evaluation.metrics.
     Bộ nhãn có thể là VIDEO TỔNG HỢP (tự sinh, có ground-truth) hoặc NHÃN THẬT nạp
     từ JSON trỏ vào video trong data/videos/ — cùng một harness, cắm dữ liệu thật là chạy.

Vì sao tách hàm thuần khỏi `main()`: `main()` nạp model SigLIP thật (nặng, cần GPU)
nên KHÔNG test tự động; còn logic đo/chấm phải kiểm được offline bằng encoder mock.

Chạy thật:  python -m evaluation.bench_retrieval
            python -m evaluation.bench_retrieval --labels evaluation/labels.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank

BENCH_DIR = Path("evaluation/benchmarks")


# ============================ 1. THROUGHPUT EMBEDDING =========================
def measure_embed_throughput(
    encoder, raws: list, batch_size: int = 256, repeat: int = 1,
) -> dict:
    """Đo frame/giây khi embed LẺ vs THEO LÔ trên cùng encoder → tỉ lệ tăng tốc (A1).

    `encoder` chỉ cần có `.embed(raw)`; nếu có thêm `.embed_batch(raws, batch_size)`
    thì đo luôn nhánh lô và tính speedup. Trả dict số liệu (lưu JSON để tham chiếu).
    """
    n = len(raws) * repeat
    if n == 0:
        return {"n_frames": 0}

    t0 = time.perf_counter()
    for _ in range(repeat):
        for r in raws:
            encoder.embed(r)
    seq_s = time.perf_counter() - t0
    seq_fps = n / seq_s if seq_s > 0 else float("inf")
    out = {"n_frames": len(raws), "repeat": repeat,
           "sequential_fps": round(seq_fps, 2)}

    batch_fn = getattr(encoder, "embed_batch", None)
    if callable(batch_fn):
        t0 = time.perf_counter()
        for _ in range(repeat):
            batch_fn(raws, batch_size)
        bat_s = time.perf_counter() - t0
        bat_fps = n / bat_s if bat_s > 0 else float("inf")
        out["batched_fps"] = round(bat_fps, 2)
        out["batch_size"] = batch_size
        out["speedup"] = (round(bat_fps / seq_fps, 2)
                          if 0 < seq_fps < float("inf") else None)
    return out


# ========================= 2. ĐỘ CHÍNH XÁC TRÊN NHÃN =========================
@dataclass
class Label:
    """Một mẫu đánh giá: câu query + cửa sổ thời gian ĐÚNG trong 1 video.

    Dùng cửa sổ thời gian (thay vì liệt kê từng frame id) vì keyframe id phụ thuộc
    tham số lấy mẫu/dedup — nhãn theo THỜI GIAN ổn định qua mọi cấu hình, đúng tinh
    thần so cấu hình ở Mục 11.3.
    """

    query: str
    video_id: str
    time_window: tuple[float, float]   # [t_start, t_end) giây — vùng chứa đáp án


def relevant_ids(entry, label: Label) -> set[str]:
    """Tập keyframe id 'đúng' = các frame của đúng video rơi vào cửa sổ thời gian."""
    lo, hi = label.time_window
    return {
        rid for rid, r in entry.raws.items()
        if r.video_id == label.video_id and lo <= r.timestamp < hi
    }


def evaluate_labeled(
    engine, entry, labels: Sequence[Label], ks: Sequence[int] = (1, 5),
    top_k: int = 10, rerank: bool = False,
) -> dict:
    """Chấm Recall@K / hit@K / MRR trên bộ nhãn. Trả per-query + tổng hợp.

    hit@k = KIS Top-k accuracy (có ít nhất 1 frame đúng trong top-k). recall@k hữu
    ích cho AVS (nhiều frame đúng). MRR thưởng việc đẩy frame đúng lên cao."""
    per_query = []
    for lab in labels:
        rel = relevant_ids(entry, lab)
        results = engine.search(entry, lab.query, top_k=top_k, rerank=rerank)
        retrieved = [c.keyframe_id for c in results]
        row = {"query": lab.query, "video_id": lab.video_id,
               "n_relevant": len(rel), "mrr": round(reciprocal_rank(retrieved, rel), 4)}
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(retrieved, rel, k)
            row[f"recall@{k}"] = round(recall_at_k(retrieved, rel, k), 4)
        per_query.append(row)

    agg: dict = {"n_queries": len(per_query)}
    if per_query:
        agg["mrr"] = round(mean(r["mrr"] for r in per_query), 4)
        for k in ks:
            agg[f"hit@{k}"] = round(mean(r[f"hit@{k}"] for r in per_query), 4)
            agg[f"recall@{k}"] = round(mean(r[f"recall@{k}"] for r in per_query), 4)
    return {"aggregate": agg, "per_query": per_query}


def load_labels(path: str | Path) -> list[Label]:
    """Nạp nhãn thật từ JSON: [{"query","video_id","time_window":[lo,hi]}, ...]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Label(query=d["query"], video_id=d["video_id"],
                  time_window=tuple(d["time_window"])) for d in data]


# =============================== quét cấu hình ===============================
def sweep_configs(
    config_factories: dict, video_paths: Sequence, labels: Sequence[Label],
    out_dir: str | Path, ks: Sequence[int] = (1, 5), top_k: int = 10,
) -> list[dict]:
    """Index + chấm NHIỀU cấu hình engine trên CÙNG bộ nhãn → so recall/latency.

    Mục đích (blueprint Mục 11.3): quét tham số [PROVISIONAL] (sample_every_s, efSearch,
    embed_batch_size…) rồi vẽ đường cong recall–latency, chọn điểm "khuỷu tay" — nơi
    tăng thêm chi phí không còn đáng tăng recall. KHÔNG chốt tham số bằng cảm tính.

    `config_factories`: dict[tên -> hàm() trả về VideoSearchEngine đã set encoder].
    Mỗi cấu hình được index LẠI (vì lấy mẫu/encoder khác nhau đổi cả index). Trả 1
    dòng số liệu / cấu hình: thời gian index, số keyframe, và các metric tổng hợp."""
    results: list[dict] = []
    for name, factory in config_factories.items():
        engine = factory()
        t0 = time.perf_counter()
        entry = engine.index_dataset(list(video_paths), out_dir)
        t_index = time.perf_counter() - t0
        acc = evaluate_labeled(engine, entry, labels, ks=ks, top_k=top_k)
        results.append({
            "config": name, "index_seconds": round(t_index, 2),
            "num_indexed": entry.num_indexed, **acc["aggregate"],
        })
    return results


# ============================ synthetic ground-truth =========================
def make_labeled_video(path: Path, video_id: str = "gt_scenes"):
    """Sinh video 3 cảnh có nhãn chữ + trả (video_path, [Label]) — dùng khi CHƯA có
    nhãn thật, để harness chạy được ngay end-to-end (mock-first)."""
    import cv2
    import numpy as np

    fps, spc = 10, 2.0
    scenes = [("CAT", ["a photo of a cat", "một con mèo"], (30, 30, 30)),
              ("DOG", ["a photo of a dog", "một con chó"], (200, 200, 200)),
              ("CAR", ["a photo of a car", "một chiếc xe hơi"], (60, 20, 120))]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (224, 224))
    assert vw.isOpened()
    labels: list[Label] = []
    t = 0.0
    for label, queries, bg in scenes:
        frame = np.zeros((224, 224, 3), np.uint8)
        frame[:] = bg
        cv2.putText(frame, label, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (0, 255, 255), 6)
        for _ in range(int(fps * spc)):
            vw.write(frame)
        for q in queries:
            labels.append(Label(query=q, video_id=video_id, time_window=(t, t + spc)))
        t += spc
    vw.release()
    return path, labels


# ================================== main =====================================
def main(  # pragma: no cover - model thật
    labels_path: Optional[str] = None, sweep: bool = False,
) -> None:
    """Chạy benchmark thật (nạp SigLIP). Đo throughput A1 + Recall/MRR, lưu JSON."""
    from retrieval.video_engine import VideoSearchEngine

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    video_dir = Path("data/videos")
    engine = VideoSearchEngine(enable_ocr=False)

    if labels_path:
        labels = load_labels(labels_path)
        vids = sorted({l.video_id for l in labels})
        video_paths = [video_dir / f"{v}.mp4" for v in vids]
        entry = engine.index_dataset(video_paths, BENCH_DIR / "frames")
    else:
        video, labels = make_labeled_video(BENCH_DIR / "gt_scenes.mp4")
        video_paths = [video]
        entry = engine.index_video(video, BENCH_DIR / "frames")

    # 1) Throughput: đo trên chính encoder đầu của engine, dùng raws đã cắt.
    raws = list(entry.raws.values())
    thr = measure_embed_throughput(engine._load_encoders()[0], raws)
    print(f"[throughput] tuần tự {thr.get('sequential_fps')} fps | "
          f"lô {thr.get('batched_fps')} fps | tăng tốc ×{thr.get('speedup')}")

    # 2) Độ chính xác.
    acc = evaluate_labeled(engine, entry, labels)
    print(f"[accuracy] {acc['aggregate']}")

    report = {"throughput": thr, "accuracy": acc}

    # 3) (tuỳ chọn) Quét sample_every_s để chọn điểm "khuỷu tay" (Mục 11.3).
    if sweep:
        factories = {
            f"every_{s}s": (lambda s=s: VideoSearchEngine(sample_every_s=s, enable_ocr=False))
            for s in (0.5, 1.0, 2.0)
        }
        rows = sweep_configs(factories, video_paths, labels, BENCH_DIR / "sweep_frames")
        for r in rows:
            print(f"[sweep] {r['config']}: hit@1={r.get('hit@1')} hit@5={r.get('hit@5')} "
                  f"mrr={r.get('mrr')} | index {r['index_seconds']}s | {r['num_indexed']} kf")
        report["sweep_sample_every_s"] = rows

    out = BENCH_DIR / "retrieval_benchmark.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Đã lưu] {out}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Benchmark truy xuất video (throughput + accuracy).")
    p.add_argument("--labels", default=None, help="JSON nhãn thật (mặc định: video tổng hợp).")
    p.add_argument("--sweep", action="store_true", help="Quét sample_every_s (0.5/1/2s).")
    args = p.parse_args()
    main(args.labels, sweep=args.sweep)
