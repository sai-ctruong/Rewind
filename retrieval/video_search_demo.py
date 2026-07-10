"""Demo TÌM KIẾM TRÊN VIDEO THẬT — bản độ chính xác cao (không cần API key).

Dùng retrieval/video_engine.py::VideoSearchEngine:
  - Ensemble 2 encoder (SigLIP2 + SigLIP multilingual) fuse bằng RRF (Mục 2.1).
  - Query prompt ensemble: trung bình embedding nhiều biến thể câu (Mục 4.2).
  - Lấy mẫu dày 0.5s + dedup ngữ nghĩa (Mục 5.1).

Lần chạy đầu tải 2 model (~750MB tổng) — cần internet để tải, KHÔNG cần API key.
CPU chạy được (indexing chậm hơn bản 1-encoder ~2x, đổi lại chính xác hơn).

Chạy:
    python -m retrieval.video_search_demo video.mp4 "người đang đi bộ" --topk 5
    python -m retrieval.video_search_demo video.mp4 "xe hơi" --single   # 1 encoder, nhanh
"""
from __future__ import annotations

import sys

from retrieval.video_engine import DEFAULT_ENCODERS, VideoSearchEngine


def _cli(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Tìm bằng chữ trên video thật (ensemble SigLIP).")
    p.add_argument("video", help="Đường dẫn file video")
    p.add_argument("query", help="Câu truy vấn (tiếng Việt/Anh)")
    p.add_argument("--out", default="artifacts/frames")
    p.add_argument("--every", type=float, default=0.5, help="Lấy mẫu mỗi N giây")
    p.add_argument("--max", type=int, default=120, help="Trần số keyframe lấy mẫu")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--single", action="store_true",
                   help="Chỉ dùng 1 encoder (SigLIP2) — nhanh hơn, kém chính xác hơn")
    args = p.parse_args(argv)

    names = DEFAULT_ENCODERS[:1] if args.single else DEFAULT_ENCODERS
    engine = VideoSearchEngine(encoder_names=names, sample_every_s=args.every,
                               max_frames=args.max)
    print(f"• Encoder: {', '.join(names)}")
    print("• Đang cắt keyframe + embed (lần đầu tải model, CPU có thể chậm)…")
    entry = engine.index_video(args.video, args.out)
    print(f"• Lấy mẫu {entry.num_sampled} frame -> sau dedup còn {entry.num_indexed} vào index.")

    results = engine.search(entry, args.query, top_k=args.topk)
    print(f"\nKết quả cho: “{args.query}”")
    for rank, c in enumerate(results, 1):
        img = entry.raws[c.keyframe_id].image_path
        srcs = ",".join(f"{k}#{v}" for k, v in sorted(c.source_ranks.items()))
        print(f"  #{rank}  t={c.timestamp:.1f}s  rrf={c.score:.4f}  [{srcs}]  {img}")


if __name__ == "__main__":
    _cli(sys.argv[1:])
