"""Demo TÌM KIẾM TRÊN VIDEO THẬT bằng SigLIP (CLAUDE.md Phase 2/3, không cần API key).

Nối trọn mắt xích còn thiếu:
    video.mp4 → extract_keyframes (cv2) → SigLIP embed ảnh → KeyframeIndex (Faiss HNSW)
              → encode_text query bằng SigLIP → coarse search cross-modal → top keyframe

Đây là bản THẬT (không mock): dùng model SigLIP local (miễn phí). Lần chạy đầu sẽ TẢI
model (~vài trăm MB) — chỉ cần internet để tải, KHÔNG cần API key. CPU chạy được (chậm).

LƯU Ý THIẾT KẾ: blueprint dự kiến CLIP feature do BTC cấp. Khi CHƯA có, ta dùng SigLIP
cho tầng dense (đặt vào field clip_embedding). Khi có CLIP feature BTC, thêm nó vào để
ensemble 2 encoder (Mục 2.1) — phần index/search không đổi.

Chạy:
    python -m retrieval.video_search_demo video.mp4 "người đang đi bộ" --topk 5
"""
from __future__ import annotations

import sys
from pathlib import Path

from ingestion.build_index import KeyframeIndex
from ingestion.embed_siglip import SiglipEncoder
from ingestion.schemas import KeyframeRecord, RawKeyframe
from ingestion.video_ingest import extract_keyframes
from retrieval.coarse_retriever import CoarseRetriever

# SigLIP ĐA NGÔN NGỮ: query tiếng Việt lẫn tiếng Anh đều được (100+ ngôn ngữ). Đổi
# sang "google/siglip-base-patch16-224" nếu chỉ cần tiếng Anh (nhẹ/nhanh hơn chút).
DEMO_MODEL = "google/siglip-base-patch16-256-multilingual"


def build_index_from_video(
    video_path: str,
    out_dir: str = "artifacts/frames",
    sample_every_s: float = 1.0,
    max_frames: int | None = 60,
    model_name: str = DEMO_MODEL,
) -> tuple[KeyframeIndex, dict[str, RawKeyframe], SiglipEncoder]:
    """Trích keyframe -> embed SigLIP -> dựng KeyframeIndex. Trả (index, raw theo id, encoder)."""
    raws = extract_keyframes(video_path, out_dir, sample_every_s=sample_every_s,
                             max_frames=max_frames)
    if not raws:
        raise RuntimeError("Không trích được keyframe nào từ video.")
    print(f"• Đã trích {len(raws)} keyframe. Đang tải SigLIP + embed (có thể chậm trên CPU)…")
    encoder = SiglipEncoder(model_name=model_name)
    records: list[KeyframeRecord] = []
    for r in raws:
        records.append(KeyframeRecord(
            id=r.id, video_id=r.video_id, timestamp=r.timestamp,
            clip_embedding=encoder.embed(r),   # SigLIP thay CLIP khi chưa có feature BTC
        ))
    index = KeyframeIndex.build(records)
    return index, {r.id: r for r in raws}, encoder


def search_video(index, encoder, query: str, top_k: int = 5) -> list:
    """Mã hoá query bằng SigLIP rồi coarse search cross-modal trên index."""
    qvec = encoder.encode_text(query)
    return CoarseRetriever(index).search(query_clip_vec=qvec, top_k=top_k)


def _cli(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Tìm kiếm bằng chữ trên video thật (SigLIP).")
    p.add_argument("video", help="Đường dẫn file video")
    p.add_argument("query", help="Câu truy vấn (tiếng Việt/Anh)")
    p.add_argument("--out", default="artifacts/frames")
    p.add_argument("--every", type=float, default=1.0)
    p.add_argument("--max", type=int, default=60)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args(argv)

    index, raws, encoder = build_index_from_video(
        args.video, args.out, args.every, args.max)
    results = search_video(index, encoder, args.query, args.topk)
    print(f"\nKết quả cho: “{args.query}”")
    for rank, c in enumerate(results, 1):
        img = raws[c.keyframe_id].image_path
        print(f"  #{rank}  t={c.timestamp:.1f}s  score={c.score:.3f}  {img}")


if __name__ == "__main__":
    _cli(sys.argv[1:])
