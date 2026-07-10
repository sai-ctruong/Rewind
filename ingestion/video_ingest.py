"""Trích xuất keyframe TỪ FILE VIDEO thật (CLAUDE.md — ô "Video → Keyframe" ở đầu
sơ đồ Mục 3, phần upstream mà blueprint giả định BTC cấp sẵn).

VÌ SAO CÓ FILE NÀY: phần còn lại của hệ thống bắt đầu TỪ keyframe (KeyframeRecord/
RawKeyframe). Module này lấp mắt xích còn thiếu: đọc .mp4 -> chọn keyframe đại diện
-> lưu ảnh ra đĩa -> sinh RawKeyframe (có image_path + timestamp thật) để đưa vào
pipeline embed/OCR/caption/index như bình thường.

CHIẾN LƯỢC CHỌN KEYFRAME:
  - Lấy mẫu ĐỀU theo thời gian (mỗi `sample_every_s` giây) — đủ phủ nội dung.
  - BỎ frame gần trùng liên tiếp bằng tương quan histogram màu (cảnh tĩnh sinh nhiều
    frame giống hệt) -> giảm số keyframe mà không mất nội dung. Đây là bước dedup THÔ
    ở mức pixel; dedup tinh hơn ở mức ngữ nghĩa (embedding) làm sau ở ingestion/dedup.py.

Chỉ cần OpenCV (cv2) — KHÔNG cần GPU/API. Chạy:
    python -m ingestion.video_ingest path/to/video.mp4 --out artifacts/frames --every 1.0
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .schemas import RawKeyframe


def _color_histogram(frame):
    """Histogram màu HSV (H×S) đã chuẩn hoá — dùng so sánh độ giống giữa 2 frame."""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def extract_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    video_id: Optional[str] = None,
    sample_every_s: float = 1.0,
    duplicate_threshold: float = 0.985,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 90,
) -> list[RawKeyframe]:
    """Cắt keyframe đại diện từ 1 video và lưu ảnh ra đĩa.

    Args:
        video_path: đường dẫn file video (.mp4/.avi/...).
        out_dir: thư mục gốc để lưu ảnh keyframe (tạo subfolder theo video_id).
        video_id: định danh video (mặc định = tên file không đuôi).
        sample_every_s: khoảng thời gian giữa 2 lần lấy mẫu (giây).
        duplicate_threshold: tương quan histogram (0..1); >= ngưỡng này so với keyframe
            giữ gần nhất thì coi là gần trùng -> BỎ (cảnh tĩnh). 1.0 = giống hệt.
        max_frames: trần số keyframe (None = không giới hạn).
        jpeg_quality: chất lượng JPG lưu ra.

    Returns:
        Danh sách RawKeyframe (id, video_id, timestamp, image_path) — sẵn sàng đưa vào
        IngestionPipeline (embed SigLIP, OCR/ASR, caption...).
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Không thấy file video: {video_path}")
    video_id = video_id or video_path.stem
    frame_dir = Path(out_dir) / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV không mở được video: {video_path}. Kiểm tra codec/đường dẫn."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0  # fallback khi metadata fps thiếu
    step = max(1, int(round(fps * sample_every_s)))

    keyframes: list[RawKeyframe] = []
    last_hist = None
    frame_idx = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:  # điểm lấy mẫu đều
                hist = _color_histogram(frame)
                is_dup = (
                    last_hist is not None
                    and float(cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL))
                    >= duplicate_threshold
                )
                if not is_dup:
                    ts = frame_idx / fps
                    img_path = frame_dir / f"{kept:06d}.jpg"
                    cv2.imwrite(str(img_path), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    keyframes.append(RawKeyframe(
                        id=f"{video_id}/{kept}", video_id=video_id,
                        timestamp=round(ts, 3), image_path=str(img_path),
                    ))
                    last_hist = hist
                    kept += 1
                    if max_frames is not None and kept >= max_frames:
                        break
            frame_idx += 1
    finally:
        cap.release()
    return keyframes


def _cli(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Trích xuất keyframe từ video.")
    p.add_argument("video", help="Đường dẫn file video")
    p.add_argument("--out", default="artifacts/frames", help="Thư mục lưu keyframe")
    p.add_argument("--every", type=float, default=1.0, help="Lấy mẫu mỗi N giây")
    p.add_argument("--max", type=int, default=None, help="Trần số keyframe")
    args = p.parse_args(argv)

    frames = extract_keyframes(args.video, args.out, sample_every_s=args.every,
                               max_frames=args.max)
    print(f"Đã trích {len(frames)} keyframe -> {args.out}")
    for kf in frames[:10]:
        print(f"  {kf.id}  t={kf.timestamp}s  {kf.image_path}")
    if len(frames) > 10:
        print(f"  … và {len(frames) - 10} keyframe nữa")


if __name__ == "__main__":
    _cli(sys.argv[1:])
