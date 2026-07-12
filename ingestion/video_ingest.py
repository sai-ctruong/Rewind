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

BACKEND DECODE (A3 — gỡ nút thắt CPU khi quét dataset lớn):
  - "decord": nếu cài `decord`, decode CHỈ các frame lấy mẫu (get_batch theo index) và
    có thể chạy trên GPU (NVDEC) -> nhanh gấp nhiều lần cho video dài. Cần build decord
    có CUDA để dùng GPU; không có CUDA thì decord vẫn decode-theo-index trên CPU (vẫn
    nhanh hơn đọc tuần tự toàn bộ).
  - "cv2" (mặc định, luôn có): tối ưu bằng `grab()` để BỎ QUA frame không lấy mẫu và chỉ
    `retrieve()` (giải mã + copy) tại điểm mẫu -> tránh giải mã đầy đủ mọi frame.
  - "auto": ưu tiên decord (GPU) nếu import được, ngược lại rơi về cv2.
Cài GPU decode (tuỳ chọn):  pip install decord   (bản CUDA nếu muốn NVDEC).

Chạy:  python -m ingestion.video_ingest path/to/video.mp4 --out artifacts/frames --every 1.0
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional

from .schemas import RawKeyframe


def _color_histogram(frame):
    """Histogram màu HSV (H×S) đã chuẩn hoá — dùng so sánh độ giống giữa 2 frame."""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _probe_fps(video_path: Path) -> float:
    """Đọc FPS nhanh bằng cv2 (rẻ). Fallback 25.0 khi metadata thiếu."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV không mở được video: {video_path}. Kiểm tra codec/đường dẫn."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps if fps > 0 else 25.0


def _decode_samples_cv2(video_path: Path, step: int) -> Iterator[tuple[int, "object"]]:
    """Yield (frame_idx, frame_BGR) CHỈ tại điểm lấy mẫu.

    Tối ưu quan trọng: dùng `grab()` (chỉ tiến con trỏ, KHÔNG giải mã đầy đủ + copy)
    cho các frame BỎ QUA, và `retrieve()` (giải mã) chỉ tại mốc lấy mẫu -> tránh chi
    phí giải mã/màu/copy cho ~(step-1)/step số frame."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV không mở được video: {video_path}. Kiểm tra codec/đường dẫn."
        )
    frame_idx = 0
    try:
        while True:
            if not cap.grab():          # tiến tới frame kế; hết video -> dừng
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()   # chỉ giải mã tại điểm mẫu
                if not ok:
                    break
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def _decode_samples_decord(
    video_path: Path, step: int, use_gpu: bool,
) -> Iterator[tuple[int, "object"]]:  # pragma: no cover - cần cài decord + (CUDA)
    """Yield (frame_idx, frame_BGR) chỉ tại điểm mẫu, dùng decord.

    decord.get_batch(indices) chỉ giải mã ĐÚNG các frame yêu cầu (seek tới keyframe
    gần nhất rồi decode tới đích) và với ctx=gpu(0) dùng NVDEC. Giải mã theo cụm nhỏ
    để chặn bộ nhớ. Trả BGR để khớp phần còn lại (histogram/encode dùng cv2 BGR)."""
    import cv2
    import decord

    ctx = decord.cpu(0)
    if use_gpu:
        try:
            ctx = decord.gpu(0)
            decord.VideoReader(str(video_path), ctx=ctx)  # thử mở trên GPU
        except Exception:
            ctx = decord.cpu(0)  # không có NVDEC -> vẫn decode-theo-index trên CPU
    vr = decord.VideoReader(str(video_path), ctx=ctx)
    indices = list(range(0, len(vr), step))
    for start in range(0, len(indices), 64):     # cụm 64 để chặn RAM/VRAM
        chunk = indices[start:start + 64]
        batch = vr.get_batch(chunk).asnumpy()     # (b,H,W,3) RGB
        for j, fi in enumerate(chunk):
            yield fi, cv2.cvtColor(batch[j], cv2.COLOR_RGB2BGR)


def _iter_samples(
    video_path: Path, step: int, backend: str, use_gpu: bool,
) -> Iterator[tuple[int, "object"]]:
    """Chọn backend decode và yield (frame_idx, frame_BGR) tại điểm mẫu.

    backend: "cv2" | "decord" | "auto" (ưu tiên decord nếu import được, else cv2)."""
    if backend in ("auto", "decord"):
        import importlib.util

        if importlib.util.find_spec("decord") is not None:
            yield from _decode_samples_decord(video_path, step, use_gpu)
            return
        if backend == "decord":
            raise ImportError(
                "backend='decord' nhưng chưa cài decord. `pip install decord` "
                "(bản CUDA nếu muốn NVDEC), hoặc dùng backend='cv2'/'auto'."
            )
    yield from _decode_samples_cv2(video_path, step)


def iter_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    video_id: Optional[str] = None,
    sample_every_s: float = 1.0,
    duplicate_threshold: float = 0.985,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 90,
    save_images: bool = False,
    decode_backend: str = "auto",
    use_gpu: bool = True,
) -> Iterator[RawKeyframe]:
    """Bản GENERATOR của extract_keyframes: YIELD từng RawKeyframe theo dòng (streaming).

    VÌ SAO STREAMING (A4): cho phép PIPELINE decode ‖ embed — consumer bắt đầu embed các
    keyframe đầu NGAY khi producer còn đang decode phần sau của video, thay vì phải chờ
    cắt xong CẢ video mới embed. `extract_keyframes` chỉ là `list(iter_keyframes(...))`.

    Tham số & hành vi giống hệt extract_keyframes (xem đó)."""
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Không thấy file video: {video_path}")
    video_id = video_id or video_path.stem
    frame_dir = Path(out_dir) / video_id
    if save_images:
        frame_dir.mkdir(parents=True, exist_ok=True)

    fps = _probe_fps(video_path)
    step = max(1, int(round(fps * sample_every_s)))
    src = str(video_path)

    last_hist = None
    kept = 0
    # Iterator chỉ trả frame TẠI ĐIỂM MẪU (backend tự bỏ qua frame giữa) -> không giải
    # mã thừa. frame_idx là số thứ tự frame THẬT trong video (để tính timestamp + seek lại).
    for frame_idx, frame in _iter_samples(video_path, step, decode_backend, use_gpu):
        hist = _color_histogram(frame)
        is_dup = (
            last_hist is not None
            and float(cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL))
            >= duplicate_threshold
        )
        if is_dup:
            continue
        ts = frame_idx / fps
        # source_video + frame_idx: cho phép DECODE LẠI frame từ video gốc khi index
        # được nạp từ đĩa (A2 — không lưu ảnh nặng theo index).
        if save_images:  # ghi file .jpg ra đĩa
            img_path = frame_dir / f"{kept:06d}.jpg"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            kf = RawKeyframe(id=f"{video_id}/{kept}", video_id=video_id,
                             timestamp=round(ts, 3), image_path=str(img_path),
                             source_video=src, frame_idx=frame_idx)
        else:  # giữ JPEG trong RAM — KHÔNG chạm đĩa
            ok_enc, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok_enc:
                continue
            kf = RawKeyframe(id=f"{video_id}/{kept}", video_id=video_id,
                             timestamp=round(ts, 3), image_bytes=buf.tobytes(),
                             source_video=src, frame_idx=frame_idx)
        yield kf
        last_hist = hist
        kept += 1
        if max_frames is not None and kept >= max_frames:
            return


def extract_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    video_id: Optional[str] = None,
    sample_every_s: float = 1.0,
    duplicate_threshold: float = 0.985,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 90,
    save_images: bool = False,
    decode_backend: str = "auto",
    use_gpu: bool = True,
) -> list[RawKeyframe]:
    """Cắt keyframe đại diện từ 1 video (bản thu về LIST — bọc quanh `iter_keyframes`).

    MẶC ĐỊNH XỬ LÝ TRONG RAM (save_images=False): mỗi keyframe được nén JPEG và giữ ở
    `image_bytes` (~vài chục KB/frame) thay vì ghi từng file .jpg ra đĩa. Video dài
    sinh hàng nghìn frame -> ghi đĩa vừa chậm vừa tốn hàng GB dung lượng; giữ trong RAM
    rồi embed xong là bỏ (engine còn xóa bytes của frame bị dedup loại). Đặt
    save_images=True nếu thực sự cần file ảnh trên đĩa (vd tool CLI debug).

    DECODE (A3): chỉ giải mã các frame LẤY MẪU (xem docstring module). Muốn xử lý theo
    dòng (pipeline A4) thì dùng `iter_keyframes` trực tiếp.

    Args:
        video_path: đường dẫn file video (.mp4/.avi/...).
        out_dir: thư mục gốc để lưu ảnh keyframe (chỉ dùng khi save_images=True).
        video_id: định danh video (mặc định = tên file không đuôi).
        sample_every_s: khoảng thời gian giữa 2 lần lấy mẫu (giây).
        duplicate_threshold: tương quan histogram (0..1); >= ngưỡng này so với keyframe
            giữ gần nhất thì coi là gần trùng -> BỎ (cảnh tĩnh). 1.0 = giống hệt.
        max_frames: trần số keyframe (None = không giới hạn).
        jpeg_quality: chất lượng JPG (áp cho cả bytes trong RAM lẫn file trên đĩa).
        save_images: True = ghi file .jpg ra đĩa (image_path); False = giữ JPEG trong
            RAM (image_bytes), không đụng đĩa.
        decode_backend: "auto" | "cv2" | "decord" — nguồn giải mã frame (xem module).
        use_gpu: dùng NVDEC (chỉ áp cho decord; không có CUDA thì tự về CPU).

    Returns:
        Danh sách RawKeyframe — sẵn sàng đưa vào pipeline embed SigLIP, OCR/ASR, caption.
    """
    return list(iter_keyframes(
        video_path, out_dir, video_id=video_id, sample_every_s=sample_every_s,
        duplicate_threshold=duplicate_threshold, max_frames=max_frames,
        jpeg_quality=jpeg_quality, save_images=save_images,
        decode_backend=decode_backend, use_gpu=use_gpu,
    ))


def _cli(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Trích xuất keyframe từ video.")
    p.add_argument("video", help="Đường dẫn file video")
    p.add_argument("--out", default="artifacts/frames", help="Thư mục lưu keyframe")
    p.add_argument("--every", type=float, default=1.0, help="Lấy mẫu mỗi N giây")
    p.add_argument("--max", type=int, default=None, help="Trần số keyframe")
    args = p.parse_args(argv)

    frames = extract_keyframes(args.video, args.out, sample_every_s=args.every,
                               max_frames=args.max, save_images=True)
    print(f"Đã trích {len(frames)} keyframe -> {args.out}")
    for kf in frames[:10]:
        print(f"  {kf.id}  t={kf.timestamp}s  {kf.image_path}")
    if len(frames) > 10:
        print(f"  … và {len(frames) - 10} keyframe nữa")


if __name__ == "__main__":
    _cli(sys.argv[1:])
