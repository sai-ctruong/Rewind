"""Unit test cho ingestion/video_ingest — trích keyframe từ VIDEO THẬT.

Tự tạo một video tổng hợp bằng cv2 (3 cảnh màu khác nhau, mỗi cảnh nhiều frame gần
trùng) rồi kiểm: trích đúng số keyframe đại diện (bỏ frame trùng), lưu ảnh ra đĩa,
timestamp tăng dần. Chỉ cần OpenCV — KHÔNG cần torch/model/API.

SigLIP (bản thật, tải model nặng) KHÔNG test ở đây; kiểm qua script demo thủ công
retrieval/video_search_demo.py.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # bỏ qua nếu môi trường chưa cài OpenCV

from ingestion.video_ingest import (  # noqa: E402
    _iter_samples, extract_keyframes,
)
from ingestion.schemas import RawKeyframe  # noqa: E402


def _make_video(path, colors, frames_per_color=8, fps=10, size=64):
    """Tạo video: mỗi màu là 1 'cảnh' gồm nhiều frame giống hệt nhau."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    assert vw.isOpened(), "Không mở được VideoWriter (thiếu codec mp4v?)"
    for color in colors:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = color  # BGR
        for _ in range(frames_per_color):
            vw.write(frame)
    vw.release()


def test_extract_representative_keyframes_in_memory(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    # 3 cảnh: đỏ, xanh lá, xanh dương (BGR).
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10, fps=10)

    frame_dir = tmp_path / "frames"
    kfs = extract_keyframes(video, frame_dir, sample_every_s=0.3)  # mặc định: RAM
    # Lấy mẫu ~mỗi 3 frame; các frame cùng cảnh gần trùng bị bỏ -> ~1 keyframe/cảnh.
    assert 3 <= len(kfs) <= 5
    assert all(isinstance(k, RawKeyframe) for k in kfs)
    # MẶC ĐỊNH: ảnh giữ trong RAM (image_bytes), KHÔNG ghi file ra đĩa.
    for k in kfs:
        assert k.image_path is None
        assert k.image_bytes and len(k.image_bytes) > 0
    # image_bytes là JPEG hợp lệ -> decode lại được thành ảnh.
    from ingestion.schemas import load_cv2_image
    img = load_cv2_image(kfs[0])
    assert img is not None and img.ndim == 3
    # Không tạo thư mục frame trên đĩa khi không lưu ảnh.
    assert not frame_dir.exists()
    # Timestamp tăng dần, id đúng định dạng.
    ts = [k.timestamp for k in kfs]
    assert ts == sorted(ts)
    assert kfs[0].id == "clip/0" and kfs[0].video_id == "clip"


def test_extract_save_images_writes_disk(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10, fps=10)
    kfs = extract_keyframes(video, tmp_path / "frames", sample_every_s=0.3,
                            save_images=True)
    from pathlib import Path
    for k in kfs:
        assert k.image_bytes is None
        assert k.image_path and Path(k.image_path).is_file()


def test_max_frames_caps(tmp_path) -> None:
    video = tmp_path / "v.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)], frames_per_color=10)
    kfs = extract_keyframes(video, tmp_path / "f", sample_every_s=0.2, max_frames=2)
    assert len(kfs) == 2


def test_duplicate_dropping(tmp_path) -> None:
    video = tmp_path / "static.mp4"
    # 1 cảnh tĩnh duy nhất -> mọi mẫu gần trùng -> chỉ giữ 1 keyframe.
    _make_video(video, [(120, 60, 200)], frames_per_color=30, fps=10)
    kfs = extract_keyframes(video, tmp_path / "f", sample_every_s=0.2)
    assert len(kfs) == 1  # cảnh tĩnh -> 1 đại diện


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_keyframes(tmp_path / "khong_co.mp4", tmp_path / "f")


def test_cv2_backend_only_samples_step_frames(tmp_path) -> None:
    """A3: iterator cv2 chỉ trả frame TẠI ĐIỂM MẪU (0, step, 2*step, …)."""
    video = tmp_path / "v.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0)], frames_per_color=12, fps=10)
    step = 5
    idxs = [fi for fi, _frame in _iter_samples(video, step, "cv2", use_gpu=False)]
    assert idxs == list(range(0, 24, step))     # 24 frame -> 0,5,10,15,20


def test_backend_cv2_matches_auto(tmp_path) -> None:
    """cv2 và auto (decord vắng -> về cv2) cho cùng keyframe."""
    video = tmp_path / "v.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10, fps=10)
    a = extract_keyframes(video, tmp_path / "f", sample_every_s=0.3, decode_backend="auto")
    b = extract_keyframes(video, tmp_path / "f", sample_every_s=0.3, decode_backend="cv2")
    assert [k.frame_idx for k in a] == [k.frame_idx for k in b]
    assert len(a) == len(b) >= 3


def test_decord_backend_missing_raises(tmp_path) -> None:
    """Yêu cầu backend='decord' khi chưa cài -> báo lỗi rõ ràng (không im lặng)."""
    import importlib.util

    if importlib.util.find_spec("decord") is not None:
        pytest.skip("decord đã cài — không kiểm nhánh thiếu")
    video = tmp_path / "v.mp4"
    _make_video(video, [(0, 0, 255)], frames_per_color=8)
    with pytest.raises(ImportError):
        extract_keyframes(video, tmp_path / "f", decode_backend="decord")


def test_custom_video_id(tmp_path) -> None:
    video = tmp_path / "abc.mp4"
    _make_video(video, [(10, 20, 30), (200, 100, 50)], frames_per_color=8)
    kfs = extract_keyframes(video, tmp_path / "f", video_id="my_vid", sample_every_s=0.3)
    assert all(k.video_id == "my_vid" for k in kfs)
