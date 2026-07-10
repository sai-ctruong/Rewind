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

from ingestion.video_ingest import extract_keyframes  # noqa: E402
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


def test_extract_representative_keyframes(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    # 3 cảnh: đỏ, xanh lá, xanh dương (BGR).
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10, fps=10)

    kfs = extract_keyframes(video, tmp_path / "frames", sample_every_s=0.3)
    # Lấy mẫu ~mỗi 3 frame; các frame cùng cảnh gần trùng bị bỏ -> ~1 keyframe/cảnh.
    assert 3 <= len(kfs) <= 5
    assert all(isinstance(k, RawKeyframe) for k in kfs)
    # Ảnh keyframe thực sự được lưu ra đĩa.
    for k in kfs:
        assert k.image_path and __import__("pathlib").Path(k.image_path).is_file()
    # Timestamp tăng dần, id đúng định dạng.
    ts = [k.timestamp for k in kfs]
    assert ts == sorted(ts)
    assert kfs[0].id == "clip/0" and kfs[0].video_id == "clip"


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


def test_custom_video_id(tmp_path) -> None:
    video = tmp_path / "abc.mp4"
    _make_video(video, [(10, 20, 30), (200, 100, 50)], frames_per_color=8)
    kfs = extract_keyframes(video, tmp_path / "f", video_id="my_vid", sample_every_s=0.3)
    assert all(k.video_id == "my_vid" for k in kfs)
