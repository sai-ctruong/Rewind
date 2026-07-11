"""Unit test cho retrieval/video_engine — engine tìm kiếm video độ chính xác cao.

Không tải model thật (nặng): bơm MOCK ENCODER đọc MÀU TRUNG BÌNH của ảnh keyframe
làm "ngữ nghĩa" (embed) và map từ khoá màu trong câu (encode_text). Nhờ đó kiểm được
TRỌN pipeline: cắt keyframe -> embed 2 encoder -> dedup ngữ nghĩa -> index 2 slot ->
query prompt ensemble -> RRF fuse -> top-k, với ngữ nghĩa thật (màu khớp màu).

Cần OpenCV để dựng video tổng hợp (importorskip như test_video_ingest).
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ingestion.schemas import RawKeyframe  # noqa: E402
from retrieval.video_engine import VideoSearchEngine  # noqa: E402


# ------------------------- video + mock encoder helpers -----------------------
def _make_video(path, colors, frames_per_color=8, fps=10, size=64):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    assert vw.isOpened()
    for color in colors:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = color  # BGR
        for _ in range(frames_per_color):
            vw.write(frame)
    vw.release()


class ColorMockEncoder:
    """Encoder giả: embedding = hướng màu trung bình của ảnh (unit vector 3D + pad).

    encode_text: bắt từ khoá màu trong câu -> cùng không gian với ảnh. `salt` xoay
    nhẹ không gian để 2 instance mô phỏng 2 encoder KHÁC nhau (ensemble thật sự).
    """

    KEYWORDS = {"đỏ": (0, 0, 255), "red": (0, 0, 255),
                "xanh lá": (0, 255, 0), "green": (0, 255, 0),
                "xanh dương": (255, 0, 0), "blue": (255, 0, 0)}

    def __init__(self, dim: int = 8, salt: float = 0.0):
        self.dim = dim
        self.salt = salt
        self.text_calls: list[str] = []

    def _vec(self, bgr) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[:3] = np.asarray(bgr, dtype=np.float32)
        v[3] = self.salt  # dịch không gian theo encoder
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, raw: RawKeyframe) -> np.ndarray:
        img = cv2.imread(raw.image_path)
        return self._vec(img.reshape(-1, 3).mean(axis=0))

    def encode_text(self, text: str) -> np.ndarray:
        self.text_calls.append(text)
        low = text.lower()
        for kw, bgr in sorted(self.KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in low:
                return self._vec(bgr)
        return self._vec((1, 1, 1))  # không nhận ra màu -> hướng trung tính


@pytest.fixture()
def engine_and_entry(tmp_path):
    video = tmp_path / "scenes.mp4"
    # 3 cảnh: đỏ, xanh lá, xanh dương — mỗi cảnh 10 frame tĩnh (nhiều bản sao).
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return engine, entry


# ---------------------------------- tests -------------------------------------
def test_index_dedups_static_scenes(engine_and_entry) -> None:
    _, entry = engine_and_entry
    # Cảnh tĩnh -> nhiều frame lấy mẫu nhưng dedup ngữ nghĩa gộp về ~1 đại diện/cảnh.
    assert entry.num_sampled >= entry.num_indexed
    assert 3 <= entry.num_indexed <= 5


def test_search_correct_scene_per_color(engine_and_entry) -> None:
    engine, entry = engine_and_entry
    # Cảnh theo thời gian: đỏ ~[0,1)s, xanh lá ~[1,2)s, xanh dương ~[2,3)s.
    for query, (lo, hi) in [("màu đỏ", (0.0, 1.0)), ("green", (1.0, 2.0)),
                             ("xanh dương", (2.0, 3.0))]:
        results = engine.search(entry, query, top_k=3)
        assert results, query
        assert lo <= results[0].timestamp < hi, (
            f"query {query!r}: top-1 t={results[0].timestamp} ngoài cảnh [{lo},{hi})"
        )


def test_ensemble_uses_both_encoders(engine_and_entry) -> None:
    engine, entry = engine_and_entry
    results = engine.search(entry, "red", top_k=3)
    # RRF phải nhận ranked list từ CẢ 2 slot (clip = encoder 1, siglip = encoder 2).
    assert set(results[0].source_ranks.keys()) == {"clip", "siglip"}


def test_query_prompt_ensemble_calls_all_templates(engine_and_entry) -> None:
    engine, entry = engine_and_entry
    enc1 = engine._encoders[0]
    before = len(enc1.text_calls)
    engine.encode_query("màu đỏ")
    called = enc1.text_calls[before:]
    # Mỗi template 1 lần gọi, câu gốc nằm trong biến thể đầu.
    assert len(called) == len(engine.query_templates)
    assert any("màu đỏ" in c for c in called)


def test_encode_query_returns_unit_vectors(engine_and_entry) -> None:
    engine, _ = engine_and_entry
    for v in engine.encode_query("blue"):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_index_dataset_searches_across_videos(tmp_path) -> None:
    # 2 video khác nhau: video A có cảnh đỏ + xanh lá; video B có cảnh xanh dương.
    va = tmp_path / "vidA.mp4"; vb = tmp_path / "vidB.mp4"
    _make_video(va, [(0, 0, 255), (0, 255, 0)], frames_per_color=10)
    _make_video(vb, [(255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])

    entry = engine.index_dataset([va, vb], tmp_path / "frames")
    assert entry.video_id == "__dataset__"
    # Keyframe từ CẢ 2 video đều có trong raws (id toàn cục '{video}/{n}').
    vids = {r.video_id for r in entry.raws.values()}
    assert vids == {"vidA", "vidB"}

    # Query "xanh dương" -> phải trả keyframe từ VIDEO B (chỉ B có cảnh xanh dương).
    res = engine.search(entry, "xanh dương", top_k=3)
    assert res[0].video_id == "vidB"
    # Query "đỏ" -> keyframe từ video A.
    res2 = engine.search(entry, "màu đỏ", top_k=3)
    assert res2[0].video_id == "vidA"


class SignMockOcr:
    """OCR giả: trả token DUY NHẤT theo màu cảnh (không phải tên màu) -> kiểm BM25
    trên OCR text hoạt động (dense encoder KHÔNG biết token này)."""

    def extract(self, raw: RawKeyframe):
        b, g, r = cv2.imread(raw.image_path).reshape(-1, 3).mean(axis=0)  # BGR
        if r > 150 and g < 100 and b < 100:
            return "bienbao SPECIALREDSIGN"
        if g > 150 and r < 100:
            return "bienbao SPECIALGREENSIGN"
        if b > 150 and r < 100:
            return "bienbao SPECIALBLUESIGN"
        return None


def test_ocr_text_search_finds_sign(tmp_path) -> None:
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    engine.set_ocr(SignMockOcr())            # bật OCR (mock)
    entry = engine.index_video(video, tmp_path / "frames")

    # Tìm bằng CHỮ trên biển — token chỉ có trong OCR, dense không biết.
    res = engine.search(entry, "SPECIALGREENSIGN", top_k=3)
    assert res, "phải có kết quả từ BM25 trên OCR"
    assert 1.0 <= res[0].timestamp < 2.0     # cảnh xanh lá ~[1,2)s
    assert "bm25" in res[0].source_ranks     # kết quả đến từ BM25 (OCR text)


def test_single_encoder_mode(tmp_path) -> None:
    video = tmp_path / "v.mp4"
    _make_video(video, [(0, 0, 255), (255, 0, 0)], frames_per_color=8)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=30, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder()])   # chỉ 1 encoder
    entry = engine.index_video(video, tmp_path / "f")
    results = engine.search(entry, "red", top_k=2)
    assert results and results[0].timestamp < 1.0
    assert set(results[0].source_ranks.keys()) == {"clip"}  # không có slot siglip
