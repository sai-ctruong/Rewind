"""Test tầng VLM rerank — parse điểm + luồng tích hợp vào VideoSearchEngine.

Không tải VLM thật (nặng ~4GB): bơm MOCK RERANKER "nhìn" màu trung bình ảnh keyframe
để chấm khớp với từ khoá màu trong câu. Nhờ đó kiểm được: engine.search(rerank=True)
chạy coarse -> rerank -> trả RerankedCandidate đúng thứ tự, và reranker thực sự được
gọi với ĐƯỜNG DẪN ẢNH (đúng hợp đồng: VLM cần ảnh thật).
"""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.vlm_rerank import _parse_score

cv2 = pytest.importorskip("cv2")

from retrieval.fine_rerank import RerankedCandidate, Reranker  # noqa: E402
from retrieval.video_engine import VideoSearchEngine  # noqa: E402
from tests.test_video_engine import ColorMockEncoder, _make_video  # noqa: E402


# ------------------------------- _parse_score --------------------------------
@pytest.mark.parametrize("ans,expected", [
    ("8", 0.8), ("Điểm: 10", 1.0), ("0", 0.0), ("khớp 7/10", 0.7),
    ("15", 1.0),          # clamp về 1.0
    ("không rõ", 0.0),    # không có số -> 0
    ("9.5", 0.95),
])
def test_parse_score(ans, expected) -> None:
    assert _parse_score(ans) == pytest.approx(expected)


# --------------------------- mock VLM reranker -------------------------------
class ColorMockReranker(Reranker):
    """Chấm điểm = độ giống giữa màu trung bình ảnh (đọc từ context=đường dẫn) và
    màu trong câu. Mô phỏng VLM 'nhìn ảnh + hiểu chữ' mà không cần model."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def score(self, query_text: str, keyframe_id: str, context: object) -> tuple[float, str]:
        self.calls.append((keyframe_id, str(context)))
        img = cv2.imread(str(context))
        if img is None:
            return 0.0, "thiếu ảnh"
        mean = img.reshape(-1, 3).mean(axis=0).astype(np.float32)  # BGR
        mean /= (np.linalg.norm(mean) + 1e-9)
        low = query_text.lower()
        for kw, bgr in ColorMockEncoder.KEYWORDS.items():
            if kw in low:
                tgt = np.asarray(bgr, np.float32); tgt /= np.linalg.norm(tgt)
                return float(max(0.0, mean @ tgt)), f"mock màu {kw}"
        return 0.0, "không nhận ra màu"


@pytest.fixture()
def engine_entry(tmp_path):
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, rerank_pool=6, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return engine, entry


def test_rerank_path_returns_reranked_candidates(engine_entry) -> None:
    engine, entry = engine_entry
    mock = ColorMockReranker()
    engine.set_reranker(mock)
    results = engine.search(entry, "màu đỏ", top_k=3, rerank=True)
    assert results and isinstance(results[0], RerankedCandidate)
    assert results[0].timestamp < 1.0                 # cảnh đỏ ~[0,1)s -> top-1
    assert results[0].explanation and "mock" in results[0].explanation
    # Reranker phải nhận ĐƯỜNG DẪN ẢNH (kết thúc .jpg), không phải text.
    assert mock.calls and mock.calls[0][1].endswith(".jpg")


def test_rerank_false_skips_reranker(engine_entry) -> None:
    engine, entry = engine_entry
    mock = ColorMockReranker()
    engine.set_reranker(mock)
    engine.search(entry, "green", top_k=3, rerank=False)
    assert mock.calls == []                           # rerank=False -> KHÔNG gọi VLM


def test_rerank_reorders_toward_query(engine_entry) -> None:
    engine, entry = engine_entry
    engine.set_reranker(ColorMockReranker())
    # Mỗi màu -> top-1 đúng cảnh tương ứng, chứng minh rerank hiểu câu.
    for q, (lo, hi) in [("xanh dương", (2.0, 3.0)), ("green", (1.0, 2.0))]:
        res = engine.search(entry, q, top_k=3, rerank=True)
        assert lo <= res[0].timestamp < hi, (q, res[0].timestamp)
