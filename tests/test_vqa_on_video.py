"""Unit test cho cầu nối VQA → VIDEO THẬT (retrieval/vqa_module.answer_on_video).

Dùng video mock 3 cảnh + ColorMockEncoder: câu hỏi định vị được cảnh đúng (qua search),
rồi answerer trả lời trên cửa sổ quanh cảnh đó. Bơm caption giả vào entry để mô phỏng
"video đã bật Caption lúc index" — nếu không có chữ, mock không có gì để suy luận
(giới hạn thật, được test tường minh bên dưới).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")

from retrieval.video_engine import VideoSearchEngine  # noqa: E402
from retrieval.vqa_module import (  # noqa: E402
    ClaudeVqaAnswerer, MockVqaAnswerer, answer_on_video, entry_records)
from tests.test_video_engine import ColorMockEncoder, _make_video  # noqa: E402


@pytest.fixture()
def engine_entry(tmp_path):
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return engine, entry


# ------------------------------ dựng record -----------------------------------
def test_entry_records_merge_text_signals(engine_entry) -> None:
    _, entry = engine_entry
    kid = next(iter(entry.raws))
    entry.caption_by_id[kid] = "một chiếc xe màu đỏ"
    entry.ocr_by_id[kid] = "STOP"
    recs = entry_records(entry)
    r = next(r for r in recs if r.id == kid)
    assert r.llm_caption == "một chiếc xe màu đỏ" and r.ocr_text == "STOP"
    assert [x.timestamp for x in recs] == sorted(x.timestamp for x in recs)  # theo thời gian


# ------------------------------ trả lời trên video ----------------------------
def test_answer_locates_window_around_matching_scene(engine_entry) -> None:
    engine, entry = engine_entry
    ans, info = answer_on_video(engine, entry, "cảnh màu đỏ", window_s=0.5,
                                answerer=MockVqaAnswerer())
    assert info["center_time"] < 1.0          # tâm cửa sổ rơi vào cảnh đỏ [0,1)s
    assert info["frame_ids"]                  # có frame trong cửa sổ
    assert ans.used_frame_ids


def test_answer_counts_from_caption_on_real_entry(engine_entry) -> None:
    # CHÍNH CÂU HỎI định vị cửa sổ (đúng blueprint bước [6]) -> câu hỏi phải chứa đủ
    # dấu hiệu để tìm ra cảnh; caption đặt lên đúng frame mà câu hỏi đó tìm tới.
    engine, entry = engine_entry
    question = "Có bao nhiêu xe màu đỏ?"
    hit = engine.search(entry, question, top_k=1)[0]
    entry.caption_by_id[hit.keyframe_id] = "Có 3 chiếc xe màu đỏ trên đường."
    ans, info = answer_on_video(engine, entry, question, window_s=0.3,
                                answerer=MockVqaAnswerer())
    assert hit.keyframe_id in ans.used_frame_ids   # cửa sổ phủ đúng frame có caption
    assert ans.value == 3


def test_answer_without_text_signal_is_honest(engine_entry) -> None:
    """Video chưa bật Caption/OCR/ASR -> mock KHÔNG bịa: nói không có mô tả."""
    engine, entry = engine_entry
    ans, _ = answer_on_video(engine, entry, "Chuyện gì đang diễn ra?",
                             answerer=MockVqaAnswerer())
    assert "không có mô tả" in ans.answer


def test_answer_requires_question(engine_entry) -> None:
    engine, entry = engine_entry
    with pytest.raises(ValueError):
        answer_on_video(engine, entry, "   ")


# ------------------------------ Claude vision nhận ẢNH ------------------------
class _FakeClaude:
    def __init__(self, text="2"):
        self._text = text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def test_claude_answerer_receives_real_images(engine_entry) -> None:
    """Hồi quy: ClaudeVqaAnswerer đọc `f.image_path` — mà KeyframeRecord KHÔNG có field
    đó, nên trước đây nó không bao giờ gửi được ảnh nào. Giờ ảnh đi qua map `images`."""
    engine, entry = engine_entry
    fake = _FakeClaude("2")
    ans, _ = answer_on_video(engine, entry, "Có bao nhiêu người?", window_s=0.4,
                             answerer=ClaudeVqaAnswerer(client=fake))
    content = fake.captured["messages"][0]["content"]
    n_img = sum(1 for c in content if c.get("type") == "image")
    assert n_img > 0                      # ẢNH THẬT đã được gửi cho Claude
    assert ans.value == 2
