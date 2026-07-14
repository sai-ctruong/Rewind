"""Unit test cho G4 — RAG Reader (retrieval/vqa_module: Reader/MockReader/ClaudeReader).

Reader nhận KẾT QUẢ TOOL đã chuẩn hoá (list dict) + entry (tra caption/OCR/ASR/ảnh) rồi
sinh câu trả lời GROUNDED có trích dẫn keyframe_id. MockReader chạy offline; ClaudeReader
test cấu trúc request bằng fake client (không cần API key/mạng).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrieval.vqa_module import ClaudeReader, MockReader, ReaderAnswer


def _entry(caption=None, ocr=None, asr=None, raws=None):
    return SimpleNamespace(
        caption_by_id=caption or {}, ocr_by_id=ocr or {}, asr_by_id=asr or {},
        raws=raws or {})


def _res(kid, video="v1", ts=1.0, score=0.9):
    return {"keyframe_id": kid, "video_id": video, "timestamp": ts, "score": score}


# --------------------------------- MockReader --------------------------------
def test_mock_reader_cites_top_k_ids() -> None:
    entry = _entry(caption={"a": "người tưới hoa", "b": "trẻ em chơi bóng"})
    results = [_res("a", ts=2.0), _res("b", ts=5.0), _res("c", ts=9.0)]
    ans = MockReader().read("tưới hoa", results, entry, top_k=2)
    assert isinstance(ans, ReaderAnswer)
    assert ans.cited_frame_ids == ["a", "b"]        # đúng top-2, không kèm c
    assert "[a]" in ans.answer and "người tưới hoa" in ans.answer
    assert "2.0s" in ans.answer                      # có dẫn chứng thời gian


def test_mock_reader_uses_ocr_asr_when_no_caption() -> None:
    entry = _entry(ocr={"a": "BIEN BAO STOP"}, asr={"a": "dừng lại đi"})
    ans = MockReader().read("biển báo", [_res("a")], entry)
    assert "BIEN BAO STOP" in ans.answer and "dừng lại đi" in ans.answer


def test_mock_reader_empty_results() -> None:
    ans = MockReader().read("gì đó", [], _entry())
    assert ans.cited_frame_ids == [] and "Không tìm thấy" in ans.answer


def test_mock_reader_without_text_signal_still_grounds_by_location() -> None:
    ans = MockReader().read("q", [_res("a", video="vid9", ts=3.5)], _entry())
    assert "[a]" in ans.answer and "vid9" in ans.answer  # vẫn dẫn id + vị trí


# --------------------------------- ClaudeReader ------------------------------
class _FakeClaude:
    def __init__(self, text):
        self._text = text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def test_claude_reader_builds_images_and_returns_answer() -> None:
    raws = {"a": SimpleNamespace(image_bytes=b"\xff\xd8jpeg"),
            "b": SimpleNamespace(image_bytes=None)}   # b mất ảnh -> chỉ text
    entry = _entry(caption={"a": "mèo trên ghế"}, raws=raws)
    fake = _FakeClaude("Kết quả khớp nhất là [a]: mèo trên ghế.")
    reader = ClaudeReader(client=fake)

    ans = reader.read("mèo", [_res("a"), _res("b")], entry, top_k=2)
    assert ans.cited_frame_ids == ["a", "b"]
    assert "[a]" in ans.answer
    # có đúng 1 khối ảnh (chỉ a có image_bytes) trong content gửi đi
    content = fake.captured["messages"][0]["content"]
    assert sum(1 for c in content if c.get("type") == "image") == 1


def test_claude_reader_requires_key_without_client(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeReader().read("q", [_res("a")], _entry(raws={"a": SimpleNamespace(image_bytes=b"x")}))
