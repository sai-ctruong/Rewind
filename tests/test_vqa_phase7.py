"""Unit test Phase 7 — vqa_module (CLAUDE.md Mục 8).

DoD Phase 7: "Trả lời đúng ví dụ case study VQA (đếm số nến, xác định người tặng quà)".

Ta test 2 ví dụ case-study đó + retrieve temporal window + các nhánh intent + guard.
Chạy offline bằng MockVqaAnswerer (khai thác llm_caption đã sinh lúc indexing — Mục 2.4).
"""
from __future__ import annotations

import numpy as np
import pytest

from ingestion.schemas import KeyframeRecord
from retrieval.vqa_module import (
    ClaudeVqaAnswerer,
    MockVqaAnswerer,
    VqaModule,
    retrieve_temporal_window,
)


def _rec(kf_id: str, video: str, ts: float, caption: str, objects=None) -> KeyframeRecord:
    return KeyframeRecord(
        id=kf_id, video_id=video, timestamp=ts,
        clip_embedding=np.zeros(4, dtype=np.float32),
        objects=objects or [],
        llm_caption=caption,
    )


# -----------------------------------------------------------------------------
# DoD case 1: đếm số nến
# -----------------------------------------------------------------------------
def test_counting_candles() -> None:
    records = [
        _rec("v/0", "birthday", 0.0, "Mọi người quây quần quanh bàn tiệc."),
        _rec("v/1", "birthday", 5.0,
             "Một chiếc bánh sinh nhật với 5 ngọn nến đang cháy.", objects=["bánh", "nến"]),
        _rec("v/2", "birthday", 8.0, "Cô gái chuẩn bị thổi nến."),
    ]
    ans = VqaModule().answer(
        "Trong video có bao nhiêu ngọn nến trên bánh sinh nhật?",
        records, video_id="birthday",
    )
    assert ans.value == 5
    assert ans.answer == "5"
    assert "v/1" in ans.used_frame_ids


def test_counting_candles_word_number() -> None:
    records = [
        _rec("v/1", "b", 5.0, "Chiếc bánh có ba ngọn nến nhỏ.", objects=["nến"]),
    ]
    ans = VqaModule().answer("Có bao nhiêu ngọn nến?", records)
    assert ans.value == 3  # số viết bằng chữ 'ba'


# -----------------------------------------------------------------------------
# DoD case 2: xác định người tặng quà
# -----------------------------------------------------------------------------
def test_identify_gift_giver() -> None:
    records = [
        _rec("g/0", "party", 10.0, "Cả nhóm đang trò chuyện vui vẻ."),
        _rec("g/1", "party", 15.0,
             "Người đàn ông áo xanh đang tặng quà cho cô gái.", objects=["người", "quà"]),
    ]
    ans = VqaModule().answer("Ai là người tặng quà?", records, video_id="party")
    assert "người đàn ông" in ans.answer.lower()
    assert "g/1" in ans.used_frame_ids


# -----------------------------------------------------------------------------
# Temporal window retrieval
# -----------------------------------------------------------------------------
def test_temporal_window_filters_by_center_and_video() -> None:
    records = [
        _rec("a", "V1", 0.0, "x"),
        _rec("b", "V1", 20.0, "y"),
        _rec("c", "V1", 25.0, "z"),
        _rec("d", "V2", 22.0, "w"),
    ]
    window = retrieve_temporal_window(records, center_time=22.0, window_s=5.0, video_id="V1")
    ids = [r.id for r in window]
    assert ids == ["b", "c"]          # trong [17,27] & video V1, sắp theo thời gian
    assert "d" not in ids             # khác video bị loại


def test_window_no_center_returns_all_sorted() -> None:
    records = [_rec("b", "V", 20.0, "y"), _rec("a", "V", 5.0, "x")]
    window = retrieve_temporal_window(records)
    assert [r.id for r in window] == ["a", "b"]


# -----------------------------------------------------------------------------
# Các nhánh khác + guard
# -----------------------------------------------------------------------------
def test_count_fallback_when_no_number() -> None:
    records = [
        _rec("v/0", "V", 0.0, "Có một chú chó.", objects=["người"]),
        _rec("v/1", "V", 1.0, "Hai người đang đi bộ.", objects=["người"]),
    ]
    # 'người' xuất hiện: caption frame 2 có 'hai' + 'người' -> đếm được 2.
    ans = VqaModule().answer("Có bao nhiêu người?", records)
    assert ans.value == 2


def test_describe_intent() -> None:
    records = [_rec("v/0", "V", 0.0, "Một cảnh hoàng hôn đẹp.")]
    ans = VqaModule().answer("Chuyện gì đang diễn ra?", records)
    assert "hoàng hôn" in ans.answer


def test_empty_window_returns_no_data() -> None:
    records = [_rec("v/0", "V", 0.0, "x")]
    ans = VqaModule().answer("bao nhiêu nến?", records, video_id="khac", )
    assert "không có dữ liệu" in ans.answer


def test_mock_answerer_direct() -> None:
    frames = [_rec("v/1", "V", 5.0, "Bàn có 7 ngọn nến.", objects=["nến"])]
    ans = MockVqaAnswerer().answer("bao nhiêu nến?", frames)
    assert ans.value == 7


def test_claude_vqa_guard() -> None:
    with pytest.raises((ImportError, RuntimeError)):
        ClaudeVqaAnswerer()
