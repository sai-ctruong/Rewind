"""Unit test cho G3 — Session Memory (retrieval/session_memory).

Kiểm bộ nhớ phiên độc lập: episodic append-only + đọc theo độ mới; semantic feedback
tích luỹ + quy tắc "phản hồi mới thắng"; facts. (Tích hợp với Agent xem test_search_agent.)
"""
from __future__ import annotations

from retrieval.session_memory import SessionMemory, Turn


def test_episodic_append_only_and_recency() -> None:
    m = SessionMemory()
    for q in ["a", "b", "c", "d"]:
        m.record(Turn(query=q))
    assert m.num_turns == 4
    assert m.recent_queries(2) == ["c", "d"]        # đọc theo độ mới
    assert [t.query for t in m.episodic] == ["a", "b", "c", "d"]  # giữ nguyên thứ tự


def test_feedback_accumulates_across_turns() -> None:
    m = SessionMemory()
    m.record(Turn(query="q1", positive_ids=["x"]))
    m.record(Turn(query="q2", positive_ids=["y"], negative_ids=["z"]))
    assert set(m.positive_ids) == {"x", "y"}
    assert m.negative_ids == ["z"]
    assert m.has_feedback()
    assert m.feedback_context() == {"positive_ids": ["x", "y"], "negative_ids": ["z"]}


def test_latest_feedback_wins_on_conflict() -> None:
    m = SessionMemory()
    m.note_feedback(positive_ids=["x"])
    assert m.positive_ids == ["x"]
    # người dùng đổi ý: x thành KHÔNG thích -> rời positive, sang negative
    m.note_feedback(negative_ids=["x"])
    assert m.positive_ids == [] and m.negative_ids == ["x"]
    # đổi ý lần nữa -> quay lại positive
    m.note_feedback(positive_ids=["x"])
    assert m.positive_ids == ["x"] and m.negative_ids == []


def test_facts_semantic_store() -> None:
    m = SessionMemory()
    m.remember("target", "móc khoá đỏ")
    assert m.recall("target") == "móc khoá đỏ"
    assert m.recall("khong_co", default="?") == "?"


def test_summary_shape() -> None:
    m = SessionMemory()
    m.record(Turn(query="q1", positive_ids=["a"]))
    m.remember("k", 1)
    s = m.summary()
    assert s["turns"] == 1 and s["positive_ids"] == ["a"] and s["facts"] == {"k": 1}


def test_recent_zero_is_empty() -> None:
    m = SessionMemory()
    m.record(Turn(query="q"))
    assert m.recent(0) == []
