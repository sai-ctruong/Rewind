"""Unit test Phase 4 — query_understanding + query_expansion (CLAUDE.md Mục 8).

DoD Phase 4: "JSON output đúng schema Mục 7 cho cả 5 câu query mẫu".

Ta test trên 5 câu lấy từ case-study trong CLAUDE.md (Mục 4.1) + KISC demo:
kiểm (a) StructuredQuery hợp lệ schema cho CẢ 5 câu, (b) các trường parse chắc chắn
(temporal_order, color, query_type, location) đúng như mong đợi.

Chạy offline bằng MockQueryUnderstander/MockQueryExpander (chưa có API — mock-first).
"""
from __future__ import annotations

import pytest

from ingestion.schemas import QUERY_TYPES, StructuredQuery
from retrieval.query_expansion import MockQueryExpander
from retrieval.query_understanding import (
    ClaudeQueryUnderstander,
    MockQueryUnderstander,
)

# 5 câu case-study.
Q1 = "Tôi làm rơi chùm chìa khóa có móc khóa gấu bông màu hồng ở quầy bán hoa quả."
Q2 = "Người đàn ông cởi mũ trước khi vào phòng."
Q3 = "Tìm tất cả các cảnh có trẻ em đang tưới hoa trong công viên."
Q4 = "Trong video có bao nhiêu ngọn nến trên bánh sinh nhật?"
Q5 = "Tôi gặp một người bạn cũ ở quán cà phê ngoài trời tuần trước, anh ấy mặc áo màu xanh dương."
SAMPLES = [Q1, Q2, Q3, Q4, Q5]


def _assert_valid_schema(sq: StructuredQuery) -> None:
    """Kiểm StructuredQuery đúng schema Mục 7 (kiểu + ràng buộc)."""
    assert isinstance(sq.raw_text, str) and sq.raw_text
    assert isinstance(sq.objects, list) and all(isinstance(x, str) for x in sq.objects)
    assert isinstance(sq.actions, list) and all(isinstance(x, str) for x in sq.actions)
    assert sq.location is None or isinstance(sq.location, str)
    assert isinstance(sq.attributes, dict)
    assert sq.time_constraint is None or isinstance(sq.time_constraint, str)
    if sq.temporal_order is not None:
        assert isinstance(sq.temporal_order, list)
        for ev in sq.temporal_order:
            assert set(ev.keys()) >= {"event", "order"}
            assert isinstance(ev["event"], str) and isinstance(ev["order"], int)
    assert sq.query_type in QUERY_TYPES


# -----------------------------------------------------------------------------
# DoD: cả 5 câu ra StructuredQuery hợp lệ schema
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("query", SAMPLES)
def test_all_samples_produce_valid_schema(query: str) -> None:
    sq = MockQueryUnderstander().parse(query)
    _assert_valid_schema(sq)


# -----------------------------------------------------------------------------
# Kiểm các trường parse chắc chắn cho từng câu
# -----------------------------------------------------------------------------
def test_q1_color_objects_location() -> None:
    sq = MockQueryUnderstander().parse(Q1)
    assert sq.query_type == "KIS_textual"
    assert "hồng" in sq.attributes.get("color", [])
    assert {"chìa khóa", "móc khóa", "gấu bông"}.issubset(set(sq.objects))
    assert sq.location is not None and "quầy" in sq.location


def test_q2_temporal_order() -> None:
    sq = MockQueryUnderstander().parse(Q2)
    assert sq.temporal_order is not None
    events = [(e["event"], e["order"]) for e in sq.temporal_order]
    # "cởi mũ" phải xảy ra TRƯỚC "vào phòng".
    order_of = {e: o for e, o in events}
    coi = next(e for e in order_of if "cởi mũ" in e)
    vao = next(e for e in order_of if "vào phòng" in e)
    assert order_of[coi] < order_of[vao]


def test_q3_avs_location_action() -> None:
    sq = MockQueryUnderstander().parse(Q3)
    assert sq.query_type == "AVS"          # "tất cả các cảnh"
    assert sq.location == "công viên"
    assert "tưới" in sq.actions


def test_q4_vqa_type_no_bad_location() -> None:
    sq = MockQueryUnderstander().parse(Q4)
    assert sq.query_type == "VQA"          # "bao nhiêu"
    # Không bắt nhầm 'video ...' làm địa điểm.
    assert sq.location is None or "video" not in sq.location.lower()
    assert {"nến", "bánh"} & set(" ".join(sq.objects).split()) or "nến" in " ".join(sq.objects)


def test_q5_time_color_location() -> None:
    sq = MockQueryUnderstander().parse(Q5)
    assert sq.time_constraint == "tuần trước"
    assert "xanh dương" in sq.attributes.get("color", [])
    assert sq.location is not None and "quán cà phê" in sq.location


def test_query_type_hint_overrides() -> None:
    sq = MockQueryUnderstander().parse("một câu bất kỳ", query_type_hint="KIS_video")
    assert sq.query_type == "KIS_video"


# -----------------------------------------------------------------------------
# Query expansion (Mục 4.2)
# -----------------------------------------------------------------------------
def test_expansion_includes_original_and_count() -> None:
    expander = MockQueryExpander()
    variants = expander.expand(Q5, num_variants=4)
    assert variants[0] == Q5                    # câu gốc luôn đứng đầu
    assert len(variants) <= 4
    assert len(set(variants)) == len(variants)  # không trùng


def test_expansion_generates_multiple_variants() -> None:
    expander = MockQueryExpander()
    variants = expander.expand("Tìm video người đàn ông", num_variants=4)
    assert len(variants) >= 2                    # có sinh thêm biến thể


def test_expansion_invalid_count_raises() -> None:
    with pytest.raises(ValueError):
        MockQueryExpander().expand("x", num_variants=0)


# -----------------------------------------------------------------------------
# Guard bản thật khi chưa có anthropic/key
# -----------------------------------------------------------------------------
def test_claude_understander_guard() -> None:
    with pytest.raises((ImportError, RuntimeError)):
        ClaudeQueryUnderstander()
