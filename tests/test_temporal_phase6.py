"""Unit test Phase 6 — temporal_check (CLAUDE.md Mục 8, Mục 4.5).

DoD Phase 6: "Phân biệt đúng 2 trường hợp thứ tự khác nhau trong bộ test tự tạo"
(case 'cởi mũ trước khi vào phòng').

Ta test: (a) đúng thứ tự -> có match, sai thứ tự -> rỗng (hard constraint), (b) không
gộp xuyên video, (c) chuỗi 3 sự kiện, (d) xếp theo score, (e) guard đầu vào.
Thuần logic, offline.
"""
from __future__ import annotations

import pytest

from retrieval.coarse_retriever import Candidate
from retrieval.query_understanding import MockQueryUnderstander
from retrieval.temporal_check import (
    has_valid_ordering,
    temporal_consistency_filter,
)


def _c(kf_id: str, video: str, ts: float, score: float = 0.0) -> Candidate:
    return Candidate(
        keyframe_id=kf_id, row=0, score=score, video_id=video,
        timestamp=ts, source_ranks={},
    )


# -----------------------------------------------------------------------------
# DoD: phân biệt 2 trường hợp thứ tự (dùng chính parser Phase 4 cho Q2)
# -----------------------------------------------------------------------------
Q2 = "Người đàn ông cởi mũ trước khi vào phòng."


def test_distinguish_two_temporal_orders() -> None:
    order = MockQueryUnderstander().parse(Q2).temporal_order
    assert order is not None
    # "cởi mũ" (t=10) THỰC SỰ xảy ra trước "vào phòng" (t=30).
    cands = {
        "cởi mũ": [_c("kf_coi", "V1", 10.0)],
        "vào phòng": [_c("kf_vao", "V1", 30.0)],
    }

    # Trường hợp 1: query "cởi mũ TRƯỚC KHI vào phòng" -> KHỚP.
    matches = temporal_consistency_filter(order, cands)
    assert len(matches) == 1
    steps = matches[0].steps
    assert [s.event for s in steps] == ["cởi mũ", "vào phòng"]
    assert steps[0].timestamp < steps[1].timestamp

    # Trường hợp 2: đảo thứ tự yêu cầu ("vào phòng trước cởi mũ") -> KHÔNG khớp,
    # vì trong dữ liệu vào phòng (30) KHÔNG xảy ra trước cởi mũ (10).
    reversed_order = [
        {"event": "vào phòng", "order": 1},
        {"event": "cởi mũ", "order": 2},
    ]
    assert temporal_consistency_filter(reversed_order, cands) == []
    # has_valid_ordering phản ánh đúng sự phân biệt:
    assert has_valid_ordering(order, cands) is True
    assert has_valid_ordering(reversed_order, cands) is False


# -----------------------------------------------------------------------------
# Hard constraint: sai thứ tự bị loại dù cùng video
# -----------------------------------------------------------------------------
def test_wrong_time_order_filtered_out() -> None:
    order = [{"event": "A", "order": 1}, {"event": "B", "order": 2}]
    # A ở t=100, B ở t=50 -> A KHÔNG trước B -> rỗng.
    cands = {"A": [_c("a", "V", 100.0)], "B": [_c("b", "V", 50.0)]}
    assert temporal_consistency_filter(order, cands) == []


def test_multiple_valid_pairs() -> None:
    order = [{"event": "A", "order": 1}, {"event": "B", "order": 2}]
    cands = {
        "A": [_c("a1", "V", 10.0), _c("a2", "V", 40.0)],
        "B": [_c("b1", "V", 30.0), _c("b2", "V", 50.0)],
    }
    matches = temporal_consistency_filter(order, cands)
    # Cặp hợp lệ (a.t < b.t): (10,30),(10,50),(40,50) — KHÔNG có (40,30).
    pairs = {(m.steps[0].timestamp, m.steps[1].timestamp) for m in matches}
    assert pairs == {(10.0, 30.0), (10.0, 50.0), (40.0, 50.0)}


# -----------------------------------------------------------------------------
# Không gộp xuyên video
# -----------------------------------------------------------------------------
def test_no_cross_video_matches() -> None:
    order = [{"event": "A", "order": 1}, {"event": "B", "order": 2}]
    cands = {"A": [_c("a", "V1", 10.0)], "B": [_c("b", "V2", 30.0)]}
    assert temporal_consistency_filter(order, cands) == []


# -----------------------------------------------------------------------------
# Chuỗi 3 sự kiện
# -----------------------------------------------------------------------------
def test_three_event_chain() -> None:
    order = [
        {"event": "E1", "order": 1},
        {"event": "E2", "order": 2},
        {"event": "E3", "order": 3},
    ]
    cands = {
        "E1": [_c("e1", "V", 10.0)],
        "E2": [_c("e2", "V", 20.0)],
        "E3": [_c("e3", "V", 30.0)],
    }
    matches = temporal_consistency_filter(order, cands)
    assert len(matches) == 1
    assert [s.timestamp for s in matches[0].steps] == [10.0, 20.0, 30.0]

    # Đảo E2, E3 về thời gian (E3 sớm hơn E2) -> chuỗi tăng dần gãy -> rỗng.
    cands_bad = {
        "E1": [_c("e1", "V", 10.0)],
        "E2": [_c("e2", "V", 30.0)],
        "E3": [_c("e3", "V", 20.0)],
    }
    assert temporal_consistency_filter(order, cands_bad) == []


# -----------------------------------------------------------------------------
# Xếp theo score + guard
# -----------------------------------------------------------------------------
def test_sorted_by_total_score() -> None:
    order = [{"event": "A", "order": 1}, {"event": "B", "order": 2}]
    cands = {
        "A": [_c("a1", "V", 10.0, score=0.9), _c("a2", "V", 20.0, score=0.1)],
        "B": [_c("b1", "V", 30.0, score=0.9)],
    }
    matches = temporal_consistency_filter(order, cands)
    # (a1,b1) tổng 1.8 phải đứng trên (a2,b1) tổng 1.0.
    assert matches[0].steps[0].keyframe_id == "a1"
    assert matches[0].total_score > matches[1].total_score


def test_missing_event_candidates_returns_empty() -> None:
    order = [{"event": "A", "order": 1}, {"event": "B", "order": 2}]
    assert temporal_consistency_filter(order, {"A": [_c("a", "V", 1.0)], "B": []}) == []


def test_requires_two_events() -> None:
    with pytest.raises(ValueError, match=">= 2 sự kiện"):
        temporal_consistency_filter([{"event": "A", "order": 1}], {"A": [_c("a", "V", 1.0)]})


def test_max_results_cap() -> None:
    order = [{"event": "A", "order": 1}, {"event": "B", "order": 2}]
    # 10 A sớm + 10 B muộn -> 100 cặp hợp lệ; cap 5.
    cands = {
        "A": [_c(f"a{i}", "V", float(i)) for i in range(10)],
        "B": [_c(f"b{i}", "V", float(100 + i)) for i in range(10)],
    }
    matches = temporal_consistency_filter(order, cands, max_results=5)
    assert len(matches) == 5
