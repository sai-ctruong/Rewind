"""Unit test Phase 5 — fusion (RRF) + fine_rerank (LVLM verifier). CLAUDE.md Mục 8.

DoD Phase 5: "Trên tập test có ground-truth nhỏ (~20 cặp query-answer), Top-1 accuracy
được đo và ghi log."

Ngoài DoD ta test: RRF (có trọng số), time budget (bơm đồng hồ giả -> tất định, không
sleep thật), early stopping (tái dùng is_confident_enough), giới hạn max_candidates /
final_top_k. Chạy offline (MockReranker), không API.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.build_index import KeyframeIndex
from ingestion.build_records import IngestionPipeline, make_sample_video, searchable_text
from ingestion.schemas import RawKeyframe
from retrieval.coarse_retriever import Candidate, CoarseRetriever
from retrieval.fine_rerank import (
    FineReranker,
    MockReranker,
    RerankConfig,
    Reranker,
)
from retrieval.fusion import fuse_to_sorted_rows, reciprocal_rank_fusion


# =============================== FUSION =======================================
def test_rrf_basic_ordering() -> None:
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [1, 3, 2]}, k=60)
    order = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
    assert order[0][0] == 1  # hạng 1 ở cả 2 nguồn


def test_rrf_weights_shift_ranking() -> None:
    # row 5 chỉ đứng đầu ở nguồn 'b'; tăng trọng số 'b' phải đẩy 5 lên.
    lists = {"a": [1, 2], "b": [5, 6]}
    no_w = dict(reciprocal_rank_fusion(lists))
    weighted = dict(reciprocal_rank_fusion(lists, weights={"b": 5.0}))
    assert weighted[5][0] > no_w[5][0]


def test_fuse_to_sorted_rows() -> None:
    rows = fuse_to_sorted_rows({"a": [7, 8], "b": [7, 9]})
    assert rows[0][0] == 7
    assert rows[0][1] >= rows[1][1]  # đã sắp giảm dần


# =========================== FINE RERANK — cơ chế ==============================
def _mk_candidates(n: int) -> list[Candidate]:
    return [
        Candidate(
            keyframe_id=f"kf{i}", row=i, score=1.0 / (i + 1),
            video_id="v", timestamp=float(i), source_ranks={},
        )
        for i in range(n)
    ]


class _FakeClock:
    """Đồng hồ giả tăng đều mỗi lần gọi -> test time budget tất định, không sleep."""

    def __init__(self, step: float = 1.0):
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def test_time_budget_stops_early() -> None:
    cands = _mk_candidates(100)
    ctx = {c.keyframe_id: "person" for c in cands}
    fr = FineReranker(
        MockReranker(),
        RerankConfig(max_candidates=100, time_budget_s=10.0, early_stop=False),
    )
    out = fr.rerank("person", cands, ctx, clock=_FakeClock(step=1.0))
    # Với đồng hồ giả tăng ~3/iter, budget 10 -> dừng sau ~3 ứng viên, KHÔNG chấm hết 100.
    assert 0 < len(out) < 100


def test_early_stopping_when_top1_dominates() -> None:
    cands = _mk_candidates(100)
    # Chỉ kf0 khớp query hoàn toàn; còn lại không khớp -> Top-1 vượt trội.
    ctx = {c.keyframe_id: ("mèo con dễ thương" if c.keyframe_id == "kf0" else "xe tải")
           for c in cands}
    fr = FineReranker(
        MockReranker(),
        RerankConfig(max_candidates=100, early_stop=True,
                     confidence_margin=0.15, min_evaluated_before_stop=5),
    )
    out = fr.rerank("mèo con dễ thương", cands, ctx)
    assert out[0].keyframe_id == "kf0"
    assert len(out) < 100  # đã dừng sớm, không chấm hết


def test_final_top_k_limits_output() -> None:
    cands = _mk_candidates(50)
    ctx = {c.keyframe_id: "person car" for c in cands}
    fr = FineReranker(
        MockReranker(),
        RerankConfig(max_candidates=50, final_top_k=10, early_stop=False),
    )
    out = fr.rerank("person car", cands, ctx)
    assert len(out) <= 10


def test_scores_normalized_0_1() -> None:
    cands = _mk_candidates(3)
    ctx = {c.keyframe_id: "person car street" for c in cands}
    fr = FineReranker(MockReranker(), RerankConfig(early_stop=False))
    out = fr.rerank("person car", cands, ctx)
    assert all(0.0 <= c.score <= 1.0 for c in out)


def test_claude_reranker_guard() -> None:
    from retrieval.fine_rerank import ClaudeReranker

    with pytest.raises((ImportError, RuntimeError)):
        ClaudeReranker()


# =========================== DoD: Top-1 accuracy ==============================
def _build_ground_truth(n: int = 20):
    """Sinh ~n keyframe, mỗi cái có 1 object ĐỘC NHẤT (widgetK) + object nền chung.
    Query cho mỗi keyframe nhắc tới widget độc nhất -> có đáp án đúng xác định."""
    shared = ["person", "table"]
    records = []
    queries = []  # (query_text, target_id)
    raws = []
    for i in range(n):
        uid = f"widget{i}"
        raws.append(
            RawKeyframe(
                id=f"gt/{i}",
                video_id=f"vid{i % 4}",
                timestamp=float(i),
                objects=[uid, *shared],
            )
        )
        queries.append((f"tìm cảnh có {uid} và person", f"gt/{i}"))
    records = IngestionPipeline().build(raws)
    return records, queries


def test_top1_accuracy_measured_and_logged(capsys, tmp_path) -> None:
    records, queries = _build_ground_truth(20)
    index = KeyframeIndex.build(records)
    coarse = CoarseRetriever(index)
    contexts = {r.id: searchable_text(r) for r in records}
    fr = FineReranker(
        MockReranker(),
        RerankConfig(max_candidates=100, final_top_k=5, early_stop=False),
    )

    correct = 0
    for query_text, target_id in queries:
        cands = coarse.search(query_text=query_text, top_k=100)
        reranked = fr.rerank(query_text, cands, contexts)
        if reranked and reranked[0].keyframe_id == target_id:
            correct += 1

    top1 = correct / len(queries)
    # GHI LOG (DoD yêu cầu "được đo và ghi log").
    print(f"[Phase5] Top-1 accuracy = {top1:.3f} ({correct}/{len(queries)})")
    result = {"metric": "top1_accuracy", "value": top1, "n": len(queries)}
    out_file = tmp_path / "phase5_rerank_top1.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    assert 0.0 <= top1 <= 1.0
    assert top1 >= 0.8, f"Top-1 accuracy {top1} thấp bất thường trên ground-truth mock"
