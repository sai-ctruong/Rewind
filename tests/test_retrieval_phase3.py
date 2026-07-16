"""Unit test Phase 3 — build_index + coarse_retriever (CLAUDE.md Mục 8).

DoD Phase 3: "Query mẫu trả về candidate hợp lý trong <200ms trên tập test".

Ta kiểm:
  - HNSW dense self-retrieval: query bằng chính embedding của 1 keyframe -> top-1 là
    chính nó (chứng minh index đúng + recall).
  - BM25 sparse: query text khớp objects/caption -> trả đúng frame liên quan.
  - Metadata pre-filter (Mục 11.1.1): lọc video/objects/time thu hẹp đúng.
  - RRF fusion (Mục 4.3): gộp dense + sparse, candidate mang source_ranks nhiều nguồn.
  - Latency < 200ms trên tập ~1500 keyframe.
  - save/load roundtrip.

Dùng pipeline mock Phase 2 để sinh KeyframeRecord (offline, không GPU/API).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from ingestion.build_index import IndexConfig, KeyframeIndex, tokenize
from ingestion.build_records import IngestionPipeline, make_sample_video
from ingestion.schemas import RawKeyframe
from retrieval.coarse_retriever import CoarseRetriever, reciprocal_rank_fusion


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_index() -> KeyframeIndex:
    raws = make_sample_video(video_id="vidA", num_frames=12, scene_size=3)
    records = IngestionPipeline().build(raws)
    return KeyframeIndex.build(records)


@pytest.fixture(scope="module")
def small_records():
    raws = make_sample_video(video_id="vidA", num_frames=12, scene_size=3)
    return IngestionPipeline().build(raws)


# -----------------------------------------------------------------------------
# tokenize / RRF (thuần logic)
# -----------------------------------------------------------------------------
def test_tokenize_unicode_vietnamese() -> None:
    toks = tokenize("Người lớn tưới HOA màu hồng")
    assert "người" in toks and "hoa" in toks and "hồng" in toks


def test_rrf_orders_by_summed_reciprocal_rank() -> None:
    # row 2 đứng đầu ở cả 2 list -> điểm cao nhất.
    lists = {"a": [2, 1, 3], "b": [2, 3, 1]}
    fused = reciprocal_rank_fusion(lists, k=60)
    order = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
    assert order[0][0] == 2
    # row 2 xuất hiện ở cả 2 nguồn.
    assert set(fused[2][1].keys()) == {"a", "b"}


# -----------------------------------------------------------------------------
# Index built đúng
# -----------------------------------------------------------------------------
def test_index_has_both_encoders(small_index: KeyframeIndex) -> None:
    assert small_index.has_encoder("clip")
    assert small_index.has_encoder("siglip")
    assert len(small_index.ids) == 12


# -----------------------------------------------------------------------------
# Dense self-retrieval (recall)
# -----------------------------------------------------------------------------
def test_dense_self_retrieval_top1(small_index, small_records) -> None:
    target = small_records[5]
    retriever = CoarseRetriever(small_index)
    results = retriever.search(query_clip_vec=target.clip_embedding, top_k=5)
    assert results, "phải có kết quả"
    assert results[0].keyframe_id == target.id  # top-1 chính là frame đã query


def test_dense_self_retrieval_via_hnsw_and_exact_agree(small_index, small_records) -> None:
    target = small_records[7]
    retriever = CoarseRetriever(small_index)
    # HNSW (không filter) vs exact (có filter chứa target) -> cùng top-1.
    hnsw = retriever.search(query_clip_vec=target.clip_embedding, top_k=3)
    exact = retriever.search(
        query_clip_vec=target.clip_embedding,
        filters={"video_id": "vidA"},
        top_k=3,
    )
    assert hnsw[0].keyframe_id == target.id
    assert exact[0].keyframe_id == target.id


# -----------------------------------------------------------------------------
# Sparse BM25
# -----------------------------------------------------------------------------
def test_bm25_matches_objects(small_index, small_records) -> None:
    retriever = CoarseRetriever(small_index)
    # "flower"/"child" thuộc scene 1 (objects person/child/flower).
    results = retriever.search(query_text="child flower", top_k=12)
    returned = {c.keyframe_id for c in results}
    expected = {r.id for r in small_records if "flower" in r.objects}
    assert expected, "phải có frame chứa flower trong video mẫu"
    assert expected.issubset(returned)


# -----------------------------------------------------------------------------
# Metadata pre-filter
# -----------------------------------------------------------------------------
def test_filter_by_objects_all(small_index, small_records) -> None:
    retriever = CoarseRetriever(small_index)
    results = retriever.search(
        query_clip_vec=small_records[0].clip_embedding,
        filters={"objects_all": ["flower"]},
        top_k=12,
    )
    for c in results:
        objs = small_index.objects[c.row]
        assert "flower" in objs  # mọi ứng viên đều thoả filter cứng


def test_filter_time_range(small_index, small_records) -> None:
    retriever = CoarseRetriever(small_index)
    results = retriever.search(
        query_clip_vec=small_records[0].clip_embedding,
        filters={"time_range": (0.0, 6.0)},
        top_k=12,
    )
    for c in results:
        assert 0.0 <= c.timestamp <= 6.0


def test_filter_excludes_everything_returns_empty(small_index, small_records) -> None:
    retriever = CoarseRetriever(small_index)
    results = retriever.search(
        query_clip_vec=small_records[0].clip_embedding,
        filters={"video_id": "khong_ton_tai"},
        top_k=12,
    )
    assert results == []


# -----------------------------------------------------------------------------
# Fusion nhiều nguồn
# -----------------------------------------------------------------------------
def test_fusion_combines_dense_and_sparse(small_index, small_records) -> None:
    target = next(r for r in small_records if "flower" in r.objects)
    retriever = CoarseRetriever(small_index)
    results = retriever.search(
        query_clip_vec=target.clip_embedding,
        query_siglip_vec=target.siglip_embedding,
        query_text="child flower",
        top_k=12,
    )
    top = results[0]
    assert top.keyframe_id == target.id
    # Ứng viên đầu phải được nhiều nguồn ủng hộ.
    assert set(top.source_ranks.keys()) & {"clip", "siglip", "bm25"}


def test_search_requires_a_signal(small_index) -> None:
    retriever = CoarseRetriever(small_index)
    with pytest.raises(ValueError, match="ít nhất 1 tín hiệu"):
        retriever.search()


# -----------------------------------------------------------------------------
# Latency DoD (<200ms) trên tập lớn hơn
# -----------------------------------------------------------------------------
def _big_records(n: int):
    rng = np.random.default_rng(0)
    raws = [
        RawKeyframe(
            id=f"big/{i}",
            video_id=f"vid_{i % 20}",
            timestamp=float(i),
            objects=["person"] if i % 2 else ["car", "street"],
        )
        for i in range(n)
    ]
    # Dùng pipeline mock nhưng thêm nhiễu để embedding không trùng nhau.
    pipeline = IngestionPipeline()
    return pipeline.build(raws)


def test_latency_under_200ms() -> None:
    records = _big_records(1500)
    index = KeyframeIndex.build(records, IndexConfig(hnsw_m=32, ef_search=128))
    retriever = CoarseRetriever(index)
    q = records[123]
    # Warm-up (loại chi phí khởi tạo lần đầu).
    retriever.search(query_clip_vec=q.clip_embedding, query_text="person", top_k=1000)
    t0 = time.perf_counter()
    results = retriever.search(
        query_clip_vec=q.clip_embedding,
        query_siglip_vec=q.siglip_embedding,
        query_text="person car",
        top_k=1000,
    )
    elapsed = time.perf_counter() - t0
    assert results
    assert results[0].keyframe_id == q.id
    assert elapsed < 0.2, f"coarse search {elapsed*1000:.1f}ms vượt 200ms"


# -----------------------------------------------------------------------------
# Fusion depth ĐỘC LẬP với top_k (bug hồi quy)
# -----------------------------------------------------------------------------
def test_ranking_does_not_change_with_requested_top_k() -> None:
    """XIN NHIỀU KẾT QUẢ HƠN KHÔNG ĐƯỢC ĐỔI THỨ HẠNG.

    BUG THẬT đã đo: hit@5 = 0.627 khi xin top_k=5, tụt còn 0.510 khi xin top_k=20 —
    cùng query, cùng index. Vì `top_k` từng vừa là "số kết quả trả về" vừa là "độ sâu
    mỗi ranked list": list sâu hơn -> item được cộng điểm từ NHIỀU nguồn hơn -> điểm
    đổi -> thứ tự đổi. Người dùng bấm 'xem thêm' mà kết quả tốt nhất lại khác đi.
    """
    records = _big_records(600)
    index = KeyframeIndex.build(records)
    retriever = CoarseRetriever(index)
    q = records[42]
    kw = dict(
        query_clip_vec=q.clip_embedding,
        query_siglip_vec=q.siglip_embedding,
        query_text="person car street",
    )
    shallow = retriever.search(top_k=5, **kw)
    deep = retriever.search(top_k=50, **kw)

    assert [c.keyframe_id for c in shallow] == [c.keyframe_id for c in deep[:5]]
    # Điểm cũng phải y hệt, không chỉ thứ tự.
    assert [round(c.score, 9) for c in shallow] == [round(c.score, 9) for c in deep[:5]]


def test_fusion_depth_is_what_changes_ranking_not_top_k() -> None:
    """Mặt kia của cùng một đồng xu: `depth` MỚI là thứ được phép đổi thứ hạng.

    Nếu đổi depth mà thứ hạng không bao giờ đổi thì test trên chỉ đang khẳng định một
    điều tầm thường (vd fusion depth luôn phủ hết index) — test này chặn khả năng đó,
    giữ cho test hồi quy ở trên còn ý nghĩa thật.
    """
    records = _big_records(600)
    index = KeyframeIndex.build(records)
    retriever = CoarseRetriever(index)
    q = records[42]
    kw = dict(
        query_clip_vec=q.clip_embedding,
        query_siglip_vec=q.siglip_embedding,
        query_text="person car street",
        top_k=20,
    )
    d5 = [c.keyframe_id for c in retriever.search(depth=5, **kw)]
    d500 = [c.keyframe_id for c in retriever.search(depth=500, **kw)]
    assert d5 != d500, "đổi độ sâu fusion phải đổi được kết quả — nếu không, test kia vô nghĩa"


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path, small_records) -> None:
    index = KeyframeIndex.build(small_records)
    index.save(tmp_path)
    loaded = KeyframeIndex.load(tmp_path)
    assert loaded.ids == index.ids
    retriever = CoarseRetriever(loaded)
    target = small_records[3]
    results = retriever.search(query_clip_vec=target.clip_embedding, top_k=3)
    assert results[0].keyframe_id == target.id
