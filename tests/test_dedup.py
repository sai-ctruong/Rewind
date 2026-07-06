"""Unit test cho ingestion/dedup.py — DoD Phase 1 (CLAUDE.md Mục 8).

DoD Phase 1: "Test chứng minh giảm >30% số record trên dữ liệu giả mà KHÔNG mất
representative nào."

Ngoài DoD, ta test thêm các bất biến quan trọng theo Mục 1.2 (không mất recall):
  - Không frame nào bị "bốc hơi": union các cụm = toàn bộ id đầu vào.
  - Chống semantic drift: chuỗi frame biến đổi chậm KHÔNG bị gộp thành 1 (nếu bị
    gộp thì mất recall) — kiểm chứng lựa chọn anchor-based ở dedup.py.
  - Các edge case: video 1 frame, nhiều video độc lập, embedding thiếu, ngưỡng sai,
    tính tất định (deterministic).

Toàn bộ chạy offline bằng numpy (Mục 1.5).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ingestion.dedup import cosine_similarity, deduplicate_keyframes
from ingestion.schemas import KeyframeRecord


# -----------------------------------------------------------------------------
# Helpers sinh dữ liệu giả
# -----------------------------------------------------------------------------
def _kf(id_: str, video_id: str, ts: float, emb: np.ndarray) -> KeyframeRecord:
    return KeyframeRecord(
        id=id_, video_id=video_id, timestamp=ts, clip_embedding=emb
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_near_duplicate_runs(
    num_scenes: int,
    frames_per_scene: int,
    dim: int = 64,
    seed: int = 0,
) -> list[KeyframeRecord]:
    """Mô phỏng lifelog: `num_scenes` cảnh khác nhau, mỗi cảnh là một chuỗi
    `frames_per_scene` frame gần trùng (anchor + nhiễu cực nhỏ -> cosine ~ 1).
    Các cảnh khác nhau có vector nền độc lập -> cosine ~ 0 (khác ngữ nghĩa).
    """
    rng = np.random.default_rng(seed)
    records: list[KeyframeRecord] = []
    ts = 0.0
    for s in range(num_scenes):
        base = _unit(rng.standard_normal(dim))
        for f in range(frames_per_scene):
            # Nhiễu rất nhỏ để cosine với anchor luôn > 0.97.
            noisy = _unit(base + 0.01 * rng.standard_normal(dim))
            records.append(_kf(f"s{s}_f{f}", "video_A", ts, noisy))
            ts += 1.0
    return records


# -----------------------------------------------------------------------------
# cosine_similarity
# -----------------------------------------------------------------------------
def test_cosine_identical_is_one() -> None:
    v = _unit(np.array([1.0, 2.0, 3.0]))
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine_similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)


def test_cosine_zero_vector_is_safe_zero() -> None:
    # Vector 0 -> trả 0.0, KHÔNG chia cho 0, KHÔNG vô tình gộp.
    assert cosine_similarity(np.zeros(3), np.array([1.0, 1.0, 1.0])) == 0.0


# -----------------------------------------------------------------------------
# DoD Phase 1: giảm > 30% mà không mất representative
# -----------------------------------------------------------------------------
def test_reduction_exceeds_30_percent() -> None:
    records = make_near_duplicate_runs(num_scenes=5, frames_per_scene=10)
    result = deduplicate_keyframes(records, similarity_threshold=0.97)

    assert result.num_input == 50
    # 5 cảnh gần trùng -> mong đợi ~5 đại diện.
    assert result.num_output == 5
    assert result.reduction_ratio > 0.30
    # DoD nói rõ ">30%"; ở kịch bản này thực tế đạt 90%.
    assert result.reduction_ratio == pytest.approx(0.90)


def test_no_representative_lost() -> None:
    """Mỗi cụm giữ đúng 1 đại diện và mọi frame đều được quy về đúng 1 cụm."""
    records = make_near_duplicate_runs(num_scenes=4, frames_per_scene=8)
    result = deduplicate_keyframes(records, similarity_threshold=0.97)

    # Số đại diện == số cụm (không cụm nào thiếu đại diện).
    assert len(result.representatives) == result.num_clusters
    # Mọi đại diện phải được đánh dấu.
    assert all(r.is_cluster_representative for r in result.representatives)

    # Union id trong tất cả cụm = toàn bộ id đầu vào, không trùng, không mất.
    all_member_ids = [mid for members in result.cluster_members for mid in members]
    assert len(all_member_ids) == result.num_input          # không mất frame nào
    assert len(set(all_member_ids)) == result.num_input     # không frame nào bị đếm 2 lần
    assert set(all_member_ids) == {r.id for r in records}

    # Mỗi đại diện của cụm >1 frame phải có cluster_span hợp lệ (t_start <= t_end).
    for rep, members in zip(result.representatives, result.cluster_members):
        if len(members) > 1:
            assert rep.cluster_span is not None
            assert rep.cluster_span[0] <= rep.cluster_span[1]


# -----------------------------------------------------------------------------
# Chống semantic drift (bảo vệ recall — Mục 1.2)
# -----------------------------------------------------------------------------
def test_slow_drift_not_over_merged() -> None:
    """Chuỗi frame xoay chậm: mỗi cặp LIỀN TRƯỚC giống nhau (>0.97) nhưng đầu-cuối
    khác hẳn. Anchor-based phải TÁCH thành nhiều cụm, không gộp thành 1.

    Nếu bị gộp thành 1 đại diện -> frame cuối (khác ngữ nghĩa) biến mất khỏi index
    -> mất recall. Test này chốt hành vi an toàn.
    """
    dim_angles = [i * 10 for i in range(10)]  # 0,10,...,90 độ
    records = []
    for i, deg in enumerate(dim_angles):
        rad = math.radians(deg)
        v = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
        records.append(_kf(f"d_{i}", "video_drift", float(i), v))

    # Cặp liền trước cách nhau 10 độ: cos(10°)=0.985 > 0.97 (nếu so prev-based sẽ gộp hết).
    assert cosine_similarity(
        records[0].clip_embedding, records[1].clip_embedding
    ) > 0.97
    # Nhưng đầu (0°) và cuối (90°) trực giao: cos(90°)=0 < 0.97.
    assert cosine_similarity(
        records[0].clip_embedding, records[-1].clip_embedding
    ) < 0.97

    result = deduplicate_keyframes(records, similarity_threshold=0.97)
    # Phải có > 1 đại diện (không over-merge). Với bước 10°, ngưỡng 0.97 (~14°),
    # mỗi cụm gồm tối đa 2 frame -> khoảng 5 cụm.
    assert result.num_output > 1
    assert result.num_output >= 5


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------
def test_singleton_video_keeps_frame_no_span() -> None:
    rec = [_kf("only", "v1", 0.0, _unit(np.array([1.0, 0.0, 0.0])))]
    result = deduplicate_keyframes(rec)
    assert result.num_output == 1
    assert result.representatives[0].cluster_span is None  # singleton không phải "cụm"
    assert result.reduction_ratio == 0.0


def test_multiple_videos_are_independent() -> None:
    """Frame giống hệt nhau nhưng KHÁC video thì KHÔNG được gộp chung."""
    v = _unit(np.array([1.0, 1.0, 1.0]))
    records = [
        _kf("a", "vidA", 0.0, v.copy()),
        _kf("b", "vidB", 0.0, v.copy()),  # cùng vector, khác video
    ]
    result = deduplicate_keyframes(records, similarity_threshold=0.97)
    assert result.num_output == 2  # không gộp xuyên video


def test_empty_input() -> None:
    result = deduplicate_keyframes([])
    assert result.num_output == 0
    assert result.reduction_ratio == 0.0


def test_missing_embedding_raises() -> None:
    rec = KeyframeRecord(
        id="x", video_id="v", timestamp=0.0, clip_embedding=np.array([1.0, 0.0])
    )
    rec.clip_embedding = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="thiếu embedding"):
        deduplicate_keyframes([rec])


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, 2.0])
def test_invalid_threshold_raises(bad: float) -> None:
    records = make_near_duplicate_runs(2, 2)
    with pytest.raises(ValueError, match="similarity_threshold"):
        deduplicate_keyframes(records, similarity_threshold=bad)


def test_deterministic() -> None:
    records = make_near_duplicate_runs(3, 6, seed=123)
    r1 = deduplicate_keyframes(records, 0.97)
    # Chạy lại trên bản sao dữ liệu giống hệt -> kết quả trùng khớp.
    records2 = make_near_duplicate_runs(3, 6, seed=123)
    r2 = deduplicate_keyframes(records2, 0.97)
    assert [r.id for r in r1.representatives] == [r.id for r in r2.representatives]
