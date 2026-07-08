"""Score Fusion — Reciprocal Rank Fusion (CLAUDE.md Mục 4.3, Phase 5).

VÌ SAO RRF (không dùng trung bình cộng điểm thô): các nguồn tín hiệu ở tầng coarse
(CLIP cosine, SigLIP cosine, BM25) có THANG ĐO KHÁC NHAU, cộng thẳng là sai lệch. RRF
chỉ dựa trên THỨ HẠNG (rank) nên không cần chuẩn hoá thang điểm — đây là kỹ thuật
fusion chuẩn trong Information Retrieval khi gộp nhiều ranked list không đồng nhất.

    RRF_score(d) = Σ_i  w_i / (k + rank_i(d))          (k = 60 mặc định chuẩn)

Module này là NƠI CHÍNH THỨC của RRF. coarse_retriever.py import từ đây (trước đó
Phase 3 để tạm bản inline; Phase 5 gom về một chỗ để fine_rerank và các tầng khác
cùng tái dùng, tránh lệch cài đặt).
"""
from __future__ import annotations

from typing import Optional

# Hằng số RRF chuẩn (Mục 4.3, [FIXED] trong settings.yaml -> fusion.rrf_k).
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[dict[str, float]] = None,
) -> dict[int, tuple[float, dict[str, int]]]:
    """Gộp nhiều ranked list bằng RRF.

    Args:
        ranked_lists: {tên_nguồn: [row theo thứ hạng giảm dần]}.
        k: hằng số RRF (mặc định 60). k lớn -> giảm ảnh hưởng của chênh lệch hạng
            ở top; k nhỏ -> ưu ái mạnh các item hạng đầu.
        weights: trọng số mỗi nguồn (mặc định 1.0). Cho phép ưu tiên encoder mạnh
            hơn (vd SigLIP > CLIP) mà vẫn giữ tính chất "không cần chuẩn hoá thang
            điểm" của RRF — trọng số áp lên đóng góp theo hạng, không lên điểm thô.

    Returns:
        {row: (rrf_score, {nguồn: rank})} — kèm rank từng nguồn để truy vết/giải thích.

    rank tính từ 1 (item đầu list = hạng 1).
    """
    weights = weights or {}
    fused: dict[int, tuple[float, dict[str, int]]] = {}
    for source, rows in ranked_lists.items():
        w = weights.get(source, 1.0)
        for rank, row in enumerate(rows, start=1):
            score, ranks = fused.get(row, (0.0, {}))
            score += w / (k + rank)
            ranks = {**ranks, source: rank}
            fused[row] = (score, ranks)
    return fused


def fuse_to_sorted_rows(
    ranked_lists: dict[str, list[int]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[dict[str, float]] = None,
) -> list[tuple[int, float, dict[str, int]]]:
    """Tiện ích: trả list [(row, score, source_ranks)] đã sắp giảm dần theo RRF."""
    fused = reciprocal_rank_fusion(ranked_lists, k, weights)
    return sorted(
        ((row, sc, ranks) for row, (sc, ranks) in fused.items()),
        key=lambda x: x[1],
        reverse=True,
    )
