"""Tầng Coarse Retrieval — recall-first (CLAUDE.md Mục 3 bước [2], Mục 1.2).

NHIỆM VỤ: từ 1 query (đã tách thành vector CLIP/SigLIP + text + filter), trả về
top-K ứng viên (mặc định 1000) để đưa xuống fusion/rerank. Ràng buộc SỐ 1 ở tầng
này là RECALL CAO (Mục 1.2): ứng viên bị loại ở đây KHÔNG BAO GIỜ được xét lại.

LUỒNG:
  1. Metadata PRE-FILTER (Mục 11.1.1): lọc cứng theo video/time/objects -> subset.
     Lọc trước giúp thu hẹp không gian mà không mất ứng viên đúng nào.
  2. Nhiều nguồn tín hiệu tạo ranked list:
       - Dense CLIP  (ensemble encoder 1)
       - Dense SigLIP (ensemble encoder 2, Mục 2.1)
       - Sparse BM25  (objects/OCR/ASR/caption, Mục 2.4)
     Có subset (đã filter) -> EXACT search trên subset (không mất recall, Mục 11.1);
     không filter -> ANN HNSW trên toàn dataset (nhanh).
  3. FUSION bằng Reciprocal Rank Fusion (Mục 4.3): gộp các ranked list khác thang đo
     mà không cần chuẩn hoá điểm. (Phase 5 sẽ tách RRF ra retrieval/fusion.py và mở
     rộng; ở đây dùng bản tối thiểu để tầng coarse hoạt động và test được.)

VÌ SAO NHẬN VECTOR ĐÃ MÃ HOÁ (không tự encode text): mã hoá query text -> vector cần
CLIP/SigLIP TEXT ENCODER thật (model + GPU) — thuộc phần cắm dữ liệu thật. Tách phần
encode ra ngoài giúp tầng retrieval test được offline (truyền thẳng vector), và bản
thật chỉ việc bơm vector từ encoder vào.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ingestion.build_index import EncoderName, KeyframeIndex
from retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


@dataclass
class Candidate:
    """Một ứng viên coarse kèm điểm fusion và thông tin truy vết nguồn."""

    keyframe_id: str
    row: int
    score: float                       # điểm RRF sau fusion
    video_id: str
    timestamp: float
    source_ranks: dict[str, int]       # {"clip": rank, "siglip": rank, "bm25": rank}


#: Độ sâu fusion mặc định — mỗi nguồn (clip/siglip/bm25) đóng góp bấy nhiêu rank vào
#: RRF. [PROVISIONAL] cho tới khi bench_fusion_depth.py đo xong (Mục 11.3).
DEFAULT_FUSION_DEPTH = 1000


class CoarseRetriever:
    """Retriever tầng coarse trên một KeyframeIndex đã dựng (Phase 3).

    `fusion_depth` TÁCH RIÊNG khỏi `top_k` — đây là bản sửa của một BUG ĐO ĐƯỢC:
    trước đây `top_k` vừa là "số kết quả trả về" vừa là "độ sâu mỗi ranked list", nên
    xin nhiều kết quả hơn lại làm THỨ HẠNG đổi (hit@5: 0.627 khi xin top_k=5 -> 0.510
    khi xin top_k=20 — cùng query, cùng index).

    VÌ SAO đổi: điểm RRF của một item = tổng 1/(k+rank) trên các list mà nó CÓ MẶT.
    List sâu hơn -> item xuất hiện trong NHIỀU nguồn hơn -> điểm nó TĂNG. Một ảnh dense
    rank 2 + bm25 rank 12 vô hình ở depth=5 (chỉ tính dense), nhưng ở depth=20 được cộng
    thêm điểm bm25 và vọt lên trên ảnh dense rank 1. Tức là số kết quả người gọi XIN đã
    âm thầm đổi thứ tự — sai về mặt logic: xem bao nhiêu kết quả không được phép ảnh
    hưởng kết quả nào tốt nhất.
    """

    def __init__(
        self,
        index: KeyframeIndex,
        rrf_k: int = DEFAULT_RRF_K,
        fusion_depth: int = DEFAULT_FUSION_DEPTH,
    ):
        self.index = index
        self.rrf_k = rrf_k
        self.fusion_depth = fusion_depth

    # ------------------------------------------------------- metadata filter
    def _apply_filters(self, filters: Optional[dict[str, Any]]) -> Optional[list[int]]:
        """Trả danh sách row thoả filter, hoặc None nếu không có filter (= toàn bộ).

        Hỗ trợ:
          - video_id: str | video_ids: list[str]     (khớp video)
          - objects_all: list[str]                    (chứa TẤT CẢ object)
          - objects_any: list[str]                    (chứa ÍT NHẤT 1 object)
          - time_range: (start, end)                  (timestamp trong khoảng, bao gồm 2 đầu)
        """
        if not filters:
            return None
        rows = range(len(self.index.ids))
        result: list[int] = []
        vid = filters.get("video_id")
        vids = set(filters.get("video_ids", []))
        if vid:
            vids.add(vid)
        objs_all = set(filters.get("objects_all", []))
        objs_any = set(filters.get("objects_any", []))
        time_range = filters.get("time_range")

        for r in rows:
            if vids and self.index.video_ids[r] not in vids:
                continue
            if time_range is not None:
                t = self.index.timestamps[r]
                if not (time_range[0] <= t <= time_range[1]):
                    continue
            if objs_all or objs_any:
                oset = set(self.index.objects[r])
                if objs_all and not objs_all.issubset(oset):
                    continue
                if objs_any and oset.isdisjoint(objs_any):
                    continue
            result.append(r)
        return result

    # --------------------------------------------------------------- search
    def _dense_ranked_rows(
        self,
        query_vec: np.ndarray,
        encoder: EncoderName,
        candidate_rows: Optional[list[int]],
        top_k: int,
    ) -> list[int]:
        if candidate_rows is not None:
            # Đã pre-filter -> exact trên subset (không mất recall, Mục 11.1).
            pairs = self.index.exact_dense_search(
                query_vec, encoder, candidate_rows, top_k
            )
        else:
            pairs = self.index.dense_search(query_vec, encoder, top_k)
        return [row for row, _ in pairs]

    def search(
        self,
        query_clip_vec: Optional[np.ndarray] = None,
        query_siglip_vec: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 1000,
        weights: Optional[dict[str, float]] = None,
        depth: Optional[int] = None,
    ) -> list[Candidate]:
        """Trả top-K Candidate đã fusion. Ít nhất một nguồn (vec/text) phải có.

        `weights`: trọng số RRF theo nguồn ({"clip":1,"siglip":1,"bm25":3}...). Tăng
        weight BM25 giúp khớp CHỮ chính xác (biển hiệu) nổi lên khi dense (2 encoder)
        vốn áp đảo.

        `top_k`: CHỈ là số kết quả trả về — không còn ảnh hưởng thứ hạng. Nếu
        `fusion_depth` nhỏ hơn `top_k` thì trả về ít hơn top_k (xem bên dưới).
        `depth`: độ sâu fusion, mặc định `self.fusion_depth`. Chỉ truyền tay khi
        benchmark quét giá trị (bench_retrieval.py --fusion-depth)."""
        if query_clip_vec is None and query_siglip_vec is None and query_text is None:
            raise ValueError(
                "Cần ít nhất 1 tín hiệu query: clip_vec, siglip_vec, hoặc text."
            )
        candidate_rows = self._apply_filters(filters)
        if candidate_rows is not None and not candidate_rows:
            return []  # filter loại hết -> không có ứng viên

        # Độ sâu fusion CỐ ĐỊNH, KHÔNG kẹp theo top_k (xem docstring class).
        #
        # Từng viết `max(top_k, depth)` cho "an toàn" — SAI, và nó âm thầm phá thí
        # nghiệm: bench gọi top_k=100 nên mọi depth < 100 bị kẹp lên 100, đo cùng một
        # cấu hình 5 lần rồi tưởng "độ sâu không ảnh hưởng". Kẹp như vậy còn tái lập
        # đúng cái bug vừa sửa (top_k lại chi phối độ sâu).
        #
        # depth < top_k -> trả về ÍT hơn top_k. Đó là đúng: fusion sâu d trên n nguồn
        # chỉ sinh tối đa d·n ứng viên; bịa thêm cho đủ số mới là sai.
        fuse_depth = depth if depth is not None else self.fusion_depth

        ranked_lists: dict[str, list[int]] = {}
        if query_clip_vec is not None and self.index.has_encoder("clip"):
            ranked_lists["clip"] = self._dense_ranked_rows(
                query_clip_vec, "clip", candidate_rows, fuse_depth
            )
        if query_siglip_vec is not None and self.index.has_encoder("siglip"):
            ranked_lists["siglip"] = self._dense_ranked_rows(
                query_siglip_vec, "siglip", candidate_rows, fuse_depth
            )
        if query_text:
            sparse = self.index.sparse_search(query_text, fuse_depth, candidate_rows)
            ranked_lists["bm25"] = [row for row, _ in sparse]

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, self.rrf_k, weights=weights)
        ordered = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

        return [
            Candidate(
                keyframe_id=self.index.ids[row],
                row=row,
                score=score,
                video_id=self.index.video_ids[row],
                timestamp=self.index.timestamps[row],
                source_ranks=ranks,
            )
            for row, (score, ranks) in ordered
        ]
