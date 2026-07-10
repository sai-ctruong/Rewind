"""Engine tìm kiếm video độ chính xác cao (không cần API key).

Nâng cấp so với bản demo 1-encoder ban đầu, áp dụng đúng các kỹ thuật blueprint:

1. ENSEMBLE 2 ENCODER (Mục 2.1): SigLIP2 (thế hệ mới, chính xác hơn) + SigLIP
   multilingual (đời trước, phân phối lỗi khác). Mỗi encoder cho 1 ranked list,
   fuse bằng RRF qua CoarseRetriever — giảm rủi ro "cùng sai" (correlated errors).
2. QUERY PROMPT ENSEMBLE (Mục 4.2 — Multi-Query Expansion, bản không cần LLM):
   encode nhiều biến thể prompt của cùng câu truy vấn ("{q}", "a photo of {q}",
   "một bức ảnh về {q}") rồi lấy TRUNG BÌNH embedding (chuẩn hoá lại). Một cách
   diễn đạt duy nhất có thể lệch khỏi phân phối caption model đã học — trung bình
   nhiều biến thể "trúng" biểu diễn đúng thường xuyên hơn.
3. LẤY MẪU DÀY + DEDUP NGỮ NGHĨA (Mục 5.1): sample mỗi 0.5s (dày gấp đôi) để không
   bỏ lỡ khoảnh khắc ngắn, rồi gộp frame gần trùng bằng ingestion/dedup (anchor-based,
   trên embedding — tinh hơn histogram) để index không phình và không nhiễu.
4. Điểm số minh bạch: giữ source_ranks của RRF để truy vết encoder nào ủng hộ.

Mọi tham số accuracy-critical đều [PROVISIONAL] theo Mục 11.3 — chỉnh qua benchmark,
không coi là số cuối.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ingestion.build_index import KeyframeIndex, l2_normalize
from ingestion.dedup import deduplicate_keyframes
from ingestion.schemas import KeyframeRecord, RawKeyframe
from ingestion.video_ingest import extract_keyframes
from retrieval.coarse_retriever import Candidate, CoarseRetriever

# Cặp encoder mặc định cho ensemble (Mục 2.1). Cả hai đều đa ngôn ngữ:
#   - SigLIP2: bản kế nhiệm, retrieval tốt hơn SigLIP1 trên benchmark.
#   - SigLIP1 multilingual: phân phối lỗi khác đời -> ensemble có ích.
DEFAULT_ENCODERS = (
    "google/siglip2-base-patch16-256",
    "google/siglip-base-patch16-256-multilingual",
)

# Biến thể prompt cho query ensemble (Mục 4.2). Trộn Anh + Việt vì cả 2 encoder đều
# đa ngôn ngữ; template "a photo of" khớp phân phối caption model được huấn luyện.
QUERY_TEMPLATES = (
    "{q}",
    "a photo of {q}.",
    "một bức ảnh về {q}.",
)


@dataclass
class VideoIndexEntry:
    """Kết quả index 1 video: Faiss index + map id -> RawKeyframe (đường dẫn ảnh)."""

    video_id: str
    index: KeyframeIndex
    raws: dict[str, RawKeyframe]
    num_sampled: int          # số keyframe sau bước cắt thô (histogram)
    num_indexed: int          # số keyframe sau dedup ngữ nghĩa (vào index)


class VideoSearchEngine:
    """Điều phối: cắt keyframe -> embed ensemble -> dedup -> index -> search RRF.

    Model nạp LƯỜI ở lần dùng đầu (tải/khởi tạo chậm — không chặn app startup).
    """

    def __init__(
        self,
        encoder_names: Sequence[str] = DEFAULT_ENCODERS,
        sample_every_s: float = 0.5,           # [PROVISIONAL] dày hơn bản cũ (1.0s)
        max_frames: int = 120,                  # [PROVISIONAL] trần frame lấy mẫu
        dedup_threshold: float = 0.97,          # [PROVISIONAL] khớp settings.yaml
        query_templates: Sequence[str] = QUERY_TEMPLATES,
    ):
        self.encoder_names = list(encoder_names)[:2]  # KeyframeIndex có đúng 2 slot dense
        self.sample_every_s = sample_every_s
        self.max_frames = max_frames
        self.dedup_threshold = dedup_threshold
        self.query_templates = list(query_templates)
        self._encoders: Optional[list] = None   # nạp lười

    # ------------------------------------------------------------- encoders
    def _load_encoders(self) -> list:
        if self._encoders is None:
            from ingestion.embed_siglip import SiglipEncoder

            self._encoders = [SiglipEncoder(model_name=n) for n in self.encoder_names]
        return self._encoders

    def set_encoders(self, encoders: list) -> None:
        """Bơm encoder ngoài (mock trong test / model tuỳ chọn). Mỗi encoder cần
        `.embed(raw) -> vec` và `.encode_text(str) -> vec`."""
        self._encoders = list(encoders)[:2]

    # -------------------------------------------------------------- indexing
    def index_video(
        self, video_path: str | Path, out_dir: str | Path,
        video_id: Optional[str] = None,
    ) -> VideoIndexEntry:
        raws = extract_keyframes(
            video_path, out_dir, video_id=video_id,
            sample_every_s=self.sample_every_s, max_frames=self.max_frames,
        )
        if not raws:
            raise RuntimeError("Không trích được keyframe nào từ video.")
        encoders = self._load_encoders()

        records: list[KeyframeRecord] = []
        for r in raws:
            rec = KeyframeRecord(
                id=r.id, video_id=r.video_id, timestamp=r.timestamp,
                clip_embedding=encoders[0].embed(r),
            )
            if len(encoders) > 1:
                rec.siglip_embedding = encoders[1].embed(r)
            records.append(rec)

        # Dedup NGỮ NGHĨA (Mục 5.1): gộp frame gần trùng theo embedding chính (slot 1).
        # An toàn recall: đại diện giữ cluster_span nên khoảng thời gian không mất.
        num_sampled = len(records)
        result = deduplicate_keyframes(
            records, similarity_threshold=self.dedup_threshold,
            embedding_attr="clip_embedding",
        )
        kept = result.representatives

        index = KeyframeIndex.build(kept)
        vid = kept[0].video_id if kept else (video_id or Path(video_path).stem)
        return VideoIndexEntry(
            video_id=vid, index=index,
            raws={r.id: r for r in raws},
            num_sampled=num_sampled, num_indexed=len(kept),
        )

    # ---------------------------------------------------------------- query
    def encode_query(self, query: str) -> list[np.ndarray]:
        """Encode query cho TỪNG encoder, mỗi encoder = trung bình các biến thể prompt.

        Trung bình vector đơn vị rồi chuẩn hoá lại (Mục 4.2 — weighted average của
        embedding, trọng số đều)."""
        encoders = self._load_encoders()
        out: list[np.ndarray] = []
        for enc in encoders:
            variants = [
                enc.encode_text(tpl.format(q=query)) for tpl in self.query_templates
            ]
            mean = np.mean(np.stack(variants), axis=0).astype(np.float32)
            out.append(l2_normalize(mean.reshape(1, -1))[0])
        return out

    def search(
        self, entry: VideoIndexEntry, query: str, top_k: int = 8,
    ) -> list[Candidate]:
        """Search ensemble: mỗi encoder 1 ranked list -> RRF fuse (CoarseRetriever)."""
        qvecs = self.encode_query(query)
        retriever = CoarseRetriever(entry.index)
        return retriever.search(
            query_clip_vec=qvecs[0],
            query_siglip_vec=qvecs[1] if len(qvecs) > 1 else None,
            top_k=top_k,
        )
