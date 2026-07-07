"""Data schema cho tầng ingestion (CLAUDE.md Mục 7).

RÀNG BUỘC: Mục 7 cấm tự ý đổi tên field của KeyframeRecord — mọi module ingestion
và retrieval đều dựa vào đúng các tên này. Chỉ được MỞ RỘNG (thêm field mới), không
được đổi/xoá field đã khai báo.

Ghi chú thiết kế:
  - `clip_embedding` do BTC cấp sẵn nên luôn có ở Phase 2 trở đi.
  - `siglip_embedding`, `ocr_text`, `asr_text`, `llm_caption` được điền dần qua các
    bước khác nhau của pipeline ingestion (Phase 2). Vì vậy chúng để Optional với
    default None để có thể khởi tạo record "một phần" ở các bước sớm mà không phải
    bịa dữ liệu — điều này KHÔNG đổi tên field, chỉ nới lỏng tính bắt buộc lúc khởi
    tạo (đúng tinh thần cho phép "mở rộng" của Mục 7).
  - `is_cluster_representative` và `cluster_span` do bước dedup (Phase 1, Mục 5.1)
    điền: True + (t_start, t_end) cho keyframe đại diện của một cụm frame gần trùng.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RawKeyframe:
    """Đầu vào THÔ của một keyframe cho pipeline ingestion Phase 2.

    Đây là những gì ta có TRƯỚC khi làm giàu dữ liệu: keyframe do BTC trích xuất
    (id, thuộc video nào, mốc thời gian, đường dẫn ảnh gốc, danh sách object đã
    detect sẵn, và tuỳ chọn đường dẫn audio để ASR). Pipeline sẽ bổ sung embedding,
    OCR, ASR, caption để tạo ra một `KeyframeRecord` hoàn chỉnh.

    Tách RawKeyframe khỏi KeyframeRecord để phân biệt rõ "dữ liệu đầu vào" với
    "record đã làm giàu" — giúp mock/test dễ dàng và không phải bịa các field chưa có.
    """

    id: str
    video_id: str
    timestamp: float
    image_path: Optional[str] = None      # đường dẫn ảnh keyframe gốc (cho SigLIP/OCR/caption)
    audio_path: Optional[str] = None      # đoạn audio quanh timestamp (cho ASR)
    objects: list[str] = field(default_factory=list)  # Open Images 600 categories (BTC cấp)


@dataclass
class KeyframeRecord:
    """Một keyframe đã trích xuất + toàn bộ tín hiệu đa phương tiện kèm theo.

    Đây là đơn vị dữ liệu trung tâm của pipeline: đi từ ingestion (trích xuất,
    dedup, embed, caption) tới retrieval (coarse search, rerank).
    """

    id: str
    video_id: str
    timestamp: float                                # giây tính từ đầu video
    clip_embedding: np.ndarray                      # từ BTC cấp sẵn (Mục 2.1)
    siglip_embedding: Optional[np.ndarray] = None   # tự trích xuất ở Phase 2
    objects: list[str] = field(default_factory=list)  # Open Images 600 categories
    ocr_text: Optional[str] = None
    asr_text: Optional[str] = None                  # transcript audio quanh timestamp
    llm_caption: Optional[str] = None               # caption LVLM sinh lúc index (Mục 2.4)
    is_cluster_representative: bool = True           # True nếu là đại diện sau dedup
    cluster_span: Optional[tuple[float, float]] = None  # (t_start, t_end) của cụm

    def __post_init__(self) -> None:
        # Chuẩn hoá embedding về np.ndarray float32 để cosine/Faiss nhất quán.
        # Lý do float32: Faiss dùng float32 mặc định; ép sớm tránh lỗi dtype về sau.
        if self.clip_embedding is not None and not isinstance(
            self.clip_embedding, np.ndarray
        ):
            self.clip_embedding = np.asarray(self.clip_embedding, dtype=np.float32)
        if self.siglip_embedding is not None and not isinstance(
            self.siglip_embedding, np.ndarray
        ):
            self.siglip_embedding = np.asarray(self.siglip_embedding, dtype=np.float32)


# Các loại bài toán hợp lệ cho query_type (Mục 0 + Mục 7).
QUERY_TYPES = {"KIS_video", "KIS_textual", "AVS", "VQA", "KISC"}


@dataclass
class StructuredQuery:
    """Query tự nhiên đã được decompose thành cấu trúc (CLAUDE.md Mục 4.1, Mục 7).

    Bắt buộc parse query thô -> cấu trúc TRƯỚC khi retrieval (Mục 4.1), không search
    trực tiếp câu thô (giảm accuracy do semantic gap). `temporal_order` là input cho
    bước Temporal consistency check (Mục 4.5, Phase 6).

    Không đổi tên field (Mục 7). `query_type` phải thuộc QUERY_TYPES.
    """

    raw_text: str
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    location: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    time_constraint: Optional[str] = None
    temporal_order: Optional[list[dict]] = None
    query_type: str = "KIS_textual"
