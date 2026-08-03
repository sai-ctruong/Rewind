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
    image_bytes: Optional[bytes] = None   # ẢNH TRONG RAM (JPEG) — dùng khi KHÔNG ghi ảnh ra đĩa
    audio_path: Optional[str] = None      # đoạn audio quanh timestamp (cho ASR)
    objects: list[str] = field(default_factory=list)  # Open Images 600 categories (BTC cấp)
    object_detections: list[dict] = field(default_factory=list)  # label, confidence, bounding_box
    source_video: Optional[str] = None    # đường dẫn VIDEO GỐC -> decode lại frame khi cần
    frame_idx: Optional[int] = None       # số thứ tự frame trong video (seek chính xác)


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


def _decode_from_source(raw: "RawKeyframe"):
    """Decode LẠI frame gốc từ file video (BGR numpy) theo frame_idx/timestamp.

    VÌ SAO CẦN: khi index được NẠP LẠI TỪ ĐĨA (A2), ta CỐ Ý không lưu ảnh (image_bytes
    quá nặng cho triệu frame). Ảnh chỉ dựng lại KHI CẦN (hiển thị/rerank) bằng cách
    seek đúng frame trong video gốc — đổi chút thời gian decode lấy dung lượng đĩa.
    """
    if raw.source_video is None:
        return None
    import cv2

    cap = cv2.VideoCapture(raw.source_video)
    if not cap.isOpened():
        return None
    try:
        if raw.frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw.frame_idx)   # seek chính xác theo frame
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, raw.timestamp * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def load_pil_image(raw: "RawKeyframe"):
    """Trả PIL.Image RGB của keyframe: ưu tiên ảnh TRONG RAM (image_bytes) → file trên
    đĩa (image_path) → decode lại từ VIDEO GỐC (source_video, khi nạp index từ đĩa).

    VÌ SAO: pipeline xử lý frame TRONG RAM (không ghi từng frame ra đĩa — quá nhiều
    file với video dài). SigLIP/VLM đọc ảnh qua helper này để dùng được mọi nguồn mà
    không phải đổi logic ở nhiều nơi.
    """
    from io import BytesIO

    from PIL import Image

    if raw.image_bytes is not None:
        return Image.open(BytesIO(raw.image_bytes)).convert("RGB")
    if raw.image_path is not None:
        return Image.open(raw.image_path).convert("RGB")
    frame = _decode_from_source(raw)
    if frame is not None:
        import cv2
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    raise ValueError(
        f"Keyframe {raw.id!r} không có ảnh (thiếu image_bytes, image_path lẫn source_video)."
    )


def load_cv2_image(raw: "RawKeyframe"):
    """Trả ảnh dạng numpy BGR (cv2): ưu tiên RAM → đĩa → decode lại từ video gốc.

    Dùng cho EasyOCR (readtext nhận numpy array). Tránh phải ghi ảnh ra đĩa chỉ để
    OCR đọc lại.
    """
    import numpy as _np

    if raw.image_bytes is not None:
        import cv2
        arr = _np.frombuffer(raw.image_bytes, dtype=_np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if raw.image_path is not None:
        import cv2
        return cv2.imread(raw.image_path)
    frame = _decode_from_source(raw)
    if frame is not None:
        return frame
    raise ValueError(
        f"Keyframe {raw.id!r} không có ảnh (thiếu image_bytes, image_path lẫn source_video)."
    )


def frame_jpeg_bytes(raw: "RawKeyframe") -> Optional[bytes]:
    """Trả JPEG bytes của keyframe (RAM → đĩa → decode lại từ video gốc), None nếu chịu.

    VÌ SAO: cả việc PHỤC VỤ ảnh cho web lẫn GỬI ảnh cho LVLM (VQA/Reader) đều cần đúng
    một thứ — jpeg bytes — bất kể ảnh đang ở RAM, trên đĩa, hay phải dựng lại từ video.
    Gom về một chỗ để không mỗi nơi tự xoay một kiểu (và quên mất nhánh decode lại)."""
    if raw.image_bytes is not None:
        return raw.image_bytes
    try:
        img = load_cv2_image(raw)
    except ValueError:
        return None
    if img is None:
        return None
    import cv2
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None


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
