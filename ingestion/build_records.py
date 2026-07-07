"""Orchestrator ingestion Phase 2: RawKeyframe -> KeyframeRecord đầy đủ field.

VAI TRÒ: nối 4 thành phần Phase 2 (CLIP + SigLIP + OCR/ASR + LLM caption) thành một
pipeline, biến mỗi keyframe THÔ (RawKeyframe) thành một KeyframeRecord đã làm giàu
đủ field theo Mục 7. Đây chính là chỗ hiện thực DoD Phase 2.

LƯU Ý PHÂN BIỆT với build_index.py (Phase 3): file này TẠO RA record; build_index.py
sẽ NHẬN các record đó để dựng Faiss/BM25 index.

Dùng dependency injection: truyền vào các provider (mock hoặc thật). Mặc định là MOCK
để chạy offline ngay (Mục 1.5); khi có GPU/API/data, đổi sang bản thật mà KHÔNG phải
sửa hàm này — đúng nguyên tắc tách interface khỏi cài đặt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .embed_clip import ClipEmbeddingProvider, MockClipEmbeddingProvider
from .embed_siglip import MockSiglipEmbeddingProvider, SiglipEmbeddingProvider
from .llm_captioning import Captioner, MockCaptioner
from .ocr_asr_extract import (
    AsrEngine,
    MockAsrEngine,
    MockOcrEngine,
    OcrEngine,
)
from .schemas import KeyframeRecord, RawKeyframe


@dataclass
class IngestionPipeline:
    """Gom các thành phần làm giàu keyframe. Mặc định toàn bộ là MOCK.

    Vì sao dùng dataclass injection: mỗi thành phần là một interface độc lập; muốn
    bật/tắt hay thay bản thật chỉ cần gán field tương ứng. Giúp test từng nhánh và
    nâng cấp dần từng phần (vd chỉ có SigLIP thật, còn lại vẫn mock).
    """

    clip: ClipEmbeddingProvider = field(default_factory=MockClipEmbeddingProvider)
    siglip: SiglipEmbeddingProvider = field(
        default_factory=MockSiglipEmbeddingProvider
    )
    ocr: OcrEngine = field(default_factory=MockOcrEngine)
    asr: AsrEngine = field(default_factory=MockAsrEngine)
    captioner: Captioner = field(default_factory=MockCaptioner)

    def build_one(self, raw: RawKeyframe) -> KeyframeRecord:
        """Làm giàu 1 keyframe thô thành 1 KeyframeRecord đầy đủ field."""
        return KeyframeRecord(
            id=raw.id,
            video_id=raw.video_id,
            timestamp=raw.timestamp,
            clip_embedding=self.clip.embed(raw),
            siglip_embedding=self.siglip.embed(raw),
            objects=list(raw.objects),
            ocr_text=self.ocr.extract(raw),
            asr_text=self.asr.transcribe(raw),
            llm_caption=self.captioner.caption(raw),
            # is_cluster_representative / cluster_span do bước dedup (Phase 1) điền
            # sau; ở đây giữ mặc định (True / None) vì chưa gom cụm.
        )

    def build(self, raws: Iterable[RawKeyframe]) -> list[KeyframeRecord]:
        """Làm giàu cả một lô keyframe (vd toàn bộ 1 video)."""
        return [self.build_one(raw) for raw in raws]


def searchable_text(record: KeyframeRecord) -> str:
    """Gộp các tín hiệu text của 1 record để đưa vào BM25/full-text (Mục 2.4, 3).

    Gộp objects + OCR + ASR + llm_caption thành một chuỗi để index sparse. Tách
    riêng hàm này để Phase 3 (build_index) tái dùng, và để test kiểm được caption
    đã thực sự vào text tìm kiếm.
    """
    parts: list[str] = []
    if record.objects:
        parts.append(" ".join(record.objects))
    for txt in (record.ocr_text, record.asr_text, record.llm_caption):
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def make_sample_video(
    video_id: str = "sample_video",
    num_frames: int = 6,
    scene_size: int = 3,
) -> list[RawKeyframe]:
    """Tạo một 'video mẫu' THÔ để chạy pipeline mock end-to-end (thay cho data thật).

    Mô phỏng lifelog: các frame chia thành cảnh (mỗi `scene_size` frame là 1 cảnh),
    gán objects khác nhau theo cảnh để MockCaptioner sinh caption có nội dung. Đây là
    stand-in cho 'video mẫu thật' trong DoD Phase 2 khi chưa có dữ liệu BTC.
    """
    scene_objects = [
        ["person", "coffee cup", "table"],
        ["person", "child", "flower"],
        ["car", "street", "traffic light"],
    ]
    raws: list[RawKeyframe] = []
    for i in range(num_frames):
        scene = i // scene_size
        raws.append(
            RawKeyframe(
                id=f"{video_id}/{i}",
                video_id=video_id,
                timestamp=float(i * 2),
                image_path=None,  # mock không cần ảnh thật
                audio_path=None,
                objects=scene_objects[scene % len(scene_objects)],
            )
        )
    return raws
