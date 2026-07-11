"""Trích xuất OCR (chữ trong ảnh) và ASR (lời nói trong audio) — Phase 2, Mục 6.

VÌ SAO CẦN: ngoài tín hiệu thị giác (embedding), chữ hiện trên khung hình (biển
hiệu, phụ đề) và lời thoại là tín hiệu NGỮ NGHĨA mạnh để retrieval kiểu sparse
(BM25) ở tầng coarse (Mục 3, bước [2]). OCR/ASR + llm_caption cùng nhau lấp khoảng
trống mà embedding thuần thị giác bỏ sót.

THIẾT KẾ (ABC + Mock + bản thật lazy-import, Mục 1.5):
  - OCR thật: PaddleOCR; ASR thật: Whisper. Cả hai lazy-import để mock-first chạy
    được khi chưa cài.
  - Mock: trả text rỗng/tất định để pipeline hoàn chỉnh mà không cần ảnh/audio thật.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .schemas import RawKeyframe


# =============================== OCR ==========================================
class OcrEngine(ABC):
    """Interface: trích xuất chữ trong ảnh keyframe."""

    @abstractmethod
    def extract(self, raw: RawKeyframe) -> Optional[str]:
        ...


class MockOcrEngine(OcrEngine):
    """Mock OCR: mặc định trả None (đa số keyframe không có chữ).

    Có thể nạp sẵn `canned` = {keyframe_id: text} để mô phỏng vài frame có chữ,
    phục vụ test nhánh "có OCR".
    """

    def __init__(self, canned: Optional[dict[str, str]] = None):
        self.canned = canned or {}

    def extract(self, raw: RawKeyframe) -> Optional[str]:
        return self.canned.get(raw.id)


class EasyOcrEngine(OcrEngine):
    """Bản THẬT: EasyOCR (lazy-import) — đọc chữ trên keyframe (biển hiệu, phụ đề...).

    VÌ SAO EasyOCR (thay vì PaddleOCR): dùng torch nên TỰ chạy GPU (đã có torch CUDA),
    dễ cài trên Python 3.14, hỗ trợ tiếng Việt + Anh (cùng bảng Latin). Model reader
    nạp LƯỜI (tải ~100MB lần đầu). Kết quả nối các đoạn text có độ tin cậy >= min_conf
    thành 1 chuỗi để đưa vào BM25 (tìm chữ/biển hiệu).
    """

    def __init__(self, langs: tuple[str, ...] = ("vi", "en"), gpu: Optional[bool] = None,
                 min_conf: float = 0.3):
        try:
            import easyocr
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "EasyOcrEngine cần 'easyocr'. Cài: pip install easyocr. "
                "(Đang mock-first — dùng MockOcrEngine để test offline.)"
            ) from e
        if gpu is None:  # tự bật GPU nếu có CUDA
            try:
                import torch
                gpu = bool(torch.cuda.is_available())
            except ImportError:
                gpu = False
        self.min_conf = min_conf
        self._reader = easyocr.Reader(list(langs), gpu=gpu)

    def extract(self, raw: RawKeyframe) -> Optional[str]:  # pragma: no cover - bản thật
        if raw.image_path is None:
            return None
        results = self._reader.readtext(raw.image_path)
        texts = [t for (_box, t, conf) in results if conf >= self.min_conf and t.strip()]
        text = " ".join(texts).strip()
        return text or None


class PaddleOcrEngine(OcrEngine):
    """Bản THẬT: PaddleOCR (lazy-import). Hỗ trợ tiếng Việt + tiếng Anh."""

    def __init__(self, lang: str = "vi"):
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:  # pragma: no cover - chỉ khi dùng bản thật
            raise ImportError(
                "PaddleOcrEngine cần 'paddleocr' + 'paddlepaddle'. Cài: "
                "pip install paddleocr paddlepaddle. (Đang mock-first — dùng "
                "MockOcrEngine để test offline.)"
            ) from e
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract(self, raw: RawKeyframe) -> Optional[str]:  # pragma: no cover
        if raw.image_path is None:
            return None
        result = self._ocr.ocr(raw.image_path, cls=True)
        if not result or not result[0]:
            return None
        # Ghép các đoạn text nhận diện được theo thứ tự.
        lines = [line[1][0] for line in result[0] if line and line[1]]
        text = " ".join(lines).strip()
        return text or None


# =============================== ASR ==========================================
class AsrEngine(ABC):
    """Interface: chép lại lời nói trong đoạn audio quanh keyframe."""

    @abstractmethod
    def transcribe(self, raw: RawKeyframe) -> Optional[str]:
        ...


class MockAsrEngine(AsrEngine):
    """Mock ASR: trả None mặc định, hoặc text đóng hộp theo id."""

    def __init__(self, canned: Optional[dict[str, str]] = None):
        self.canned = canned or {}

    def transcribe(self, raw: RawKeyframe) -> Optional[str]:
        return self.canned.get(raw.id)


class WhisperAsrEngine(AsrEngine):
    """Bản THẬT: OpenAI Whisper (lazy-import)."""

    def __init__(self, model_size: str = "small", language: Optional[str] = "vi"):
        try:
            import whisper
        except ImportError as e:  # pragma: no cover - chỉ khi dùng bản thật
            raise ImportError(
                "WhisperAsrEngine cần 'openai-whisper'. Cài: "
                "pip install openai-whisper. (Đang mock-first — dùng MockAsrEngine "
                "để test offline.)"
            ) from e
        self.language = language
        self._model = whisper.load_model(model_size)

    def transcribe(self, raw: RawKeyframe) -> Optional[str]:  # pragma: no cover
        if raw.audio_path is None:
            return None
        result = self._model.transcribe(raw.audio_path, language=self.language)
        text = (result.get("text") or "").strip()
        return text or None
