"""Sinh caption tự nhiên cho keyframe bằng LVLM lúc indexing (CLAUDE.md Mục 2.4).

VÌ SAO QUAN TRỌNG (Mục 2.4): object detector (600 category rời rạc) chỉ ra "person",
"child", "flower" riêng lẻ, KHÔNG nắm được QUAN HỆ ngữ nghĩa như "người lớn đang
hướng dẫn trẻ em tưới hoa" — đúng thách thức AVS. Caption tự nhiên do LVLM sinh lấp
trực tiếp khoảng trống này, rồi đưa vào BM25/full-text index cùng OCR/ASR.

VÌ SAO CHẠY Ở INDEXING (offline): gọi LVLM cho HÀNG TRIỆU keyframe chỉ khả thi vì
BTC cấp API miễn phí (Mục 2.3). Ràng buộc là độ trễ LÚC THI, không phải lúc index —
nên caption sinh sẵn trước ngày thi, không tính vào latency thi đấu (Mục 11.2).

THIẾT KẾ (ABC + Mock + Claude lazy, Mục 1.5): mock-first vì hiện CHƯA có API key.
`ClaudeCaptioner` lazy-import `anthropic` và đọc key từ biến môi trường -> khi có
key chỉ việc thay MockCaptioner bằng ClaudeCaptioner.
"""
from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .schemas import RawKeyframe

# Prompt theo đúng mô tả Mục 2.4.
CAPTION_PROMPT = (
    "Mô tả chi tiết những gì đang diễn ra trong ảnh này bằng 1-2 câu, chú ý: "
    "người, hành động, địa điểm, màu sắc, đồ vật, mối quan hệ tương tác giữa các "
    "đối tượng."
)


class Captioner(ABC):
    """Interface: sinh caption tự nhiên cho một RawKeyframe."""

    @abstractmethod
    def caption(self, raw: RawKeyframe) -> Optional[str]:
        ...


class MockCaptioner(Captioner):
    """Mock: dựng caption tất định TỪ danh sách object có sẵn của keyframe.

    Không gọi API. Dù thô hơn LVLM thật, nó vẫn cho pipeline một `llm_caption`
    khác None để test end-to-end đầy đủ field (DoD Phase 2). Nếu keyframe không có
    object nào, trả một câu mô tả trống hợp lệ thay vì None (đảm bảo field luôn đầy).
    """

    def caption(self, raw: RawKeyframe) -> Optional[str]:
        if raw.objects:
            objs = ", ".join(raw.objects)
            return f"Cảnh có {objs} tại video {raw.video_id} (mô tả mock)."
        return f"Cảnh trong video {raw.video_id} lúc {raw.timestamp:.0f}s (mô tả mock)."


class ClaudeCaptioner(Captioner):
    """Bản THẬT: gọi Claude vision qua SDK `anthropic` (lazy-import).

    Đọc API key từ biến môi trường (mặc định ANTHROPIC_API_KEY). KHÔNG hard-code
    key trong repo (Mục 2.3). Khi BTC cấp key, set biến môi trường rồi dùng class
    này thay MockCaptioner.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 200,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - chỉ khi dùng bản thật
            raise ImportError(
                "ClaudeCaptioner cần SDK 'anthropic'. Cài: pip install anthropic. "
                "(Đang mock-first — dùng MockCaptioner để test offline.)"
            ) from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Thiếu API key: đặt biến môi trường {api_key_env}. "
                "(BTC cấp key thì set vào đây; hiện chưa có nên dùng MockCaptioner.)"
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _media_type(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/jpeg")

    def caption(self, raw: RawKeyframe) -> Optional[str]:  # pragma: no cover
        if raw.image_path is None:
            raise ValueError(
                f"Keyframe {raw.id!r} thiếu image_path — LVLM caption cần ảnh gốc."
            )
        img_bytes = Path(raw.image_path).read_bytes()
        b64 = base64.standard_b64encode(img_bytes).decode("ascii")
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": self._media_type(raw.image_path),
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            ],
        )
        parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        text = " ".join(parts).strip()
        return text or None
