"""Multi-Query Expansion — tăng recall (CLAUDE.md Mục 4.2, Phase 4).

VÌ SAO: một câu mô tả duy nhất có thể không khớp cách model đã học biểu diễn khái niệm
đó. Sinh 3-5 biến thể diễn đạt (đồng nghĩa, góc nhìn khác — "Generative Query Expansion")
rồi union kết quả / trung bình embedding trước fusion -> tăng khả năng "trúng" biểu diễn
đúng, tức tăng RECALL ở tầng coarse (đúng ưu tiên Mục 1.2).

THIẾT KẾ (ABC + Mock + Claude lazy): bản THẬT nhờ Claude paraphrase; bản MOCK dùng luật
đồng nghĩa + template để sinh biến thể tất định, chạy offline. Cả hai LUÔN giữ câu gốc
làm 1 biến thể (không đánh mất truy vấn nguyên bản).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

# Từ điển đồng nghĩa nhỏ cho mock (tiếng Việt). Bản thật để LLM lo.
_SYNONYMS = {
    "tìm": "truy tìm",
    "video": "đoạn phim",
    "người đàn ông": "nam giới",
    "người phụ nữ": "nữ giới",
    "quán cà phê": "tiệm cà phê",
    "màu xanh dương": "màu lam",
    "trẻ em": "em bé",
    "chìa khóa": "chùm chìa khóa",
}


class QueryExpander(ABC):
    """Interface: raw text -> danh sách biến thể (gồm cả câu gốc)."""

    @abstractmethod
    def expand(self, raw_text: str, num_variants: int = 4) -> list[str]:
        ...


class MockQueryExpander(QueryExpander):
    """Bản MOCK: sinh biến thể tất định bằng thay đồng nghĩa + template.

    Đảm bảo: (1) câu gốc luôn là phần tử đầu, (2) không trùng lặp, (3) trả đúng
    tối đa `num_variants` phần tử (có thể ít hơn nếu không sinh đủ biến thể khác nhau).
    """

    def expand(self, raw_text: str, num_variants: int = 4) -> list[str]:
        if num_variants < 1:
            raise ValueError("num_variants phải >= 1")
        variants: list[str] = [raw_text]

        # 1) Thay từng cặp đồng nghĩa (mỗi cặp -> 1 biến thể).
        low = raw_text.lower()
        for src, dst in _SYNONYMS.items():
            if src in low:
                # Thay giữ nguyên phần còn lại (thay ở dạng lowercase để đơn giản).
                variant = low.replace(src, dst)
                if variant not in (v.lower() for v in variants):
                    variants.append(variant)
            if len(variants) >= num_variants:
                break

        # 2) Nếu chưa đủ, thêm template diễn đạt lại (góc nhìn khác).
        templates = [
            f"cảnh quay thể hiện: {raw_text}",
            f"hình ảnh mô tả {raw_text}",
            f"tìm khoảnh khắc: {raw_text}",
        ]
        for t in templates:
            if len(variants) >= num_variants:
                break
            if t not in variants:
                variants.append(t)

        return variants[:num_variants]


EXPANSION_PROMPT = """Sinh {n} cách diễn đạt KHÁC NHAU cho cùng một truy vấn tìm kiếm
video (đồng nghĩa, đổi góc nhìn, giữ nguyên ý). Trả về mỗi biến thể trên một dòng, KHÔNG
đánh số, KHÔNG giải thích. Bao gồm cả câu gốc.
Truy vấn gốc: "{query}"
"""


class ClaudeQueryExpander(QueryExpander):
    """Bản THẬT: nhờ Claude paraphrase (anthropic lazy, key từ env)."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 300,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ClaudeQueryExpander cần SDK 'anthropic'. Cài: pip install anthropic. "
                "(Đang mock-first — dùng MockQueryExpander để test offline.)"
            ) from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Thiếu API key: đặt biến môi trường {api_key_env}.")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def expand(self, raw_text: str, num_variants: int = 4) -> list[str]:  # pragma: no cover
        prompt = EXPANSION_PROMPT.format(n=num_variants, query=raw_text)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        lines = [ln.strip("-•* \t") for ln in text.splitlines() if ln.strip()]
        # Đảm bảo có câu gốc và loại trùng, giữ thứ tự.
        out: list[str] = []
        for ln in [raw_text, *lines]:
            if ln and ln not in out:
                out.append(ln)
        return out[:num_variants]
