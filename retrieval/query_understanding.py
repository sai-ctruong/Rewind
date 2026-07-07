"""Query Understanding — decompose query tự nhiên -> StructuredQuery (Mục 4.1, Phase 4).

VÌ SAO BẮT BUỘC: search trực tiếp câu thô làm giảm accuracy do "semantic gap". Tách
query thành {objects, actions, location, attributes, time, temporal_order, type} cho
phép: (a) lọc metadata cứng (Mục 11.1.1), (b) kiểm tra thứ tự thời gian (Mục 4.5),
(c) định tuyến đúng loại bài toán (Mục 3 bước [6]).

THIẾT KẾ (ABC + Mock + Claude lazy, Mục 1.5): bản THẬT gọi Claude trả JSON theo schema;
bản MOCK dùng heuristic tiếng Việt (lexicon + luật) để chạy/test offline khi chưa có
API. Mock KHÔNG nhằm thay LLM về độ tinh vi — chỉ để pipeline hoạt động và kiểm được
đúng schema + các trường parse chắc chắn (temporal_order, color, query_type...).
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

from ingestion.schemas import QUERY_TYPES, StructuredQuery

# ---- Lexicon nhỏ cho heuristic mock (đủ cho các câu case-study Mục 4.1) --------
_COLOR_WORDS = {
    "đỏ", "xanh", "xanh dương", "xanh lá", "vàng", "hồng", "tím", "cam",
    "đen", "trắng", "nâu", "xám",
}
_TIME_WORDS = {
    "hôm qua", "hôm nay", "tuần trước", "tuần này", "tháng trước", "tháng này",
    "năm ngoái", "sáng nay", "tối qua", "hai tuần trước",
}
_OBJECT_WORDS = {
    "chìa khóa", "móc khóa", "gấu bông", "mũ", "hoa", "nến", "bánh", "áo",
    "áo sơ mi", "xe", "chìa khoá", "quà", "bánh sinh nhật", "kính",
}
_ACTION_WORDS = {
    "cởi", "vào", "tưới", "mặc", "đánh rơi", "làm rơi", "rơi", "gặp", "tặng",
    "cầm", "đi", "chạy", "ngồi", "đứng",
}
_LOCATION_MARKERS = ("ở ", "tại ", "trong ", "trên ")


class QueryUnderstander(ABC):
    """Interface: raw text -> StructuredQuery."""

    @abstractmethod
    def parse(self, raw_text: str, query_type_hint: str | None = None) -> StructuredQuery:
        ...


def _detect_query_type(text: str) -> str:
    """Suy loại bài toán từ dấu hiệu ngôn ngữ (heuristic)."""
    low = text.lower()
    if any(w in low for w in ("bao nhiêu", "đếm", "mấy ", "số lượng")):
        return "VQA"
    if any(w in low for w in ("tất cả", "những cảnh", "các cảnh", "mọi cảnh")):
        return "AVS"
    if "video mẫu" in low or "đoạn video mẫu" in low:
        return "KIS_video"
    return "KIS_textual"


def _detect_temporal_order(text: str) -> list[dict] | None:
    """Bắt cấu trúc 'A TRƯỚC KHI B' / 'A SAU KHI B' -> danh sách sự kiện có order.

    Đây là input cho Temporal consistency check (Mục 4.5). Chỉ xử lý 2 mẫu phổ biến;
    LLM thật sẽ tổng quát hơn.
    """
    low = text.lower()
    for marker, first_is_earlier in (("trước khi", True), ("sau khi", False)):
        if marker in low:
            before, after = low.split(marker, 1)
            e1 = _clean_event(before)
            e2 = _clean_event(after)
            if not e1 or not e2:
                return None
            # "A trước khi B": A xảy ra trước (order 1). "A sau khi B": B trước.
            if first_is_earlier:
                return [{"event": e1, "order": 1}, {"event": e2, "order": 2}]
            return [{"event": e2, "order": 1}, {"event": e1, "order": 2}]
    return None


def _clean_event(fragment: str) -> str:
    """Rút gọn một mệnh đề sự kiện: bỏ chủ ngữ dài dòng, giữ cụm động từ chính."""
    frag = fragment.strip(" ,.;")
    # Bỏ các marker chủ ngữ đầu câu thường gặp để giữ cụm hành động.
    for lead in ("người đàn ông ", "người phụ nữ ", "cô ấy ", "anh ấy ", "họ ", "tôi "):
        if frag.startswith(lead):
            frag = frag[len(lead):]
    return frag.strip()


def _scan_lexicon(text: str, lexicon: set[str]) -> list[str]:
    """Trả các cụm trong lexicon xuất hiện trong text (ưu tiên cụm dài trước)."""
    low = text.lower()
    found: list[str] = []
    for phrase in sorted(lexicon, key=len, reverse=True):
        if phrase in low and not any(phrase in f for f in found):
            found.append(phrase)
    return found


# Cụm KHÔNG phải địa điểm (loại nhầm khi marker 'trong' đứng trước 'video').
_NON_LOCATION = {"video", "đoạn video", "phim", "đoạn phim", "ảnh", "hình"}


def _detect_location(text: str) -> str | None:
    """Bắt địa điểm sau các marker 'ở/tại/trong/trên' tới hết mệnh đề.

    Có lọc nhiễu: bỏ đuôi mệnh đề sau ' có ', loại các cụm không phải nơi chốn
    (vd 'trong video'), và loại cụm chứa từ nghi vấn (câu VQA) hoặc quá dài.
    """
    low = text.lower()
    for marker in _LOCATION_MARKERS:
        i = low.find(marker)
        if i == -1:
            continue
        rest = text[i + len(marker):]
        # Cắt tới dấu phẩy/chấm hoặc liên từ tiếp theo.
        loc = re.split(r"[,.;?]| và | rồi | trước | sau ", rest)[0].strip()
        # Bỏ mệnh đề phụ sau ' có ' (vd 'video có bao nhiêu...').
        loc = re.split(r"\s+có\s+", loc)[0].strip()
        # Cắt cụm thời gian dính đuôi (vd '...ngoài trời tuần trước' -> '...ngoài trời').
        low_loc = loc.lower()
        for tw in sorted(_TIME_WORDS, key=len, reverse=True):
            j = low_loc.find(tw)
            if j != -1:
                loc = loc[:j].strip()
                low_loc = loc.lower()
                break
        if not loc or low_loc in _NON_LOCATION:
            continue
        if any(w in low_loc for w in ("bao nhiêu", "đếm", "mấy")):
            continue
        if len(loc.split()) > 6:
            continue
        return loc
    return None


class MockQueryUnderstander(QueryUnderstander):
    """Bản MOCK heuristic — chạy offline, không API."""

    def parse(self, raw_text: str, query_type_hint: str | None = None) -> StructuredQuery:
        colors = _scan_lexicon(raw_text, _COLOR_WORDS)
        times = _scan_lexicon(raw_text, _TIME_WORDS)
        attributes: dict = {}
        if colors:
            attributes["color"] = colors
        qtype = query_type_hint or _detect_query_type(raw_text)
        if qtype not in QUERY_TYPES:
            qtype = "KIS_textual"
        return StructuredQuery(
            raw_text=raw_text,
            objects=_scan_lexicon(raw_text, _OBJECT_WORDS),
            actions=_scan_lexicon(raw_text, _ACTION_WORDS),
            location=_detect_location(raw_text),
            attributes=attributes,
            time_constraint=times[0] if times else None,
            temporal_order=_detect_temporal_order(raw_text),
            query_type=qtype,
        )


# Prompt cho bản thật: ép Claude trả đúng JSON schema Mục 4.1/7.
UNDERSTANDING_PROMPT = """Bạn là bộ phân tích truy vấn cho hệ thống truy xuất video.
Phân tích câu truy vấn sau và trả về DUY NHẤT một JSON hợp lệ (không giải thích thêm)
theo schema:
{{
  "objects": [chuỗi],
  "actions": [chuỗi],
  "location": chuỗi hoặc null,
  "attributes": {{"color": [chuỗi], ...}},
  "time_constraint": chuỗi hoặc null,
  "temporal_order": [{{"event": chuỗi, "order": số}}] hoặc null,
  "query_type": một trong {types}
}}
Với mệnh đề thứ tự thời gian (vd "cởi mũ TRƯỚC KHI vào phòng"), phải điền temporal_order.
Truy vấn: "{query}"
"""


class ClaudeQueryUnderstander(QueryUnderstander):
    """Bản THẬT: gọi Claude trả JSON (anthropic lazy-import, key từ env)."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 500,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - chỉ khi dùng bản thật
            raise ImportError(
                "ClaudeQueryUnderstander cần SDK 'anthropic'. Cài: pip install anthropic. "
                "(Đang mock-first — dùng MockQueryUnderstander để test offline.)"
            ) from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Thiếu API key: đặt biến môi trường {api_key_env}.")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def parse(self, raw_text: str, query_type_hint: str | None = None) -> StructuredQuery:  # pragma: no cover
        prompt = UNDERSTANDING_PROMPT.format(
            types=sorted(QUERY_TYPES), query=raw_text
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        data = json.loads(_extract_json(text))
        qtype = data.get("query_type") or query_type_hint or "KIS_textual"
        if qtype not in QUERY_TYPES:
            qtype = "KIS_textual"
        return StructuredQuery(
            raw_text=raw_text,
            objects=list(data.get("objects") or []),
            actions=list(data.get("actions") or []),
            location=data.get("location"),
            attributes=dict(data.get("attributes") or {}),
            time_constraint=data.get("time_constraint"),
            temporal_order=data.get("temporal_order"),
            query_type=qtype,
        )


def _extract_json(text: str) -> str:
    """Rút khối JSON đầu tiên khỏi output LLM (phòng khi model thêm chữ thừa)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Không tìm thấy JSON trong output LLM: {text!r}")
    return text[start : end + 1]
