"""Video Question Answering (CLAUDE.md Mục 3 bước [6] VQA, Phase 7).

BÀI TOÁN: video dài + câu hỏi -> câu trả lời text CÓ SUY LUẬN (đếm số lượng, xác định
thứ tự/thời gian, nhận biết ai làm gì). Khác retrieval thuần: không chỉ tìm keyframe
mà phải TRẢ LỜI dựa trên nội dung một CỬA SỔ THỜI GIAN (temporal window) liên quan.

LUỒNG (Mục 3):
  1. Xác định cửa sổ thời gian liên quan tới câu hỏi (retrieve temporal window) — đưa
     đủ ngữ cảnh chuỗi frame cho LVLM suy luận (đếm nến trên bánh cần nhìn khung hình,
     xác định người tặng quà cần thấy chuỗi hành động).
  2. Gọi LVLM trả lời câu hỏi trên cửa sổ đó.

THIẾT KẾ (ABC + Mock + Claude lazy, Mục 1.5): MockVqaAnswerer suy luận bằng heuristic
trên caption/objects (offline) — đủ để chạy/đo pipeline và trả đúng ví dụ case-study.
ClaudeVqaAnswerer (bản thật) gửi nhiều ảnh keyframe + câu hỏi cho Claude vision.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ingestion.build_index import tokenize
from ingestion.schemas import KeyframeRecord

# Số đếm tiếng Việt dạng chữ (cho MockVqaAnswerer). "năm" cũng nghĩa là year nhưng
# trong ngữ cảnh đếm vật thể thì hiểu là 5 — chấp nhận cho mock.
_NUM_WORDS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}
# Object noun hay được hỏi (đủ cho case-study). Có thể mở rộng.
_COUNTABLE_NOUNS = {"nến", "người", "quà", "hoa", "bánh", "xe", "cây", "đèn", "ghế"}
# Trợ động từ cần lược bỏ khi rút chủ ngữ.
_AUX = {"đang", "đã", "sẽ", "vẫn", "cũng", "thì", "là"}


@dataclass
class VqaAnswer:
    """Kết quả VQA: câu trả lời + suy luận + frame đã dùng (truy vết được)."""

    answer: str
    value: Optional[int] = None            # số đếm nếu là câu hỏi đếm
    reasoning: str = ""
    used_frame_ids: list[str] = field(default_factory=list)


def retrieve_temporal_window(
    records: Sequence[KeyframeRecord],
    center_time: Optional[float] = None,
    window_s: float = 10.0,
    video_id: Optional[str] = None,
) -> list[KeyframeRecord]:
    """Chọn các keyframe trong CỬA SỔ THỜI GIAN [center-window, center+window].

    Nếu không cho center_time -> lấy toàn bộ (đã lọc theo video nếu có video_id). Sắp
    theo timestamp để LVLM đọc đúng trình tự thời gian (cần cho suy luận thứ tự)."""
    frames = [r for r in records if video_id is None or r.video_id == video_id]
    if center_time is not None:
        frames = [r for r in frames if abs(r.timestamp - center_time) <= window_s]
    return sorted(frames, key=lambda r: r.timestamp)


def _detect_intent(question: str) -> str:
    low = question.lower()
    if any(w in low for w in ("bao nhiêu", "đếm", "mấy ", "số lượng")):
        return "count"
    if low.strip().startswith(("ai ", "ai?", "ai là")) or "người nào" in low:
        return "identify"
    return "describe"


def _target_noun(question: str) -> Optional[str]:
    for tok in tokenize(question):
        if tok in _COUNTABLE_NOUNS:
            return tok
    return None


def _as_number(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


class VqaAnswerer(ABC):
    """Interface: trả lời câu hỏi dựa trên các frame trong cửa sổ thời gian."""

    @abstractmethod
    def answer(self, question: str, frames: Sequence[KeyframeRecord]) -> VqaAnswer:
        ...


class MockVqaAnswerer(VqaAnswerer):
    """Suy luận heuristic trên caption/objects (offline, không API).

    Không thay được LVLM về khả năng "nhìn" ảnh, nhưng khai thác llm_caption (vốn do
    LVLM sinh lúc indexing — Mục 2.4) để trả lời đếm/nhận diện. Đây là lý do caption
    tự nhiên quan trọng: nó mã hoá sẵn quan hệ ngữ nghĩa để suy luận downstream."""

    def answer(self, question: str, frames: Sequence[KeyframeRecord]) -> VqaAnswer:
        intent = _detect_intent(question)
        used = [f.id for f in frames]
        if intent == "count":
            return self._answer_count(question, frames, used)
        if intent == "identify":
            return self._answer_identify(question, frames, used)
        return self._answer_describe(frames, used)

    def _answer_count(self, question, frames, used) -> VqaAnswer:
        target = _target_noun(question)
        best: Optional[int] = None
        source = ""
        if target:
            # Tìm "<số> [classifier] <target>" trong caption của các frame.
            for f in frames:
                text = f.llm_caption or ""
                toks = tokenize(text)
                for i, tok in enumerate(toks):
                    n = _as_number(tok)
                    if n is None:
                        continue
                    # Nhìn tối đa 3 token phía sau xem có target noun không.
                    if target in toks[i + 1 : i + 4]:
                        if best is None or n > best:
                            best = n
                            source = text
        if best is not None:
            return VqaAnswer(
                answer=str(best), value=best,
                reasoning=f"Đếm '{target}' từ mô tả: \"{source}\"",
                used_frame_ids=used,
            )
        # Fallback: đếm số frame trong cửa sổ có chứa target trong objects/caption.
        if target:
            cnt = sum(
                1 for f in frames
                if target in [o.lower() for o in f.objects]
                or target in (f.llm_caption or "").lower()
            )
            return VqaAnswer(
                answer=str(cnt), value=cnt,
                reasoning=f"Không thấy số rõ trong mô tả; đếm {cnt} frame có '{target}'.",
                used_frame_ids=used,
            )
        return VqaAnswer(answer="không xác định", reasoning="Không rõ đối tượng cần đếm.",
                         used_frame_ids=used)

    def _answer_identify(self, question, frames, used) -> VqaAnswer:
        # Rút động từ hành động trong câu hỏi (vd 'tặng'), tìm chủ ngữ trước động từ đó
        # trong caption.
        q_toks = tokenize(question)
        action = next((t for t in q_toks if t in {"tặng", "cầm", "mặc", "cởi", "tưới"}), None)
        if action:
            for f in frames:
                cap = f.llm_caption or ""
                m = re.search(rf"(.*?)\b{action}\b", cap, flags=re.IGNORECASE)
                if m and m.group(1).strip():
                    subject = _clean_subject(m.group(1))
                    if subject:
                        return VqaAnswer(
                            answer=subject,
                            reasoning=f"Chủ ngữ đứng trước '{action}' trong: \"{cap}\"",
                            used_frame_ids=used,
                        )
        return VqaAnswer(answer="không xác định",
                         reasoning="Không tìm thấy chủ ngữ phù hợp trong mô tả.",
                         used_frame_ids=used)

    def _answer_describe(self, frames, used) -> VqaAnswer:
        caps = [f.llm_caption for f in frames if f.llm_caption]
        return VqaAnswer(
            answer=" ".join(caps[:3]) if caps else "không có mô tả",
            reasoning="Tổng hợp mô tả các frame trong cửa sổ.",
            used_frame_ids=used,
        )


def _clean_subject(fragment: str) -> str:
    """Rút gọn chủ ngữ: bỏ trợ động từ đuôi ('đang', 'đã'...) và khoảng trắng thừa."""
    toks = fragment.strip().split()
    while toks and toks[-1].lower() in _AUX:
        toks.pop()
    return " ".join(toks).strip(" ,.;")


class VqaModule:
    """Điều phối VQA: chọn cửa sổ thời gian rồi giao cho answerer."""

    def __init__(self, answerer: Optional[VqaAnswerer] = None):
        self.answerer = answerer or MockVqaAnswerer()

    def answer(
        self,
        question: str,
        records: Sequence[KeyframeRecord],
        video_id: Optional[str] = None,
        center_time: Optional[float] = None,
        window_s: float = 10.0,
    ) -> VqaAnswer:
        frames = retrieve_temporal_window(records, center_time, window_s, video_id)
        if not frames:
            return VqaAnswer(answer="không có dữ liệu",
                             reasoning="Cửa sổ thời gian không có frame nào.")
        return self.answerer.answer(question, frames)


# --------------------------------------------------------------------- Claude
VQA_PROMPT = (
    "Dưới đây là các keyframe liên tiếp trích từ một video (theo thứ tự thời gian). "
    "Trả lời câu hỏi sau dựa trên nội dung các ảnh, suy luận cẩn thận (đếm chính xác, "
    "chú ý thứ tự thời gian nếu cần). Trả lời NGẮN GỌN.\nCâu hỏi: {question}"
)


class ClaudeVqaAnswerer(VqaAnswerer):
    """Bản THẬT: gửi nhiều ảnh keyframe + câu hỏi cho Claude vision (anthropic lazy)."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_frames: int = 8,
        max_tokens: int = 400,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ClaudeVqaAnswerer cần SDK 'anthropic'. Cài: pip install anthropic. "
                "(Đang mock-first — dùng MockVqaAnswerer để test offline.)"
            ) from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Thiếu API key: đặt biến môi trường {api_key_env}.")
        self.model = model
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def answer(self, question: str, frames: Sequence[KeyframeRecord]) -> VqaAnswer:  # pragma: no cover
        import base64
        from pathlib import Path

        # Lấy đều max_frames frame trong cửa sổ (tránh gửi quá nhiều ảnh -> tốn độ trễ).
        chosen = list(frames)
        if len(chosen) > self.max_frames:
            step = len(chosen) / self.max_frames
            chosen = [chosen[int(i * step)] for i in range(self.max_frames)]
        content: list[dict] = []
        for f in chosen:
            if not getattr(f, "image_path", None):
                continue
            b64 = base64.standard_b64encode(Path(f.image_path).read_bytes()).decode("ascii")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
        content.append({"type": "text", "text": VQA_PROMPT.format(question=question)})
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        num = re.search(r"\d+", text)
        return VqaAnswer(
            answer=text,
            value=int(num.group()) if num else None,
            reasoning="Câu trả lời từ Claude vision trên cửa sổ keyframe.",
            used_frame_ids=[f.id for f in chosen],
        )
