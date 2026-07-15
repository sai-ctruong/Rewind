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


# ============================================================================
# G4 — RAG READER (MemoriEase 3.0: "Rerank -> Reader")
# ============================================================================
#
# BỐI CẢNH (Slide Buổi 3, slide 31): sau khi Agent RERANK ra top-K keyframe, một bước
# READER TỔNG HỢP câu trả lời ngôn ngữ tự nhiên CÓ DẪN CHỨNG (grounded) — không chỉ trả
# về lưới ảnh trần. Đây là bước biến "danh sách kết quả" thành "câu trả lời có thể đọc",
# và là mắt xích để KISC/hội thoại nói "tôi tìm được X vì…".
#
# KHÁC VqaModule: VQA trả lời một CÂU HỎI trên một cửa sổ thời gian; Reader TÓM TẮT/GIẢI
# THÍCH kết quả TRUY XUẤT (top-K keyframe rời rạc, có thể khác video) theo truy vấn. Dùng
# lại đúng hạ tầng (VqaAnswer-style, Claude vision lazy) nhưng đầu vào là KẾT QUẢ TOOL đã
# chuẩn hoá (dict) + entry để tra text/ảnh — nối thẳng được vào G2 SearchAgent.


@dataclass
class ReaderAnswer:
    """Câu trả lời grounded của Reader: text + CÁC keyframe được trích dẫn (truy vết)."""

    answer: str
    cited_frame_ids: list[str] = field(default_factory=list)
    reasoning: str = ""


def _frame_text(entry, kid: str) -> str:
    """Gộp mọi tín hiệu CHỮ có cho 1 keyframe (caption VLM + OCR + ASR) để grounding.

    Reader dựa vào đây để dẫn chứng bằng NGÔN TỪ (không chỉ id). Rỗng nếu keyframe chưa
    có caption/OCR/ASR — khi đó Reader lùi về mô tả theo video/thời gian."""
    parts = [
        getattr(entry, "caption_by_id", {}).get(kid),
        getattr(entry, "ocr_by_id", {}).get(kid),
        getattr(entry, "asr_by_id", {}).get(kid),
    ]
    return " · ".join(p for p in parts if p)


class Reader(ABC):
    """Interface Reader: (truy vấn, kết quả top-K, entry) -> câu trả lời grounded."""

    @abstractmethod
    def read(self, query: str, results: Sequence[dict], entry,
             top_k: int = 5) -> ReaderAnswer:
        ...


class MockReader(Reader):
    """Reader OFFLINE tất định: dựng câu trả lời từ TÍN HIỆU CHỮ sẵn có của top-K.

    Không "nhìn" được ảnh (đó là việc của ClaudeReader), nhưng tổng hợp caption/OCR/ASR
    + video/timestamp thành câu trả lời có DẪN CHỨNG keyframe_id — đủ để chạy/đo pipeline
    G2→Reader và demo 'trả lời có giải thích' mà không cần API key."""

    def read(self, query: str, results: Sequence[dict], entry,
             top_k: int = 5) -> ReaderAnswer:
        top = list(results)[:top_k]
        if not top:
            return ReaderAnswer(answer=f"Không tìm thấy kết quả cho: \"{query}\".")
        # Kết quả có 2 DẠNG: keyframe phẳng {keyframe_id,…} hoặc CHUỖI thời gian
        # {video_id, steps:[…]} (từ search_temporal). Reader phải đọc được cả hai —
        # nếu không, một truy vấn "A trước khi B" sẽ làm sập cả vòng Agent.
        if any(it.get("steps") for it in top):
            return self._read_chains(query, top, entry, total=len(results))
        cited = [it["keyframe_id"] for it in top if it.get("keyframe_id")]
        lines: list[str] = []
        for it in top:
            kid = it.get("keyframe_id")
            if not kid:
                continue
            txt = _frame_text(entry, kid)
            where = f"video {it.get('video_id')} @ {float(it.get('timestamp') or 0):.1f}s"
            snippet = f" — {txt}" if txt else ""
            lines.append(f"[{kid}] ({where}){snippet}")
        head = (f"Tìm thấy {len(results)} kết quả liên quan đến \"{query}\". "
                f"{len(lines)} keyframe khớp nhất:")
        return ReaderAnswer(
            answer=head + "\n" + "\n".join(lines),
            cited_frame_ids=cited,
            reasoning="Tổng hợp caption/OCR/ASR + vị trí thời gian của top-K (offline).",
        )

    def _read_chains(self, query: str, top: list[dict], entry,
                     total: Optional[int] = None) -> ReaderAnswer:
        """Diễn giải kết quả CHUỖI THỜI GIAN: mỗi chuỗi là các cảnh đúng thứ tự."""
        cited: list[str] = []
        lines: list[str] = []
        for ch in top:
            steps = ch.get("steps") or []
            cited.extend(s["keyframe_id"] for s in steps if s.get("keyframe_id"))
            path = " → ".join(
                f"[{s['keyframe_id']}] {float(s.get('timestamp') or 0):.1f}s"
                f" ({s.get('event')})" for s in steps)
            lines.append(f"video {ch.get('video_id')}: {path}")
        n = total if total is not None else len(top)
        head = (f"Tìm thấy {n} chuỗi cảnh đúng thứ tự cho \"{query}\""
                + (f" — {len(top)} chuỗi khớp nhất:" if n > len(top) else ":"))
        return ReaderAnswer(
            answer=head + "\n" + "\n".join(lines), cited_frame_ids=cited,
            reasoning="Các cảnh cùng video, timestamp tăng dần đúng thứ tự yêu cầu.",
        )


READER_PROMPT = (
    "Duoi day la cac keyframe truy xuat duoc cho truy van: \"{query}\". Moi anh kem "
    "keyframe_id. Hay TRA LOI NGAN GON bang tieng Viet: mo ta ket qua khop nhat va "
    "GIAI THICH vi sao no khop truy van, TRICH DAN keyframe_id lien quan trong ngoac "
    "vuong (vi du [kf_12])."
)


class ClaudeReader(Reader):
    """Reader THẬT dùng Claude vision (anthropic lazy): gửi top-K ảnh + truy vấn, nhận
    câu trả lời grounded. Cho tiêm `client` để test cấu trúc request offline.

    Ảnh lấy từ `entry.raws[kid].image_bytes` (frame trong RAM); keyframe đã nạp lại từ
    đĩa (mất image_bytes) sẽ được BỎ QUA phần ảnh — Reader vẫn grounding bằng text +
    trích dẫn id (giảm nhẹ chứ không chặn)."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 500,
                 max_images: int = 5, client=None):
        self.model = model
        self.max_tokens = max_tokens
        self.max_images = max_images
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ClaudeReader cần ANTHROPIC_API_KEY (hoặc MockReader).")
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("Chưa cài 'anthropic'. pip install anthropic") from e
            self._client = anthropic.Anthropic()
        return self._client

    def _build_content(self, query: str, results, entry, top_k: int):
        import base64

        top = list(results)[:top_k]
        content: list[dict] = []
        cited: list[str] = []
        n_img = 0
        for it in top:
            kid = it.get("keyframe_id")
            if not kid:      # chuỗi thời gian -> không có ảnh đơn lẻ để gửi
                continue
            cited.append(kid)
            raw = getattr(entry, "raws", {}).get(kid)
            img = getattr(raw, "image_bytes", None) if raw is not None else None
            if img and n_img < self.max_images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": base64.standard_b64encode(img).decode("ascii")},
                })
                n_img += 1
            content.append({"type": "text", "text": f"keyframe_id={kid} "
                            f"({_frame_text(entry, kid) or 'khong co mo ta'})"})
        content.append({"type": "text", "text": READER_PROMPT.format(query=query)})
        return content, cited

    def read(self, query: str, results: Sequence[dict], entry,
             top_k: int = 5) -> ReaderAnswer:
        if not results:
            return ReaderAnswer(answer=f"Không tìm thấy kết quả cho: \"{query}\".")
        content, cited = self._build_content(query, results, entry, top_k)
        msg = self._get_client().messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()
        return ReaderAnswer(answer=text, cited_frame_ids=cited,
                            reasoning="Claude vision tổng hợp top-K keyframe.")
