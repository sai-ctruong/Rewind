"""Bộ lọc ẢNH hội thoại — thu hẹp dần tập keyframe THẬT qua nhiều lượt.

Ý TƯỞNG: thay vì mỗi truy vấn là một lần tìm ĐỘC LẬP trả về danh sách chữ, người dùng
bắt đầu bằng một mô tả thô -> hệ hiện một LƯỚI ẢNH THẬT (vd 20 khung hình), rồi mỗi lượt
sau lưới ẢNH CO LẠI cho tới khi còn đúng thứ cần tìm. Đây là "bộ lọc ảnh", không phải
hội thoại hỏi-đáp bằng chữ.

BA CÁCH THU HẸP (dùng ĐỒNG THỜI — quyết định thiết kế đã chốt):
  1. THÊM MÔ TẢ  — nối chi tiết mới vào truy vấn tích luỹ ("cảnh đường phố" + "áo trắng").
  2. PHẢN HỒI 👍/👎 — Rocchio dịch vector truy vấn về phía ảnh thích / xa ảnh không thích.
  3. CHỌN ẢNH ĐẠI DIỆN — khi còn mơ hồ, hệ đưa vài ảnh KHÁC NHAU hỏi "cái nào gần ý
     nhất?"; ảnh được chọn thành 👍 mạnh, các ảnh còn lại thành 👎.

VÌ SAO CÓ "POOL" (điểm mấu chốt khiến nó là BỘ LỌC chứ không phải tìm lại từ đầu):
    Mỗi lượt ta chỉ xếp hạng lại và giữ lại một phần của TẬP ỨNG VIÊN HIỆN TẠI (pool),
    không tìm lại toàn bộ index. Nhờ vậy số ảnh **đảm bảo giảm dần** và người dùng thấy
    rõ mình đang thu hẹp — đúng mô hình tinh thần "lọc". Đánh đổi: một khung hình bị loại
    sẽ không quay lại (giống mọi bộ lọc) -> luôn có `reset()` để làm lại từ đầu.

TÁI DÙNG: `SessionMemory` (G3) tích luỹ phản hồi + ghi chuỗi lượt; `engine.search`,
`engine.search_with_feedback` (Rocchio), `engine.disambiguation` (chọn ảnh đại diện).
Chạy được offline với engine mock -> test không cần GPU/API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from retrieval.session_memory import SessionMemory, Turn

# Câu hỏi khi hệ chủ động nhờ chọn ảnh (mơ hồ) — hiển thị kèm các ảnh đại diện.
PICK_QUESTION = "Cái nào gần ý bạn nhất?"


@dataclass
class FilterResult:
    """Trạng thái trả về sau mỗi lượt — đủ để UI vẽ lưới ảnh + panel hỏi lại."""

    query: str                                   # truy vấn tích luỹ (mọi mô tả đã cộng dồn)
    results: list[dict] = field(default_factory=list)   # ảnh còn lại (đã xếp hạng)
    count: int = 0                               # số ảnh còn lại
    count_before: int = 0                        # số ảnh trước lượt này (để hiện "20 → 8")
    disambiguation: list[dict] = field(default_factory=list)  # ảnh đại diện để chọn
    question: Optional[str] = None               # câu hỏi khi cần chọn ảnh
    turn: int = 0
    finished: bool = False                       # đã đủ hẹp -> dừng lọc
    memory: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query, "results": self.results, "count": self.count,
            "count_before": self.count_before, "disambiguation": self.disambiguation,
            "question": self.question, "turn": self.turn, "finished": self.finished,
            "memory": self.memory,
        }


class ImageFilterSession:
    """Một phiên lọc ảnh trên MỘT index video thật.

    Dùng:
        s = ImageFilterSession(engine, entry, start_k=20)
        s.start("cảnh đường phố")                  -> 20 ảnh
        s.refine(text="áo trắng")                  -> ~10 ảnh
        s.refine(positive=["kf_3"])                -> ~5 ảnh
        s.refine(pick="kf_7", others=["kf_2"])     -> thu hẹp quanh kf_7
    """

    def __init__(self, engine, entry, *, start_k: int = 20, shrink: float = 0.5,
                 min_k: int = 3):
        self.engine = engine
        self.entry = entry
        self.start_k = start_k
        self.shrink = shrink            # giữ lại bao nhiêu phần pool mỗi lượt
        self.min_k = min_k              # sàn: đừng thu quá hẹp khiến mất ứng viên đúng
        self.memory = SessionMemory()
        self.query_parts: list[str] = []
        self.pool: list[str] = []       # id ứng viên còn lại (thứ tự = xếp hạng hiện tại)
        self._cands_by_id: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def query(self) -> str:
        """Truy vấn TÍCH LUỸ: mọi mô tả người dùng đã thêm, nối lại."""
        return " ".join(p for p in self.query_parts if p)

    def _next_k(self) -> int:
        """Số ảnh giữ lại lượt tới — luôn NHỎ HƠN hiện tại (tới khi chạm sàn min_k)."""
        target = math.ceil(len(self.pool) * self.shrink)
        return max(self.min_k, min(target, max(len(self.pool) - 1, 1)))

    def _pack(self, cands: Sequence[Any]) -> list[dict]:
        out = []
        for c in cands:
            kid = c.keyframe_id
            out.append({
                "id": kid,
                "video_id": c.video_id,
                "timestamp": round(float(c.timestamp), 1),
                "score": round(float(c.score), 4),
                "caption": getattr(self.entry, "caption_by_id", {}).get(kid),
                "ocr": getattr(self.entry, "ocr_by_id", {}).get(kid),
            })
        return out

    def _search(self, top_k: int) -> list:
        """Tìm theo truy vấn tích luỹ, có Rocchio nếu phiên đã có phản hồi."""
        if self.memory.has_feedback():
            return self.engine.search_with_feedback(
                self.entry, self.query,
                positive_ids=self.memory.positive_ids,
                negative_ids=self.memory.negative_ids,
                top_k=top_k)
        return self.engine.search(self.entry, self.query, top_k=top_k)

    def _ask_pick(self, cands: Sequence[Any]) -> list[dict]:
        """Khi còn mơ hồ -> vài ảnh ĐA DẠNG để người dùng chọn (F1)."""
        try:
            ids = self.engine.disambiguation(self.entry, list(cands))
        except Exception:
            return []
        if not ids:
            return []
        by_id = {c.keyframe_id: c for c in cands}
        return [{"id": i, "video_id": by_id[i].video_id,
                 "timestamp": round(float(by_id[i].timestamp), 1)}
                for i in ids if i in by_id]

    # ------------------------------------------------------------------- public
    def start(self, query: str, k: Optional[int] = None) -> FilterResult:
        """Lượt 1: mô tả thô -> lưới ảnh ban đầu (pool)."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Cần một mô tả để bắt đầu lọc.")
        self.memory = SessionMemory()
        self.query_parts = [query]
        top_k = k or self.start_k
        cands = self._search(top_k)
        self._cands_by_id = {c.keyframe_id: c for c in cands}
        self.pool = [c.keyframe_id for c in cands]
        self.memory.record(Turn(query=query, route="filter_start",
                                result_ids=list(self.pool)))
        return FilterResult(
            query=self.query, results=self._pack(cands), count=len(cands),
            count_before=0, disambiguation=self._ask_pick(cands),
            question=PICK_QUESTION if len(cands) > self.min_k else None,
            turn=self.memory.num_turns, finished=len(cands) <= 1,
            memory=self.memory.summary(),
        )

    def refine(self, text: Optional[str] = None, positive: Sequence[str] = (),
               negative: Sequence[str] = (), pick: Optional[str] = None,
               others: Sequence[str] = (), k: Optional[int] = None) -> FilterResult:
        """Một lượt thu hẹp. Kết hợp được cả 3 tín hiệu trong CÙNG một lượt.

        Args:
            text: mô tả THÊM (nối vào truy vấn tích luỹ).
            positive/negative: id ảnh 👍/👎.
            pick: id ảnh người dùng chọn khi hệ hỏi "cái nào gần ý nhất?" -> 👍 mạnh.
            others: các ảnh đại diện KHÔNG được chọn -> 👎 (tín hiệu tương phản rõ).
            k: ép số ảnh giữ lại (mặc định: tự co theo `shrink`).
        """
        if not self.pool:
            raise RuntimeError("Chưa có phiên lọc. Gọi start() trước.")
        count_before = len(self.pool)

        if text and text.strip():
            self.query_parts.append(text.strip())
        pos = list(positive)
        neg = list(negative)
        if pick:
            pos.append(pick)
            neg.extend([o for o in others if o != pick])
        self.memory.note_feedback(pos, neg)

        target_k = k or self._next_k()
        # Xếp hạng lại RỘNG hơn pool rồi lọc về pool -> đảm bảo mọi thành viên pool có
        # cơ hội được chấm lại (nếu chỉ lấy đúng top_k=len(pool) thì dễ sót).
        wide = max(len(self.pool) * 5, 50)
        ranked = [c for c in self._search(wide) if c.keyframe_id in set(self.pool)]
        if not ranked:  # không ai trong pool lọt -> giữ pool cũ, chỉ cắt bớt
            ranked = [self._cands_by_id[i] for i in self.pool if i in self._cands_by_id]

        kept = ranked[:target_k]
        self._cands_by_id.update({c.keyframe_id: c for c in kept})
        self.pool = [c.keyframe_id for c in kept]
        self.memory.record(Turn(query=text or "", route="filter_refine",
                                result_ids=list(self.pool),
                                positive_ids=pos, negative_ids=neg))
        return FilterResult(
            query=self.query, results=self._pack(kept), count=len(kept),
            count_before=count_before, disambiguation=self._ask_pick(kept),
            question=PICK_QUESTION if len(kept) > self.min_k else None,
            turn=self.memory.num_turns, finished=len(kept) <= 1,
            memory=self.memory.summary(),
        )

    def reset(self) -> None:
        """Xoá phiên (pool + phản hồi + truy vấn tích luỹ) để lọc lại từ đầu."""
        self.memory = SessionMemory()
        self.query_parts = []
        self.pool = []
        self._cands_by_id = {}
