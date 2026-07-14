"""G3 — Session Memory: trí nhớ phiên (episodic + semantic) xuyên nhiều lượt.

BỐI CẢNH (Slide Buổi 3 — "Memory"):
    Agent mạnh cần NHỚ trải nghiệm và tri thức xuyên các lượt tương tác, không chỉ phản
    xạ trên prompt hiện tại. Slide phân loại theo NỘI DUNG bộ nhớ:
      - Episodic (hồi tưởng): ghi CHUỖI SỰ KIỆN theo lối *append-only*; đọc lại bằng
        heuristic (ở đây: theo độ mới — recency).
      - Semantic (ngữ nghĩa): TRI THỨC suy ra từ các sự kiện; đọc lại bằng truy xuất.
    MemoriEase 3.0 (slide 31) đưa "observation space = lịch sử trò chuyện NHIỀU LƯỢT"
    làm đầu vào chính — đó chính là thứ file này cung cấp.

VÌ SAO CẦN (nối vào Agent):
    Trước G3, mỗi truy vấn là một lượt ĐỘC LẬP; KISC chỉ nhớ trong 1 lượt. G3 cho phép
    "LƯỢT 2 NHỚ LƯỢT 1": phản hồi 👍/👎 người dùng cho ở lượt trước được TÍCH LUỸ vào
    bộ nhớ ngữ nghĩa, rồi bơm vào Rocchio (search_with_feedback) ở lượt sau — càng hỏi
    càng sát ý, đúng vòng khám phá↔khai phá.

THIẾT KẾ (offline, tất định — chạy không cần API key):
  - `episodic`: list[Turn] append-only (chuỗi sự kiện của phiên).
  - Bộ nhớ ngữ nghĩa GỌN, HỮU DỤNG NGAY: tập id LIÊN QUAN (positive) / KHÔNG (negative)
    tích luỹ, và `facts` (dict) cho tri thức tự do (vd "mục tiêu = móc khoá đỏ"). Quy tắc
    hoà giải: PHẢN HỒI MỚI THẮNG — một id đánh dấu negative sẽ bị gỡ khỏi positive và
    ngược lại (người dùng đổi ý ở lượt sau được tôn trọng).
  Việc "LLM suy luận ra facts" (bản mạnh của semantic memory) để dành cho tầng có API;
  ở đây `facts` được ghi tường minh — đủ để mang thông tin xuyên lượt mà vẫn test được.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class Turn:
    """Một sự kiện trong chuỗi episodic: người dùng hỏi gì, Agent làm gì, phản hồi gì."""

    query: str
    route: Optional[str] = None                       # tool nhánh chính đã dùng
    result_ids: list[str] = field(default_factory=list)
    positive_ids: list[str] = field(default_factory=list)  # 👍 cho KẾT QUẢ LƯỢT TRƯỚC
    negative_ids: list[str] = field(default_factory=list)  # 👎
    answer: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SessionMemory:
    """Bộ nhớ một phiên hội thoại: episodic (append-only) + semantic (feedback + facts)."""

    def __init__(self) -> None:
        self.episodic: list[Turn] = []
        # Dùng dict giữ THỨ TỰ xuất hiện (Python 3.7+), cho phép gỡ khi đổi ý.
        self._positive: dict[str, None] = {}
        self._negative: dict[str, None] = {}
        self.facts: dict[str, Any] = {}

    # -- episodic (chuỗi sự kiện) ------------------------------------------
    def record(self, turn: Turn) -> None:
        """Ghi NỐI TIẾP một lượt (append-only) + gấp phản hồi của lượt vào semantic."""
        self.episodic.append(turn)
        self.note_feedback(turn.positive_ids, turn.negative_ids)

    def recent(self, n: int = 3) -> list[Turn]:
        """Đọc lại theo heuristic ĐỘ MỚI (recency) — n lượt gần nhất."""
        return self.episodic[-n:] if n > 0 else []

    def recent_queries(self, n: int = 3) -> list[str]:
        return [t.query for t in self.recent(n)]

    @property
    def num_turns(self) -> int:
        return len(self.episodic)

    # -- semantic: phản hồi tích luỹ (relevance) ---------------------------
    def note_feedback(self, positive_ids: Sequence[str] = (),
                      negative_ids: Sequence[str] = ()) -> None:
        """Gấp phản hồi vào bộ nhớ ngữ nghĩa. PHẢN HỒI MỚI THẮNG khi mâu thuẫn."""
        for i in positive_ids:
            self._negative.pop(i, None)
            self._positive[i] = None
        for i in negative_ids:
            self._positive.pop(i, None)
            self._negative[i] = None

    @property
    def positive_ids(self) -> list[str]:
        return list(self._positive)

    @property
    def negative_ids(self) -> list[str]:
        return list(self._negative)

    def has_feedback(self) -> bool:
        return bool(self._positive or self._negative)

    def feedback_context(self) -> dict:
        """Gói phản hồi tích luỹ để bơm vào ToolRegistry.context (Planner đọc)."""
        return {"positive_ids": self.positive_ids, "negative_ids": self.negative_ids}

    # -- semantic: tri thức tự do (facts) ----------------------------------
    def remember(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    # -- tổng hợp cho hiển thị / đưa vào Planner ---------------------------
    def summary(self) -> dict:
        return {
            "turns": self.num_turns,
            "recent_queries": self.recent_queries(),
            "positive_ids": self.positive_ids,
            "negative_ids": self.negative_ids,
            "facts": dict(self.facts),
        }
