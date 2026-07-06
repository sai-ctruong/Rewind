"""
schemas.py
Các dataclass mô tả trạng thái dữ liệu dùng xuyên suốt module KISC.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Keyframe:
    """
    Đại diện cho 1 keyframe/ứng viên trong kho dữ liệu.
    Trong hệ thống thật, `attributes` sẽ được sinh ra từ: CLIP features, Objects
    (Faster R-CNN), OCR, ASR, metadata... đã được BTC cung cấp sẵn (xem slide 41).
    """
    id: str
    video_id: str
    timestamp: float  # thời điểm (giây) trong video
    attributes: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # similarity score do retriever tính (CLIP similarity, BM25...)


@dataclass
class DialogueTurn:
    """Một lượt hội thoại (của user hoặc system)."""
    turn_index: int
    speaker: str  # "user" | "system"
    text: str
    extracted_filters: Optional[Dict[str, Any]] = None


@dataclass
class DialogueState:
    """
    Trạng thái tích lũy của toàn bộ hội thoại KISC (Dialogue State Tracking).
    `filters` được cộng dồn qua từng lượt — đây chính là "bộ nhớ ngắn hạn" của
    trợ lý ảo trong suốt phiên hội thoại.
    """
    turns: List[DialogueTurn] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Keyframe] = field(default_factory=list)
    asked_attributes: List[str] = field(default_factory=list)
    finished: bool = False
