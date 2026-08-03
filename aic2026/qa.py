"""Grounded multi-frame Q&A helpers and answer normalization."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ingestion.schemas import KeyframeRecord
from retrieval.vqa_module import MockVqaAnswerer, VqaAnswer, VqaAnswerer


_NUMBER_ALIASES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "khong": "0", "mot": "1", "một": "1", "hai": "2", "ba": "3", "bon": "4",
    "bốn": "4", "nam": "5", "năm": "5", "sau": "6", "sáu": "6", "bay": "7",
    "bảy": "7", "tam": "8", "tám": "8", "chin": "9", "chín": "9", "muoi": "10", "mười": "10",
}
_ALIASES = {
    "yes": "yes", "yeah": "yes", "true": "yes", "co": "yes", "có": "yes", "dung": "yes", "đúng": "yes",
    "no": "no", "nope": "no", "false": "no", "khong": "no", "không": "no", "sai": "no",
    "xanh duong": "blue", "xanh dương": "blue", "blue": "blue",
    "do": "red", "đỏ": "red", "red": "red", "trang": "white", "trắng": "white", "white": "white",
    "den": "black", "đen": "black", "black": "black",
}


def normalize_answer(value: str, aliases: dict[str, str] | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = " ".join(text.split())
    mapping = {**_ALIASES, **(aliases or {})}
    if text in mapping:
        return mapping[text]
    words = [_NUMBER_ALIASES.get(word, word) for word in text.split()]
    return " ".join(words)


@dataclass(frozen=True)
class QAInput:
    event_description: str
    question: str
    expected_answer_type: str | None = None


@dataclass(frozen=True)
class EvidenceFrame:
    record: KeyframeRecord
    role: str
    relevance: float = 0.0


@dataclass(frozen=True)
class GroundedQAResult:
    video_id: str
    frame_id: str
    answer: str
    answer_normalized: str
    evidence_frame_ids: tuple[str, ...]
    grounding_score: float
    answer_confidence: float
    warning: str | None = None


class LocalVlmAnswerer(VqaAnswerer):
    """Adapter contract for a local VLM implementation."""

    def answer(self, question: str, frames: Sequence[KeyframeRecord], images=None) -> VqaAnswer:
        raise RuntimeError("No local VLM backend is configured.")


class OptionalApiVlmAnswerer(VqaAnswerer):
    """Explicit optional adapter; API keys are supplied by the concrete implementation."""

    def answer(self, question: str, frames: Sequence[KeyframeRecord], images=None) -> VqaAnswer:
        raise RuntimeError("No API VLM backend is configured.")


def build_retrieval_query(data: QAInput, mode: str = "event_only", question_weight: float = 0.35) -> str:
    if mode == "event_only":
        return data.event_description.strip() or data.question.strip()
    if mode == "event_plus_question":
        return " ".join(part.strip() for part in (data.event_description, data.question) if part.strip())
    if mode == "weighted_combination":
        repeats = max(1, round(question_weight * 3))
        return " ".join([data.event_description.strip()] + [data.question.strip()] * repeats).strip()
    raise ValueError(f"Unknown Q&A retrieval query mode: {mode}")


def select_diverse_evidence(
    frames: Sequence[KeyframeRecord], center_time: float, count: int = 8, query: str = ""
) -> list[EvidenceFrame]:
    ordered = sorted(frames, key=lambda frame: frame.timestamp)
    if len(ordered) <= count:
        return [EvidenceFrame(frame, "before" if frame.timestamp < center_time else "after" if frame.timestamp > center_time else "event") for frame in ordered]
    indices = {0, len(ordered) - 1, min(range(len(ordered)), key=lambda i: abs(ordered[i].timestamp - center_time))}
    query_terms = set(normalize_answer(query).split())
    scored = []
    for index, frame in enumerate(ordered):
        text = " ".join([*(frame.objects or []), frame.ocr_text or "", frame.asr_text or "", frame.llm_caption or ""])
        relevance = len(query_terms & set(normalize_answer(text).split()))
        scored.append((relevance, -abs(frame.timestamp - center_time), index))
    for _, _, index in sorted(scored, reverse=True):
        indices.add(index)
        if len(indices) >= count:
            break
    chosen = [ordered[index] for index in sorted(indices)[:count]]
    return [EvidenceFrame(frame, "before" if frame.timestamp < center_time else "after" if frame.timestamp > center_time else "event") for frame in chosen]


def confidence_from_evidence(grounding_score: float, evidence_count: int, answer: str) -> float:
    known = normalize_answer(answer) not in {"", "unknown", "khong xac dinh", "không xác định", "khong co mo ta"}
    return max(0.0, min(1.0, 0.65 * grounding_score + 0.05 * min(evidence_count, 6) + (0.05 if known else 0.0)))


MockVqaAnswerer = MockVqaAnswerer
ABLATIONS = ("single_frame", "temporal_window", "diverse_multi_frame", "multi_frame_plus_text_signals", "full_grounded_qa")
