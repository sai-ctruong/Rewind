"""Configurable multi-signal retrieval fusion for AIC keyframes."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class RetrievalSignal(str, Enum):
    CLIP = "clip"
    OBJECT = "object"
    METADATA = "metadata"
    SPARSE = "sparse"


@dataclass(frozen=True)
class CandidateEvidence:
    matched_objects: tuple[str, ...] = ()
    metadata_terms: tuple[str, ...] = ()
    sparse_terms: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass
class RankedCandidate:
    video_id: str
    frame_id: str
    timestamp: float
    keyframe_path: str | None
    dense_score: float = 0.0
    object_score: float = 0.0
    metadata_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    evidence: CandidateEvidence = field(default_factory=CandidateEvidence)


@dataclass(frozen=True)
class FusionResult:
    method: str
    candidates: tuple[RankedCandidate, ...]
    weights: Mapping[str, float]


@dataclass(frozen=True)
class FusionConfig:
    method: str = "adaptive"
    clip_weight: float = 1.0
    object_weight: float = 0.20
    metadata_weight: float = 0.10
    sparse_weight: float = 0.25
    rrf_k: int = 60
    adaptive: bool = True


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("_", " ")
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    words = []
    for word in value.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def _terms(text: str) -> set[str]:
    return set(normalize_label(text).split())


def object_match_score(query: str, detections: Sequence[Mapping], confidence_threshold: float = 0.20) -> tuple[float, tuple[str, ...]]:
    q = _terms(query)
    if not q:
        return 0.0, ()
    matches: list[tuple[str, float]] = []
    for det in detections:
        confidence = float(det.get("confidence", 0.0))
        label = normalize_label(str(det.get("label", "")))
        if confidence >= confidence_threshold and label and (_terms(label) & q):
            matches.append((label, confidence))
    if not matches:
        return 0.0, ()
    # Confidence and count both matter, with diminishing returns for repeated detections.
    confidence_sum = sum(score for _, score in matches)
    score = min(1.0, confidence_sum / math.sqrt(max(1, len(matches))))
    return score, tuple(dict.fromkeys(label for label, _ in matches))


def token_overlap_score(query: str, text: str) -> tuple[float, tuple[str, ...]]:
    q, d = _terms(query), _terms(text)
    overlap = tuple(sorted(q & d))
    return (len(overlap) / max(1, len(q))), overlap


def adaptive_weights(query: str, config: FusionConfig) -> dict[str, float]:
    weights = {
        "clip": config.clip_weight,
        "object": config.object_weight,
        "metadata": config.metadata_weight,
        "sparse": config.sparse_weight,
    }
    if not config.adaptive:
        return weights
    has_quoted_text = bool(re.search(r"['\"].+?['\"]|\b[A-Z]{2,}\b|\d{2,}", query))
    object_words = {"person", "car", "bus", "bicycle", "dog", "cat", "shirt", "hat", "phone", "xe", "nguoi", "ao", "mu"}
    has_object = bool(_terms(query) & object_words)
    action_words = {"running", "walking", "entering", "leaving", "holding", "chay", "di", "vao", "ra"}
    has_action = bool(_terms(query) & action_words)
    if has_quoted_text:
        weights["sparse"] *= 2.0
        weights["metadata"] *= 1.5
    if has_object:
        weights["object"] *= 1.75
    if has_action:
        weights["clip"] *= 1.25
    return weights


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def fuse_candidates(query: str, candidates: Sequence[RankedCandidate], config: FusionConfig | None = None) -> FusionResult:
    config = config or FusionConfig()
    items = list(candidates)
    weights = adaptive_weights(query, config)
    signals = {
        "clip": _minmax([c.dense_score for c in items]),
        "object": _minmax([c.object_score for c in items]),
        "metadata": _minmax([c.metadata_score for c in items]),
        "sparse": _minmax([c.bm25_score for c in items]),
    }
    if config.method.lower() in {"rrf", "rrf_full"}:
        totals = [0.0] * len(items)
        for name, values in signals.items():
            for rank, idx in enumerate(sorted(range(len(items)), key=lambda i: (-values[i], i)), 1):
                if values[idx] > 0:
                    totals[idx] += weights[name] / (config.rrf_k + rank)
    else:
        totals = [sum(weights[name] * signals[name][i] for name in signals) for i in range(len(items))]
    for item, score in zip(items, totals):
        item.fused_score = float(score)
    items.sort(key=lambda c: (-c.fused_score, -c.dense_score, c.video_id, c.frame_id))
    return FusionResult(config.method, tuple(items), weights)


ABLATIONS = {
    "clip_only": FusionConfig(method="weighted", object_weight=0, metadata_weight=0, sparse_weight=0, adaptive=False),
    "clip_objects": FusionConfig(method="weighted", metadata_weight=0, sparse_weight=0, adaptive=False),
    "clip_metadata": FusionConfig(method="weighted", object_weight=0, sparse_weight=0, adaptive=False),
    "clip_sparse": FusionConfig(method="weighted", object_weight=0, metadata_weight=0, adaptive=False),
    "rrf_full": FusionConfig(method="rrf", adaptive=False),
    "adaptive_full": FusionConfig(method="adaptive", adaptive=True),
}
