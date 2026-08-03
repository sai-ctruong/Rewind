"""Video-aware allocation of retrieval candidates to the official Top-100."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RankingConfig:
    top_k: int = 100
    max_frames_per_video: int = 12
    min_frame_gap: int = 30
    neighbor_offsets: tuple[int, ...] = (-2, -1, 1, 2)
    diversity_lambda: float = 0.25
    precision_head_size: int = 5
    recall_tail_size: int = 50


def video_aware_top100(
    candidates: Sequence[T],
    *,
    video_id: Callable[[T], str],
    frame_id: Callable[[T], int],
    score: Callable[[T], float],
    neighbors: Callable[[T, Sequence[int]], Sequence[T]] | None = None,
    config: RankingConfig = RankingConfig(),
) -> list[T]:
    ordered = sorted(candidates, key=lambda item: (-score(item), video_id(item), frame_id(item)))
    if not ordered:
        return []
    out: list[T] = []
    seen: set[tuple[str, int]] = set()
    per_video: dict[str, int] = {}

    def add(item: T, *, enforce_gap: bool = True) -> bool:
        if len(out) >= config.top_k:
            return False
        key = (video_id(item), frame_id(item))
        if key in seen or per_video.get(key[0], 0) >= config.max_frames_per_video:
            return False
        if enforce_gap and any(v == key[0] and abs(f - key[1]) < config.min_frame_gap for v, f in seen):
            return False
        seen.add(key)
        per_video[key[0]] = per_video.get(key[0], 0) + 1
        out.append(item)
        return True

    # Preserve the strongest hypotheses for precision-sensitive ranks.
    for item in ordered:
        add(item, enforce_gap=True)
        if len(out) >= min(config.precision_head_size, config.top_k):
            break
    # Add nearby frames because official ground truth is an interval.
    if neighbors:
        for anchor in list(out):
            for item in neighbors(anchor, config.neighbor_offsets):
                add(item, enforce_gap=False)
                if len(out) >= min(20, config.top_k):
                    break
            if len(out) >= min(20, config.top_k):
                break
    # Recall tail: temporal/video diversity first, then fill any remaining legal slots.
    for enforce_gap in (True, False):
        for item in ordered:
            add(item, enforce_gap=enforce_gap)
            if len(out) >= config.top_k:
                return out
    return out
