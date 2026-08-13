"""Video-aware allocation of retrieval candidates to the official Top-100.

Diversity here is structural, not a weighted objective: `min_frame_gap` suppresses
near-duplicate frames of one video and `max_frames_per_video` caps how much of the list
one video may own. `diversity_lambda` and `recall_tail_size` were removed in R0 because
nothing ever read them — the recall tail is a fill loop with no size of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from .rank_utility import OFFICIAL_CUTOFFS, bucket_of


T = TypeVar("T")


@dataclass(frozen=True)
class RankingConfig:
    final_top_k: int = 100
    max_frames_per_video: int = 12
    min_frame_gap: int = 30
    neighbor_offsets: tuple[int, ...] = (-2, -1, 1, 2)
    precision_head_size: int = 5
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.top_k is not None:
            object.__setattr__(self, "final_top_k", int(self.top_k))


def video_aware_top100(
    candidates: Sequence[T],
    *,
    video_id: Callable[[T], str],
    frame_id: Callable[[T], int],
    score: Callable[[T], float],
    neighbors: Callable[[T, Sequence[int]], Sequence[T]] | None = None,
    config: RankingConfig | None = None,
) -> list[T]:
    config = config or RankingConfig()
    ordered = sorted(candidates, key=lambda item: (-score(item), video_id(item), frame_id(item)))
    if not ordered:
        return []
    out: list[T] = []
    seen: set[tuple[str, int]] = set()
    per_video: dict[str, int] = {}

    def add(item: T, *, enforce_gap: bool = True) -> bool:
        if len(out) >= config.final_top_k:
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
        if len(out) >= min(config.precision_head_size, config.final_top_k):
            break
    # Add nearby frames because official ground truth is an interval.
    if neighbors:
        for anchor in list(out):
            for item in neighbors(anchor, config.neighbor_offsets):
                add(item, enforce_gap=False)
                if len(out) >= min(20, config.final_top_k):
                    break
            if len(out) >= min(20, config.final_top_k):
                break
    # Recall tail: temporal/video diversity first, then fill any remaining legal slots.
    for enforce_gap in (True, False):
        for item in ordered:
            add(item, enforce_gap=enforce_gap)
            if len(out) >= config.final_top_k:
                return out
    return out


def cutoff_aware_top100(
    candidates: Sequence[T],
    *,
    video_id: Callable[[T], str],
    frame_id: Callable[[T], int],
    score: Callable[[T], float],
    neighbors: Callable[[T, Sequence[int]], Sequence[T]] | None = None,
    config: RankingConfig | None = None,
    tail_min_videos: int = 2,
    diagnostics: dict[str, Any] | None = None,
) -> list[T]:
    """R1: allocate the Top-100 with the official cutoffs in mind. EXPERIMENTAL.

    `video_aware_top100` applies one diversity rule from rank 1 to rank 100. But the
    metric does not treat those ranks alike: rank 1 is worth five times rank 51-100, and
    a near-duplicate row sitting at rank 60 consumes a slot that could hold a different
    video's hypothesis for the same cost.

    So this allocator splits the list by cutoff bucket:

    * `1` and `2-5` — precision head. Strongest relevance evidence only, no diversity
      pressure beyond the existing duplicate and frame-gap rules. Deliberately identical
      to the baseline here, because promoting a weaker-but-different row into rank 1
      trades the most valuable slot in the metric for variety.
    * `6-20` — the baseline rule, with neighbour expansion, since ground truth is an
      interval and a neighbouring frame of a correct hit is often also correct.
    * `21-50`, `51-100` — diversity-oriented tail. Each bucket must reach at least
      `tail_min_videos` distinct videos before a video may take a second slot in it,
      so the low-value ranks buy coverage rather than repetition.

    With `adaptive_budget.enabled: false` this function is not called at all: the
    baseline allocator runs unchanged.
    """
    config = config or RankingConfig()
    ordered = sorted(candidates, key=lambda item: (-score(item), video_id(item), frame_id(item)))
    if not ordered:
        return []

    out: list[T] = []
    seen: set[tuple[str, int]] = set()
    per_video: dict[str, int] = {}
    bucket_videos: dict[str, dict[str, int]] = {}
    survival: dict[str, int] = {}

    def current_bucket() -> str:
        return bucket_of(len(out) + 1, OFFICIAL_CUTOFFS)

    def add(item: T, *, enforce_gap: bool, diversity_floor: int = 0) -> bool:
        if len(out) >= config.final_top_k:
            return False
        key = (video_id(item), frame_id(item))
        if key in seen or per_video.get(key[0], 0) >= config.max_frames_per_video:
            return False
        if enforce_gap and any(
            v == key[0] and abs(f - key[1]) < config.min_frame_gap for v, f in seen
        ):
            return False
        bucket = current_bucket()
        counts = bucket_videos.setdefault(bucket, {})
        if diversity_floor and counts.get(key[0], 0) >= 1 and len(counts) < diversity_floor:
            # This bucket has not yet reached its video floor, and this video already
            # holds a slot in it. Give a different video the chance first.
            return False
        seen.add(key)
        per_video[key[0]] = per_video.get(key[0], 0) + 1
        counts[key[0]] = counts.get(key[0], 0) + 1
        survival[bucket] = survival.get(bucket, 0) + 1
        out.append(item)
        return True

    head_size = min(int(config.precision_head_size), int(config.final_top_k))
    for item in ordered:
        if len(out) >= head_size:
            break
        add(item, enforce_gap=True)

    if neighbors:
        for anchor in list(out):
            for item in neighbors(anchor, config.neighbor_offsets):
                add(item, enforce_gap=False)
                if len(out) >= min(20, config.final_top_k):
                    break
            if len(out) >= min(20, config.final_top_k):
                break

    for item in ordered:
        if len(out) >= min(20, config.final_top_k):
            break
        add(item, enforce_gap=True)

    floor = max(1, int(tail_min_videos))
    for enforce_gap in (True, False):
        for pass_floor in (floor, 0):
            for item in ordered:
                add(item, enforce_gap=enforce_gap, diversity_floor=pass_floor)
                if len(out) >= config.final_top_k:
                    break
            if len(out) >= config.final_top_k:
                break
        if len(out) >= config.final_top_k:
            break

    if diagnostics is not None:
        diagnostics["bucket_survival"] = dict(survival)
        diagnostics["bucket_distinct_videos"] = {
            bucket: len(counts) for bucket, counts in bucket_videos.items()
        }
        diagnostics["tail_min_videos"] = floor
    return out