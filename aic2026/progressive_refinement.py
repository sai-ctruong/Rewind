"""Stage 2/3: staged MP4 sampling under the SAME hard frame budget as the fixed plan.

The fixed sampler spends its whole budget in one shot: it decides a plan of up to
`max_frames` indices, decodes them, scores them, and stops. That is the right thing to do
when nothing is known in advance, and the wrong thing when the first eight frames already
separate a clear winner from everything else.

This sampler spends the same budget in stages:

* **Stage A** — a sparse sweep across the window. If the peak is already well separated
  and stable, stop; the remaining budget is simply not spent.
* **Stage B** — zoom around the one or two strongest peaks, scoring only frames not
  already scored.
* **Stage C** — spend whatever is left near the single unresolved peak.

Invariants that hold in every stage, and are tested:

* the coarse frame is always in stage A;
* the total number of scored frames never exceeds the hard budget;
* no frame index is decoded or scored twice within one request;
* every index stays inside the real video bounds, using that video's own fps;
* a decode or scorer failure falls back to coarse behaviour rather than failing.

Whether stopping early costs quality is unknown and unmeasurable here. What is measurable
is that it costs fewer frames, and that is all this module claims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

STOP_CONFIDENT = "confident_peak"
STOP_BUDGET = "budget_exhausted"
STOP_NO_FRAMES = "no_frames_available"
STOP_STAGES_DONE = "stages_completed"
STOP_FAILED = "decode_or_scoring_failed"


@dataclass
class StageRecord:
    name: str
    planned: int = 0
    scored: int = 0
    best_index: Optional[int] = None
    best_score: float = 0.0
    margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "planned": self.planned,
            "scored": self.scored,
            "best_index": self.best_index,
            "best_score": round(self.best_score, 6),
            "margin": round(self.margin, 6),
        }


@dataclass
class ProgressiveResult:
    """What a staged sample actually did. Structural counters only."""

    frames_planned: int = 0
    frames_scored: int = 0
    budget: int = 0
    stages: list[StageRecord] = field(default_factory=list)
    scored_by_index: dict[int, float] = field(default_factory=dict)
    stop_reason: str = ""
    best_index: Optional[int] = None
    best_score: float = 0.0

    @property
    def stages_entered(self) -> int:
        return len(self.stages)

    @property
    def budget_saved(self) -> int:
        return max(0, self.budget - self.frames_scored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "frames_planned": self.frames_planned,
            "frames_scored": self.frames_scored,
            "budget_saved": self.budget_saved,
            "stages_entered": self.stages_entered,
            "stages": [item.to_dict() for item in self.stages],
            "stop_reason": self.stop_reason,
            "best_index": self.best_index,
            "best_score": round(self.best_score, 6),
            "note": (
                "Staged sampling under the same hard frame budget as the fixed sampler. "
                "Stopping early saves frames; whether it costs quality is unknown "
                "without ground truth."
            ),
        }


def sparse_plan(anchor: int, low: int, high: int, count: int) -> list[int]:
    """`count` indices spread across [low, high], with `anchor` always included."""
    if high < low or count <= 0:
        return []
    anchor = min(max(int(anchor), low), high)
    if count == 1:
        return [anchor]
    span = high - low
    if span <= 0:
        return [anchor]
    step = span / (count - 1)
    plan = {int(round(low + index * step)) for index in range(count)}
    plan.add(anchor)
    ordered = sorted(value for value in plan if low <= value <= high)
    # Keep the anchor even when trimming to the requested count.
    while len(ordered) > count:
        removable = [value for value in ordered if value != anchor]
        if not removable:
            break
        ordered.remove(max(removable, key=lambda value: abs(value - anchor)))
    return ordered


def zoom_plan(center: int, low: int, high: int, count: int, *, step: int) -> list[int]:
    """`count` indices packed tightly around `center`, nearest first."""
    if count <= 0 or high < low:
        return []
    center = min(max(int(center), low), high)
    step = max(1, int(step))
    plan = [center]
    offset = 1
    while len(plan) < count:
        earlier, later = center - offset * step, center + offset * step
        if earlier < low and later > high:
            break
        for value in (earlier, later):
            if low <= value <= high and value not in plan and len(plan) < count:
                plan.append(value)
        offset += 1
    return sorted(plan)


def _margin(scores: Sequence[float]) -> float:
    values = sorted((float(v) for v in scores if math.isfinite(float(v))), reverse=True)
    if len(values) < 2:
        return 1.0
    best = values[0]
    return 0.0 if abs(best) < 1e-9 else max(0.0, (best - values[1]) / abs(best))


def progressive_sample(
    *,
    anchor: int,
    low: int,
    high: int,
    budget: int,
    stage_frames: Sequence[int],
    stop_margin: float,
    score_frames: Callable[[Sequence[int]], dict[int, float]],
    fps: float = 25.0,
) -> ProgressiveResult:
    """Run the staged sampler. `score_frames` decodes and scores; it is injected.

    The caller owns decoding, so this function is pure control flow and is testable
    offline with a synthetic scorer.
    """
    result = ProgressiveResult(budget=max(0, int(budget)))
    if result.budget <= 0 or high < low:
        result.stop_reason = STOP_NO_FRAMES
        return result

    seen: set[int] = set()
    stages = [max(1, int(value)) for value in stage_frames] or [result.budget]
    zoom_step = max(1, int(round(max(1.0, float(fps)) / 8.0)))

    for position, requested in enumerate(stages):
        remaining = result.budget - result.frames_scored
        if remaining <= 0:
            result.stop_reason = STOP_BUDGET
            break
        take = min(requested, remaining)
        if position == 0:
            plan = sparse_plan(anchor, low, high, take)
        else:
            # Zoom on the strongest peak found so far; stage C narrows to it alone.
            center = result.best_index if result.best_index is not None else anchor
            plan = zoom_plan(center, low, high, take, step=max(1, zoom_step // position))
        plan = [index for index in plan if index not in seen]
        if not plan:
            result.stop_reason = result.stop_reason or STOP_STAGES_DONE
            break

        record = StageRecord(name=f"stage_{chr(ord('A') + position)}", planned=len(plan))
        result.frames_planned += len(plan)
        try:
            scored = score_frames(plan)
        except Exception:  # noqa: BLE001 - a failure falls back to coarse behaviour
            result.stop_reason = STOP_FAILED
            result.stages.append(record)
            break
        for index, value in (scored or {}).items():
            index = int(index)
            if index in seen:
                continue
            seen.add(index)
            result.scored_by_index[index] = float(value)
            record.scored += 1
        result.frames_scored = len(result.scored_by_index)

        if result.scored_by_index:
            best_index = max(result.scored_by_index, key=lambda key: result.scored_by_index[key])
            result.best_index = best_index
            result.best_score = result.scored_by_index[best_index]
            record.best_index = best_index
            record.best_score = result.best_score
            record.margin = _margin(list(result.scored_by_index.values()))
        result.stages.append(record)

        if record.margin >= float(stop_margin) and record.scored:
            # Well separated already: the rest of the budget is deliberately not spent.
            result.stop_reason = STOP_CONFIDENT
            break
    else:
        result.stop_reason = result.stop_reason or STOP_STAGES_DONE

    if not result.stop_reason:
        result.stop_reason = STOP_BUDGET if result.frames_scored >= result.budget else STOP_STAGES_DONE
    return result


__all__ = [
    "STOP_BUDGET",
    "STOP_CONFIDENT",
    "STOP_FAILED",
    "STOP_NO_FRAMES",
    "STOP_STAGES_DONE",
    "ProgressiveResult",
    "StageRecord",
    "progressive_sample",
    "sparse_plan",
    "zoom_plan",
]
