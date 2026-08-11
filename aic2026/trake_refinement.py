"""Bounded, event-local visual refinement of a chosen TRAKE sequence.

TRAKE refinement differs from Phase 5's KIS refinement in one decisive way: a sequence
has *several* semantic events, and each one must be scored against **its own** text. A
single query built from the whole sentence would tell you which frame looks like the
whole story, which is not the question.

The pipeline deliberately refines AFTER coarse alignment, never before: refining every
per-event candidate would multiply Phase 5's cost by the candidate depth. Only a few
already-complete sequences are refined, only a few of their events, and only a few frames
per event, under a hard per-query frame ceiling.

Everything reusable is reused: `LocalFrameRefiner` supplies the bounded sample plan,
`FrameProvider` the single OpenCV implementation, and `FrameScorer` the shared CLIP
checkpoint. No second model stack is created here.

Two rules survive from earlier phases:

* the submitted frame stays the coarse official mapped `frame_idx` -- a refined frame is
  evidence, a local score, and a reranking contribution, never the submission;
* nothing is invented: a missing MP4, a decode failure, or an unavailable scorer leaves
  the coarse sequence exactly as it was.

No accuracy is claimed. There is no AIC ground truth in this repository.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

from .frame_provider import FrameProvider
from .frame_scorer import FrameScorer
from .local_refinement import (
    FRAME_OUTPUT_PRESERVE_COARSE,
    MODE_ALWAYS,
    LocalFrameRefiner,
    LocalRefinementFrame,
    LocalRefinementRequest,
    RefinementCandidate,
    RefinementConfig,
)
from .trake import AlignmentConfig, TrakeAlignedStep, TrakePrediction

EVENT_REFINED = "refined"
EVENT_UNCHANGED = "unchanged"
EVENT_SKIPPED_BUDGET = "skipped_budget"
EVENT_NOT_SELECTED = "not_selected"
EVENT_UNAVAILABLE = "unavailable"

SEQUENCE_REFINED = "refined"
SEQUENCE_NOT_REQUESTED = "not_requested"
SEQUENCE_UNAVAILABLE = "unavailable"
SEQUENCE_BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class FrameBudget:
    """A hard ceiling on frames decoded for ONE TRAKE query, shared across sequences."""

    limit: int
    used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def take(self, count: int) -> int:
        allowed = min(int(count), self.remaining())
        self.used += allowed
        return allowed


@dataclass(frozen=True)
class TrakeEventRefinement:
    """What local refinement found for ONE event of ONE sequence."""

    event_index: int
    event_text: str
    video_id: str
    coarse_frame_idx: Optional[int]
    coarse_timestamp: Optional[float]
    status: str = EVENT_NOT_SELECTED
    best_visual_frame_idx: Optional[int] = None
    best_visual_timestamp: Optional[float] = None
    coarse_visual_score: Optional[float] = None
    best_visual_score: Optional[float] = None
    visual_gain: float = 0.0
    frames_sampled: int = 0
    frames_decoded: int = 0
    order_constrained: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": int(self.event_index),
            "event_text": self.event_text,
            "video_id": self.video_id,
            "status": self.status,
            "coarse_frame_idx": self.coarse_frame_idx,
            "coarse_timestamp": (
                None if self.coarse_timestamp is None else round(float(self.coarse_timestamp), 3)
            ),
            "best_visual_frame_idx": self.best_visual_frame_idx,
            "best_visual_timestamp": (
                None
                if self.best_visual_timestamp is None
                else round(float(self.best_visual_timestamp), 3)
            ),
            "coarse_visual_score": (
                None if self.coarse_visual_score is None else round(float(self.coarse_visual_score), 6)
            ),
            "best_visual_score": (
                None if self.best_visual_score is None else round(float(self.best_visual_score), 6)
            ),
            "visual_gain": round(float(self.visual_gain), 6),
            "frames_sampled": int(self.frames_sampled),
            "frames_decoded": int(self.frames_decoded),
            "order_constrained": bool(self.order_constrained),
            "warning": self.warning,
        }


@dataclass(frozen=True)
class TrakeSequenceRefinement:
    """Local visual evidence for a whole sequence, plus the score decomposition."""

    video_id: str
    status: str = SEQUENCE_NOT_REQUESTED
    events: tuple[TrakeEventRefinement, ...] = ()
    coarse_alignment_score: float = 0.0
    visual_gain_aggregate: float = 0.0
    final_sequence_score: float = 0.0
    order_violation_detected: bool = False
    order_violation_resolved: bool = False
    frames_decoded: int = 0
    frames_scored: int = 0
    events_refined: int = 0
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def applied(self) -> bool:
        return self.status == SEQUENCE_REFINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "status": self.status,
            "applied": self.applied,
            "coarse_alignment_score": round(float(self.coarse_alignment_score), 6),
            "visual_gain_aggregate": round(float(self.visual_gain_aggregate), 6),
            "final_sequence_score": round(float(self.final_sequence_score), 6),
            "order_violation_detected": bool(self.order_violation_detected),
            "order_violation_resolved": bool(self.order_violation_resolved),
            "frames_decoded": int(self.frames_decoded),
            "frames_scored": int(self.frames_scored),
            "events_refined": int(self.events_refined),
            "decode_ms": round(float(self.decode_ms), 3),
            "inference_ms": round(float(self.inference_ms), 3),
            "total_ms": round(float(self.total_ms), 3),
            "events": [item.to_dict() for item in self.events],
            "warnings": list(self.warnings),
        }


def local_ordered_refinement(
    per_event: Sequence[Sequence[LocalRefinementFrame]],
) -> tuple[int, ...] | None:
    """Choose one sampled frame per event, jointly, under non-decreasing time.

    Picking each event's independent visual maximum can produce a reversed sequence,
    which would display an impossible reading of the events. This small DP over the
    already-scored local frames picks the best ORDERED selection instead, so order
    safety falls out of the choice rather than being patched afterwards.

    It is `local_ordered_refinement`, not the global TRAKE alignment DP: it only ranges
    over the handful of frames already sampled inside each event's own window.
    """
    if not per_event or any(not frames for frames in per_event):
        return None
    best: list[list[tuple[float, int | None]]] = []
    for position, frames in enumerate(per_event):
        row: list[tuple[float, int | None]] = []
        for index, frame in enumerate(frames):
            if position == 0:
                row.append((float(frame.score), None))
                continue
            choice: tuple[float, int | None] = (float("-inf"), None)
            for previous_index, previous in enumerate(per_event[position - 1]):
                score = best[position - 1][previous_index][0]
                if score == float("-inf"):
                    continue
                if float(previous.timestamp) > float(frame.timestamp):
                    continue
                total = score + float(frame.score)
                if total > choice[0]:
                    choice = (total, previous_index)
            row.append(choice)
        best.append(row)
    final = best[-1]
    if all(score == float("-inf") for score, _ in final):
        return None
    # Deterministic: best total, then the earliest frame on a tie.
    end = min(
        range(len(final)),
        key=lambda index: (-final[index][0], float(per_event[-1][index].timestamp), index),
    )
    picks = [end]
    cursor = end
    for position in range(len(per_event) - 1, 0, -1):
        cursor = best[position][cursor][1]
        if cursor is None:
            return None
        picks.append(cursor)
    picks.reverse()
    return tuple(picks)


class TrakeSequenceRefiner:
    """Refines a few complete sequences, event by event, within a hard frame budget."""

    def __init__(
        self,
        config: AlignmentConfig,
        *,
        frame_provider: FrameProvider,
        scorer: Optional[FrameScorer] = None,
        source_video_for: Optional[Callable[[str, Optional[str]], Optional[str]]] = None,
        window_s: Optional[float] = None,
    ):
        self.config = config
        self.frame_provider = frame_provider
        self.scorer = scorer
        self._source_video_for = source_video_for
        # `refine_window_s` from the request wins over the configured window, which is
        # what makes the request parameter genuinely functional.
        self.window_s = float(
            config.refinement_window_s if window_s is None else window_s
        )
        self.refiner = LocalFrameRefiner(
            RefinementConfig(
                enabled=True,
                mode=MODE_ALWAYS,
                top_hypotheses=1,
                candidate_budget=1,
                window_before_s=self.window_s,
                window_after_s=self.window_s,
                fine_fps=float(config.refinement_fine_fps),
                max_frames=max(1, int(config.refinement_frames_per_event)),
                batch_size=max(1, int(config.refinement_batch_size)),
                frame_output_policy=FRAME_OUTPUT_PRESERVE_COARSE,
            ),
            frame_provider=frame_provider,
            scorer=scorer,
        )

    def scorer_available(self) -> bool:
        return self.scorer is not None

    def _source_video(self, video_id: str, keyframe_id: Optional[str]) -> Optional[str]:
        if self._source_video_for is None:
            return None
        try:
            return self._source_video_for(video_id, keyframe_id)
        except Exception:  # noqa: BLE001 - a lookup failure just means "unknown"
            return None

    def _refine_event(
        self, step: TrakeAlignedStep, budget: FrameBudget
    ) -> tuple[TrakeEventRefinement, tuple[LocalRefinementFrame, ...]]:
        """One event, scored against its OWN text. Never raises."""
        coarse_idx = None if step.candidate is None else int(step.candidate.frame_id)
        base = TrakeEventRefinement(
            event_index=step.event_index,
            event_text=step.event_text,
            video_id=step.video_id,
            coarse_frame_idx=coarse_idx,
            coarse_timestamp=step.timestamp,
        )
        if budget.exhausted:
            return replace(base, status=EVENT_SKIPPED_BUDGET), ()
        allowance = min(
            max(1, int(self.config.refinement_frames_per_event)), budget.remaining()
        )
        refiner = self.refiner
        if allowance < int(self.config.refinement_frames_per_event):
            refiner = LocalFrameRefiner(
                replace(self.refiner.config, max_frames=allowance),
                frame_provider=self.frame_provider,
                scorer=self.scorer,
            )
        candidate = RefinementCandidate(
            keyframe_id=str(step.keyframe_id),
            video_id=step.video_id,
            coarse_frame_idx=coarse_idx,
            timestamp=float(step.timestamp or 0.0),
            coarse_score=float(step.score),
            source_video=self._source_video(step.video_id, step.keyframe_id),
        )
        # The event's own text is the query. Using the whole sentence would ask a
        # different question entirely.
        result = refiner.refine(
            LocalRefinementRequest(query=step.event_text, candidates=(candidate,))
        )
        item = result.refinements[0] if result.refinements else None
        if item is None or not item.applied or not item.frames:
            warning = (item.warning if item is not None else None) or (
                result.warnings[0] if result.warnings else None
            )
            return (
                replace(
                    base,
                    status=EVENT_UNAVAILABLE,
                    frames_sampled=0 if item is None else item.sampled_frame_count,
                    warning=warning,
                ),
                (),
            )
        budget.take(item.frames_decoded)
        return (
            replace(
                base,
                status=EVENT_REFINED,
                coarse_visual_score=item.coarse_visual_score,
                frames_sampled=item.sampled_frame_count,
                frames_decoded=item.frames_decoded,
                warning=item.warning,
            ),
            item.frames,
        )

    def refine(self, prediction: TrakePrediction, budget: FrameBudget) -> TrakeSequenceRefinement:
        """Refine one complete sequence. Failure degrades to the coarse sequence."""
        started = time.perf_counter()
        coarse_score = float(prediction.coarse_alignment_score or prediction.score)
        if self.scorer is None:
            return TrakeSequenceRefinement(
                video_id=prediction.video_id,
                status=SEQUENCE_UNAVAILABLE,
                coarse_alignment_score=coarse_score,
                final_sequence_score=coarse_score,
                warnings=("No visual frame scorer is available; TRAKE refinement skipped.",),
            )
        limit = max(1, int(self.config.refinement_max_events_per_alignment))
        outcomes: list[TrakeEventRefinement] = []
        per_event_frames: list[tuple[LocalRefinementFrame, ...]] = []
        for position, step in enumerate(prediction.steps):
            if position >= limit:
                outcomes.append(
                    TrakeEventRefinement(
                        event_index=step.event_index,
                        event_text=step.event_text,
                        video_id=step.video_id,
                        coarse_frame_idx=None if step.candidate is None else int(step.candidate.frame_id),
                        coarse_timestamp=step.timestamp,
                        status=EVENT_NOT_SELECTED,
                    )
                )
                per_event_frames.append(())
                continue
            outcome, frames = self._refine_event(step, budget)
            outcomes.append(outcome)
            per_event_frames.append(frames)

        refined_positions = [i for i, frames in enumerate(per_event_frames) if frames]
        if not refined_positions:
            status = SEQUENCE_BUDGET_EXHAUSTED if budget.exhausted else SEQUENCE_UNAVAILABLE
            return TrakeSequenceRefinement(
                video_id=prediction.video_id,
                status=status,
                events=tuple(outcomes),
                coarse_alignment_score=coarse_score,
                final_sequence_score=coarse_score,
                frames_decoded=sum(item.frames_decoded for item in outcomes),
                total_ms=(time.perf_counter() - started) * 1000.0,
                warnings=("No event could be refined; the coarse sequence is unchanged.",),
            )

        # An event that was not refined still participates in the ordered choice, pinned
        # to its coarse frame, so the joint selection covers the whole sequence.
        pinned: list[tuple[LocalRefinementFrame, ...]] = []
        for position, step in enumerate(prediction.steps):
            frames = per_event_frames[position]
            if frames:
                pinned.append(tuple(sorted(frames, key=lambda f: (f.timestamp, f.frame_idx))))
                continue
            coarse_idx = 0 if step.candidate is None else int(step.candidate.frame_id)
            pinned.append(
                (
                    LocalRefinementFrame(
                        frame_idx=coarse_idx,
                        timestamp=float(step.timestamp or 0.0),
                        score=0.0,
                        is_coarse_frame=True,
                    ),
                )
            )

        independent = tuple(
            max(range(len(frames)), key=lambda i: (frames[i].score, -frames[i].frame_idx))
            for frames in pinned
        )
        independent_times = [pinned[i][choice].timestamp for i, choice in enumerate(independent)]
        order_violation = independent_times != sorted(independent_times)

        picks = local_ordered_refinement(pinned)
        resolved = bool(order_violation and picks is not None)
        if picks is None:
            # Nothing ordered exists: keep the coarse frames as the visual choice.
            picks = tuple(
                min(range(len(frames)), key=lambda i: (not frames[i].is_coarse_frame, i))
                for frames in pinned
            )

        gains: list[float] = []
        final_events: list[TrakeEventRefinement] = []
        frames_scored = 0
        for position, outcome in enumerate(outcomes):
            frames = pinned[position]
            chosen = frames[picks[position]]
            if outcome.status != EVENT_REFINED:
                final_events.append(outcome)
                continue
            frames_scored += len(per_event_frames[position])
            coarse_visual = outcome.coarse_visual_score
            gain = 0.0 if coarse_visual is None else float(chosen.score) - float(coarse_visual)
            gains.append(gain)
            constrained = bool(
                picks[position] != independent[position] and order_violation
            )
            final_events.append(
                replace(
                    outcome,
                    status=EVENT_REFINED if not chosen.is_coarse_frame or gain != 0.0 else EVENT_UNCHANGED,
                    best_visual_frame_idx=int(chosen.frame_idx),
                    best_visual_timestamp=float(chosen.timestamp),
                    best_visual_score=float(chosen.score),
                    visual_gain=gain,
                    order_constrained=constrained,
                )
            )

        aggregate = sum(gains) / len(gains) if gains else 0.0
        bounded = max(-1.0, min(1.0, aggregate))
        final_score = coarse_score + float(self.config.refinement_rerank_alpha) * bounded
        decode_ms = sum(0.0 for _ in final_events)
        return TrakeSequenceRefinement(
            video_id=prediction.video_id,
            status=SEQUENCE_REFINED,
            events=tuple(final_events),
            coarse_alignment_score=coarse_score,
            visual_gain_aggregate=aggregate,
            final_sequence_score=final_score,
            order_violation_detected=order_violation,
            order_violation_resolved=resolved,
            frames_decoded=sum(item.frames_decoded for item in final_events),
            frames_scored=frames_scored,
            events_refined=sum(
                1 for item in final_events if item.status in {EVENT_REFINED, EVENT_UNCHANGED}
            ),
            decode_ms=decode_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
        )


def apply_refinement(
    prediction: TrakePrediction, refinement: TrakeSequenceRefinement
) -> TrakePrediction:
    """Fold visual evidence into a prediction WITHOUT changing its submitted frames.

    `visual_frame_idx` records what local search preferred; `submission_frame_idx` stays
    the coarse official mapped frame, because AIC has not confirmed the frame-ID
    semantics of an arbitrary decoded frame.
    """
    by_event = {item.event_index: item for item in refinement.events}
    steps = tuple(
        replace(
            step,
            visual_frame_idx=(
                by_event[step.event_index].best_visual_frame_idx
                if step.event_index in by_event
                else None
            ),
        )
        for step in prediction.steps
    )
    updated = replace(
        prediction,
        steps=steps,
        coarse_alignment_score=refinement.coarse_alignment_score,
        visual_gain_aggregate=refinement.visual_gain_aggregate,
        final_sequence_score=refinement.final_sequence_score,
        refinement_status=refinement.status,
        score=refinement.final_sequence_score,
    )
    # The row is unchanged: same frames, same order, same length.
    assert updated.frame_ids == prediction.frame_ids
    return updated


__all__ = [
    "EVENT_NOT_SELECTED",
    "EVENT_REFINED",
    "EVENT_SKIPPED_BUDGET",
    "EVENT_UNAVAILABLE",
    "EVENT_UNCHANGED",
    "SEQUENCE_BUDGET_EXHAUSTED",
    "SEQUENCE_NOT_REQUESTED",
    "SEQUENCE_REFINED",
    "SEQUENCE_UNAVAILABLE",
    "FrameBudget",
    "TrakeEventRefinement",
    "TrakeSequenceRefinement",
    "TrakeSequenceRefiner",
    "apply_refinement",
    "local_ordered_refinement",
]
