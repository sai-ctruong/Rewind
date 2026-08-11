"""Event-preserving TRAKE alignment with beam-pruned dynamic programming.

An official TRAKE row is `video_id, frame_id_1, ..., frame_id_N` where frame *i*
corresponds to semantic event *i*. Before Phase 7 the alignment allowed an event to be
skipped and the conversion then *dropped* the skipped position:

```python
frame_ids = tuple(a.candidate.frame_id for a in alignments if a.candidate is not None)
```

so a four-event query could emit three frames, and every event after the gap silently
shifted one place to the left. That is a malformed submission row, not a partial one.

Phase 7 makes the structure incapable of expressing that:

* a `TrakeAlignment` always has exactly one step per query event, and a missing event is
  an explicit `missing` step rather than an absent one;
* `TrakePrediction` raises `TrakeStructureError` unless it holds exactly `event_count`
  frames, all from one video;
* a deterministic recovery pass tries to fill missing events from the *same video's*
  candidates for *that event*, respecting the temporal neighbours;
* anything still incomplete is discarded rather than exported.

The search is beam-pruned dynamic programming and is named `beam_dp` everywhere. It is
not exact DP, and nothing in this repository claims it is. No accuracy is claimed: this
repository has no AIC ground truth.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

# Step status: what happened to one event position.
STEP_ALIGNED = "aligned"
STEP_RECOVERED = "recovered"
STEP_MISSING = "missing"

# Alignment status: what happened to the sequence as a whole.
ALIGNMENT_COMPLETE = "complete"
ALIGNMENT_COMPLETE_WITH_RECOVERY = "complete_with_recovery"
ALIGNMENT_INCOMPLETE = "incomplete"

# The search really is beam-pruned DP. `beam_pruned_dp` is accepted as a synonym so an
# older config keeps working; `exact_dp` is deliberately NOT a valid value.
METHOD_BEAM_DP = "beam_dp"
ALIGNMENT_METHODS = (METHOD_BEAM_DP, "beam_pruned_dp")


class TrakeStructureError(ValueError):
    """Raised when a TRAKE structure would violate the one-frame-per-event invariant."""


@dataclass(frozen=True)
class TrakeEvent:
    index: int
    text: str


@dataclass(frozen=True)
class EventCandidate:
    event_index: int
    video_id: str
    keyframe_id: str
    # The OFFICIAL mapped frame_idx, as a string. Not unique within a video: 192 official
    # videos repeat a frame_idx, so equality of frame_id is never treated as corruption.
    frame_id: str
    timestamp: float
    score: float


# Kept as an alias so existing callers keep the old name.
TrakeEventCandidate = EventCandidate


@dataclass(frozen=True)
class VideoEventCandidates:
    video_id: str
    by_event: Mapping[int, tuple[EventCandidate, ...]]

    def candidates_for(self, event_index: int) -> tuple[EventCandidate, ...]:
        return tuple(self.by_event.get(int(event_index), ()))


@dataclass(frozen=True)
class AlignmentConfig:
    alignment_method: str = METHOD_BEAM_DP
    per_event_top_k: int = 40
    top_video_hypotheses: int = 20
    alignments_per_video: int = 1
    beam_width: int = 8
    min_gap_s: float = 0.001
    max_gap_s: float | None = None
    coverage_bonus: float = 0.20
    missing_event_penalty: float = 0.35
    transition_penalty: float = 0.02
    gap_penalty: float = 0.001
    sequence_overlap_threshold: float = 0.8
    final_top_k: int = 100
    # Phase 7: try to fill an event the beam search skipped, using that event's own
    # candidates from the same video.
    recover_missing_events: bool = True
    # Whether recovery must satisfy min_gap_s / max_gap_s as well as plain ordering.
    recovery_respect_gap: bool = True


@dataclass
class AlignmentState:
    score: float
    event_index: int
    candidate: EventCandidate | None
    previous: "AlignmentState | None" = None
    matched: int = 0


@dataclass(frozen=True)
class TrakeAlignedStep:
    """One event position. It always exists, even when nothing was aligned to it."""

    event_index: int
    event_text: str
    video_id: str
    status: str = STEP_MISSING
    candidate: EventCandidate | None = None
    # Phase 5 frame-ID policy, carried into TRAKE: the official mapped frame_idx is what
    # gets submitted. `visual_frame_idx` is reserved for a future locally refined frame
    # and is always None in Phase 7, because TRAKE refinement is not implemented.
    visual_frame_idx: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {STEP_ALIGNED, STEP_RECOVERED, STEP_MISSING}:
            raise TrakeStructureError(f"Unknown TRAKE step status {self.status!r}.")
        if self.status == STEP_MISSING and self.candidate is not None:
            raise TrakeStructureError("A missing TRAKE step cannot carry a candidate.")
        if self.status != STEP_MISSING and self.candidate is None:
            raise TrakeStructureError(
                f"A {self.status!r} TRAKE step must carry a candidate."
            )
        if self.candidate is not None:
            if self.candidate.video_id != self.video_id:
                raise TrakeStructureError(
                    f"Step for event {self.event_index} holds a candidate from "
                    f"{self.candidate.video_id!r} but belongs to {self.video_id!r}."
                )
            if int(self.candidate.event_index) != int(self.event_index):
                raise TrakeStructureError(
                    f"Step for event {self.event_index} holds a candidate for event "
                    f"{self.candidate.event_index}."
                )

    @property
    def is_present(self) -> bool:
        return self.candidate is not None

    @property
    def keyframe_id(self) -> str | None:
        return None if self.candidate is None else self.candidate.keyframe_id

    @property
    def coarse_official_frame_idx(self) -> str | None:
        return None if self.candidate is None else self.candidate.frame_id

    @property
    def submission_frame_idx(self) -> str | None:
        """What goes in the official row. The coarse mapped frame, never a decoded one."""
        return self.coarse_official_frame_idx

    @property
    def timestamp(self) -> float | None:
        return None if self.candidate is None else float(self.candidate.timestamp)

    @property
    def score(self) -> float:
        return 0.0 if self.candidate is None else float(self.candidate.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": int(self.event_index),
            "event_text": self.event_text,
            "video_id": self.video_id,
            "status": self.status,
            "keyframe_id": self.keyframe_id,
            "coarse_official_frame_idx": self.coarse_official_frame_idx,
            "visual_frame_idx": self.visual_frame_idx,
            "submission_frame_idx": self.submission_frame_idx,
            "timestamp": None if self.timestamp is None else round(self.timestamp, 3),
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class TrakeAlignment:
    """A full event sequence for one video. Always one step per query event."""

    video_id: str
    events: tuple[TrakeEvent, ...]
    steps: tuple[TrakeAlignedStep, ...]
    score: float = 0.0
    method: str = METHOD_BEAM_DP
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.events):
            raise TrakeStructureError(
                f"TRAKE alignment for {self.video_id!r} has {len(self.steps)} steps for "
                f"{len(self.events)} events; every event position must be present."
            )
        for position, (event, step) in enumerate(zip(self.events, self.steps)):
            if step.event_index != position or event.index != position:
                raise TrakeStructureError(
                    f"TRAKE step {position} reports event_index {step.event_index}; "
                    "step order must match event order exactly."
                )
            if step.video_id != self.video_id:
                raise TrakeStructureError(
                    f"TRAKE alignment for {self.video_id!r} contains a step from "
                    f"{step.video_id!r}; all events must come from one video."
                )

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def present_steps(self) -> tuple[TrakeAlignedStep, ...]:
        return tuple(step for step in self.steps if step.is_present)

    @property
    def missing_event_indices(self) -> tuple[int, ...]:
        return tuple(step.event_index for step in self.steps if not step.is_present)

    @property
    def recovered_event_indices(self) -> tuple[int, ...]:
        return tuple(
            step.event_index for step in self.steps if step.status == STEP_RECOVERED
        )

    @property
    def is_complete(self) -> bool:
        return not self.missing_event_indices

    @property
    def status(self) -> str:
        if not self.is_complete:
            return ALIGNMENT_INCOMPLETE
        return (
            ALIGNMENT_COMPLETE_WITH_RECOVERY
            if self.recovered_event_indices
            else ALIGNMENT_COMPLETE
        )

    @property
    def coverage(self) -> float:
        return len(self.present_steps) / max(1, self.event_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "event_count": self.event_count,
            "status": self.status,
            "score": round(float(self.score), 6),
            "method": self.method,
            "coverage": round(self.coverage, 6),
            "missing_event_indices": list(self.missing_event_indices),
            "recovered_event_indices": list(self.recovered_event_indices),
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TrakePrediction:
    """A COMPLETE event sequence, ready to become an official row.

    Construction enforces the invariant rather than trusting callers: exactly
    `event_count` frames, exactly `event_count` present steps, all from one video.
    """

    video_id: str
    frame_ids: tuple[str, ...]
    event_count: int
    alignment_status: str
    score: float
    steps: tuple[TrakeAlignedStep, ...] = field(default_factory=tuple)
    missing_event_indices: tuple[int, ...] = field(default_factory=tuple)
    recovered_event_indices: tuple[int, ...] = field(default_factory=tuple)
    method: str = METHOD_BEAM_DP
    coverage: float = 1.0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if int(self.event_count) <= 0:
            raise TrakeStructureError("A TRAKE prediction needs at least one event.")
        if len(self.frame_ids) != int(self.event_count):
            raise TrakeStructureError(
                f"TRAKE prediction for {self.video_id!r} has {len(self.frame_ids)} frames "
                f"for {self.event_count} events; an official row needs exactly one frame "
                "per event."
            )
        if self.steps and len(self.steps) != int(self.event_count):
            raise TrakeStructureError(
                f"TRAKE prediction for {self.video_id!r} has {len(self.steps)} steps for "
                f"{self.event_count} events."
            )
        if self.missing_event_indices:
            raise TrakeStructureError(
                f"TRAKE prediction for {self.video_id!r} still misses events "
                f"{list(self.missing_event_indices)}; incomplete alignments must not "
                "become predictions."
            )
        for step in self.steps:
            if not step.is_present:
                raise TrakeStructureError(
                    f"TRAKE prediction for {self.video_id!r} contains a missing step for "
                    f"event {step.event_index}."
                )
            if step.video_id != self.video_id:
                raise TrakeStructureError(
                    f"TRAKE prediction for {self.video_id!r} contains a step from "
                    f"{step.video_id!r}."
                )

    @property
    def alignments(self) -> tuple["EventAlignment", ...]:
        """Backwards-compatible view for callers written against the old model."""
        return tuple(
            EventAlignment(TrakeEvent(step.event_index, step.event_text), step.candidate)
            for step in self.steps
        )

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(float(step.timestamp or 0.0) for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "event_count": int(self.event_count),
            "frame_ids": list(self.frame_ids),
            "alignment_status": self.alignment_status,
            "recovered_event_indices": list(self.recovered_event_indices),
            "missing_event_indices": list(self.missing_event_indices),
            "method": self.method,
            "score": round(float(self.score), 6),
            "coverage": round(float(self.coverage), 6),
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EventAlignment:
    """Legacy pair type, retained so older call sites keep compiling."""

    event: TrakeEvent
    candidate: EventCandidate | None


@dataclass(frozen=True)
class VideoHypothesis:
    video_id: str
    event_coverage: float
    event_relevance: float
    order_consistency: float
    temporal_compactness: float
    missing_event_penalty: float
    transition_penalty: float
    score: float


@dataclass(frozen=True)
class TrakeAlignmentReport:
    """Complete predictions plus the structural diagnostics behind them."""

    predictions: tuple[TrakePrediction, ...] = field(default_factory=tuple)
    alignments: tuple[TrakeAlignment, ...] = field(default_factory=tuple)
    discarded: tuple[TrakeAlignment, ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def group_candidates(
    candidates_by_event: Mapping[int, Sequence[EventCandidate]]
) -> list[VideoEventCandidates]:
    grouped: dict[str, dict[int, list[EventCandidate]]] = defaultdict(lambda: defaultdict(list))
    for event_index, candidates in candidates_by_event.items():
        for candidate in candidates:
            grouped[candidate.video_id][event_index].append(candidate)
    out = []
    for video_id, by_event in grouped.items():
        frozen = {
            idx: tuple(sorted(values, key=lambda c: (c.timestamp, -c.score, c.keyframe_id)))
            for idx, values in by_event.items()
        }
        out.append(VideoEventCandidates(video_id, frozen))
    out.sort(key=lambda item: item.video_id)
    return out


def score_video_hypothesis(
    video: VideoEventCandidates, event_count: int, config: AlignmentConfig
) -> VideoHypothesis:
    """Rank a video before aligning it.

    Every value is computed from THIS video's candidates for THAT event, so a hypothesis
    can never be scored using another video's or another event's candidate. The order and
    compactness terms use each event's earliest candidate rather than the eventually
    aligned one; that is a coarse pre-filter heuristic, and changing it would mean tuning
    against quality, which is a Phase 8 concern with no ground truth available here.
    """
    present = [idx for idx in range(event_count) if video.by_event.get(idx)]
    coverage = len(present) / max(1, event_count)
    relevance = sum(max(c.score for c in video.by_event[idx]) for idx in present) / max(1, event_count)
    missing = (event_count - len(present)) * config.missing_event_penalty
    first_times = [video.by_event[idx][0].timestamp for idx in present]
    ordered = sum(a < b for a, b in zip(first_times, first_times[1:]))
    order = ordered / max(1, len(first_times) - 1)
    span = max(first_times) - min(first_times) if len(first_times) > 1 else 0.0
    compactness = 1.0 / (1.0 + span)
    score = relevance + coverage * config.coverage_bonus + order * 0.1 + compactness * 0.05 - missing
    return VideoHypothesis(video.video_id, coverage, relevance, order, compactness, missing, 0.0, score)


def _transition(previous: EventCandidate, current: EventCandidate, config: AlignmentConfig) -> float | None:
    gap = current.timestamp - previous.timestamp
    if gap < config.min_gap_s or (config.max_gap_s is not None and gap > config.max_gap_s):
        return None
    return config.transition_penalty + config.gap_penalty * gap


def align_video_beam_dp(
    events: Sequence[TrakeEvent],
    video: VideoEventCandidates,
    config: AlignmentConfig | None = None,
) -> TrakeAlignment:
    """Beam-pruned DP over one video, producing one step per event.

    Skipping an event stays available inside the search because it is how a partial
    hypothesis is represented, but the skip now materializes as an explicit `missing`
    step rather than as an absent position.
    """
    config = config or AlignmentConfig()
    ordered_events = tuple(events)
    states: list[AlignmentState] = [AlignmentState(0.0, -1, None, None, 0)]
    for event in ordered_events:
        next_states: list[AlignmentState] = []
        for previous_state in states:
            next_states.append(
                AlignmentState(
                    previous_state.score - config.missing_event_penalty,
                    event.index,
                    None,
                    previous_state,
                    previous_state.matched,
                )
            )
            for candidate in video.candidates_for(event.index):
                last = previous_state
                while last is not None and last.candidate is None:
                    last = last.previous
                penalty = 0.0
                if last is not None and last.candidate is not None:
                    transition = _transition(last.candidate, candidate, config)
                    if transition is None:
                        continue
                    penalty = transition
                next_states.append(
                    AlignmentState(
                        previous_state.score + candidate.score - penalty,
                        event.index,
                        candidate,
                        previous_state,
                        previous_state.matched + 1,
                    )
                )
        next_states.sort(
            key=lambda s: (-s.score, -s.matched, s.candidate.timestamp if s.candidate else float("inf"))
        )
        states = next_states[
            : max(config.beam_width, 1) * max(1, len(video.candidates_for(event.index)))
        ]
    best = max(
        states,
        key=lambda s: (s.score + (s.matched / max(1, len(ordered_events))) * config.coverage_bonus, s.matched),
    )
    chosen: dict[int, EventCandidate | None] = {}
    cursor: AlignmentState | None = best
    while cursor is not None and cursor.event_index >= 0:
        chosen[cursor.event_index] = cursor.candidate
        cursor = cursor.previous
    steps = tuple(
        _step_for(event, video.video_id, chosen.get(event.index), STEP_ALIGNED)
        for event in ordered_events
    )
    coverage = best.matched / max(1, len(ordered_events))
    warnings = (
        ()
        if coverage == 1.0
        else ("One or more TRAKE events could not be aligned by the beam search.",)
    )
    return TrakeAlignment(
        video_id=video.video_id,
        events=ordered_events,
        steps=steps,
        score=best.score + coverage * config.coverage_bonus,
        method=METHOD_BEAM_DP,
        warnings=warnings,
    )


def _step_for(
    event: TrakeEvent, video_id: str, candidate: EventCandidate | None, status: str
) -> TrakeAlignedStep:
    return TrakeAlignedStep(
        event_index=event.index,
        event_text=event.text,
        video_id=video_id,
        status=STEP_MISSING if candidate is None else status,
        candidate=candidate,
    )


def _bounds_for(
    steps: Sequence[TrakeAlignedStep], index: int
) -> tuple[float | None, float | None]:
    """Timestamps of the nearest present steps before and after one event position."""
    before = next(
        (
            step.timestamp
            for step in reversed(list(steps[:index]))
            if step.is_present
        ),
        None,
    )
    after = next((step.timestamp for step in steps[index + 1:] if step.is_present), None)
    return before, after


def _recovery_candidate(
    candidates: Sequence[EventCandidate],
    *,
    lower: float | None,
    upper: float | None,
    config: AlignmentConfig,
) -> EventCandidate | None:
    """Best same-event, same-video candidate that fits between its neighbours.

    Nothing is invented: if no candidate satisfies the temporal constraints, the event
    stays missing. A neighbouring event's frame, a sentinel, frame 0, or "closest
    timestamp regardless of event" are all explicitly not options.
    """
    gap = float(config.min_gap_s) if config.recovery_respect_gap else 0.0
    maximum = config.max_gap_s if config.recovery_respect_gap else None
    viable: list[EventCandidate] = []
    for candidate in candidates:
        timestamp = float(candidate.timestamp)
        if lower is not None:
            if timestamp - lower < gap:
                continue
            if maximum is not None and timestamp - lower > float(maximum):
                continue
        if upper is not None:
            if upper - timestamp < gap:
                continue
            if maximum is not None and upper - timestamp > float(maximum):
                continue
        viable.append(candidate)
    if not viable:
        return None
    # Deterministic: best score, then earliest, then id.
    return min(viable, key=lambda c: (-float(c.score), float(c.timestamp), c.keyframe_id))


def recover_missing_events(
    alignment: TrakeAlignment,
    video: VideoEventCandidates,
    config: AlignmentConfig | None = None,
) -> TrakeAlignment:
    """Fill events the beam search skipped, using only this video's own candidates.

    Missing positions are visited in ascending order and each one re-reads its
    neighbours from the partially recovered sequence, so consecutive gaps stay ordered
    with respect to each other. A recovered step is marked `recovered`, never `aligned`.
    """
    config = config or AlignmentConfig()
    if alignment.is_complete or not config.recover_missing_events:
        return alignment
    if video.video_id != alignment.video_id:
        raise TrakeStructureError(
            f"Cannot recover events for {alignment.video_id!r} using candidates from "
            f"{video.video_id!r}."
        )
    steps = list(alignment.steps)
    recovered = 0
    for index in list(alignment.missing_event_indices):
        lower, upper = _bounds_for(steps, index)
        candidate = _recovery_candidate(
            video.candidates_for(index), lower=lower, upper=upper, config=config
        )
        if candidate is None:
            continue
        steps[index] = TrakeAlignedStep(
            event_index=index,
            event_text=alignment.events[index].text,
            video_id=alignment.video_id,
            status=STEP_RECOVERED,
            candidate=candidate,
        )
        recovered += 1
    if not recovered:
        return alignment
    warnings = list(alignment.warnings)
    warnings.append(f"{recovered} TRAKE event(s) were filled by deterministic recovery.")
    updated = TrakeAlignment(
        video_id=alignment.video_id,
        events=alignment.events,
        steps=tuple(steps),
        score=alignment.score,
        method=alignment.method,
        warnings=tuple(warnings),
    )
    if not is_temporally_ordered(updated, config):
        # A recovery that breaks the sequence is not a recovery. Keep the original.
        return alignment
    return updated


def is_temporally_ordered(alignment: TrakeAlignment, config: AlignmentConfig | None = None) -> bool:
    """Are the present steps non-decreasing in time, and gap-compliant?

    Ordering is judged on TIMESTAMPS, never on frame-ID uniqueness: 192 official videos
    repeat a `frame_idx`, so two events sharing a frame ID is legitimate data.
    """
    config = config or AlignmentConfig()
    times = [step.timestamp for step in alignment.steps if step.is_present]
    minimum = float(config.min_gap_s) if config.recovery_respect_gap else 0.0
    for previous, current in zip(times, times[1:]):
        if current < previous:
            return False
        if current - previous < minimum:
            return False
        if config.max_gap_s is not None and current - previous > float(config.max_gap_s):
            return False
    return True


def to_complete_prediction(
    alignment: TrakeAlignment, config: AlignmentConfig | None = None
) -> TrakePrediction | None:
    """Convert a COMPLETE alignment into an official-shaped prediction, else None.

    Returning None is the whole point: an incomplete sequence must never be reshaped
    into a shorter row.
    """
    if not alignment.is_complete:
        return None
    if not is_temporally_ordered(alignment, config):
        return None
    frame_ids = tuple(str(step.submission_frame_idx) for step in alignment.steps)
    return TrakePrediction(
        video_id=alignment.video_id,
        frame_ids=frame_ids,
        event_count=alignment.event_count,
        alignment_status=alignment.status,
        score=float(alignment.score),
        steps=alignment.steps,
        recovered_event_indices=alignment.recovered_event_indices,
        method=alignment.method,
        coverage=alignment.coverage,
        warnings=alignment.warnings,
    )


def align_trake(
    events: Sequence[str],
    candidates_by_event: Mapping[int, Sequence[EventCandidate]],
    config: AlignmentConfig | None = None,
    max_results: int = 100,
) -> TrakeAlignmentReport:
    """Full pipeline: rank videos, align, recover, discard incomplete, rank, report."""
    config = config or AlignmentConfig()
    parsed = tuple(TrakeEvent(index, text.strip()) for index, text in enumerate(events))
    videos = group_candidates(candidates_by_event)
    ranked = sorted(
        ((score_video_hypothesis(video, len(parsed), config), video) for video in videos),
        key=lambda pair: (-pair[0].score, pair[0].video_id),
    )[: max(1, int(config.top_video_hypotheses))]

    alignments: list[TrakeAlignment] = []
    discarded: list[TrakeAlignment] = []
    initial_missing = 0
    recovered_events = 0
    # Why a position could not be recovered, so "0 recovered" is explainable rather than
    # indistinguishable from a broken recovery path.
    without_candidates = 0
    with_rejected_candidates = 0
    for _, video in ranked:
        alignment = align_video_beam_dp(parsed, video, config)
        initial_missing += len(alignment.missing_event_indices)
        alignment = recover_missing_events(alignment, video, config)
        recovered_events += len(alignment.recovered_event_indices)
        for index in alignment.missing_event_indices:
            if video.candidates_for(index):
                with_rejected_candidates += 1
            else:
                without_candidates += 1
        alignments.append(alignment)

    predictions: list[TrakePrediction] = []
    for alignment in alignments:
        prediction = to_complete_prediction(alignment, config)
        if prediction is None:
            discarded.append(alignment)
            continue
        predictions.append(prediction)

    predictions.sort(key=lambda p: (-p.score, p.video_id, p.frame_ids))
    out: list[TrakePrediction] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for prediction in predictions:
        key = (prediction.video_id, prediction.frame_ids)
        if key in seen:
            continue
        seen.add(key)
        out.append(prediction)
        if len(out) >= max(1, min(100, int(max_results))):
            break

    diagnostics = {
        "event_count": len(parsed),
        "event_candidate_counts": {
            index: len(candidates_by_event.get(index, ())) for index in range(len(parsed))
        },
        "videos_considered": len(videos),
        "video_hypotheses_considered": len(ranked),
        "initial_complete_alignments": sum(
            1 for a in alignments if not a.missing_event_indices or a.recovered_event_indices
        ),
        "initial_missing_events": initial_missing,
        "recovered_events": recovered_events,
        "remaining_missing_events": sum(len(a.missing_event_indices) for a in alignments),
        # Breakdown of what stayed missing: the event was never retrieved for that video,
        # or its candidates existed but did not fit between the neighbours.
        "missing_without_candidates": without_candidates,
        "missing_with_rejected_candidates": with_rejected_candidates,
        "initial_incomplete_alignments": sum(1 for a in alignments if not a.is_complete),
        "discarded_incomplete_alignments": len(discarded),
        "returned_complete_predictions": len(out),
        "alignment_method": METHOD_BEAM_DP,
        "beam_width": int(config.beam_width),
        "recover_missing_events": bool(config.recover_missing_events),
        # Structural invariants. All three must be zero.
        "malformed_prediction_count": sum(
            1 for p in out if len(p.frame_ids) != p.event_count
        ),
        "wrong_event_count_prediction_count": sum(
            1 for p in out if len(p.frame_ids) != len(parsed)
        ),
        "cross_video_step_count": sum(
            1 for p in out for step in p.steps if step.video_id != p.video_id
        ),
    }
    return TrakeAlignmentReport(
        predictions=tuple(out),
        alignments=tuple(alignments),
        discarded=tuple(discarded),
        diagnostics=diagnostics,
    )


def joint_trake_alignment(
    events: Sequence[str],
    candidates_by_event: Mapping[int, Sequence[EventCandidate]],
    config: AlignmentConfig | None = None,
    max_results: int = 100,
) -> list[TrakePrediction]:
    """Complete predictions only. Incomplete sequences are discarded, never shortened."""
    return list(align_trake(events, candidates_by_event, config, max_results).predictions)


# `align_video_dp` was the pre-Phase-7 entry point. It now returns an event-preserving
# alignment; the name is kept so existing imports do not break.
align_video_dp = align_video_beam_dp


ABLATIONS = (
    "independent_search", "hard_order_filter", "greedy_alignment", "beam_dp_alignment",
    "beam_dp_alignment_recovery", "full_event_coverage_beam_dp",
)

__all__ = [
    "ABLATIONS",
    "ALIGNMENT_COMPLETE",
    "ALIGNMENT_COMPLETE_WITH_RECOVERY",
    "ALIGNMENT_INCOMPLETE",
    "ALIGNMENT_METHODS",
    "METHOD_BEAM_DP",
    "STEP_ALIGNED",
    "STEP_MISSING",
    "STEP_RECOVERED",
    "AlignmentConfig",
    "AlignmentState",
    "EventAlignment",
    "EventCandidate",
    "TrakeAlignedStep",
    "TrakeAlignment",
    "TrakeAlignmentReport",
    "TrakeEvent",
    "TrakeEventCandidate",
    "TrakePrediction",
    "TrakeStructureError",
    "VideoEventCandidates",
    "VideoHypothesis",
    "align_trake",
    "align_video_beam_dp",
    "align_video_dp",
    "group_candidates",
    "is_temporally_ordered",
    "joint_trake_alignment",
    "recover_missing_events",
    "score_video_hypothesis",
    "to_complete_prediction",
]
