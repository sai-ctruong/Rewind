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
    beam_width: int = 8
    min_gap_s: float = 0.001
    max_gap_s: float | None = None
    coverage_bonus: float = 0.20
    missing_event_penalty: float = 0.35
    transition_penalty: float = 0.02
    gap_penalty: float = 0.001
    final_top_k: int = 100
    # Phase 7: try to fill an event the beam search skipped, using that event's own
    # candidates from the same video.
    recover_missing_events: bool = True
    # Whether recovery must satisfy min_gap_s / max_gap_s as well as plain ordering.
    recovery_respect_gap: bool = True

    # --- Phase 8: k-best, diversity, adaptive depth -------------------------------
    # How many distinct sequences the search enumerates per video...
    k_best_per_video: int = 4
    # ...and how many of them may survive the diversity filter into the final list.
    max_alignments_per_video: int = 3
    # Two sequences of one video are near-identical unless at least this many event
    # positions use a different frame, AND some event moved at least this far in time.
    min_sequence_difference_events: int = 1
    min_sequence_time_distance_s: float = 1.0
    # `per_event_top_k` is the INITIAL retrieval depth. When too few videos can cover
    # every event, only the poorly covered events are re-retrieved at these depths.
    candidate_depth_expansion: tuple[int, ...] = (120, 300)
    candidate_depth_max: int = 400
    # Expansion stops once this many videos can cover every event.
    target_complete_video_hypotheses: int = 12
    # Safety bound for the exact-DP reference; it is a test oracle, not a default.
    exact_dp_max_states: int = 400_000

    # --- Phase 8: bounded local video refinement of a chosen sequence -------------
    refinement_enabled: bool = False
    refinement_top_alignment_budget: int = 3
    refinement_max_events_per_alignment: int = 4
    refinement_frames_per_event: int = 8
    refinement_fine_fps: float = 2.0
    refinement_window_s: float = 2.0
    refinement_batch_size: int = 8
    refinement_rerank_alpha: float = 0.10
    # Hard ceiling on frames decoded for one TRAKE query, whatever the other settings.
    refinement_max_frames_per_query: int = 96


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
    # Phase 8: position in the final list, which variant of its video this is, and the
    # score decomposition once local visual refinement has run.
    rank: int = 0
    sequence_variant_id: int = 0
    coarse_alignment_score: float = 0.0
    visual_gain_aggregate: float = 0.0
    final_sequence_score: float = 0.0
    refinement_status: str = "not_requested"

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
            "rank": int(self.rank),
            "sequence_variant_id": int(self.sequence_variant_id),
            "event_count": int(self.event_count),
            "frame_ids": list(self.frame_ids),
            "alignment_status": self.alignment_status,
            "recovered_event_indices": list(self.recovered_event_indices),
            "missing_event_indices": list(self.missing_event_indices),
            "method": self.method,
            "score": round(float(self.score), 6),
            "coarse_alignment_score": round(float(self.coarse_alignment_score), 6),
            "visual_gain_aggregate": round(float(self.visual_gain_aggregate), 6),
            "final_sequence_score": round(float(self.final_sequence_score), 6),
            "refinement_status": self.refinement_status,
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
    """Complete predictions plus the structural diagnostics behind them.

    `alignments` holds the COMPLETE, diversity-filtered sequences that survived; a video
    whose best variant stayed incomplete contributes to `discarded` instead. Since
    Phase 8 one video may appear in `alignments` more than once.
    """

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


def alignment_objective(alignment: TrakeAlignment, config: AlignmentConfig | None = None) -> float:
    """Recompute a sequence's score from the steps it ACTUALLY holds.

    Phase 7 carried the beam's running score, which stopped describing the sequence once
    recovery replaced a step. Ranking now reads the chosen steps, so a video is never
    ranked on evidence it did not end up using.
    """
    config = config or AlignmentConfig()
    total = 0.0
    matched = 0
    previous: EventCandidate | None = None
    for step in alignment.steps:
        if step.candidate is None:
            total -= config.missing_event_penalty
            continue
        matched += 1
        total += float(step.candidate.score)
        if previous is not None:
            transition = _transition(previous, step.candidate, config)
            # A transition the search would have refused still costs the plain penalty
            # rather than silently scoring as free.
            total -= config.transition_penalty if transition is None else transition
        previous = step.candidate
    coverage = matched / max(1, alignment.event_count)
    return total + coverage * config.coverage_bonus


def _rescored(alignment: TrakeAlignment, config: AlignmentConfig) -> TrakeAlignment:
    return TrakeAlignment(
        video_id=alignment.video_id,
        events=alignment.events,
        steps=alignment.steps,
        score=alignment_objective(alignment, config),
        method=alignment.method,
        warnings=alignment.warnings,
    )


def _paths_from_states(
    states: Sequence[AlignmentState],
    events: Sequence[TrakeEvent],
    video_id: str,
    config: AlignmentConfig,
    *,
    limit: int,
) -> list[TrakeAlignment]:
    """Reconstruct distinct event paths from the surviving search states.

    k-best comes from enumerating genuinely different histories the beam kept, not from
    re-running the search with perturbed inputs. Two states that end at the same
    candidate but chose different earlier events are different sequences and both
    survive, because each state keeps its own `previous` chain.
    """
    ordered = sorted(
        states,
        key=lambda s: (
            -(s.score + (s.matched / max(1, len(events))) * config.coverage_bonus),
            -s.matched,
            s.candidate.timestamp if s.candidate else float("inf"),
            s.candidate.keyframe_id if s.candidate else "",
        ),
    )
    out: list[TrakeAlignment] = []
    seen: set[tuple[str | None, ...]] = set()
    for state in ordered:
        chosen: dict[int, EventCandidate | None] = {}
        cursor: AlignmentState | None = state
        while cursor is not None and cursor.event_index >= 0:
            chosen[cursor.event_index] = cursor.candidate
            cursor = cursor.previous
        signature = tuple(
            None if chosen.get(event.index) is None else chosen[event.index].keyframe_id
            for event in events
        )
        if signature in seen:
            continue
        seen.add(signature)
        steps = tuple(
            _step_for(event, video_id, chosen.get(event.index), STEP_ALIGNED)
            for event in events
        )
        alignment = TrakeAlignment(
            video_id=video_id,
            events=tuple(events),
            steps=steps,
            score=0.0,
            method=METHOD_BEAM_DP,
        )
        out.append(_rescored(alignment, config))
        if len(out) >= max(1, int(limit)):
            break
    return out


def align_video_k_best_beam(
    events: Sequence[TrakeEvent],
    video: VideoEventCandidates,
    config: AlignmentConfig | None = None,
    *,
    k: int = 1,
) -> list[TrakeAlignment]:
    """Up to `k` distinct event sequences for one video, best first.

    Deterministic: identical inputs give identical output, including tie order. Every
    returned alignment is event-preserving, single-video, and free of duplicates.
    """
    config = config or AlignmentConfig()
    ordered_events = tuple(events)
    wanted = max(1, int(k))
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
            key=lambda s: (
                -s.score,
                -s.matched,
                s.candidate.timestamp if s.candidate else float("inf"),
                s.candidate.keyframe_id if s.candidate else "",
            )
        )
        # Keeping k times the usual beam is what makes several distinct completions
        # survive to the end rather than being pruned in favour of one dominant path.
        width = max(config.beam_width, 1) * wanted
        states = next_states[: width * max(1, len(video.candidates_for(event.index)))]
    return _paths_from_states(states, ordered_events, video.video_id, config, limit=wanted)


def align_video_beam_dp(
    events: Sequence[TrakeEvent],
    video: VideoEventCandidates,
    config: AlignmentConfig | None = None,
) -> TrakeAlignment:
    """Single best beam-DP sequence for one video (the Phase 7 entry point)."""
    config = config or AlignmentConfig()
    best = align_video_k_best_beam(events, video, config, k=1)
    if best:
        alignment = best[0]
        if alignment.is_complete:
            return alignment
        return TrakeAlignment(
            video_id=alignment.video_id,
            events=alignment.events,
            steps=alignment.steps,
            score=alignment.score,
            method=alignment.method,
            warnings=("One or more TRAKE events could not be aligned by the beam search.",),
        )
    return TrakeAlignment(
        video_id=video.video_id,
        events=tuple(events),
        steps=tuple(_step_for(event, video.video_id, None, STEP_ALIGNED) for event in events),
        score=0.0,
        method=METHOD_BEAM_DP,
        warnings=("No TRAKE event could be aligned for this video.",),
    )


def align_video_exact_dp(
    events: Sequence[TrakeEvent],
    video: VideoEventCandidates,
    config: AlignmentConfig | None = None,
) -> TrakeAlignment | None:
    """Exhaustive DP over the same objective. A TEST ORACLE, not the default path.

    The objective is Markovian in `(event_index, last present candidate, matched count)`,
    so the exact optimum is polynomial rather than exponential. It is still bounded by
    `exact_dp_max_states` and returns `None` when the bound would be exceeded, because a
    reference that silently degrades is worse than no reference.

    Nothing in production calls this: the shipped search is beam-pruned `beam_dp`.
    """
    config = config or AlignmentConfig()
    ordered_events = tuple(events)
    total_candidates = sum(len(video.candidates_for(e.index)) for e in ordered_events)
    estimate = len(ordered_events) * (total_candidates + 1) * (len(ordered_events) + 1)
    if estimate > max(1, int(config.exact_dp_max_states)):
        return None

    # key -> (score, chosen candidates so far)
    best: dict[tuple[str | None, int], tuple[float, tuple[EventCandidate | None, ...]]] = {
        (None, 0): (0.0, ())
    }
    lookup: dict[str, EventCandidate] = {}
    for event in ordered_events:
        nxt: dict[tuple[str | None, int], tuple[float, tuple[EventCandidate | None, ...]]] = {}

        def offer(key, value) -> None:
            current = nxt.get(key)
            if current is None or value[0] > current[0] or (
                value[0] == current[0] and value[1] < current[1]
            ):
                nxt[key] = value

        for (last_id, matched), (score, path) in best.items():
            offer((last_id, matched), (score - config.missing_event_penalty, path + (None,)))
            last = lookup.get(last_id) if last_id else None
            for candidate in video.candidates_for(event.index):
                penalty = 0.0
                if last is not None:
                    transition = _transition(last, candidate, config)
                    if transition is None:
                        continue
                    penalty = transition
                lookup[candidate.keyframe_id] = candidate
                offer(
                    (candidate.keyframe_id, matched + 1),
                    (score + float(candidate.score) - penalty, path + (candidate,)),
                )
        best = nxt
        if not best:
            return None

    def objective(item) -> float:
        (_, matched), (score, _) = item
        return score + (matched / max(1, len(ordered_events))) * config.coverage_bonus

    winner = max(best.items(), key=lambda item: (objective(item), -item[0][1]))
    (_, _), (_, path) = winner
    steps = tuple(
        _step_for(event, video.video_id, path[position], STEP_ALIGNED)
        for position, event in enumerate(ordered_events)
    )
    alignment = TrakeAlignment(
        video_id=video.video_id,
        events=ordered_events,
        steps=steps,
        score=0.0,
        method="exact_dp_reference",
    )
    return _rescored(alignment, config)


def sequence_signature(alignment: TrakeAlignment | TrakePrediction) -> tuple[str, ...]:
    steps = alignment.steps
    return tuple(str(step.submission_frame_idx) for step in steps)


def sequences_are_near_duplicates(
    first: TrakeAlignment, second: TrakeAlignment, config: AlignmentConfig
) -> bool:
    """Deterministic near-duplicate rule for two sequences of the same video.

    Two sequences are near-identical unless enough event positions changed frame AND at
    least one event moved appreciably in time. `[100, 200, 300]` versus `[100, 201, 300]`
    is one alternative; `[120, 240, 340]` is a genuinely different reading.
    """
    if first.video_id != second.video_id:
        return False
    differing = sum(
        1
        for a, b in zip(first.steps, second.steps)
        if a.submission_frame_idx != b.submission_frame_idx
    )
    if differing < max(1, int(config.min_sequence_difference_events)):
        return True
    shift = max(
        (
            abs(float(a.timestamp or 0.0) - float(b.timestamp or 0.0))
            for a, b in zip(first.steps, second.steps)
        ),
        default=0.0,
    )
    return shift < float(config.min_sequence_time_distance_s)


def select_diverse_alignments(
    alignments: Sequence[TrakeAlignment], config: AlignmentConfig
) -> list[TrakeAlignment]:
    """Keep the strongest sequences of one video, dropping near-identical variants."""
    kept: list[TrakeAlignment] = []
    for alignment in sorted(
        alignments, key=lambda a: (-a.score, sequence_signature(a))
    ):
        if any(sequences_are_near_duplicates(existing, alignment, config) for existing in kept):
            continue
        kept.append(alignment)
        if len(kept) >= max(1, int(config.max_alignments_per_video)):
            break
    return kept


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
    score = float(alignment.score)
    return TrakePrediction(
        video_id=alignment.video_id,
        frame_ids=frame_ids,
        event_count=alignment.event_count,
        alignment_status=alignment.status,
        score=score,
        steps=alignment.steps,
        recovered_event_indices=alignment.recovered_event_indices,
        method=alignment.method,
        coverage=alignment.coverage,
        warnings=alignment.warnings,
        # Until refinement runs, the final score IS the coarse alignment score.
        coarse_alignment_score=score,
        final_sequence_score=score,
    )


def _empty_alignment(
    events: Sequence[TrakeEvent], video: VideoEventCandidates
) -> TrakeAlignment:
    return TrakeAlignment(
        video_id=video.video_id,
        events=tuple(events),
        steps=tuple(_step_for(event, video.video_id, None, STEP_ALIGNED) for event in events),
        score=0.0,
        method=METHOD_BEAM_DP,
        warnings=("No TRAKE event could be aligned for this video.",),
    )


def _select_final_sequences(
    predictions: Sequence[TrakePrediction],
    config: AlignmentConfig,
    *,
    max_results: int,
) -> list[TrakePrediction]:
    """Fill the final list by score, balancing video coverage against sequence variety.

    One strong video may contribute several readings, but not fill the list with them:
    a first pass takes the best sequence of each video in score order, and later passes
    admit further sequences per video up to `max_alignments_per_video`. Deterministic
    throughout, and never more than `final_top_k`.
    """
    limit = max(1, min(100, int(max_results), int(config.final_top_k)))
    ordered = sorted(
        predictions, key=lambda p: (-float(p.score), p.video_id, p.frame_ids)
    )
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[TrakePrediction] = []
    for prediction in ordered:
        key = (prediction.video_id, prediction.frame_ids)
        if key in seen:
            continue
        seen.add(key)
        unique.append(prediction)

    out: list[TrakePrediction] = []
    per_video: dict[str, int] = {}
    for allowance in range(1, max(1, int(config.max_alignments_per_video)) + 1):
        for prediction in unique:
            if len(out) >= limit:
                break
            if prediction in out:
                continue
            if per_video.get(prediction.video_id, 0) >= allowance:
                continue
            per_video[prediction.video_id] = per_video.get(prediction.video_id, 0) + 1
            out.append(prediction)
        if len(out) >= limit:
            break
    return [
        replace(prediction, rank=position)
        for position, prediction in enumerate(out[:limit], start=1)
    ]


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
    generated = 0
    duplicates_removed = 0
    # Why a position could not be recovered, so "0 recovered" is explainable rather than
    # indistinguishable from a broken recovery path.
    without_candidates = 0
    with_rejected_candidates = 0
    # Counted over each video's BEST variant, so they describe videos rather than the
    # k-times-larger variant population.
    initial_incomplete = 0
    remaining_missing = 0
    videos_with_full_coverage = 0
    for _, video in ranked:
        variants = align_video_k_best_beam(
            parsed, video, config, k=max(1, int(config.k_best_per_video))
        )
        generated += len(variants)
        repaired: list[TrakeAlignment] = []
        for position, variant in enumerate(variants):
            if position == 0:
                initial_missing += len(variant.missing_event_indices)
            recovered = recover_missing_events(variant, video, config)
            if recovered is not variant:
                recovered = _rescored(recovered, config)
            if position == 0:
                recovered_events += len(recovered.recovered_event_indices)
                remaining_missing += len(recovered.missing_event_indices)
                if not recovered.is_complete:
                    initial_incomplete += 1
                for index in recovered.missing_event_indices:
                    if video.candidates_for(index):
                        with_rejected_candidates += 1
                    else:
                        without_candidates += 1
            repaired.append(recovered)
        complete = [item for item in repaired if item.is_complete]
        if complete:
            videos_with_full_coverage += 1
        if not complete:
            # Report the best variant so the diagnostics can explain the discard.
            discarded.append(repaired[0] if repaired else _empty_alignment(parsed, video))
            continue
        kept = select_diverse_alignments(complete, config)
        duplicates_removed += len(complete) - len(kept)
        alignments.extend(kept)

    predictions: list[TrakePrediction] = []
    for alignment in alignments:
        prediction = to_complete_prediction(alignment, config)
        if prediction is None:
            discarded.append(alignment)
            continue
        predictions.append(prediction)

    out = _select_final_sequences(predictions, config, max_results=max_results)
    per_video: dict[str, int] = {}
    for prediction in out:
        per_video[prediction.video_id] = per_video.get(prediction.video_id, 0) + 1

    diagnostics = {
        "event_count": len(parsed),
        "event_candidate_counts": {
            index: len(candidates_by_event.get(index, ())) for index in range(len(parsed))
        },
        "videos_considered": len(videos),
        "video_hypotheses_considered": len(ranked),
        "initial_complete_alignments": videos_with_full_coverage,
        "videos_with_full_event_coverage": videos_with_full_coverage,
        "initial_missing_events": initial_missing,
        "recovered_events": recovered_events,
        "remaining_missing_events": remaining_missing,
        # Breakdown of what stayed missing: the event was never retrieved for that video,
        # or its candidates existed but did not fit between the neighbours.
        "missing_without_candidates": without_candidates,
        "missing_with_rejected_candidates": with_rejected_candidates,
        "initial_incomplete_alignments": initial_incomplete,
        "discarded_incomplete_alignments": len(discarded),
        "returned_complete_predictions": len(out),
        "alignment_method": METHOD_BEAM_DP,
        "beam_width": int(config.beam_width),
        "recover_missing_events": bool(config.recover_missing_events),
        "k_best_per_video": int(config.k_best_per_video),
        "k_best_alignments_generated": generated,
        "unique_alignments": len(alignments),
        "sequence_duplicates_removed": duplicates_removed,
        "complete_sequences_generated": len(predictions),
        "unique_sequences_generated": len({(p.video_id, p.frame_ids) for p in predictions}),
        "videos_with_multiple_alignments": sum(1 for count in per_video.values() if count > 1),
        "max_sequences_from_one_video": max(per_video.values(), default=0),
        "unordered_submission_sequence_count": sum(
            1
            for prediction in out
            if list(prediction.timestamps) != sorted(prediction.timestamps)
        ),
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
    "align_video_exact_dp",
    "align_video_k_best_beam",
    "alignment_objective",
    "group_candidates",
    "select_diverse_alignments",
    "sequence_signature",
    "sequences_are_near_duplicates",
    "is_temporally_ordered",
    "joint_trake_alignment",
    "recover_missing_events",
    "score_video_hypothesis",
    "to_complete_prediction",
]
