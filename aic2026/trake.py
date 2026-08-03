"""Joint event-aware TRAKE alignment with monotonic dynamic programming."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class TrakeEvent:
    index: int
    text: str


@dataclass(frozen=True)
class EventCandidate:
    event_index: int
    video_id: str
    keyframe_id: str
    frame_id: str
    timestamp: float
    score: float


@dataclass(frozen=True)
class VideoEventCandidates:
    video_id: str
    by_event: Mapping[int, tuple[EventCandidate, ...]]


@dataclass(frozen=True)
class AlignmentConfig:
    min_gap_s: float = 0.001
    max_gap_s: float | None = None
    coverage_bonus: float = 0.20
    missing_event_penalty: float = 0.35
    transition_penalty: float = 0.02
    gap_penalty: float = 0.001
    beam_width: int = 8
    top_video_hypotheses: int = 20


@dataclass
class AlignmentState:
    score: float
    event_index: int
    candidate: EventCandidate | None
    previous: "AlignmentState | None" = None
    matched: int = 0


@dataclass(frozen=True)
class EventAlignment:
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
class TrakePrediction:
    video_id: str
    alignments: tuple[EventAlignment, ...]
    score: float
    coverage: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(a.candidate.frame_id for a in self.alignments if a.candidate is not None)


def group_candidates(candidates_by_event: Mapping[int, Sequence[EventCandidate]]) -> list[VideoEventCandidates]:
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
    return out


def score_video_hypothesis(video: VideoEventCandidates, event_count: int, config: AlignmentConfig) -> VideoHypothesis:
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


def align_video_dp(events: Sequence[TrakeEvent], video: VideoEventCandidates, config: AlignmentConfig = AlignmentConfig()) -> TrakePrediction:
    # Each event keeps a bounded list of best paths ending at each candidate (or a missing-event state).
    states: list[AlignmentState] = [AlignmentState(0.0, -1, None, None, 0)]
    for event in events:
        next_states: list[AlignmentState] = []
        for previous_state in states:
            skip = AlignmentState(
                previous_state.score - config.missing_event_penalty,
                event.index,
                None,
                previous_state,
                previous_state.matched,
            )
            next_states.append(skip)
            for candidate in video.by_event.get(event.index, ()):
                last = previous_state
                while last is not None and last.candidate is None:
                    last = last.previous
                penalty = 0.0
                if last is not None and last.candidate is not None:
                    transition = _transition(last.candidate, candidate, config)
                    if transition is None:
                        continue
                    penalty = transition
                next_states.append(AlignmentState(
                    previous_state.score + candidate.score - penalty,
                    event.index,
                    candidate,
                    previous_state,
                    previous_state.matched + 1,
                ))
        next_states.sort(key=lambda s: (-s.score, -s.matched, s.candidate.timestamp if s.candidate else float("inf")))
        states = next_states[: max(config.beam_width, 1) * max(1, len(video.by_event.get(event.index, ())))]
    best = max(states, key=lambda s: (s.score + (s.matched / max(1, len(events))) * config.coverage_bonus, s.matched))
    path: list[EventAlignment] = []
    cursor: AlignmentState | None = best
    while cursor is not None and cursor.event_index >= 0:
        path.append(EventAlignment(events[cursor.event_index], cursor.candidate))
        cursor = cursor.previous
    path.reverse()
    coverage = best.matched / max(1, len(events))
    final_score = best.score + coverage * config.coverage_bonus
    warnings = () if coverage == 1.0 else ("One or more TRAKE events could not be aligned.",)
    return TrakePrediction(video.video_id, tuple(path), final_score, coverage, warnings)


def joint_trake_alignment(
    events: Sequence[str],
    candidates_by_event: Mapping[int, Sequence[EventCandidate]],
    config: AlignmentConfig = AlignmentConfig(),
    max_results: int = 100,
) -> list[TrakePrediction]:
    parsed = tuple(TrakeEvent(i, text.strip()) for i, text in enumerate(events))
    videos = group_candidates(candidates_by_event)
    hypotheses = sorted(
        ((score_video_hypothesis(video, len(parsed), config), video) for video in videos),
        key=lambda pair: (-pair[0].score, pair[0].video_id),
    )[: config.top_video_hypotheses]
    predictions = [align_video_dp(parsed, video, config) for _, video in hypotheses]
    predictions.sort(key=lambda p: (-p.score, -p.coverage, p.video_id, p.frame_ids))
    out: list[TrakePrediction] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for prediction in predictions:
        key = (prediction.video_id, prediction.frame_ids)
        if key not in seen and prediction.frame_ids:
            seen.add(key)
            out.append(prediction)
        if len(out) >= min(100, max_results):
            break
    return out


ABLATIONS = (
    "independent_search", "hard_order_filter", "greedy_alignment", "dp_alignment",
    "dp_alignment_refine", "full_event_coverage_dp_refine",
)
