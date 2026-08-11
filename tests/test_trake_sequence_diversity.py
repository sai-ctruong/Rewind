"""Sequence diversity: several readings of a video, but not the same one repeated.

Top-100 is worth little if it holds twenty sequences that differ by one frame index.
These tests fix the deterministic near-duplicate rule and the final selection order.
"""
from __future__ import annotations

import pytest

from aic2026.config import ConfigError, app_config_from_dict
from aic2026.trake import (
    ALIGNMENT_COMPLETE,
    STEP_ALIGNED,
    AlignmentConfig,
    EventCandidate,
    TrakeAlignedStep,
    TrakeAlignment,
    TrakeEvent,
    align_trake,
    select_diverse_alignments,
    sequence_signature,
    sequences_are_near_duplicates,
)


def candidate(event: int, video: str, frame: int, time: float, score: float = 0.9):
    return EventCandidate(event, video, f"{video}/kf_{frame:06d}", str(frame), time, score)


def events(count: int) -> tuple[TrakeEvent, ...]:
    return tuple(TrakeEvent(i, f"event {i}") for i in range(count))


def alignment(video: str, frames: list[tuple[int, float]], score: float = 1.0) -> TrakeAlignment:
    steps = tuple(
        TrakeAlignedStep(
            event_index=index,
            event_text=f"event {index}",
            video_id=video,
            status=STEP_ALIGNED,
            candidate=candidate(index, video, frame, time),
        )
        for index, (frame, time) in enumerate(frames)
    )
    return TrakeAlignment(
        video_id=video, events=events(len(frames)), steps=steps, score=score
    )


# ------------------------------------------------------------ near-duplicates


def test_an_exact_duplicate_is_a_near_duplicate() -> None:
    first = alignment("V", [(100, 1.0), (200, 10.0), (300, 20.0)])
    second = alignment("V", [(100, 1.0), (200, 10.0), (300, 20.0)])
    assert sequences_are_near_duplicates(first, second, AlignmentConfig()) is True


def test_a_one_frame_nudge_is_a_near_duplicate() -> None:
    # [100, 200, 300] vs [100, 201, 300]: one event changed frame, by 0.1s.
    first = alignment("V", [(100, 1.0), (200, 10.0), (300, 20.0)])
    second = alignment("V", [(100, 1.0), (201, 10.1), (300, 20.0)])
    assert sequences_are_near_duplicates(first, second, AlignmentConfig()) is True


def test_a_genuinely_different_reading_is_kept() -> None:
    # [120, 240, 340] moves every event by seconds.
    first = alignment("V", [(100, 1.0), (200, 10.0), (300, 20.0)])
    second = alignment("V", [(120, 5.0), (240, 14.0), (340, 25.0)])
    assert sequences_are_near_duplicates(first, second, AlignmentConfig()) is False


def test_sequences_of_different_videos_are_never_near_duplicates() -> None:
    first = alignment("V", [(100, 1.0), (200, 10.0)])
    second = alignment("W", [(100, 1.0), (200, 10.0)])
    assert sequences_are_near_duplicates(first, second, AlignmentConfig()) is False


def test_the_time_threshold_is_configurable_and_deterministic() -> None:
    first = alignment("V", [(100, 1.0), (200, 10.0)])
    second = alignment("V", [(100, 1.0), (210, 11.5)])
    assert sequences_are_near_duplicates(first, second, AlignmentConfig(min_sequence_time_distance_s=3.0)) is True
    assert sequences_are_near_duplicates(first, second, AlignmentConfig(min_sequence_time_distance_s=1.0)) is False


def test_requiring_more_differing_events_suppresses_single_event_variants() -> None:
    first = alignment("V", [(100, 1.0), (200, 10.0), (300, 20.0)])
    second = alignment("V", [(100, 1.0), (250, 15.0), (300, 20.0)])
    lenient = AlignmentConfig(min_sequence_difference_events=1, min_sequence_time_distance_s=1.0)
    strict = AlignmentConfig(min_sequence_difference_events=2, min_sequence_time_distance_s=1.0)
    assert sequences_are_near_duplicates(first, second, lenient) is False
    assert sequences_are_near_duplicates(first, second, strict) is True


# ----------------------------------------------------------------- selection


def test_selection_keeps_the_strongest_and_drops_near_identical_variants() -> None:
    strong = alignment("V", [(100, 1.0), (200, 10.0), (300, 20.0)], score=2.0)
    nudge = alignment("V", [(100, 1.0), (201, 10.1), (300, 20.0)], score=1.9)
    distinct = alignment("V", [(120, 5.0), (240, 14.0), (340, 25.0)], score=1.5)
    kept = select_diverse_alignments([nudge, strong, distinct], AlignmentConfig())
    assert [round(item.score, 3) for item in kept] == [2.0, 1.5]
    assert sequence_signature(kept[0]) == ("100", "200", "300")


def test_selection_respects_the_per_video_cap() -> None:
    variants = [
        alignment("V", [(100 + 50 * i, 1.0 + 6.0 * i), (400 + 50 * i, 30.0 + 6.0 * i)], score=2.0 - 0.1 * i)
        for i in range(5)
    ]
    kept = select_diverse_alignments(variants, AlignmentConfig(max_alignments_per_video=2))
    assert len(kept) == 2
    assert kept[0].score >= kept[1].score


def test_selection_is_deterministic_under_ties() -> None:
    first = alignment("V", [(100, 1.0), (200, 10.0)], score=1.0)
    second = alignment("V", [(300, 30.0), (400, 40.0)], score=1.0)
    a = select_diverse_alignments([first, second], AlignmentConfig(max_alignments_per_video=2))
    b = select_diverse_alignments([second, first], AlignmentConfig(max_alignments_per_video=2))
    assert [sequence_signature(x) for x in a] == [sequence_signature(x) for x in b]


# ---------------------------------------------------------------- end to end


def spread_video(video: str) -> dict[int, list[EventCandidate]]:
    """Two well-separated choices per event, so several readings genuinely differ."""
    return {
        0: [candidate(0, video, 100, 1.0, 0.90), candidate(0, video, 150, 6.0, 0.86)],
        1: [candidate(1, video, 200, 12.0, 0.90), candidate(1, video, 250, 18.0, 0.84)],
        2: [candidate(2, video, 300, 24.0, 0.90), candidate(2, video, 350, 31.0, 0.82)],
    }


def test_duplicate_sequences_never_reach_the_output() -> None:
    candidates = spread_video("V")
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig(k_best_per_video=8))
    keys = [(p.video_id, p.frame_ids) for p in report.predictions]
    assert len(keys) == len(set(keys))
    assert report.diagnostics["unique_sequences_generated"] == report.diagnostics[
        "complete_sequences_generated"
    ]


def test_diversity_counters_are_reported() -> None:
    report = align_trake(
        ["a", "b", "c"], spread_video("V"), AlignmentConfig(k_best_per_video=8, max_alignments_per_video=2)
    )
    diagnostics = report.diagnostics
    assert diagnostics["k_best_alignments_generated"] >= diagnostics["unique_alignments"]
    assert diagnostics["sequence_duplicates_removed"] >= 0
    assert diagnostics["max_sequences_from_one_video"] <= 2
    assert diagnostics["k_best_per_video"] == 8


def test_final_selection_spreads_across_videos_before_deepening_one() -> None:
    candidates: dict[int, list[EventCandidate]] = {}
    for index in range(3):
        candidates[index] = list(spread_video("V")[index]) + list(spread_video("W")[index])
    # W's candidates are identical in score, so both videos can produce sequences.
    report = align_trake(
        ["a", "b", "c"],
        candidates,
        AlignmentConfig(k_best_per_video=6, max_alignments_per_video=3),
        max_results=2,
    )
    assert len(report.predictions) == 2
    # The first pass takes one sequence per video before a second from either.
    assert len({p.video_id for p in report.predictions}) == 2


def test_config_rejects_an_impossible_diversity_setup() -> None:
    def build(**trake):
        return app_config_from_dict({"aic2026": {"trake": trake}})

    assert build(k_best_per_video=5, max_alignments_per_video=3).trake.k_best_per_video == 5
    with pytest.raises(ConfigError, match="k_best_per_video must be >="):
        build(k_best_per_video=2, max_alignments_per_video=5)
    with pytest.raises(ConfigError, match="min_difference_events"):
        build(sequence_diversity={"difference_events": 0})
    with pytest.raises(ConfigError, match="min_time_distance_s"):
        build(sequence_diversity={"time_distance_s": -1.0})


def test_nested_diversity_block_is_flattened() -> None:
    config = app_config_from_dict(
        {
            "aic2026": {
                "trake": {"sequence_diversity": {"difference_events": 2, "time_distance_s": 4.0}}
            }
        }
    )
    assert config.trake.min_sequence_difference_events == 2
    assert config.trake.min_sequence_time_distance_s == 4.0


def test_shipped_trake_defaults() -> None:
    from aic2026.config import load_app_config

    config = load_app_config("configs/settings.yaml")
    assert config.trake.k_best_per_video == 4
    assert config.trake.max_alignments_per_video == 3
    assert config.trake.candidate_depth_expansion == (120, 300)
    assert config.trake.candidate_depth_max == 400
    assert config.trake.refinement_enabled is False
    assert config.trake.refinement_max_frames_per_query == 96
