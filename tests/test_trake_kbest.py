"""k-best enumeration, the exact-DP reference, and split video/alignment scoring.

Phase 7 returned one sequence per video, which on real data left most videos silent.
These tests pin down the enumeration: distinct, deterministic, duplicate-free, and still
subject to every Phase 7 structural invariant.
"""
from __future__ import annotations

import pytest

from aic2026.trake import (
    METHOD_BEAM_DP,
    AlignmentConfig,
    EventCandidate,
    TrakeEvent,
    VideoEventCandidates,
    align_trake,
    align_video_beam_dp,
    align_video_exact_dp,
    align_video_k_best_beam,
    alignment_objective,
    group_candidates,
    score_video_hypothesis,
)


def candidate(event: int, video: str, frame: int, time: float, score: float = 0.9):
    return EventCandidate(event, video, f"{video}/kf_{frame:06d}", str(frame), time, score)


def events(count: int) -> tuple[TrakeEvent, ...]:
    return tuple(TrakeEvent(i, f"event {i}") for i in range(count))


def video_with(video: str, by_event: dict[int, list[EventCandidate]]) -> VideoEventCandidates:
    return VideoEventCandidates(video, {k: tuple(v) for k, v in by_event.items()})


def two_choices(video: str = "V") -> VideoEventCandidates:
    """Three events, two well-separated choices each: 8 orderable combinations."""
    return video_with(
        video,
        {
            0: [candidate(0, video, 100, 1.0, 0.90), candidate(0, video, 110, 3.0, 0.85)],
            1: [candidate(1, video, 200, 10.0, 0.90), candidate(1, video, 210, 14.0, 0.80)],
            2: [candidate(2, video, 300, 20.0, 0.90), candidate(2, video, 310, 26.0, 0.70)],
        },
    )


# --------------------------------------------------------------------- k-best


def test_k_best_returns_several_distinct_sequences() -> None:
    variants = align_video_k_best_beam(events(3), two_choices(), AlignmentConfig(), k=4)
    assert len(variants) >= 2
    signatures = [tuple(s.submission_frame_idx for s in v.steps) for v in variants]
    assert len(signatures) == len(set(signatures)), "no duplicate sequences"
    assert all(v.is_complete for v in variants)
    assert all(len(v.steps) == 3 for v in variants)


def test_k_best_is_deterministic() -> None:
    first = align_video_k_best_beam(events(3), two_choices(), AlignmentConfig(), k=4)
    second = align_video_k_best_beam(events(3), two_choices(), AlignmentConfig(), k=4)
    assert [tuple(s.submission_frame_idx for s in v.steps) for v in first] == [
        tuple(s.submission_frame_idx for s in v.steps) for v in second
    ]
    assert [round(v.score, 9) for v in first] == [round(v.score, 9) for v in second]


def test_k_best_is_sorted_by_score() -> None:
    variants = align_video_k_best_beam(events(3), two_choices(), AlignmentConfig(), k=6)
    scores = [v.score for v in variants]
    assert scores == sorted(scores, reverse=True)


def test_the_top_k_best_variant_matches_the_single_best_search() -> None:
    single = align_video_beam_dp(events(3), two_choices(), AlignmentConfig())
    best = align_video_k_best_beam(events(3), two_choices(), AlignmentConfig(), k=4)[0]
    assert [s.submission_frame_idx for s in single.steps] == [
        s.submission_frame_idx for s in best.steps
    ]


def test_k_of_one_returns_exactly_one() -> None:
    assert len(align_video_k_best_beam(events(3), two_choices(), AlignmentConfig(), k=1)) == 1


def test_every_k_best_variant_stays_single_video_and_event_preserving() -> None:
    variants = align_video_k_best_beam(events(3), two_choices("A"), AlignmentConfig(), k=5)
    for variant in variants:
        assert {step.video_id for step in variant.steps} == {"A"}
        assert [step.event_index for step in variant.steps] == [0, 1, 2]
        times = [step.timestamp for step in variant.steps]
        assert times == sorted(times)


# ------------------------------------------------------------- exact reference


def test_exact_dp_reference_agrees_with_a_wide_beam() -> None:
    config = AlignmentConfig(beam_width=64)
    exact = align_video_exact_dp(events(3), two_choices(), config)
    assert exact is not None
    beam = align_video_beam_dp(events(3), two_choices(), config)
    assert [s.submission_frame_idx for s in beam.steps] == [
        s.submission_frame_idx for s in exact.steps
    ]
    assert beam.score == pytest.approx(exact.score, abs=1e-9)


def test_exact_dp_reference_agrees_on_a_larger_random_but_fixed_case() -> None:
    # Deterministic pseudo-data: five events, four candidates each.
    video = video_with(
        "V",
        {
            index: [
                candidate(
                    index,
                    "V",
                    1000 * index + 10 * choice,
                    5.0 * index + 1.3 * choice,
                    0.5 + 0.1 * ((index * 7 + choice * 3) % 5),
                )
                for choice in range(4)
            ]
            for index in range(5)
        },
    )
    config = AlignmentConfig(beam_width=128)
    exact = align_video_exact_dp(events(5), video, config)
    assert exact is not None
    beam = align_video_beam_dp(events(5), video, config)
    assert beam.score == pytest.approx(exact.score, abs=1e-9)


def test_exact_dp_refuses_rather_than_degrading_when_unbounded() -> None:
    big = video_with(
        "V",
        {
            index: [candidate(index, "V", index * 100 + i, float(i), 0.5) for i in range(60)]
            for index in range(6)
        },
    )
    assert align_video_exact_dp(events(6), big, AlignmentConfig(exact_dp_max_states=50)) is None


def test_the_exact_reference_is_never_the_shipped_method() -> None:
    exact = align_video_exact_dp(events(3), two_choices(), AlignmentConfig())
    assert exact is not None and exact.method == "exact_dp_reference"
    assert align_video_beam_dp(events(3), two_choices(), AlignmentConfig()).method == METHOD_BEAM_DP
    for prediction in align_trake(["a", "b", "c"], {
        index: list(two_choices().candidates_for(index)) for index in range(3)
    }, AlignmentConfig()).predictions:
        assert prediction.method == METHOD_BEAM_DP


# ------------------------------------------------------------------- scoring


def test_alignment_score_is_computed_from_the_chosen_steps() -> None:
    config = AlignmentConfig(missing_event_penalty=0.5, transition_penalty=0.0, gap_penalty=0.0)
    video = video_with(
        "V",
        {
            0: [candidate(0, "V", 100, 1.0, 0.9)],
            1: [candidate(1, "V", 200, 5.0, 0.8)],
        },
    )
    alignment = align_video_beam_dp(events(2), video, config)
    # 0.9 + 0.8 + coverage bonus, straight from the two steps actually used.
    assert alignment_objective(alignment, config) == pytest.approx(
        0.9 + 0.8 + config.coverage_bonus
    )
    assert alignment.score == pytest.approx(alignment_objective(alignment, config))


def test_pre_alignment_scoring_is_separate_from_the_final_score() -> None:
    # Video A's first candidate for event 1 is early and weak; its BEST candidate is a
    # good one later. Pre-alignment scoring sees the earliest, final scoring sees the
    # chosen step.
    video = video_with(
        "A",
        {
            0: [candidate(0, "A", 100, 1.0, 0.9)],
            1: [candidate(1, "A", 200, 2.0, 0.1), candidate(1, "A", 210, 9.0, 0.95)],
        },
    )
    config = AlignmentConfig()
    pre = score_video_hypothesis(video, 2, config)
    alignment = align_video_beam_dp(events(2), video, config)
    chosen = [step.submission_frame_idx for step in alignment.steps]
    assert chosen == ["100", "210"], "the strong later candidate should win the alignment"
    # The final score reflects the chosen 0.95 candidate, not the 0.1 earliest one.
    assert alignment.score == pytest.approx(alignment_objective(alignment, config))
    assert alignment.score > pre.event_relevance
    # And the pre-alignment score never reads another video's candidates.
    assert pre.video_id == "A"


def test_final_scores_use_actual_steps_across_videos() -> None:
    candidates = {
        0: [candidate(0, "A", 100, 1.0, 0.90), candidate(0, "B", 100, 1.0, 0.60)],
        1: [candidate(1, "A", 200, 5.0, 0.20), candidate(1, "B", 200, 5.0, 0.85)],
    }
    report = align_trake(["a", "b"], candidates, AlignmentConfig())
    scores = {p.video_id: p.score for p in report.predictions}
    # B wins on the sum of its ACTUAL steps (0.60 + 0.85) over A's (0.90 + 0.20).
    assert scores["B"] > scores["A"]
    assert report.predictions[0].video_id == "B"


def test_score_decomposition_is_exposed() -> None:
    report = align_trake(
        ["a", "b", "c"],
        {index: list(two_choices().candidates_for(index)) for index in range(3)},
        AlignmentConfig(),
    )
    prediction = report.predictions[0]
    payload = prediction.to_dict()
    assert payload["coarse_alignment_score"] == pytest.approx(payload["score"])
    assert payload["final_sequence_score"] == pytest.approx(payload["score"])
    assert payload["visual_gain_aggregate"] == 0.0
    assert payload["refinement_status"] == "not_requested"
    assert payload["rank"] >= 1


# ----------------------------------------------------- multiple per video


def test_one_video_can_contribute_several_sequences() -> None:
    config = AlignmentConfig(k_best_per_video=6, max_alignments_per_video=3)
    candidates = {index: list(two_choices().candidates_for(index)) for index in range(3)}
    report = align_trake(["a", "b", "c"], candidates, config)
    assert len(report.predictions) >= 2
    assert {p.video_id for p in report.predictions} == {"V"}
    signatures = {p.frame_ids for p in report.predictions}
    assert len(signatures) == len(report.predictions)
    assert report.diagnostics["videos_with_multiple_alignments"] == 1
    assert report.diagnostics["max_sequences_from_one_video"] >= 2


def test_per_video_maximum_is_enforced() -> None:
    config = AlignmentConfig(k_best_per_video=8, max_alignments_per_video=2)
    candidates = {index: list(two_choices().candidates_for(index)) for index in range(3)}
    report = align_trake(["a", "b", "c"], candidates, config)
    assert len([p for p in report.predictions if p.video_id == "V"]) <= 2


def test_a_video_never_starves_the_others() -> None:
    # V has many variants; W has one. W must still appear.
    candidates: dict[int, list[EventCandidate]] = {
        index: list(two_choices("V").candidates_for(index)) for index in range(3)
    }
    for index in range(3):
        candidates[index].append(candidate(index, "W", 900 + index, 2.0 * index + 1.0, 0.55))
    config = AlignmentConfig(k_best_per_video=6, max_alignments_per_video=3)
    report = align_trake(["a", "b", "c"], candidates, config)
    assert {p.video_id for p in report.predictions} >= {"V", "W"}
    # The strongest sequence still leads.
    assert report.predictions[0].video_id == "V"


def test_ranks_are_assigned_in_order() -> None:
    candidates = {index: list(two_choices().candidates_for(index)) for index in range(3)}
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig(k_best_per_video=4))
    assert [p.rank for p in report.predictions] == list(range(1, len(report.predictions) + 1))


def test_final_top_k_is_respected_by_k_best() -> None:
    candidates = {index: list(two_choices().candidates_for(index)) for index in range(3)}
    report = align_trake(
        ["a", "b", "c"],
        candidates,
        AlignmentConfig(k_best_per_video=8, max_alignments_per_video=8),
        max_results=2,
    )
    assert len(report.predictions) <= 2


def test_every_k_best_prediction_still_satisfies_phase_7_invariants() -> None:
    candidates = {index: list(two_choices().candidates_for(index)) for index in range(3)}
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig(k_best_per_video=6))
    assert report.predictions
    assert report.diagnostics["malformed_prediction_count"] == 0
    assert report.diagnostics["wrong_event_count_prediction_count"] == 0
    assert report.diagnostics["cross_video_step_count"] == 0
    assert report.diagnostics["unordered_submission_sequence_count"] == 0
    for prediction in report.predictions:
        assert len(prediction.frame_ids) == 3
        assert prediction.missing_event_indices == ()
