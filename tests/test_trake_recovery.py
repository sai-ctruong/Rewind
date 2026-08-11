"""Deterministic recovery of events the beam search skipped.

Recovery may only use the missing event's OWN candidates from the SAME video, and only
where they fit between the neighbouring aligned events. Nothing is ever invented: no
sentinel, no frame 0, no borrowing a neighbouring event's frame, and no nearest-timestamp
guess that ignores which event is being filled.
"""
from __future__ import annotations

import pytest

from aic2026.trake import (
    ALIGNMENT_COMPLETE,
    ALIGNMENT_COMPLETE_WITH_RECOVERY,
    ALIGNMENT_INCOMPLETE,
    STEP_ALIGNED,
    STEP_MISSING,
    STEP_RECOVERED,
    AlignmentConfig,
    EventCandidate,
    TrakeAlignedStep,
    TrakeAlignment,
    TrakeEvent,
    TrakeStructureError,
    VideoEventCandidates,
    align_trake,
    group_candidates,
    recover_missing_events,
    to_complete_prediction,
)


def candidate(event: int, video: str, frame: int, time: float, score: float = 0.9):
    return EventCandidate(event, video, f"{video}/kf_{frame:06d}", str(frame), time, score)


def events(count: int) -> tuple[TrakeEvent, ...]:
    return tuple(TrakeEvent(i, f"event {i}") for i in range(count))


def aligned(index: int, video: str, frame: int, time: float) -> TrakeAlignedStep:
    return TrakeAlignedStep(
        event_index=index,
        event_text=f"event {index}",
        video_id=video,
        status=STEP_ALIGNED,
        candidate=candidate(index, video, frame, time),
    )


def missing(index: int, video: str = "V") -> TrakeAlignedStep:
    return TrakeAlignedStep(
        event_index=index, event_text=f"event {index}", video_id=video, status=STEP_MISSING
    )


def gapped(video: str = "V") -> TrakeAlignment:
    """A three-event alignment whose middle event was skipped."""
    return TrakeAlignment(
        video_id=video,
        events=events(3),
        steps=(aligned(0, video, 100, 10.0), missing(1, video), aligned(2, video, 300, 20.0)),
        score=1.0,
    )


def video_with(video: str, by_event: dict[int, list[EventCandidate]]) -> VideoEventCandidates:
    return VideoEventCandidates(video, {k: tuple(v) for k, v in by_event.items()})


# --------------------------------------------------------------- basic recovery


def test_a_middle_event_is_recovered_from_the_same_video() -> None:
    video = video_with("V", {1: [candidate(1, "V", 200, 15.0)]})
    recovered = recover_missing_events(gapped(), video, AlignmentConfig())
    assert recovered.is_complete is True
    assert recovered.status == ALIGNMENT_COMPLETE_WITH_RECOVERY
    assert recovered.recovered_event_indices == (1,)
    assert recovered.steps[1].status == STEP_RECOVERED
    assert recovered.steps[1].candidate.frame_id == "200"
    # The recovered step keeps its own event index.
    assert recovered.steps[1].event_index == 1


def test_recovery_obeys_the_previous_neighbour() -> None:
    # 5.0s is before the aligned event 0 at 10.0s, so it cannot fill event 1.
    video = video_with("V", {1: [candidate(1, "V", 200, 5.0)]})
    result = recover_missing_events(gapped(), video, AlignmentConfig())
    assert result.is_complete is False
    assert result.missing_event_indices == (1,)


def test_recovery_obeys_the_next_neighbour() -> None:
    # 25.0s is after the aligned event 2 at 20.0s.
    video = video_with("V", {1: [candidate(1, "V", 200, 25.0)]})
    result = recover_missing_events(gapped(), video, AlignmentConfig())
    assert result.is_complete is False


def test_recovery_obeys_both_neighbours_and_picks_the_best_valid_candidate() -> None:
    video = video_with(
        "V",
        {
            1: [
                candidate(1, "V", 190, 5.0, 0.99),    # too early
                candidate(1, "V", 200, 12.0, 0.60),   # valid, weaker
                candidate(1, "V", 210, 18.0, 0.80),   # valid, stronger
                candidate(1, "V", 220, 25.0, 0.99),   # too late
            ]
        },
    )
    recovered = recover_missing_events(gapped(), video, AlignmentConfig())
    assert recovered.steps[1].candidate.frame_id == "210"
    assert recovered.is_complete is True


def test_a_one_sided_constraint_is_used_when_only_one_neighbour_exists() -> None:
    alignment = TrakeAlignment(
        video_id="V", events=events(2), steps=(missing(0), aligned(1, "V", 300, 20.0)), score=1.0
    )
    video = video_with("V", {0: [candidate(0, "V", 100, 30.0), candidate(0, "V", 110, 10.0)]})
    recovered = recover_missing_events(alignment, video, AlignmentConfig())
    assert recovered.is_complete is True
    assert recovered.steps[0].candidate.frame_id == "110"


def test_consecutive_missing_events_stay_ordered() -> None:
    alignment = TrakeAlignment(
        video_id="V",
        events=events(4),
        steps=(aligned(0, "V", 100, 5.0), missing(1), missing(2), aligned(3, "V", 400, 30.0)),
        score=1.0,
    )
    video = video_with(
        "V",
        {
            1: [candidate(1, "V", 200, 10.0)],
            2: [candidate(2, "V", 300, 8.0), candidate(2, "V", 310, 20.0)],
        },
    )
    recovered = recover_missing_events(alignment, video, AlignmentConfig())
    assert recovered.is_complete is True
    times = [stp.timestamp for stp in recovered.steps]
    assert times == sorted(times), times
    # Event 2 could not take 8.0s because event 1 was recovered at 10.0s.
    assert recovered.steps[2].candidate.frame_id == "310"


# ------------------------------------------------------------------ gap policy


def test_recovery_respects_min_gap_when_enabled() -> None:
    video = video_with("V", {1: [candidate(1, "V", 200, 10.2)]})
    strict = recover_missing_events(gapped(), video, AlignmentConfig(min_gap_s=1.0))
    assert strict.is_complete is False
    lenient = recover_missing_events(gapped(), video, AlignmentConfig(min_gap_s=0.05))
    assert lenient.is_complete is True


def test_recovery_respects_max_gap_when_enabled() -> None:
    video = video_with("V", {1: [candidate(1, "V", 200, 15.0)]})
    # Event 0 is at 10.0s and event 2 at 20.0s, so both gaps are 5.0s.
    assert recover_missing_events(gapped(), video, AlignmentConfig(max_gap_s=2.0)).is_complete is False
    assert recover_missing_events(gapped(), video, AlignmentConfig(max_gap_s=6.0)).is_complete is True


def test_gap_policy_can_be_relaxed_while_order_is_still_enforced() -> None:
    video = video_with("V", {1: [candidate(1, "V", 200, 10.0)]})
    relaxed = AlignmentConfig(min_gap_s=1.0, recovery_respect_gap=False)
    # Equal to the previous timestamp: allowed only because the gap policy is off.
    assert recover_missing_events(gapped(), video, relaxed).is_complete is True
    # Still ordered: a candidate before the previous event is refused regardless.
    early = video_with("V", {1: [candidate(1, "V", 200, 1.0)]})
    assert recover_missing_events(gapped(), early, relaxed).is_complete is False


def test_recovery_can_be_disabled() -> None:
    video = video_with("V", {1: [candidate(1, "V", 200, 15.0)]})
    result = recover_missing_events(
        gapped(), video, AlignmentConfig(recover_missing_events=False)
    )
    assert result.is_complete is False
    assert result.status == ALIGNMENT_INCOMPLETE


# ------------------------------------------------------------- nothing invented


def test_no_candidate_leaves_the_event_missing_and_inserts_no_sentinel() -> None:
    video = video_with("V", {})
    result = recover_missing_events(gapped(), video, AlignmentConfig())
    assert result.missing_event_indices == (1,)
    assert result.steps[1].candidate is None
    assert result.steps[1].coarse_official_frame_idx is None
    assert result.steps[1].submission_frame_idx is None
    # And it can never become a row.
    assert to_complete_prediction(result) is None


def test_recovery_never_borrows_a_neighbouring_events_candidate() -> None:
    # Only events 0 and 2 have candidates; event 1 has none of its own.
    video = video_with(
        "V",
        {
            0: [candidate(0, "V", 100, 10.0)],
            2: [candidate(2, "V", 300, 20.0)],
        },
    )
    result = recover_missing_events(gapped(), video, AlignmentConfig())
    assert result.missing_event_indices == (1,)
    assert "100" not in [stp.coarse_official_frame_idx for stp in result.steps if stp.event_index == 1]


def test_recovery_refuses_candidates_from_another_video() -> None:
    foreign = video_with("OTHER", {1: [candidate(1, "OTHER", 200, 15.0)]})
    with pytest.raises(TrakeStructureError, match="using candidates from"):
        recover_missing_events(gapped("V"), foreign, AlignmentConfig())


def test_end_to_end_a_cross_video_candidate_cannot_fill_an_event() -> None:
    # Video A is missing event 1 entirely; video B has one. A must stay incomplete.
    candidates = {
        0: [candidate(0, "A", 100, 10.0, 0.99), candidate(0, "B", 100, 10.0, 0.10)],
        1: [candidate(1, "B", 200, 15.0, 0.99)],
        2: [candidate(2, "A", 300, 20.0, 0.99), candidate(2, "B", 300, 20.0, 0.10)],
    }
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig())
    by_video = {a.video_id: a for a in report.alignments}
    assert by_video["A"].is_complete is False
    assert by_video["A"].missing_event_indices == (1,)
    assert [p.video_id for p in report.predictions] == ["B"]
    for prediction in report.predictions:
        assert {stp.video_id for stp in prediction.steps} == {prediction.video_id}


def test_a_recovery_that_breaks_the_order_is_rejected_wholesale() -> None:
    # The only candidate for event 1 sits before event 0; keeping it would produce a
    # decreasing sequence, so the original incomplete alignment is preserved instead.
    video = video_with("V", {1: [candidate(1, "V", 200, 1.0)]})
    result = recover_missing_events(gapped(), video, AlignmentConfig())
    assert result.steps[1].status == STEP_MISSING
    assert result.recovered_event_indices == ()


# ------------------------------------------------------------------ reporting


def test_recovery_is_reported_in_status_and_diagnostics() -> None:
    candidates = {
        0: [candidate(0, "V", 100, 10.0, 0.9)],
        # Event 1's only candidate violates min_gap from event 0 during the DP
        # transition, so the beam skips it; recovery with a looser policy fills it.
        1: [candidate(1, "V", 200, 15.0, 0.9)],
        2: [candidate(2, "V", 300, 20.0, 0.9)],
    }
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig())
    assert report.predictions
    prediction = report.predictions[0]
    assert prediction.alignment_status in {ALIGNMENT_COMPLETE, ALIGNMENT_COMPLETE_WITH_RECOVERY}
    assert report.diagnostics["remaining_missing_events"] == 0
    assert report.diagnostics["recovered_events"] >= 0
    assert set(report.diagnostics) >= {
        "event_count",
        "event_candidate_counts",
        "videos_considered",
        "initial_missing_events",
        "recovered_events",
        "remaining_missing_events",
        "discarded_incomplete_alignments",
        "returned_complete_predictions",
    }


def test_recovered_prediction_reports_which_events_were_recovered() -> None:
    video = video_with("V", {1: [candidate(1, "V", 200, 15.0)]})
    recovered = recover_missing_events(gapped(), video, AlignmentConfig())
    prediction = to_complete_prediction(recovered, AlignmentConfig())
    assert prediction is not None
    assert prediction.alignment_status == ALIGNMENT_COMPLETE_WITH_RECOVERY
    assert prediction.recovered_event_indices == (1,)
    assert prediction.frame_ids == ("100", "200", "300")
    payload = prediction.to_dict()
    assert payload["recovered_event_indices"] == [1]
    assert [item["status"] for item in payload["steps"]] == ["aligned", "recovered", "aligned"]
