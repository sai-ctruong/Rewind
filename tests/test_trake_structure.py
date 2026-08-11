"""Event preservation and the one-frame-per-event invariant.

Before Phase 7 a skipped event was dropped from the row, which shortened the sequence
and shifted every later event's label. These tests make both failures unrepresentable.
"""
from __future__ import annotations

import pytest

from aic2026.trake import (
    ALIGNMENT_COMPLETE,
    ALIGNMENT_INCOMPLETE,
    METHOD_BEAM_DP,
    STEP_ALIGNED,
    STEP_MISSING,
    STEP_RECOVERED,
    AlignmentConfig,
    EventCandidate,
    TrakeAlignedStep,
    TrakeAlignment,
    TrakeEvent,
    TrakePrediction,
    TrakeStructureError,
    align_trake,
    align_video_beam_dp,
    group_candidates,
    is_temporally_ordered,
    joint_trake_alignment,
    to_complete_prediction,
)


def candidate(event: int, video: str, frame: int, time: float, score: float = 0.9):
    return EventCandidate(event, video, f"{video}/kf_{frame:06d}", str(frame), time, score)


def events(count: int) -> tuple[TrakeEvent, ...]:
    return tuple(TrakeEvent(i, f"event {i}") for i in range(count))


def step(index: int, video: str = "V", *, present: bool = True, time: float = 0.0):
    return TrakeAlignedStep(
        event_index=index,
        event_text=f"event {index}",
        video_id=video,
        status=STEP_ALIGNED if present else STEP_MISSING,
        candidate=candidate(index, video, 100 + index, time) if present else None,
    )


# ------------------------------------------------------------ event preservation


def test_alignment_always_has_one_step_per_event() -> None:
    # Event 1 has no candidate anywhere: the position must still exist.
    grouped = group_candidates(
        {0: [candidate(0, "V", 10, 1.0)], 1: [], 2: [candidate(2, "V", 30, 3.0)]}
    )
    alignment = align_video_beam_dp(events(3), grouped[0], AlignmentConfig())
    assert len(alignment.steps) == 3
    assert [item.event_index for item in alignment.steps] == [0, 1, 2]
    assert alignment.steps[1].status == STEP_MISSING
    assert alignment.missing_event_indices == (1,)
    assert alignment.is_complete is False
    assert alignment.status == ALIGNMENT_INCOMPLETE


def test_a_missing_middle_event_stays_at_its_own_index() -> None:
    grouped = group_candidates(
        {
            0: [candidate(0, "V", 10, 1.0)],
            1: [],
            2: [candidate(2, "V", 30, 3.0)],
            3: [candidate(3, "V", 40, 4.0)],
        }
    )
    alignment = align_video_beam_dp(events(4), grouped[0], AlignmentConfig())
    assert alignment.missing_event_indices == (1,)
    # Events 2 and 3 keep their own positions instead of sliding left.
    assert alignment.steps[2].candidate.frame_id == "30"
    assert alignment.steps[3].candidate.frame_id == "40"


def test_event_labels_cannot_shift() -> None:
    grouped = group_candidates(
        {0: [candidate(0, "V", 10, 1.0)], 1: [], 2: [candidate(2, "V", 30, 3.0)]}
    )
    alignment = align_video_beam_dp(events(3), grouped[0], AlignmentConfig())
    for position, item in enumerate(alignment.steps):
        assert item.event_index == position
        assert item.event_text == f"event {position}"


def test_alignment_rejects_a_step_list_of_the_wrong_length() -> None:
    with pytest.raises(TrakeStructureError, match="every event position must be present"):
        TrakeAlignment(video_id="V", events=events(3), steps=(step(0), step(1)))


def test_alignment_rejects_out_of_order_step_indices() -> None:
    with pytest.raises(TrakeStructureError, match="step order must match"):
        TrakeAlignment(video_id="V", events=events(2), steps=(step(1), step(0)))


def test_alignment_rejects_a_step_from_another_video() -> None:
    with pytest.raises(TrakeStructureError, match="all events must come from one video"):
        TrakeAlignment(
            video_id="V", events=events(2), steps=(step(0), step(1, video="OTHER"))
        )


def test_a_step_cannot_hold_another_events_candidate() -> None:
    with pytest.raises(TrakeStructureError, match="holds a candidate for event"):
        TrakeAlignedStep(
            event_index=0,
            event_text="e",
            video_id="V",
            status=STEP_ALIGNED,
            candidate=candidate(1, "V", 10, 1.0),
        )


def test_a_step_cannot_hold_a_foreign_video_candidate() -> None:
    with pytest.raises(TrakeStructureError, match="belongs to"):
        TrakeAlignedStep(
            event_index=0,
            event_text="e",
            video_id="V",
            status=STEP_ALIGNED,
            candidate=candidate(0, "OTHER", 10, 1.0),
        )


def test_a_missing_step_cannot_carry_a_candidate() -> None:
    with pytest.raises(TrakeStructureError, match="cannot carry a candidate"):
        TrakeAlignedStep(
            event_index=0, event_text="e", video_id="V", status=STEP_MISSING,
            candidate=candidate(0, "V", 10, 1.0),
        )
    with pytest.raises(TrakeStructureError, match="must carry a candidate"):
        TrakeAlignedStep(event_index=0, event_text="e", video_id="V", status=STEP_ALIGNED)


# --------------------------------------------------------------- complete output


def test_prediction_requires_one_frame_per_event() -> None:
    with pytest.raises(TrakeStructureError, match="exactly one frame per event"):
        TrakePrediction(
            video_id="V",
            frame_ids=("1", "2"),
            event_count=3,
            alignment_status=ALIGNMENT_COMPLETE,
            score=1.0,
        )


def test_prediction_rejects_a_missing_event() -> None:
    with pytest.raises(TrakeStructureError, match="still misses events"):
        TrakePrediction(
            video_id="V",
            frame_ids=("1", "2"),
            event_count=2,
            alignment_status=ALIGNMENT_INCOMPLETE,
            score=1.0,
            missing_event_indices=(1,),
        )


def test_prediction_rejects_a_cross_video_step() -> None:
    with pytest.raises(TrakeStructureError, match="contains a step from"):
        TrakePrediction(
            video_id="V",
            frame_ids=("100", "101"),
            event_count=2,
            alignment_status=ALIGNMENT_COMPLETE,
            score=1.0,
            steps=(step(0), step(1, video="OTHER", time=2.0)),
        )


def test_incomplete_alignment_never_becomes_a_prediction() -> None:
    alignment = TrakeAlignment(
        video_id="V", events=events(3), steps=(step(0), step(1, present=False), step(2, time=2.0))
    )
    assert to_complete_prediction(alignment) is None


def test_four_events_always_emit_four_frames() -> None:
    candidates = {
        index: [candidate(index, "V", 100 + index * 10, float(index) + 1.0)]
        for index in range(4)
    }
    predictions = joint_trake_alignment(["a", "b", "c", "d"], candidates)
    assert predictions
    for prediction in predictions:
        assert prediction.event_count == 4
        assert len(prediction.frame_ids) == 4
        assert len(prediction.steps) == 4


def test_an_incomplete_video_is_discarded_not_shortened() -> None:
    candidates = {
        0: [candidate(0, "A", 10, 1.0, 0.99), candidate(0, "B", 10, 1.0, 0.5)],
        1: [candidate(1, "B", 20, 2.0, 0.5)],
        2: [candidate(2, "B", 30, 3.0, 0.5)],
    }
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig())
    # Video A can only cover event 0, so it is discarded rather than emitting one frame.
    assert [p.video_id for p in report.predictions] == ["B"]
    assert report.predictions[0].frame_ids == ("10", "20", "30")
    assert [a.video_id for a in report.discarded] == ["A"]
    assert report.diagnostics["discarded_incomplete_alignments"] == 1


def test_zero_complete_alignments_returns_an_empty_result_not_a_bad_row() -> None:
    candidates = {0: [candidate(0, "A", 10, 1.0)], 1: [], 2: []}
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig())
    assert report.predictions == ()
    assert report.diagnostics["returned_complete_predictions"] == 0
    assert report.diagnostics["discarded_incomplete_alignments"] == 1


# ------------------------------------------------------------------- invariants


def test_structural_diagnostics_are_zero() -> None:
    candidates = {
        index: [
            candidate(index, video, 100 + index * 10, float(index) + 1.0)
            for video in ("A", "B")
        ]
        for index in range(3)
    }
    report = align_trake(["a", "b", "c"], candidates, AlignmentConfig())
    assert report.predictions
    assert report.diagnostics["malformed_prediction_count"] == 0
    assert report.diagnostics["wrong_event_count_prediction_count"] == 0
    assert report.diagnostics["cross_video_step_count"] == 0


def test_all_steps_of_a_prediction_share_one_video() -> None:
    candidates = {
        index: [
            candidate(index, "A", 100 + index * 10, float(index) + 1.0, 0.9),
            candidate(index, "B", 200 + index * 10, float(index) + 1.0, 0.8),
        ]
        for index in range(3)
    }
    for prediction in joint_trake_alignment(["a", "b", "c"], candidates):
        assert {stp.video_id for stp in prediction.steps} == {prediction.video_id}
        assert {stp.candidate.video_id for stp in prediction.steps} == {prediction.video_id}


# ------------------------------------------------------------- temporal ordering


def test_temporal_order_is_preserved() -> None:
    for prediction in joint_trake_alignment(
        ["a", "b", "c"],
        {index: [candidate(index, "V", 10 + index, float(index) + 1.0)] for index in range(3)},
    ):
        times = [stp.timestamp for stp in prediction.steps]
        assert times == sorted(times)


def test_duplicate_frame_idx_is_valid_when_timestamps_advance() -> None:
    # 192 official videos repeat a frame_idx; equality is data, not corruption.
    candidates = {
        0: [candidate(0, "V", 500, 1.0)],
        1: [candidate(1, "V", 500, 5.0)],
    }
    predictions = joint_trake_alignment(["a", "b"], candidates)
    assert predictions
    assert predictions[0].frame_ids == ("500", "500")
    assert predictions[0].event_count == 2


def test_a_decreasing_sequence_is_rejected() -> None:
    alignment = TrakeAlignment(
        video_id="V",
        events=events(2),
        steps=(
            TrakeAlignedStep(0, "a", "V", STEP_ALIGNED, candidate(0, "V", 10, 9.0)),
            TrakeAlignedStep(1, "b", "V", STEP_ALIGNED, candidate(1, "V", 20, 1.0)),
        ),
    )
    assert is_temporally_ordered(alignment, AlignmentConfig()) is False
    assert to_complete_prediction(alignment, AlignmentConfig()) is None


def test_min_gap_is_enforced_when_configured() -> None:
    alignment = TrakeAlignment(
        video_id="V",
        events=events(2),
        steps=(
            TrakeAlignedStep(0, "a", "V", STEP_ALIGNED, candidate(0, "V", 10, 1.0)),
            TrakeAlignedStep(1, "b", "V", STEP_ALIGNED, candidate(1, "V", 20, 1.2)),
        ),
    )
    assert is_temporally_ordered(alignment, AlignmentConfig(min_gap_s=0.5)) is False
    assert is_temporally_ordered(alignment, AlignmentConfig(min_gap_s=0.1)) is True


# ---------------------------------------------------------------- method naming


def test_the_method_is_reported_as_beam_dp_not_exact_dp() -> None:
    candidates = {index: [candidate(index, "V", 10 + index, float(index) + 1.0)] for index in range(2)}
    report = align_trake(["a", "b"], candidates, AlignmentConfig())
    assert report.diagnostics["alignment_method"] == METHOD_BEAM_DP
    assert METHOD_BEAM_DP == "beam_dp"
    for prediction in report.predictions:
        assert prediction.method == "beam_dp"
    for alignment in report.alignments:
        assert alignment.method == "beam_dp"


def test_no_source_file_claims_exact_dp() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    checked = [
        root / "aic2026" / "trake.py",
        root / "aic2026" / "engine.py",
        root / "aic2026" / "config.py",
        root / "configs" / "settings.yaml",
        root / "ui" / "app.py",
        root / "ui" / "index.html",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8").lower()
        # The phrase may appear only while explicitly denying it.
        for line in text.splitlines():
            if "exact dp" in line or "exact_dp" in line:
                assert any(
                    word in line for word in ("not ", "never", "phase 8", "deliberately")
                ), f"{path.name} claims exact DP: {line.strip()}"


def test_alignment_config_defaults_are_honest() -> None:
    config = AlignmentConfig()
    assert config.alignment_method == "beam_dp"
    assert config.recover_missing_events is True
    assert config.recovery_respect_gap is True
