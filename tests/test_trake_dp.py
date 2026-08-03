import pytest

from aic2026.metrics import FrameRange, RankedAnswer, TRAKEGroundTruth, trake_r_score
from aic2026.trake import AlignmentConfig, EventCandidate, joint_trake_alignment


def c(event, video, frame, time, score):
    return EventCandidate(event, video, f"{video}/{frame}", str(frame), time, score)


def test_dp_enforces_order_and_beats_incomplete_strong_video() -> None:
    candidates = {
        0: [c(0, "A", 10, 10, 0.9), c(0, "B", 10, 1, 0.6)],
        1: [c(1, "A", 20, 5, 0.9), c(1, "B", 20, 2, 0.6)],
        2: [c(2, "B", 30, 3, 0.6)],
    }
    out = joint_trake_alignment(["a", "b", "c"], candidates, AlignmentConfig(missing_event_penalty=0.5))
    assert out[0].video_id == "B"
    assert out[0].frame_ids == ("10", "20", "30")


def test_dp_gap_constraints_and_determinism() -> None:
    candidates = {0: [c(0, "V", 1, 1, 1)], 1: [c(1, "V", 2, 20, 1), c(1, "V", 3, 4, 0.8)]}
    cfg = AlignmentConfig(min_gap_s=1, max_gap_s=5)
    first = joint_trake_alignment(["a", "b"], candidates, cfg)
    second = joint_trake_alignment(["a", "b"], candidates, cfg)
    assert first == second and first[0].frame_ids == ("1", "3")


def test_partial_r_score_three_of_four() -> None:
    gt = TRAKEGroundTruth("V", tuple(FrameRange(i, i) for i in (1, 2, 3, 4)))
    assert trake_r_score(RankedAnswer("V", ("1", "2", "30", "4")), gt) == pytest.approx(0.75)
