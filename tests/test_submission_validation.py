"""Structural validation of the three official submission schemas.

"Valid" here means the FORMAT is right. It never means an answer or a frame is correct:
this repository has no AIC ground truth, and the validator has no way to know.
"""
from __future__ import annotations

import pytest

from aic2026.submission_validation import (
    DUPLICATE_KIS_ROW,
    DUPLICATE_QA_ROW,
    DUPLICATE_TRAKE_SEQUENCE,
    EMPTY_SUBMISSION,
    INVALID_FRAME_ID,
    INVALID_VIDEO_ID,
    NEGATIVE_FRAME_ID,
    QA_EMPTY_ANSWER,
    QA_NON_SUBMITTABLE_STATUS,
    RESULT_BATCH_TASK_MISMATCH,
    STALE_RESULT_GENERATION,
    TOO_MANY_ROWS,
    TRAKE_EVENT_COUNT_MISMATCH,
    TRAKE_MISSING_FRAME,
    UNKNOWN_TASK,
    SubmissionRow,
    SubmissionValidationError,
    is_submittable_answer,
    validate_submission,
    validate_submission_or_raise,
)


def codes(result) -> set[str]:
    return {issue.code for issue in result.errors}


# ----------------------------------------------------------------------- KIS


def test_a_valid_kis_row_passes() -> None:
    result = validate_submission("kis", [["L21_V001", "1234"]])
    assert result.valid is True
    assert result.row_count == 1
    assert result.rows[0].csv_row() == ["L21_V001", "1234"]


def test_kis_rejects_a_negative_frame() -> None:
    result = validate_submission("kis", [["L21_V001", "-5"]])
    assert result.valid is False
    assert NEGATIVE_FRAME_ID in codes(result)


@pytest.mark.parametrize("frame", ["abc", "", "1.5", None, "12a", " "])
def test_kis_rejects_a_malformed_frame(frame) -> None:
    result = validate_submission("kis", [["L21_V001", frame]])
    assert result.valid is False
    assert INVALID_FRAME_ID in codes(result)


def test_kis_rejects_a_bad_video_id() -> None:
    for video in ("", "  ", "../etc/passwd", "a/b"):
        result = validate_submission("kis", [[video, "10"]])
        assert result.valid is False
        assert INVALID_VIDEO_ID in codes(result)


def test_kis_rejects_extra_columns() -> None:
    result = validate_submission("kis", [["L21_V001", "10", "20"]])
    assert result.valid is False


def test_kis_duplicate_removal_preserves_the_first_rank() -> None:
    rows = [["A", "10"], ["B", "20"], ["A", "10"], ["C", "30"]]
    result = validate_submission("kis", rows)
    assert result.valid is True
    assert result.duplicates_removed == 1
    assert [row.csv_row() for row in result.rows] == [["A", "10"], ["B", "20"], ["C", "30"]]
    assert DUPLICATE_KIS_ROW in {issue.code for issue in result.warnings}


def test_kis_duplicates_can_be_treated_as_errors() -> None:
    result = validate_submission(
        "kis", [["A", "10"], ["A", "10"]], remove_duplicates=False
    )
    assert result.valid is False
    assert DUPLICATE_KIS_ROW in codes(result)


def test_a_different_frame_of_the_same_video_is_not_a_duplicate() -> None:
    result = validate_submission("kis", [["A", "10"], ["A", "11"]])
    assert result.valid is True
    assert result.duplicates_removed == 0


# ----------------------------------------------------------------------- Q&A


def test_a_valid_qa_row_passes() -> None:
    result = validate_submission("qa", [["L21_V001", "10", "red"]])
    assert result.valid is True
    assert result.rows[0].csv_row() == ["L21_V001", "10", "red"]


@pytest.mark.parametrize("answer", ["", "   ", "\t", None])
def test_qa_rejects_an_empty_answer(answer) -> None:
    result = validate_submission("qa", [["L21_V001", "10", answer]])
    assert result.valid is False
    assert QA_EMPTY_ANSWER in codes(result)


def test_qa_rejects_a_row_without_an_answer_column() -> None:
    result = validate_submission("qa", [["L21_V001", "10"]])
    assert result.valid is False
    assert QA_EMPTY_ANSWER in codes(result)


def test_qa_rejects_a_backend_failure_even_though_the_text_is_non_empty() -> None:
    row = SubmissionRow(
        video_id="L21_V001", frame_ids=("10",), answer="unknown", qa_status="backend_failed"
    )
    result = validate_submission("qa", [row])
    assert result.valid is False
    assert QA_NON_SUBMITTABLE_STATUS in codes(result)


def test_qa_rejects_an_abstention() -> None:
    row = SubmissionRow(
        video_id="L21_V001", frame_ids=("10",), answer="unknown", qa_status="abstained"
    )
    result = validate_submission("qa", [row])
    assert result.valid is False
    # An abstention is the system saying it has no answer; exporting it would fabricate one.
    assert QA_NON_SUBMITTABLE_STATUS in codes(result)


def test_qa_rejects_bare_unknown_text_with_no_status() -> None:
    assert validate_submission("qa", [["V", "1", "unknown"]]).valid is False
    assert validate_submission("qa", [["V", "1", "không xác định"]]).valid is False


def test_a_manually_supplied_answer_is_submittable() -> None:
    row = SubmissionRow(
        video_id="L21_V001", frame_ids=("10",), answer="đỏ", qa_status="manual"
    )
    assert validate_submission("qa", [row]).valid is True


def test_a_backend_dumping_its_input_is_rejected_as_too_long() -> None:
    """Found by the real smoke: a non-visual mock echoed a whole YouTube description.

    Four kilobytes of channel boilerplate is structurally "non-empty text" but is
    obviously not an answer, and it must not reach a CSV.
    """
    from aic2026.submission_validation import MAX_ANSWER_LENGTH, QA_ANSWER_TOO_LONG

    dump = "media_title 60 giây sáng " + ("tin tuc " * 600)
    assert len(dump) > MAX_ANSWER_LENGTH
    result = validate_submission("qa", [["L21_V025", "11771", dump]])
    assert result.valid is False
    issue = next(i for i in result.errors if i.code == QA_ANSWER_TOO_LONG)
    assert issue.expected == MAX_ANSWER_LENGTH
    # A normal short answer of any language is unaffected.
    assert validate_submission("qa", [["V", "1", "người đang chạy"]]).valid is True


def test_a_non_visual_backend_answer_is_not_exportable() -> None:
    """A mock backend cannot answer a question about a video; only a human can."""
    from aic2026.submission_validation import submission_rows_from_qa

    class Prediction:
        video_id = "L21_V001"
        frame_id = "10"
        answer = "a plausible looking string"
        refinement = None
        trake = None
        qa = {
            "submission_frame_idx": 10,
            "answer_status": "answered",
            "backend_type": "mock",
            "backend_visual": False,
        }

    rows = submission_rows_from_qa([Prediction()])
    assert rows[0].qa_status == "mock_backend"
    result = validate_submission("qa", rows)
    assert result.valid is False
    assert QA_NON_SUBMITTABLE_STATUS in codes(result)


def test_a_visual_backend_answer_is_exportable() -> None:
    from aic2026.submission_validation import submission_rows_from_qa

    class Prediction:
        video_id = "L21_V001"
        frame_id = "10"
        answer = "red"
        refinement = None
        trake = None
        qa = {
            "submission_frame_idx": 10,
            "answer_status": "answered",
            "backend_type": "api",
            "backend_visual": True,
        }

    rows = submission_rows_from_qa([Prediction()])
    assert rows[0].qa_status == "answered"
    assert validate_submission("qa", rows).valid is True


def test_is_submittable_answer_matrix() -> None:
    assert is_submittable_answer("red") is True
    assert is_submittable_answer("red", "answered") is True
    assert is_submittable_answer("red", "abstained") is False
    assert is_submittable_answer("red", "backend_failed") is False
    assert is_submittable_answer("red", "visual_unavailable") is False
    assert is_submittable_answer("red", "mock_backend") is False
    assert is_submittable_answer("", "answered") is False
    assert is_submittable_answer("unknown") is False
    assert is_submittable_answer(None) is False


def test_qa_duplicate_policy_uses_the_normalized_answer() -> None:
    rows = [["A", "1", "red"], ["A", "1", "  RED  "], ["A", "1", "blue"]]
    result = validate_submission("qa", rows)
    assert result.valid is True
    assert result.duplicates_removed == 1
    assert [row.csv_row() for row in result.rows] == [["A", "1", "red"], ["A", "1", "blue"]]
    assert DUPLICATE_QA_ROW in {issue.code for issue in result.warnings}


# --------------------------------------------------------------------- TRAKE


def test_n_events_produce_n_frames() -> None:
    result = validate_submission("trake", [["V", "1", "2", "3", "4"]], event_count=4)
    assert result.valid is True
    assert result.rows[0].csv_row() == ["V", "1", "2", "3", "4"]


def test_trake_rejects_too_few_frames() -> None:
    result = validate_submission("trake", [["V", "1", "2", "3"]], event_count=4)
    assert result.valid is False
    issue = next(i for i in result.errors if i.code == TRAKE_EVENT_COUNT_MISMATCH)
    assert issue.expected == 4 and issue.actual == 3


def test_trake_rejects_too_many_frames() -> None:
    result = validate_submission("trake", [["V", "1", "2", "3", "4", "5"]], event_count=4)
    assert result.valid is False
    assert TRAKE_EVENT_COUNT_MISMATCH in codes(result)


def test_trake_rejects_a_none_frame() -> None:
    result = validate_submission("trake", [["V", "1", None, "3"]], event_count=3)
    assert result.valid is False
    assert TRAKE_MISSING_FRAME in codes(result) or INVALID_FRAME_ID in codes(result)


def test_trake_rejects_a_negative_frame() -> None:
    result = validate_submission("trake", [["V", "1", "-2", "3"]], event_count=3)
    assert result.valid is False
    assert NEGATIVE_FRAME_ID in codes(result)


def test_a_repeated_frame_inside_one_sequence_is_allowed() -> None:
    # 192 official videos repeat a frame_idx, so this is legitimate data.
    result = validate_submission("trake", [["V", "500", "500", "700"]], event_count=3)
    assert result.valid is True
    assert result.rows[0].csv_row() == ["V", "500", "500", "700"]


def test_a_duplicate_full_sequence_is_detected() -> None:
    rows = [["V", "1", "2"], ["V", "1", "2"], ["V", "1", "3"]]
    result = validate_submission("trake", rows, event_count=2)
    assert result.valid is True
    assert result.duplicates_removed == 1
    assert DUPLICATE_TRAKE_SEQUENCE in {issue.code for issue in result.warnings}
    assert [row.csv_row() for row in result.rows] == [["V", "1", "2"], ["V", "1", "3"]]


def test_the_event_count_is_inferred_only_when_rows_agree() -> None:
    agreeing = validate_submission("trake", [["V", "1", "2"], ["W", "3", "4"]])
    assert agreeing.valid is True
    assert agreeing.event_count == 2
    disagreeing = validate_submission("trake", [["V", "1", "2"], ["W", "3", "4", "5"]])
    assert disagreeing.valid is False
    assert TRAKE_EVENT_COUNT_MISMATCH in codes(disagreeing)


# -------------------------------------------------- shared rules and generation


def test_an_empty_submission_is_rejected() -> None:
    for task in ("kis", "qa", "trake"):
        result = validate_submission(task, [])
        assert result.valid is False
        assert EMPTY_SUBMISSION in codes(result)


def test_an_unknown_task_is_rejected() -> None:
    result = validate_submission("avs", [["V", "1"]])
    assert result.valid is False
    assert UNKNOWN_TASK in codes(result)


def test_at_most_100_rows_are_exported() -> None:
    rows = [["V", str(index)] for index in range(150)]
    result = validate_submission("kis", rows)
    assert result.valid is True
    assert result.row_count == 100
    assert result.truncated == 50
    assert TOO_MANY_ROWS in {issue.code for issue in result.warnings}
    # Truncation keeps rank order, taking the first 100.
    assert result.rows[0].csv_row() == ["V", "0"]
    assert result.rows[-1].csv_row() == ["V", "99"]


def test_the_row_cap_can_be_lowered_but_never_raised() -> None:
    rows = [["V", str(index)] for index in range(150)]
    assert validate_submission("kis", rows, max_rows=10).row_count == 10
    assert validate_submission("kis", rows, max_rows=1000).row_count == 100


def test_a_stale_generation_is_rejected() -> None:
    result = validate_submission(
        "kis", [["V", "1"]], active_generation=3, result_generation=2
    )
    assert result.valid is False
    issue = next(i for i in result.errors if i.code == STALE_RESULT_GENERATION)
    assert issue.expected == 3 and issue.actual == 2


def test_a_matching_generation_passes() -> None:
    assert validate_submission(
        "kis", [["V", "1"]], active_generation=3, result_generation=3
    ).valid is True


def test_a_task_mismatch_between_batch_and_export_is_rejected() -> None:
    result = validate_submission("kis", [["V", "1"]], batch_task="trake")
    assert result.valid is False
    assert RESULT_BATCH_TASK_MISMATCH in codes(result)


def test_validate_or_raise_carries_the_result() -> None:
    with pytest.raises(SubmissionValidationError) as info:
        validate_submission_or_raise("kis", [["V", "-1"]])
    assert info.value.result.valid is False
    assert NEGATIVE_FRAME_ID in codes(info.value.result)
    assert validate_submission_or_raise("kis", [["V", "1"]]).valid is True


def test_the_report_says_format_not_correctness() -> None:
    payload = validate_submission("kis", [["V", "1"]]).to_dict()
    assert payload["valid"] is True
    note = payload["note"].lower()
    assert "format" in note
    assert "correct" in note and "no aic ground truth" in note
