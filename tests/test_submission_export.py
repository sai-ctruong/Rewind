"""CSV serialization: atomic, UTF-8, correctly quoted, and never from visual frames."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.submission_validation import (
    SubmissionRow,
    SubmissionValidationError,
    read_submission_csv,
    submission_rows_from_kis,
    submission_rows_from_qa,
    submission_rows_from_trake,
    validate_submission,
    validation_report_path,
    write_submission_csv,
    write_validation_report,
)


class FakePrediction:
    """Shaped like `AICPrediction` for the adapters, with no engine required."""

    def __init__(
        self,
        video_id,
        frame_id,
        *,
        answer=None,
        refinement=None,
        qa=None,
        trake=None,
        event_frame_ids=(),
        keyframe_id="",
    ):
        self.video_id = video_id
        self.frame_id = frame_id
        self.answer = answer
        self.refinement = refinement
        self.qa = qa
        self.trake = trake
        self.event_frame_ids = list(event_frame_ids)
        self.keyframe_id = keyframe_id


# ------------------------------------------------------- submission frame policy


def test_a_refined_visual_frame_never_replaces_the_submission_frame(tmp_path: Path) -> None:
    """coarse 100, visual 125 -> the CSV must contain 100."""
    prediction = FakePrediction(
        "L21_V001",
        "100",
        refinement={"applied": True, "best_visual_frame_idx": 125, "coarse_official_frame_idx": 100},
    )
    rows = submission_rows_from_kis([prediction])
    assert rows[0].frame_ids == ("100",)
    assert rows[0].visual_frame_ids == (125,)
    result = validate_submission("kis", rows)
    path = write_submission_csv(result, tmp_path / "kis.csv")
    assert read_submission_csv(path) == [["L21_V001", "100"]]
    assert "125" not in path.read_text(encoding="utf-8")


def test_trake_exports_submission_frames_not_visual_ones(tmp_path: Path) -> None:
    prediction = FakePrediction(
        "L21_V001",
        "10",
        trake={
            "steps": [
                {"submission_frame_idx": "10", "visual_frame_idx": 18, "coarse_official_frame_idx": "10"},
                {"submission_frame_idx": "20", "visual_frame_idx": 27, "coarse_official_frame_idx": "20"},
                {"submission_frame_idx": "30", "visual_frame_idx": None, "coarse_official_frame_idx": "30"},
            ]
        },
    )
    rows = submission_rows_from_trake([prediction])
    assert rows[0].frame_ids == ("10", "20", "30")
    result = validate_submission("trake", rows, event_count=3)
    path = write_submission_csv(result, tmp_path / "trake.csv")
    assert read_submission_csv(path) == [["L21_V001", "10", "20", "30"]]
    text = path.read_text(encoding="utf-8")
    assert "18" not in text and "27" not in text


def test_qa_exports_the_submission_frame(tmp_path: Path) -> None:
    prediction = FakePrediction(
        "L21_V001",
        "42",
        answer="red",
        qa={
            "submission_frame_idx": 42,
            "best_visual_frame_idx": 55,
            "answer_status": "answered",
        },
    )
    rows = submission_rows_from_qa([prediction])
    assert rows[0].frame_ids == ("42",)
    assert rows[0].qa_status == "answered"
    result = validate_submission("qa", rows)
    path = write_submission_csv(result, tmp_path / "qa.csv")
    assert read_submission_csv(path) == [["L21_V001", "42", "red"]]


def test_official_frames_are_never_parsed_out_of_internal_keyframe_ids(tmp_path: Path) -> None:
    """`L21_V001/kf_000123` encodes the ORDINAL; parsing it would submit 123, not 5000."""
    prediction = FakePrediction("L21_V001", "5000", keyframe_id="L21_V001/kf_000123")
    rows = submission_rows_from_kis([prediction])
    assert rows[0].frame_ids == ("5000",)
    result = validate_submission("kis", rows)
    path = write_submission_csv(result, tmp_path / "kis.csv")
    text = path.read_text(encoding="utf-8")
    assert "5000" in text
    assert "123" not in text
    assert "kf_" not in text


# ------------------------------------------------------------------- atomicity


def test_a_validation_failure_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "kis.csv"
    result = validate_submission("kis", [["V", "-1"]])
    assert result.valid is False
    with pytest.raises(SubmissionValidationError):
        write_submission_csv(result, target)
    assert not target.exists()


def test_a_failed_export_never_replaces_a_good_file(tmp_path: Path) -> None:
    target = tmp_path / "kis.csv"
    good = validate_submission("kis", [["V", "10"]])
    write_submission_csv(good, target)
    original = target.read_text(encoding="utf-8")

    bad = validate_submission("kis", [["V", "abc"]])
    with pytest.raises(SubmissionValidationError):
        write_submission_csv(bad, target)
    assert target.read_text(encoding="utf-8") == original


def test_no_temporary_files_survive_a_successful_export(tmp_path: Path) -> None:
    result = validate_submission("kis", [["V", "10"]])
    write_submission_csv(result, tmp_path / "kis.csv")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kis.csv"]


# ------------------------------------------------------------------ encoding


@pytest.mark.parametrize("answer", ["đỏ", "có", "không", "người đang chạy", "xanh dương"])
def test_vietnamese_answers_survive_a_round_trip(tmp_path: Path, answer: str) -> None:
    result = validate_submission("qa", [["L21_V001", "10", answer]])
    path = write_submission_csv(result, tmp_path / "qa.csv")
    assert read_submission_csv(path) == [["L21_V001", "10", answer]]
    # Explicitly UTF-8, not a locale-dependent encoding.
    assert path.read_bytes().decode("utf-8").strip().endswith(answer)


def test_an_answer_containing_a_comma_is_quoted(tmp_path: Path) -> None:
    result = validate_submission("qa", [["V", "1", "a red car, parked"]])
    path = write_submission_csv(result, tmp_path / "qa.csv")
    assert '"a red car, parked"' in path.read_text(encoding="utf-8")
    # And it reads back as ONE field, not two.
    assert read_submission_csv(path) == [["V", "1", "a red car, parked"]]


def test_an_answer_containing_quotes_and_newlines_round_trips(tmp_path: Path) -> None:
    answer = 'he said "xin chào"\nand left'
    result = validate_submission("qa", [["V", "1", answer]])
    path = write_submission_csv(result, tmp_path / "qa.csv")
    assert read_submission_csv(path) == [["V", "1", answer]]


# ------------------------------------------------------------------- contents


def test_no_header_row_is_written(tmp_path: Path) -> None:
    result = validate_submission("kis", [["V", "10"], ["W", "20"]])
    path = write_submission_csv(result, tmp_path / "kis.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    # The competition format is bare rows; a header would become a submitted row.
    assert lines == ["V,10", "W,20"]


def test_csv_ordering_is_deterministic(tmp_path: Path) -> None:
    rows = [["C", "3"], ["A", "1"], ["B", "2"]]
    first = write_submission_csv(validate_submission("kis", rows), tmp_path / "a.csv")
    second = write_submission_csv(validate_submission("kis", rows), tmp_path / "b.csv")
    # Rank order is preserved exactly; validation never reorders.
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert read_submission_csv(first) == [["C", "3"], ["A", "1"], ["B", "2"]]


def test_at_most_100_rows_reach_the_file(tmp_path: Path) -> None:
    rows = [["V", str(index)] for index in range(250)]
    result = validate_submission("kis", rows)
    path = write_submission_csv(result, tmp_path / "kis.csv")
    assert len(read_submission_csv(path)) == 100


def test_an_empty_submission_is_never_written(tmp_path: Path) -> None:
    target = tmp_path / "kis.csv"
    with pytest.raises(SubmissionValidationError):
        write_submission_csv(validate_submission("kis", []), target)
    assert not target.exists()


# ------------------------------------------------------------------- sidecar


def test_the_sidecar_report_is_written_beside_the_csv(tmp_path: Path) -> None:
    result = validate_submission("kis", [["V", "10"], ["V", "10"], ["W", "20"]])
    target = tmp_path / "kis.csv"
    write_submission_csv(result, target)
    report_path = write_validation_report(
        result,
        validation_report_path(target),
        metadata={"runtime_generation": 3, "config_hash": "abc123", "task": "kis"},
    )
    assert report_path == tmp_path / "kis.validation.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert payload["task"] == "kis"
    assert payload["row_count"] == 2
    assert payload["row_count_before"] == 3
    assert payload["duplicates_removed"] == 1
    assert payload["runtime_generation"] == 3
    assert payload["config_hash"] == "abc123"
    assert "no aic ground truth" in payload["note"].lower()


def test_the_report_contains_no_absolute_paths_by_default(tmp_path: Path) -> None:
    result = validate_submission("kis", [["V", "10"]])
    target = tmp_path / "kis.csv"
    write_submission_csv(result, target)
    report = write_validation_report(result, validation_report_path(target))
    assert str(tmp_path) not in report.read_text(encoding="utf-8")
