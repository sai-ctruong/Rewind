"""Row-scoped manual editing.

The Phase 0 bug: editing one frame rewrote every matching numeric value across every
task, because edits were addressed by VALUE. These tests fix the identity model in place
— edits address `row_id` (and `event_index` for TRAKE), so two rows sharing a frame
number are two different rows.
"""
from __future__ import annotations

import pytest

from aic2026.result_batch import (
    ResultBatchStore,
    ResultEditError,
    apply_edit,
    build_result_batch,
    reset_batch,
    reset_row,
)
from aic2026.submission_validation import validate_submission


class FakePrediction:
    def __init__(self, video_id, frame_id, *, answer=None, qa=None, trake=None, keyframe_id=""):
        self.video_id = video_id
        self.frame_id = frame_id
        self.answer = answer
        self.qa = qa
        self.trake = trake
        self.refinement = None
        self.event_frame_ids = []
        self.keyframe_id = keyframe_id


def kis_batch(frames=("100", "100", "250")):
    """Deliberately repeats frame 100 across two DIFFERENT rows."""
    return build_result_batch(
        "kis",
        [FakePrediction(f"L21_V00{index + 1}", frame) for index, frame in enumerate(frames)],
        query="a person",
        runtime_generation=1,
    )


def qa_batch():
    return build_result_batch(
        "qa",
        [
            FakePrediction("L21_V001", "100", answer="red", qa={"submission_frame_idx": 100, "answer_status": "answered"}),
            FakePrediction("L21_V001", "180", answer="red", qa={"submission_frame_idx": 180, "answer_status": "answered"}),
            FakePrediction("L21_V002", "100", answer="blue", qa={"submission_frame_idx": 100, "answer_status": "answered"}),
        ],
        query="what colour?",
        runtime_generation=1,
    )


def trake_batch():
    def steps(values):
        return {"steps": [{"submission_frame_idx": v, "visual_frame_idx": None} for v in values]}

    return build_result_batch(
        "trake",
        [
            FakePrediction("L21_V001", "100", trake=steps(["100", "200", "300"])),
            FakePrediction("L21_V002", "100", trake=steps(["100", "400", "500"])),
        ],
        query="a ; b ; c",
        runtime_generation=1,
        event_count=3,
    )


# ------------------------------------------------------------------ isolation


def test_editing_one_kis_row_leaves_the_other_rows_alone() -> None:
    batch = kis_batch()
    assert [row.submission_frames for row in batch.rows] == [("100",), ("100",), ("250",)]
    updated = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value="999")
    # Row 0 changed; row 1 holds the SAME original value 100 and must be untouched.
    assert [row.submission_frames for row in updated.rows] == [("999",), ("100",), ("250",)]
    assert updated.rows[0].edited is True
    assert updated.rows[1].edited is False


def test_editing_a_kis_row_cannot_reach_qa_or_trake() -> None:
    store = ResultBatchStore()
    kis = store.put(kis_batch())
    qa = store.put(qa_batch())
    trake = store.put(trake_batch())

    store.update(apply_edit(kis, row_id=kis.rows[0].row_id, field_name="frame", value="777"))

    # Separate batches: a KIS edit is structurally incapable of touching them.
    assert [row.submission_frames for row in store.get(qa.result_id).rows] == [
        ("100",), ("180",), ("100",)
    ]
    assert [row.submission_frames for row in store.get(trake.result_id).rows] == [
        ("100", "200", "300"), ("100", "400", "500")
    ]
    assert store.get(kis.result_id).rows[0].submission_frames == ("777",)


def test_editing_one_trake_event_leaves_every_other_event_alone() -> None:
    batch = trake_batch()
    updated = apply_edit(
        batch, row_id=batch.rows[0].row_id, field_name="frame", value="222", event_index=1
    )
    assert updated.rows[0].submission_frames == ("100", "222", "300")
    # The other sequence shares frame 100 at event 0 and is untouched.
    assert updated.rows[1].submission_frames == ("100", "400", "500")


def test_editing_event_zero_does_not_touch_the_same_value_in_another_sequence() -> None:
    batch = trake_batch()
    updated = apply_edit(
        batch, row_id=batch.rows[0].row_id, field_name="frame", value="111", event_index=0
    )
    assert updated.rows[0].submission_frames[0] == "111"
    assert updated.rows[1].submission_frames[0] == "100"


def test_editing_one_qa_answer_leaves_the_other_rows_alone() -> None:
    batch = qa_batch()
    updated = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="answer", value="đỏ")
    answers = [row.answer.current_value for row in updated.rows]
    # Rows 1 and 2 also said "red"/"blue"; only row 0 changes.
    assert answers == ["đỏ", "red", "blue"]
    assert updated.rows[0].edited is True and updated.rows[1].edited is False


def test_a_qa_edit_does_not_change_the_frame() -> None:
    batch = qa_batch()
    updated = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="answer", value="xanh")
    assert updated.rows[0].submission_frames == ("100",)


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("value", ["-1", "abc", "", "   ", "1.5"])
def test_an_invalid_frame_edit_is_rejected_immediately(value) -> None:
    batch = kis_batch()
    with pytest.raises(ResultEditError) as info:
        apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value=value)
    assert info.value.error_code in {"INVALID_FRAME_ID", "NEGATIVE_FRAME_ID"}


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_an_empty_answer_edit_is_rejected(value) -> None:
    batch = qa_batch()
    with pytest.raises(ResultEditError) as info:
        apply_edit(batch, row_id=batch.rows[0].row_id, field_name="answer", value=value)
    assert info.value.error_code == "QA_EMPTY_ANSWER"


def test_editing_an_unknown_row_is_rejected() -> None:
    with pytest.raises(ResultEditError) as info:
        apply_edit(kis_batch(), row_id="nope", field_name="frame", value="1")
    assert info.value.error_code == "UNKNOWN_ROW"


def test_an_out_of_range_event_index_is_rejected() -> None:
    batch = trake_batch()
    with pytest.raises(ResultEditError) as info:
        apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value="1", event_index=9)
    assert info.value.error_code == "INVALID_EVENT_INDEX"


def test_an_unknown_field_is_rejected() -> None:
    batch = kis_batch()
    with pytest.raises(ResultEditError) as info:
        apply_edit(batch, row_id=batch.rows[0].row_id, field_name="score", value="1")
    assert info.value.error_code == "INVALID_FIELD"


def test_a_kis_row_has_no_answer_to_edit() -> None:
    batch = kis_batch()
    with pytest.raises(ResultEditError):
        apply_edit(batch, row_id=batch.rows[0].row_id, field_name="answer", value="x")


def test_trake_edits_always_preserve_the_event_count() -> None:
    batch = trake_batch()
    updated = apply_edit(
        batch, row_id=batch.rows[0].row_id, field_name="frame", value="222", event_index=2
    )
    assert all(len(row.submission_frames) == 3 for row in updated.rows)
    assert validate_submission("trake", updated.to_submission_rows(), event_count=3).valid


# ----------------------------------------------------------------- provenance


def test_an_edit_records_its_provenance() -> None:
    batch = kis_batch()
    updated = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value="999")
    cell = updated.rows[0].frames[0]
    assert cell.original_value == "100"
    assert cell.current_value == "999"
    assert cell.edited is True
    assert cell.edited_at is not None
    assert updated.manual_edit_count == 1
    assert updated.rows[0].to_dict()["edited"] is True


def test_editing_back_to_the_original_clears_the_edited_flag() -> None:
    batch = kis_batch()
    once = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value="999")
    twice = apply_edit(once, row_id=batch.rows[0].row_id, field_name="frame", value="100")
    assert twice.rows[0].frames[0].edited is False
    assert twice.manual_edit_count == 0


def test_a_manually_edited_answer_is_marked_manual_for_export() -> None:
    batch = build_result_batch(
        "qa",
        [FakePrediction("V", "1", answer="unknown", qa={"submission_frame_idx": 1, "answer_status": "abstained"})],
        query="q",
        runtime_generation=1,
    )
    # An abstention is not exportable...
    assert validate_submission("qa", batch.to_submission_rows()).valid is False
    # ...but a deliberate human answer is, and is marked as manual rather than answered.
    edited = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="answer", value="đỏ")
    rows = edited.to_submission_rows()
    assert rows[0].qa_status == "manual"
    assert validate_submission("qa", rows).valid is True


# ---------------------------------------------------------------- undo/reset


def test_resetting_one_row_restores_only_that_row() -> None:
    batch = kis_batch()
    edited = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value="999")
    edited = apply_edit(edited, row_id=edited.rows[2].row_id, field_name="frame", value="888")
    restored = reset_row(edited, edited.rows[0].row_id)
    assert restored.rows[0].submission_frames == ("100",)
    assert restored.rows[0].edited is False
    # The other edit survives.
    assert restored.rows[2].submission_frames == ("888",)


def test_resetting_the_batch_restores_everything() -> None:
    batch = trake_batch()
    edited = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="frame", value="222", event_index=1)
    edited = apply_edit(edited, row_id=edited.rows[1].row_id, field_name="frame", value="999", event_index=2)
    restored = reset_batch(edited)
    assert [row.submission_frames for row in restored.rows] == [
        ("100", "200", "300"), ("100", "400", "500")
    ]
    assert restored.manual_edit_count == 0


def test_resetting_a_qa_answer_restores_it() -> None:
    batch = qa_batch()
    edited = apply_edit(batch, row_id=batch.rows[0].row_id, field_name="answer", value="đỏ")
    restored = reset_row(edited, edited.rows[0].row_id)
    assert restored.rows[0].answer.current_value == "red"
    assert restored.rows[0].answer.edited is False


# --------------------------------------------------------------------- store


def test_the_store_returns_batches_by_id_and_reports_unknown_ones() -> None:
    store = ResultBatchStore()
    batch = store.put(kis_batch())
    assert store.get(batch.result_id).result_id == batch.result_id
    with pytest.raises(ResultEditError) as info:
        store.get("rb_nope")
    assert info.value.error_code == "UNKNOWN_RESULT_BATCH"


def test_the_store_is_bounded() -> None:
    store = ResultBatchStore(limit=2)
    first = store.put(kis_batch())
    store.put(kis_batch())
    store.put(kis_batch())
    with pytest.raises(ResultEditError):
        store.get(first.result_id)


def test_batch_metadata_identifies_the_generation_and_config() -> None:
    batch = build_result_batch(
        "kis",
        [FakePrediction("V", "1")],
        query="a person",
        runtime_generation=4,
        config_hash="cfg123",
        selected_video_ids_hash="ids456",
    )
    metadata = batch.metadata()
    assert metadata["task"] == "kis"
    assert metadata["query"] == "a person"
    assert metadata["runtime_generation"] == 4
    assert metadata["config_hash"] == "cfg123"
    assert metadata["selected_video_ids_hash"] == "ids456"
    assert metadata["manual_edit_count"] == 0
    assert metadata["created_at"]
