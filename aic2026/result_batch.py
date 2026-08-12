"""Result batches with row-scoped manual edits.

The Phase 0 audit found that editing one frame in the UI rewrote every matching numeric
value across every task:

```javascript
Object.keys(state.rows).forEach(task => state.rows[task] = state.rows[task].map(
  row => row.map((value, index) => index > 0 && String(value) === old ? edit.value : value)))
```

Editing a KIS frame `100` therefore also rewrote a Q&A frame `100` and every TRAKE event
frame `100`, silently corrupting two other submissions. The cause is that edits were
addressed by *value*. Here they are addressed by *identity*:

    result_id + row_id            for KIS and Q&A
    result_id + row_id + event    for TRAKE

Two rows sharing a frame number are two different rows, and nothing about editing one can
reach the other. Every edit keeps its original value, so a single row or the whole batch
can be restored without reloading anything.

This is session state, deliberately: no database, no persistence layer.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from .submission_validation import (
    SUBMISSION_TASKS,
    SubmissionRow,
    is_submittable_answer,
    submission_rows_for,
)

FIELD_FRAME = "frame"
FIELD_ANSWER = "answer"
EDITABLE_FIELDS = (FIELD_FRAME, FIELD_ANSWER)


class ResultEditError(ValueError):
    """Raised when an edit is malformed or would target something that does not exist."""

    def __init__(self, message: str, *, error_code: str = "INVALID_EDIT") -> None:
        super().__init__(message)
        self.error_code = error_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EditedValue:
    """One editable cell, remembering what it started as."""

    original_value: str
    current_value: str
    edited: bool = False
    edited_at: Optional[str] = None

    def with_value(self, value: str) -> "EditedValue":
        text = str(value)
        return EditedValue(
            original_value=self.original_value,
            current_value=text,
            edited=text != self.original_value,
            edited_at=_now() if text != self.original_value else self.edited_at,
        )

    def reset(self) -> "EditedValue":
        return EditedValue(self.original_value, self.original_value, False, None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResultRow:
    """One submission row, addressable by a stable `row_id`."""

    row_id: str
    video_id: str
    rank: int
    frames: tuple[EditedValue, ...]
    answer: Optional[EditedValue] = None
    qa_status: Optional[str] = None
    keyframe_id: Optional[str] = None
    # Evidence only. Kept beside the submission frames so the UI can show both, and
    # deliberately never used when building a submission row.
    visual_frame_ids: tuple[Optional[int], ...] = ()

    @property
    def edited(self) -> bool:
        return any(item.edited for item in self.frames) or bool(
            self.answer is not None and self.answer.edited
        )

    @property
    def submission_frames(self) -> tuple[str, ...]:
        return tuple(item.current_value for item in self.frames)

    def to_submission_row(self) -> SubmissionRow:
        return SubmissionRow(
            video_id=self.video_id,
            frame_ids=self.submission_frames,
            answer=None if self.answer is None else self.answer.current_value,
            rank=self.rank,
            qa_status="manual" if (self.answer is not None and self.answer.edited) else self.qa_status,
            edited=self.edited,
            visual_frame_ids=self.visual_frame_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "video_id": self.video_id,
            "rank": self.rank,
            "edited": self.edited,
            "frames": [item.to_dict() for item in self.frames],
            "submission_frames": list(self.submission_frames),
            "answer": None if self.answer is None else self.answer.to_dict(),
            "qa_status": self.qa_status,
            "keyframe_id": self.keyframe_id,
            "visual_frame_ids": list(self.visual_frame_ids),
        }


@dataclass(frozen=True)
class ResultBatch:
    """One task's results, identified so an export can be checked against them."""

    result_id: str
    task: str
    query: str
    runtime_generation: int
    config_hash: str = ""
    selected_video_ids_hash: str = ""
    created_at: str = field(default_factory=_now)
    event_count: Optional[int] = None
    rows: tuple[ResultRow, ...] = ()

    @property
    def manual_edit_count(self) -> int:
        return sum(1 for row in self.rows if row.edited)

    def row(self, row_id: str) -> ResultRow:
        for item in self.rows:
            if item.row_id == row_id:
                return item
        raise ResultEditError(f"Unknown row {row_id!r}.", error_code="UNKNOWN_ROW")

    def to_submission_rows(self) -> list[SubmissionRow]:
        return [row.to_submission_row() for row in self.rows]

    def metadata(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "task": self.task,
            "query": self.query,
            "runtime_generation": self.runtime_generation,
            "config_hash": self.config_hash,
            "selected_video_ids_hash": self.selected_video_ids_hash,
            "created_at": self.created_at,
            "event_count": self.event_count,
            "manual_edit_count": self.manual_edit_count,
        }

    def to_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        payload = self.metadata()
        payload["row_count"] = len(self.rows)
        if include_rows:
            payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


def build_result_batch(
    task: str,
    predictions: Sequence[Any],
    *,
    query: str,
    runtime_generation: int,
    config_hash: str = "",
    selected_video_ids_hash: str = "",
    event_count: Optional[int] = None,
    result_id: Optional[str] = None,
) -> ResultBatch:
    """Turn engine predictions into an editable batch of official rows."""
    task_name = str(task).strip().lower()
    if task_name not in SUBMISSION_TASKS:
        raise ResultEditError(f"Unknown task {task!r}.", error_code="UNKNOWN_TASK")
    submission_rows = submission_rows_for(task_name, predictions)
    rows: list[ResultRow] = []
    for index, (submission, prediction) in enumerate(zip(submission_rows, predictions)):
        rows.append(
            ResultRow(
                # Position-based and stable for the life of the batch, so an edit can
                # name exactly one row even when two rows share a frame number.
                row_id=f"r{index:04d}",
                video_id=submission.video_id,
                rank=submission.rank or index + 1,
                frames=tuple(EditedValue(value, value) for value in submission.frame_ids),
                answer=(
                    None
                    if submission.answer is None
                    else EditedValue(str(submission.answer), str(submission.answer))
                ),
                qa_status=submission.qa_status,
                keyframe_id=getattr(prediction, "keyframe_id", None),
                visual_frame_ids=submission.visual_frame_ids,
            )
        )
    return ResultBatch(
        result_id=result_id or f"rb_{uuid.uuid4().hex[:12]}",
        task=task_name,
        query=str(query),
        runtime_generation=int(runtime_generation),
        config_hash=str(config_hash),
        selected_video_ids_hash=str(selected_video_ids_hash),
        event_count=event_count,
        rows=tuple(rows),
    )


def _validated_frame(value: Any) -> str:
    """Reject an obviously bad frame immediately rather than at export time."""
    text = str(value).strip()
    if not text:
        raise ResultEditError("A frame id cannot be empty.", error_code="INVALID_FRAME_ID")
    if not text.lstrip("+-").isdigit():
        raise ResultEditError(
            f"A frame id must be an integer, got {value!r}.", error_code="INVALID_FRAME_ID"
        )
    if int(text) < 0:
        raise ResultEditError(
            f"A frame id cannot be negative, got {value!r}.", error_code="NEGATIVE_FRAME_ID"
        )
    return str(int(text))


def apply_edit(
    batch: ResultBatch,
    *,
    row_id: str,
    field_name: str,
    value: Any,
    event_index: Optional[int] = None,
) -> ResultBatch:
    """Edit exactly ONE cell of ONE row.

    Nothing is matched by value, so another row holding the same frame number — in this
    task or any other — is untouched by construction rather than by care.
    """
    target = batch.row(row_id)
    name = str(field_name).strip().lower()
    if name not in EDITABLE_FIELDS:
        raise ResultEditError(
            f"Field {field_name!r} is not editable; expected one of {', '.join(EDITABLE_FIELDS)}.",
            error_code="INVALID_FIELD",
        )

    if name == FIELD_FRAME:
        position = 0 if event_index is None else int(event_index)
        if position < 0 or position >= len(target.frames):
            raise ResultEditError(
                f"Row {row_id!r} has {len(target.frames)} frame(s); event index "
                f"{position} is out of range.",
                error_code="INVALID_EVENT_INDEX",
            )
        cleaned = _validated_frame(value)
        frames = list(target.frames)
        frames[position] = frames[position].with_value(cleaned)
        updated = replace(target, frames=tuple(frames))
    else:
        if target.answer is None:
            raise ResultEditError(
                f"Row {row_id!r} has no answer field to edit.", error_code="INVALID_FIELD"
            )
        text = str(value)
        if not text.strip():
            raise ResultEditError(
                "An answer cannot be empty or whitespace.", error_code="QA_EMPTY_ANSWER"
            )
        updated = replace(target, answer=target.answer.with_value(text))

    rows = tuple(updated if row.row_id == row_id else row for row in batch.rows)
    return replace(batch, rows=rows)


def reset_row(batch: ResultBatch, row_id: str) -> ResultBatch:
    """Restore one row to what the engine produced."""
    target = batch.row(row_id)
    restored = replace(
        target,
        frames=tuple(item.reset() for item in target.frames),
        answer=None if target.answer is None else target.answer.reset(),
    )
    rows = tuple(restored if row.row_id == row_id else row for row in batch.rows)
    return replace(batch, rows=rows)


def reset_batch(batch: ResultBatch) -> ResultBatch:
    """Restore every row of the batch."""
    rows = tuple(
        replace(
            row,
            frames=tuple(item.reset() for item in row.frames),
            answer=None if row.answer is None else row.answer.reset(),
        )
        for row in batch.rows
    )
    return replace(batch, rows=rows)


class ResultBatchStore:
    """A small, bounded, in-memory store of recent batches. Session state only."""

    def __init__(self, limit: int = 24):
        self._lock = threading.RLock()
        self._batches: dict[str, ResultBatch] = {}
        self._order: list[str] = []
        self.limit = max(1, int(limit))

    def put(self, batch: ResultBatch) -> ResultBatch:
        with self._lock:
            if batch.result_id in self._batches:
                self._order.remove(batch.result_id)
            self._batches[batch.result_id] = batch
            self._order.append(batch.result_id)
            while len(self._order) > self.limit:
                self._batches.pop(self._order.pop(0), None)
            return batch

    def get(self, result_id: str) -> ResultBatch:
        with self._lock:
            batch = self._batches.get(str(result_id))
        if batch is None:
            raise ResultEditError(
                f"Unknown result batch {result_id!r}; re-run the query.",
                error_code="UNKNOWN_RESULT_BATCH",
            )
        return batch

    def update(self, batch: ResultBatch) -> ResultBatch:
        with self._lock:
            self._batches[batch.result_id] = batch
            return batch

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()
            self._order.clear()


__all__ = [
    "EDITABLE_FIELDS",
    "FIELD_ANSWER",
    "FIELD_FRAME",
    "EditedValue",
    "ResultBatch",
    "ResultBatchStore",
    "ResultEditError",
    "ResultRow",
    "apply_edit",
    "build_result_batch",
    "reset_batch",
    "reset_row",
]
