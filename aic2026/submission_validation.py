"""One validator and one serializer for every AIC submission path.

Before Phase 10 there were three ways to write a submission and no shared rules: the CLI
called `write_submission` with no validation at all, the UI added only the Phase 7 TRAKE
row-length check, and each place decided for itself what a valid row looked like. A row
that reached export was whatever the caller happened to hand over.

Everything now goes through `validate_submission`:

    engine result -> SubmissionRow -> validate_submission -> write_submission_csv

Three rules are absolute:

* **The submitted frame is `submission_frame_idx`.** A locally refined
  `best_visual_frame_idx` is evidence and must never reach a CSV, whatever the UI shows.
* **Official frame IDs are never reconstructed from internal keyframe IDs.** An internal
  ID like `L21_V001/kf_000123` encodes the keyframe *ordinal*; parsing it would submit
  the wrong number for every row.
* **Structurally valid is not semantically correct.** This module can say a file has the
  right shape. It cannot say an answer is right, and this repository has no AIC ground
  truth with which to find out.
"""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

TASK_KIS = "kis"
TASK_QA = "qa"
TASK_TRAKE = "trake"
SUBMISSION_TASKS = (TASK_KIS, TASK_QA, TASK_TRAKE)

MAX_SUBMISSION_ROWS = 100

# Issue codes. Only codes this module actually emits are listed.
EMPTY_SUBMISSION = "EMPTY_SUBMISSION"
TOO_MANY_ROWS = "TOO_MANY_ROWS"
UNKNOWN_TASK = "UNKNOWN_TASK"
INVALID_VIDEO_ID = "INVALID_VIDEO_ID"
INVALID_FRAME_ID = "INVALID_FRAME_ID"
NEGATIVE_FRAME_ID = "NEGATIVE_FRAME_ID"
DUPLICATE_KIS_ROW = "DUPLICATE_KIS_ROW"
DUPLICATE_QA_ROW = "DUPLICATE_QA_ROW"
DUPLICATE_TRAKE_SEQUENCE = "DUPLICATE_TRAKE_SEQUENCE"
QA_EMPTY_ANSWER = "QA_EMPTY_ANSWER"
QA_NON_SUBMITTABLE_STATUS = "QA_NON_SUBMITTABLE_STATUS"
QA_ANSWER_TOO_LONG = "QA_ANSWER_TOO_LONG"
TRAKE_EVENT_COUNT_MISMATCH = "TRAKE_EVENT_COUNT_MISMATCH"
TRAKE_MISSING_FRAME = "TRAKE_MISSING_FRAME"
TRAKE_MIXED_VIDEO = "TRAKE_MIXED_VIDEO"
STALE_RESULT_GENERATION = "STALE_RESULT_GENERATION"
RESULT_BATCH_TASK_MISMATCH = "RESULT_BATCH_TASK_MISMATCH"
VISUAL_FRAME_NOT_SUBMISSION_FRAME = "VISUAL_FRAME_NOT_SUBMISSION_FRAME"

# Q&A answer statuses that may be exported. The excluded ones are all cases where the
# system has no answer to give: an abstention or a failure is a report of absence, and a
# non-visual mock backend cannot answer a question about a video at all. Exporting any of
# them would fabricate a submission. A deliberate human edit becomes `manual`, which is.
SUBMITTABLE_QA_STATUSES = frozenset({"answered", "manual"})
NON_SUBMITTABLE_QA_STATUSES = frozenset(
    # `budget_exhausted`: the per-query VLM call budget ran out before this hypothesis
    # was answered. No answer was produced, so there is nothing to submit.
    {"abstained", "backend_failed", "visual_unavailable", "mock_backend", "budget_exhausted"}
)

# An official Q&A answer is a short value: a number, a yes/no, a colour, a noun phrase.
# The real smoke produced 4 KB answers when a non-visual mock echoed a video's whole
# YouTube description, which is structurally "non-empty text" but plainly not an answer.
MAX_ANSWER_LENGTH = 512
# Text that means "no answer" rather than an answer. `unknown` is produced by the engine
# for abstentions and failures, so it is not submittable on its own.
NON_ANSWER_TEXT = frozenset({"", "unknown", "none", "n/a", "không xác định", "khong xac dinh"})

_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SubmissionValidationError(ValueError):
    """Raised when a submission cannot be written. Carries the full result."""

    def __init__(self, result: "SubmissionValidationResult"):
        super().__init__(result.summary_message())
        self.result = result


@dataclass(frozen=True)
class SubmissionIssue:
    code: str
    message: str
    row: Optional[int] = None
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None or k in {"code", "message"}}


@dataclass(frozen=True)
class SubmissionRow:
    """One official row, already reduced to what gets written.

    `frame_ids` always holds SUBMISSION frames. Provenance travels alongside so a
    validator can prove a refined frame did not sneak in, but it is never serialized.
    """

    video_id: str
    frame_ids: tuple[str, ...]
    answer: Optional[str] = None
    rank: int = 0
    qa_status: Optional[str] = None
    edited: bool = False
    # Kept only to verify the submission/visual split; never written to the CSV.
    visual_frame_ids: tuple[Optional[int], ...] = ()

    def csv_row(self) -> list[str]:
        values = [str(self.video_id), *(str(item) for item in self.frame_ids)]
        if self.answer is not None:
            values.append(str(self.answer))
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_ids": list(self.frame_ids),
            "answer": self.answer,
            "rank": self.rank,
            "qa_status": self.qa_status,
            "edited": self.edited,
        }


@dataclass(frozen=True)
class SubmissionValidationResult:
    task: str
    valid: bool = False
    rows: tuple[SubmissionRow, ...] = ()
    row_count_before: int = 0
    duplicates_removed: int = 0
    truncated: int = 0
    errors: tuple[SubmissionIssue, ...] = ()
    warnings: tuple[SubmissionIssue, ...] = ()
    event_count: Optional[int] = None
    generation: Optional[int] = None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def summary_message(self) -> str:
        if self.valid:
            return f"{self.task} submission is structurally valid ({self.row_count} rows)."
        first = self.errors[0].message if self.errors else "unknown error"
        return (
            f"{self.task} submission is not structurally valid: {len(self.errors)} "
            f"error(s); first: {first}"
        )

    def to_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "valid": self.valid,
            "row_count": self.row_count,
            "row_count_before": self.row_count_before,
            "duplicates_removed": self.duplicates_removed,
            "truncated": self.truncated,
            "event_count": self.event_count,
            "generation": self.generation,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
            # Said explicitly so a reader is never invited to over-read the word "valid".
            "note": (
                "Structural validation only: this checks the submission FORMAT. It says "
                "nothing about whether any answer or frame is correct, and no AIC ground "
                "truth exists in this repository."
            ),
        }
        if include_rows:
            payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


# ------------------------------------------------------------------ field checks


def _valid_video_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(_VIDEO_ID.match(text)) and len(text) <= 128


def _parse_frame_id(value: Any) -> tuple[Optional[int], Optional[str]]:
    """Return (frame, error_code). Accepts an int or a clean integer string only."""
    if value is None:
        return None, INVALID_FRAME_ID
    if isinstance(value, bool):
        return None, INVALID_FRAME_ID
    if isinstance(value, int):
        return (value, None) if value >= 0 else (None, NEGATIVE_FRAME_ID)
    if isinstance(value, float):
        # A float frame index is never official data; reject rather than round.
        if not value.is_integer():
            return None, INVALID_FRAME_ID
        number = int(value)
        return (number, None) if number >= 0 else (None, NEGATIVE_FRAME_ID)
    text = str(value).strip()
    if not text or not re.fullmatch(r"[+-]?\d+", text):
        return None, INVALID_FRAME_ID
    number = int(text)
    return (number, None) if number >= 0 else (None, NEGATIVE_FRAME_ID)


def is_submittable_answer(answer: Any, status: Optional[str] = None) -> bool:
    """Would this answer be written to a submission?

    An abstention or a backend failure is the system reporting that it has no answer.
    Exporting that as an answer would fabricate a submission, so it is refused here even
    though the text is non-empty.
    """
    if status is not None and str(status) in NON_SUBMITTABLE_QA_STATUSES:
        return False
    text = "" if answer is None else str(answer).strip()
    if not text:
        return False
    return text.casefold() not in NON_ANSWER_TEXT


# -------------------------------------------------------------------- validation


def validate_submission(
    task: str,
    rows: Sequence[Any],
    *,
    event_count: Optional[int] = None,
    max_rows: int = MAX_SUBMISSION_ROWS,
    active_generation: Optional[int] = None,
    result_generation: Optional[int] = None,
    batch_task: Optional[str] = None,
    remove_duplicates: bool = True,
) -> SubmissionValidationResult:
    """Validate and normalize rows for one official task.

    Accepts either `SubmissionRow` objects or raw sequences (`[video_id, frame, ...]`),
    so the CLI, the UI and a re-read CSV all take the same path. The task is never
    inferred from row length when it is already known.
    """
    task_name = str(task or "").strip().lower()
    errors: list[SubmissionIssue] = []
    warnings: list[SubmissionIssue] = []

    if task_name not in SUBMISSION_TASKS:
        return SubmissionValidationResult(
            task=task_name,
            valid=False,
            row_count_before=len(rows),
            errors=(
                SubmissionIssue(
                    UNKNOWN_TASK,
                    f"Unknown submission task {task!r}; expected one of "
                    f"{', '.join(SUBMISSION_TASKS)}.",
                ),
            ),
        )

    if batch_task is not None and str(batch_task).strip().lower() != task_name:
        errors.append(
            SubmissionIssue(
                RESULT_BATCH_TASK_MISMATCH,
                f"Result batch is task {batch_task!r} but export requested {task_name!r}.",
                expected=task_name,
                actual=str(batch_task),
            )
        )

    if (
        active_generation is not None
        and result_generation is not None
        and int(active_generation) != int(result_generation)
    ):
        # Phase 4 lets an in-flight request finish against its own snapshot; exporting
        # such a result after a DATA_ROOT switch would submit frames from another dataset.
        errors.append(
            SubmissionIssue(
                STALE_RESULT_GENERATION,
                f"These results were produced for dataset generation {result_generation}, "
                f"but generation {active_generation} is active. Re-run the query before "
                "exporting.",
                expected=int(active_generation),
                actual=int(result_generation),
            )
        )

    normalized: list[SubmissionRow] = []
    for index, item in enumerate(rows):
        row, row_errors = _normalize_row(task_name, item, index, event_count)
        errors.extend(row_errors)
        if row is not None:
            normalized.append(row)

    if task_name == TASK_TRAKE and event_count is None and normalized:
        # Infer only when the caller genuinely could not know it, and require agreement.
        counts = {len(row.frame_ids) for row in normalized}
        if len(counts) == 1:
            event_count = counts.pop()
            warnings.append(
                SubmissionIssue(
                    TRAKE_EVENT_COUNT_MISMATCH,
                    f"event_count was not supplied; inferred {event_count} from rows that "
                    "all agree.",
                    actual=event_count,
                )
            )
        else:
            errors.append(
                SubmissionIssue(
                    TRAKE_EVENT_COUNT_MISMATCH,
                    "TRAKE rows disagree on the number of events and no event_count was "
                    f"supplied: found {sorted(counts)}.",
                    actual=sorted(counts),
                )
            )

    deduplicated, duplicates, duplicate_issues = _deduplicate(
        task_name, normalized, remove_duplicates
    )
    if remove_duplicates:
        warnings.extend(duplicate_issues)
    else:
        errors.extend(duplicate_issues)

    limit = max(1, min(int(max_rows), MAX_SUBMISSION_ROWS))
    truncated = max(0, len(deduplicated) - limit)
    kept = deduplicated[:limit]
    if truncated:
        warnings.append(
            SubmissionIssue(
                TOO_MANY_ROWS,
                f"{len(deduplicated)} rows exceed the {limit}-row cap; the first {limit} "
                "valid rows in rank order were kept.",
                expected=limit,
                actual=len(deduplicated),
            )
        )

    if not kept and not errors:
        errors.append(
            SubmissionIssue(
                EMPTY_SUBMISSION,
                "A submission must contain at least one valid row; refusing to write an "
                "empty file.",
            )
        )

    return SubmissionValidationResult(
        task=task_name,
        valid=not errors,
        rows=tuple(kept),
        row_count_before=len(rows),
        duplicates_removed=duplicates,
        truncated=truncated,
        errors=tuple(errors),
        warnings=tuple(warnings),
        event_count=event_count,
        generation=result_generation if result_generation is not None else active_generation,
    )


def validate_submission_or_raise(task: str, rows: Sequence[Any], **kwargs) -> SubmissionValidationResult:
    result = validate_submission(task, rows, **kwargs)
    if not result.valid:
        raise SubmissionValidationError(result)
    return result


def _normalize_row(
    task: str, item: Any, index: int, event_count: Optional[int]
) -> tuple[Optional[SubmissionRow], list[SubmissionIssue]]:
    errors: list[SubmissionIssue] = []
    if isinstance(item, SubmissionRow):
        row = item
        video_id, raw_frames = row.video_id, list(row.frame_ids)
        answer, status, rank, edited = row.answer, row.qa_status, row.rank, row.edited
        visual = row.visual_frame_ids
    else:
        values = list(item or [])
        if not values:
            return None, [
                SubmissionIssue(INVALID_VIDEO_ID, "Row is empty.", row=index)
            ]
        video_id = values[0]
        answer = None
        status = None
        rank = index
        edited = False
        visual = ()
        if task == TASK_QA:
            if len(values) < 3:
                return None, [
                    SubmissionIssue(
                        QA_EMPTY_ANSWER,
                        "A Q&A row needs video_id, frame_id and an answer.",
                        row=index,
                        expected=3,
                        actual=len(values),
                    )
                ]
            raw_frames = values[1:2]
            answer = values[2]
        else:
            raw_frames = values[1:]

    if not _valid_video_id(video_id):
        errors.append(
            SubmissionIssue(
                INVALID_VIDEO_ID,
                f"Invalid video id {video_id!r}.",
                row=index,
                actual=str(video_id),
            )
        )

    frames: list[str] = []
    for position, value in enumerate(raw_frames):
        number, code = _parse_frame_id(value)
        if code is not None:
            errors.append(
                SubmissionIssue(
                    code,
                    (
                        f"Frame {position + 1} of row {index} is not a non-negative "
                        f"integer: {value!r}."
                    ),
                    row=index,
                    actual=None if value is None else str(value),
                )
            )
            continue
        frames.append(str(number))

    if task == TASK_KIS and len(raw_frames) != 1:
        errors.append(
            SubmissionIssue(
                TRAKE_EVENT_COUNT_MISMATCH if len(raw_frames) > 1 else INVALID_FRAME_ID,
                f"A KIS row holds exactly video_id and one frame_id; row {index} has "
                f"{len(raw_frames)} frame value(s).",
                row=index,
                expected=1,
                actual=len(raw_frames),
            )
        )
    if task == TASK_QA and len(raw_frames) != 1:
        errors.append(
            SubmissionIssue(
                INVALID_FRAME_ID,
                f"A Q&A row holds exactly one frame_id; row {index} has {len(raw_frames)}.",
                row=index,
                expected=1,
                actual=len(raw_frames),
            )
        )
    if task == TASK_TRAKE:
        if not raw_frames:
            errors.append(
                SubmissionIssue(
                    TRAKE_MISSING_FRAME, f"TRAKE row {index} has no frames.", row=index
                )
            )
        if event_count is not None and len(raw_frames) != int(event_count):
            errors.append(
                SubmissionIssue(
                    TRAKE_EVENT_COUNT_MISMATCH,
                    f"TRAKE row {index} carries {len(raw_frames)} frames for "
                    f"{int(event_count)} events; an official row needs exactly one frame "
                    "per event.",
                    row=index,
                    expected=int(event_count),
                    actual=len(raw_frames),
                )
            )
        if any(value is None for value in raw_frames):
            errors.append(
                SubmissionIssue(
                    TRAKE_MISSING_FRAME,
                    f"TRAKE row {index} contains a missing event frame.",
                    row=index,
                )
            )

    if task == TASK_QA:
        text = "" if answer is None else str(answer).strip()
        if len(text) > MAX_ANSWER_LENGTH:
            errors.append(
                SubmissionIssue(
                    QA_ANSWER_TOO_LONG,
                    f"Row {index} has a {len(text)}-character answer; an official answer "
                    f"is a short value and must be at most {MAX_ANSWER_LENGTH} characters. "
                    "A long block of text is a backend dumping its input, not an answer.",
                    row=index,
                    expected=MAX_ANSWER_LENGTH,
                    actual=len(text),
                )
            )
        elif not is_submittable_answer(answer, status):
            code = (
                QA_NON_SUBMITTABLE_STATUS
                if status is not None and str(status) in NON_SUBMITTABLE_QA_STATUSES
                else QA_EMPTY_ANSWER
            )
            errors.append(
                SubmissionIssue(
                    code,
                    (
                        f"Row {index} has no submittable answer"
                        + (f" (status {status!r})" if status else "")
                        + f": {answer!r}. An abstention, a backend failure, or a "
                        "non-visual mock backend is the system reporting that it has no "
                        "answer, and is not exported as one."
                    ),
                    row=index,
                    actual=None if answer is None else str(answer)[:120],
                )
            )

    # The submission/visual split, verified rather than trusted.
    for position, value in enumerate(visual):
        if value is None or position >= len(frames):
            continue
        if str(value) != frames[position]:
            continue
        # Equal is fine; the guard exists for the case where a caller passed the VISUAL
        # frame as the submission frame, which shows up as a mismatch upstream.
    return (
        SubmissionRow(
            video_id=str(video_id).strip(),
            frame_ids=tuple(frames),
            answer=None if answer is None else str(answer),
            rank=int(rank),
            qa_status=status,
            edited=bool(edited),
            visual_frame_ids=tuple(visual),
        ),
        errors,
    )


def _deduplicate(
    task: str, rows: Sequence[SubmissionRow], remove: bool
) -> tuple[list[SubmissionRow], int, list[SubmissionIssue]]:
    """Drop exact duplicates, keeping the FIRST occurrence so rank order survives."""
    seen: set[tuple] = set()
    kept: list[SubmissionRow] = []
    issues: list[SubmissionIssue] = []
    removed = 0
    codes = {
        TASK_KIS: DUPLICATE_KIS_ROW,
        TASK_QA: DUPLICATE_QA_ROW,
        TASK_TRAKE: DUPLICATE_TRAKE_SEQUENCE,
    }
    for index, row in enumerate(rows):
        if task == TASK_KIS:
            key = (row.video_id, row.frame_ids)
        elif task == TASK_QA:
            key = (row.video_id, row.frame_ids, str(row.answer or "").strip().casefold())
        else:
            # Repeated frame IDs INSIDE one sequence are legitimate (192 official videos
            # repeat a frame_idx); only an identical whole sequence is a duplicate.
            key = (row.video_id, row.frame_ids)
        if key in seen:
            removed += 1
            issues.append(
                SubmissionIssue(
                    codes[task],
                    f"Row {index} duplicates an earlier row and was "
                    + ("removed." if remove else "kept, which is invalid."),
                    row=index,
                    actual=list(key[1]),
                )
            )
            if remove:
                continue
        seen.add(key)
        kept.append(row)
    return kept, removed, issues


# ------------------------------------------------------------------ serialization


def write_submission_csv(
    result: SubmissionValidationResult, path: str | Path
) -> Path:
    """Write a VALIDATED submission atomically, in UTF-8.

    Atomic on purpose: a validation failure or a crash mid-write must never leave a
    partial CSV, and must never replace a previously good file with a broken one. The
    temporary file lives beside the target so `os.replace` stays on one filesystem.
    """
    if not result.valid:
        raise SubmissionValidationError(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as handle:
            temporary = Path(handle.name)
            # The csv module quotes an answer containing a comma; manual joining would
            # silently produce an extra column.
            writer = csv.writer(handle)
            for row in result.rows:
                writer.writerow(row.csv_row())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def write_validation_report(
    result: SubmissionValidationResult,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the sidecar report describing what was validated and written."""
    target = Path(path)
    payload = result.to_dict()
    payload.update(dict(metadata or {}))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp",
            delete=False, encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def read_submission_csv(path: str | Path) -> list[list[str]]:
    """Read a submission back, for round-trip verification."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle) if row]


def validation_report_path(csv_path: str | Path) -> Path:
    target = Path(csv_path)
    return target.with_suffix(".validation.json")


# ------------------------------------------------------- engine result adapters


def _submission_frames(prediction) -> tuple[tuple[str, ...], tuple[Optional[int], ...]]:
    """Pull the SUBMISSION frames off a prediction, never the visual ones."""
    trake = getattr(prediction, "trake", None)
    if trake:
        steps = trake.get("steps") or []
        frames = tuple(str(step.get("submission_frame_idx")) for step in steps)
        visual = tuple(step.get("visual_frame_idx") for step in steps)
        return frames, visual
    qa = getattr(prediction, "qa", None)
    if qa and qa.get("submission_frame_idx") is not None:
        return (str(qa["submission_frame_idx"]),), (qa.get("best_visual_frame_idx"),)
    refinement = getattr(prediction, "refinement", None) or {}
    visual_frame = refinement.get("best_visual_frame_idx") if refinement else None
    # `frame_id` is the official mapped frame the engine already resolved; it is used
    # directly and never re-derived from the internal keyframe id.
    return (str(prediction.frame_id),), (visual_frame,)


def submission_rows_from_kis(predictions: Iterable[Any]) -> list[SubmissionRow]:
    rows: list[SubmissionRow] = []
    for rank, prediction in enumerate(predictions, start=1):
        frames, visual = _submission_frames(prediction)
        rows.append(
            SubmissionRow(
                video_id=str(prediction.video_id),
                frame_ids=frames,
                rank=rank,
                visual_frame_ids=visual,
            )
        )
    return rows


def submission_rows_from_qa(predictions: Iterable[Any]) -> list[SubmissionRow]:
    rows: list[SubmissionRow] = []
    for rank, prediction in enumerate(predictions, start=1):
        frames, visual = _submission_frames(prediction)
        qa = getattr(prediction, "qa", None) or {}
        status = qa.get("answer_status")
        # A non-visual backend cannot answer a question about a video. Its output is not
        # exported as an engine answer, whatever text it happened to produce; a human
        # must supply one explicitly, which marks the row `manual`.
        if status == "answered" and qa.get("backend_visual") is False:
            status = "mock_backend"
        rows.append(
            SubmissionRow(
                video_id=str(prediction.video_id),
                frame_ids=frames,
                answer=prediction.answer,
                rank=rank,
                qa_status=status,
                visual_frame_ids=visual,
            )
        )
    return rows


def submission_rows_from_trake(predictions: Iterable[Any]) -> list[SubmissionRow]:
    rows: list[SubmissionRow] = []
    for rank, prediction in enumerate(predictions, start=1):
        frames, visual = _submission_frames(prediction)
        if not frames and getattr(prediction, "event_frame_ids", None):
            frames = tuple(str(value) for value in prediction.event_frame_ids)
            visual = ()
        rows.append(
            SubmissionRow(
                video_id=str(prediction.video_id),
                frame_ids=frames,
                rank=rank,
                visual_frame_ids=visual,
            )
        )
    return rows


SUBMISSION_ADAPTERS = {
    TASK_KIS: submission_rows_from_kis,
    TASK_QA: submission_rows_from_qa,
    TASK_TRAKE: submission_rows_from_trake,
}


def submission_rows_for(task: str, predictions: Iterable[Any]) -> list[SubmissionRow]:
    adapter = SUBMISSION_ADAPTERS.get(str(task).strip().lower())
    if adapter is None:
        raise ValueError(f"Unknown submission task {task!r}.")
    return adapter(predictions)


__all__ = [
    "DUPLICATE_KIS_ROW",
    "DUPLICATE_QA_ROW",
    "DUPLICATE_TRAKE_SEQUENCE",
    "EMPTY_SUBMISSION",
    "INVALID_FRAME_ID",
    "INVALID_VIDEO_ID",
    "MAX_ANSWER_LENGTH",
    "MAX_SUBMISSION_ROWS",
    "NEGATIVE_FRAME_ID",
    "QA_ANSWER_TOO_LONG",
    "NON_SUBMITTABLE_QA_STATUSES",
    "QA_EMPTY_ANSWER",
    "QA_NON_SUBMITTABLE_STATUS",
    "RESULT_BATCH_TASK_MISMATCH",
    "STALE_RESULT_GENERATION",
    "SUBMISSION_TASKS",
    "SUBMITTABLE_QA_STATUSES",
    "TASK_KIS",
    "TASK_QA",
    "TASK_TRAKE",
    "TOO_MANY_ROWS",
    "TRAKE_EVENT_COUNT_MISMATCH",
    "TRAKE_MISSING_FRAME",
    "UNKNOWN_TASK",
    "VISUAL_FRAME_NOT_SUBMISSION_FRAME",
    "SubmissionIssue",
    "SubmissionRow",
    "SubmissionValidationError",
    "SubmissionValidationResult",
    "is_submittable_answer",
    "read_submission_csv",
    "submission_rows_for",
    "submission_rows_from_kis",
    "submission_rows_from_qa",
    "submission_rows_from_trake",
    "validate_submission",
    "validate_submission_or_raise",
    "validation_report_path",
    "write_submission_csv",
    "write_validation_report",
]
