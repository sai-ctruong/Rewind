"""Official-style AIC 2026 preliminary-round metrics and submission export."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from evaluation.metrics import normalize_answer, token_f1

TOP_KS = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    def contains(self, frame_id: int | str) -> bool:
        try:
            value = int(float(frame_id))
        except (TypeError, ValueError):
            return False
        return self.start <= value <= self.end


@dataclass(frozen=True)
class KISGroundTruth:
    video_id: str
    ranges: tuple[FrameRange, ...]


@dataclass(frozen=True)
class QAGroundTruth:
    video_id: str
    ranges: tuple[FrameRange, ...]
    answers: tuple[str, ...]


@dataclass(frozen=True)
class TRAKEGroundTruth:
    video_id: str
    event_ranges: tuple[FrameRange, ...]


@dataclass(frozen=True)
class RankedAnswer:
    video_id: str
    frame_ids: tuple[str, ...]
    answer: Optional[str] = None

    @classmethod
    def from_row(cls, row: Sequence[object], *, has_answer: bool = False) -> "RankedAnswer":
        if has_answer:
            if len(row) < 3:
                raise ValueError("Q&A rows require video_id, frame_id, answer")
            return cls(str(row[0]), (str(row[1]),), str(row[2]))
        if len(row) < 2:
            raise ValueError("Rows require at least video_id and one frame_id")
        return cls(str(row[0]), tuple(str(x) for x in row[1:]))


def _frame_in_any(frame_id: str, ranges: Iterable[FrameRange]) -> bool:
    return any(r.contains(frame_id) for r in ranges)


def kis_r_score(pred: RankedAnswer, gt: KISGroundTruth) -> float:
    if pred.video_id != gt.video_id or not pred.frame_ids:
        return 0.0
    return 1.0 if _frame_in_any(pred.frame_ids[0], gt.ranges) else 0.0


def qa_answer_match(prediction: str, golds: Sequence[str], semantic_matcher: Optional[Callable[[str, str], bool]] = None) -> bool:
    for gold in golds:
        if semantic_matcher is not None and semantic_matcher(prediction, gold):
            return True
        if normalize_answer(prediction) == normalize_answer(gold):
            return True
        if token_f1(prediction, gold) >= 0.95:
            return True
    return False


def qa_r_score(
    pred: RankedAnswer,
    gt: QAGroundTruth,
    *,
    semantic_matcher: Optional[Callable[[str, str], bool]] = None,
) -> float:
    if pred.video_id != gt.video_id or not pred.frame_ids or pred.answer is None:
        return 0.0
    if not _frame_in_any(pred.frame_ids[0], gt.ranges):
        return 0.0
    return 1.0 if qa_answer_match(pred.answer, gt.answers, semantic_matcher) else 0.0


def is_structurally_valid_trake_row(
    frame_ids: Sequence[object], event_count: int
) -> bool:
    """One frame per event, exactly. Nothing shorter is a partial answer."""
    return int(event_count) > 0 and len(tuple(frame_ids)) == int(event_count)


def trake_r_score(pred: RankedAnswer, gt: TRAKEGroundTruth) -> float:
    """Fraction of events whose submitted frame falls in that event's range.

    A row with the wrong number of frames scores 0, not a partial credit. `zip` used to
    silently truncate here, so a three-frame row answering a four-event query was scored
    over the three events it happened to cover and looked merely imperfect rather than
    malformed. Length is a structural precondition, checked before any scoring.
    """
    if pred.video_id != gt.video_id:
        return 0.0
    if not gt.event_ranges:
        return 0.0
    if not is_structurally_valid_trake_row(pred.frame_ids, len(gt.event_ranges)):
        return 0.0
    hits = sum(
        1
        for frame_id, frame_range in zip(pred.frame_ids, gt.event_ranges)
        if frame_range.contains(frame_id)
    )
    return hits / len(gt.event_ranges)


def ranked_r_scores(task: str, predictions: Sequence[RankedAnswer], gt) -> list[float]:
    task = task.lower()
    if task == "kis":
        return [kis_r_score(p, gt) for p in predictions[:100]]
    if task in {"qa", "q&a", "vqa"}:
        return [qa_r_score(p, gt) for p in predictions[:100]]
    if task == "trake":
        return [trake_r_score(p, gt) for p in predictions[:100]]
    raise ValueError(f"Unknown AIC task: {task}")


def r_at_k(r_scores: Sequence[float], k: int) -> float:
    if not r_scores:
        return 0.0
    return max(r_scores[:k], default=0.0)


def final_score_from_r_scores(r_scores: Sequence[float], top_ks: Sequence[int] = TOP_KS) -> dict[str, float]:
    vals = {f"R@{k}": r_at_k(r_scores, k) for k in top_ks}
    vals["Final Score"] = sum(vals.values()) / len(top_ks) if top_ks else 0.0
    return vals


class GroundTruthRequired(RuntimeError):
    """Raised when a semantic metric is requested without official ground truth.

    Structural diagnostics (candidate counts, complete-sequence yield, latency, malformed
    rows) need no labels and are reported freely. R@k and Final Score are *semantic*
    metrics: without official AIC labels there is nothing to compare against, and
    substituting synthetic or self-generated labels would manufacture a result. This
    refuses instead.
    """

    def __init__(self, task: str = "", detail: str = "") -> None:
        super().__init__(
            "GROUND_TRUTH_REQUIRED: official AIC ground truth is needed to compute "
            f"semantic metrics{f' for {task}' if task else ''}. "
            "Structural diagnostics are available without it. "
            + detail
        )
        self.task = task
        self.error_code = "GROUND_TRUTH_REQUIRED"


def require_ground_truth(gt: Any, task: str = "") -> Any:
    """Refuse to score without real labels rather than degrading to a fake baseline."""
    if gt is None:
        raise GroundTruthRequired(task)
    ranges = getattr(gt, "ranges", None)
    event_ranges = getattr(gt, "event_ranges", None)
    if not ranges and not event_ranges:
        raise GroundTruthRequired(
            task, "The supplied ground truth carries no frame ranges."
        )
    return gt


def evaluate_query(task: str, predictions: Sequence[RankedAnswer], gt) -> dict[str, float]:
    """Official R@k and Final Score. Requires real ground truth; see `require_ground_truth`."""
    require_ground_truth(gt, task)
    return final_score_from_r_scores(ranked_r_scores(task, predictions, gt))


def mean_report(per_query_reports: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_query_reports:
        return {f"R@{k}": 0.0 for k in TOP_KS} | {"Final Score": 0.0, "n": 0}
    keys = [f"R@{k}" for k in TOP_KS] + ["Final Score"]
    out = {key: sum(r.get(key, 0.0) for r in per_query_reports) / len(per_query_reports) for key in keys}
    out["n"] = len(per_query_reports)
    return out


def parse_ground_truth_row(task: str, row: dict) -> KISGroundTruth | QAGroundTruth | TRAKEGroundTruth:
    task = task.lower()
    video_id = str(row["video_id"])
    if task == "trake":
        ranges: list[FrameRange] = []
        i = 1
        while f"start_{i}" in row and f"end_{i}" in row:
            ranges.append(FrameRange(int(row[f"start_{i}"]), int(row[f"end_{i}"])))
            i += 1
        return TRAKEGroundTruth(video_id, tuple(ranges))
    ranges = (FrameRange(int(row["start"]), int(row["end"])),)
    if task in {"qa", "q&a", "vqa"}:
        answers = tuple(a.strip() for a in str(row.get("answer", "")).split("|") if a.strip())
        return QAGroundTruth(video_id, ranges, answers)
    return KISGroundTruth(video_id, ranges)


class SubmissionStructureError(ValueError):
    """Raised when a row would be written with the wrong number of columns."""


def write_submission(
    rows: Sequence[Sequence[object]],
    path: str | Path,
    *,
    max_rows: int = 100,
    require_row_length: int | None = None,
) -> Path:
    """Write submission rows, optionally refusing a malformed shape.

    `require_row_length` is a local structural safety check, not the full submission
    validator (Phase 11). It exists so a TRAKE export cannot silently write a row with
    fewer frames than the query had events: the caller passes `1 + event_count` and a
    mismatch raises instead of producing a plausible-looking bad file.
    """
    path = Path(path)
    kept = list(rows[:max_rows])
    if require_row_length is not None:
        expected = int(require_row_length)
        for position, row in enumerate(kept):
            if len(list(row)) != expected:
                raise SubmissionStructureError(
                    f"Row {position} has {len(list(row))} columns but {expected} are "
                    "required; refusing to write a structurally invalid submission."
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for row in kept:
            writer.writerow(list(row))
    return path