"""Ground-truth files: schema, provenance, and the rules that keep them honest.

Two label sources exist and are never conflated:

`official`
    labels issued by the AIC organisers. None are present in this repository.

`private_dev`
    labels a human annotated locally for development. They are legitimate for measuring
    progress and illegitimate for claiming an AIC score. Every report built from them
    must carry `PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE`.

Two rules are enforced in code rather than left to discipline:

1. A file must declare its `label_source`. An undeclared file is refused, because an
   unlabelled provenance eventually gets quoted as official.
2. Nothing here generates a target. A label written by this system from its own
   predictions is not evidence about this system; `annotated_by: system` is rejected
   outright.

Without any file at all, semantic evaluation still raises `GROUND_TRUTH_REQUIRED`. This
module adds a way to *supply* labels; it never invents one.
"""
from __future__ import annotations

import json
from collections.abc import Mapping as Mapping_types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from aic2026.metrics import (
    FrameRange,
    GroundTruthRequired,
    KISGroundTruth,
    QAGroundTruth,
    TRAKEGroundTruth,
)

LABEL_SOURCE_OFFICIAL = "official"
LABEL_SOURCE_PRIVATE_DEV = "private_dev"
LABEL_SOURCES = (LABEL_SOURCE_OFFICIAL, LABEL_SOURCE_PRIVATE_DEV)

PRIVATE_GT_BANNER = "PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE"
OFFICIAL_GT_BANNER = "OFFICIAL AIC GROUND TRUTH"

SUPPORTED_TASKS = ("kis", "qa", "trake")

# A label produced by the system being measured is circular, so it is refused rather
# than trusted. Human annotators identify themselves however they like.
FORBIDDEN_ANNOTATORS = frozenset({"system", "engine", "model", "auto", "prediction", "self"})


class GroundTruthSchemaError(ValueError):
    """Raised when a ground-truth file is malformed, unlabelled, or self-generated."""


@dataclass(frozen=True)
class GroundTruthEntry:
    """One labelled query, in the shape the official metrics already understand."""

    query_id: str
    task: str
    video_id: str
    label_source: str
    query: str = ""
    frame_ranges: tuple[tuple[int, int], ...] = ()
    event_frame_ranges: tuple[tuple[int, int], ...] = ()
    events: tuple[str, ...] = ()
    event_text: str = ""
    question: str = ""
    answers: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_official(self) -> bool:
        return self.label_source == LABEL_SOURCE_OFFICIAL

    def to_metric_gt(self):
        """Convert to the metric object for this task. No scoring happens here."""
        if self.task == "kis":
            return KISGroundTruth(
                self.video_id, tuple(FrameRange(a, b) for a, b in self.frame_ranges)
            )
        if self.task == "qa":
            return QAGroundTruth(
                self.video_id,
                tuple(FrameRange(a, b) for a, b in self.frame_ranges),
                tuple(self.answers),
            )
        if self.task == "trake":
            return TRAKEGroundTruth(
                self.video_id, tuple(FrameRange(a, b) for a, b in self.event_frame_ranges)
            )
        raise GroundTruthSchemaError(f"Unknown task {self.task!r}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query_id": self.query_id,
            "task": self.task,
            "video_id": self.video_id,
            "label_source": self.label_source,
        }
        for name in ("query", "event_text", "question", "notes"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        if self.frame_ranges:
            payload["frame_ranges"] = [list(item) for item in self.frame_ranges]
        if self.event_frame_ranges:
            payload["event_frame_ranges"] = [list(item) for item in self.event_frame_ranges]
        if self.events:
            payload["events"] = list(self.events)
        if self.answers:
            payload["answers"] = list(self.answers)
        return payload


@dataclass(frozen=True)
class GroundTruthSet:
    """A file's worth of labels plus the provenance every report must repeat."""

    label_source: str
    entries: tuple[GroundTruthEntry, ...] = ()
    dataset: str = ""
    split: str = ""
    annotated_by: str = ""
    created_at: str = ""
    path: Optional[str] = None
    notes: str = ""

    def __iter__(self) -> Iterator[GroundTruthEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_official(self) -> bool:
        return self.label_source == LABEL_SOURCE_OFFICIAL

    @property
    def banner(self) -> str:
        return OFFICIAL_GT_BANNER if self.is_official else PRIVATE_GT_BANNER

    def for_task(self, task: str) -> tuple[GroundTruthEntry, ...]:
        return tuple(item for item in self.entries if item.task == str(task).lower())

    def counts(self) -> dict[str, int]:
        return {task: len(self.for_task(task)) for task in SUPPORTED_TASKS}

    def provenance(self) -> dict[str, Any]:
        """What every report built from this set has to state."""
        return {
            "label_source": self.label_source,
            "official": self.is_official,
            "banner": self.banner,
            "dataset": self.dataset,
            "split": self.split,
            "annotated_by": self.annotated_by,
            "created_at": self.created_at,
            "path": self.path,
            "entries": len(self.entries),
            "counts": self.counts(),
        }


def _ranges(value: Any, field_name: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise GroundTruthSchemaError(f"{field_name} must be a list of [start, end] pairs")
    out: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, Mapping_types):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise GroundTruthSchemaError(
                f"{field_name} entries must be [start, end] or {{start, end}}, got {item!r}"
            )
        try:
            start_i, end_i = int(start), int(end)
        except (TypeError, ValueError):
            raise GroundTruthSchemaError(
                f"{field_name} bounds must be integers, got {item!r}"
            ) from None
        if start_i < 0 or end_i < start_i:
            raise GroundTruthSchemaError(
                f"{field_name} needs 0 <= start <= end, got [{start_i}, {end_i}]"
            )
        out.append((start_i, end_i))
    return tuple(out)


def parse_entry(row: Any, *, label_source: str, index: int = 0) -> GroundTruthEntry:
    """Validate one labelled query. Every task keeps its own required fields."""
    if not isinstance(row, Mapping_types):
        raise GroundTruthSchemaError(f"entry {index} must be an object, got {type(row).__name__}")
    task = str(row.get("task", "")).strip().lower()
    if task not in SUPPORTED_TASKS:
        raise GroundTruthSchemaError(
            f"entry {index} has task {task!r}; supported tasks are {list(SUPPORTED_TASKS)}"
        )
    video_id = str(row.get("video_id", "")).strip()
    if not video_id:
        raise GroundTruthSchemaError(f"entry {index} is missing video_id")
    query_id = str(row.get("query_id", "") or f"{task}_{index}")

    frame_ranges = _ranges(row.get("frame_ranges"), f"entry {index} frame_ranges")
    event_ranges = _ranges(row.get("event_frame_ranges"), f"entry {index} event_frame_ranges")
    events = tuple(str(item) for item in (row.get("events") or ()))
    answers = tuple(
        str(item).strip() for item in (row.get("answers") or ()) if str(item).strip()
    )

    if task in {"kis", "qa"} and not frame_ranges:
        raise GroundTruthSchemaError(
            f"entry {index} ({task}) needs at least one frame range; a label with no "
            "range cannot decide whether a prediction is correct"
        )
    if task == "qa" and not answers:
        raise GroundTruthSchemaError(f"entry {index} (qa) needs at least one answer")
    if task == "trake":
        if not events:
            raise GroundTruthSchemaError(f"entry {index} (trake) needs its ordered events")
        if len(event_ranges) != len(events):
            raise GroundTruthSchemaError(
                f"entry {index} (trake) has {len(events)} event(s) but "
                f"{len(event_ranges)} event frame range(s); TRAKE is scored per event, so "
                "they must correspond one to one"
            )
    return GroundTruthEntry(
        query_id=query_id,
        task=task,
        video_id=video_id,
        label_source=label_source,
        query=str(row.get("query", "")),
        frame_ranges=frame_ranges,
        event_frame_ranges=event_ranges,
        events=events,
        event_text=str(row.get("event_text", "")),
        question=str(row.get("question", "")),
        answers=answers,
        notes=str(row.get("notes", "")),
    )


def parse_ground_truth(payload: Any, *, path: Optional[str] = None) -> GroundTruthSet:
    """Validate a whole ground-truth document, provenance first."""
    if not isinstance(payload, Mapping_types):
        raise GroundTruthSchemaError(
            "a ground-truth file must be an object with label_source and entries"
        )
    label_source = str(payload.get("label_source", "")).strip().lower()
    if label_source not in LABEL_SOURCES:
        raise GroundTruthSchemaError(
            f"label_source must be one of {list(LABEL_SOURCES)}, got "
            f"{payload.get('label_source')!r}. An undeclared source eventually gets "
            "quoted as an official AIC score."
        )
    annotated_by = str(payload.get("annotated_by", "")).strip()
    if annotated_by.lower() in FORBIDDEN_ANNOTATORS:
        raise GroundTruthSchemaError(
            f"annotated_by={annotated_by!r} is refused: a label produced by the system "
            "being measured is circular, not evidence."
        )
    rows = payload.get("entries")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise GroundTruthSchemaError("a ground-truth file must contain a non-empty entries list")
    entries = tuple(
        parse_entry(row, label_source=label_source, index=index)
        for index, row in enumerate(rows)
    )
    seen: set[str] = set()
    for entry in entries:
        if entry.query_id in seen:
            raise GroundTruthSchemaError(f"duplicate query_id {entry.query_id!r}")
        seen.add(entry.query_id)
    return GroundTruthSet(
        label_source=label_source,
        entries=entries,
        dataset=str(payload.get("dataset", "")),
        split=str(payload.get("split", "")),
        annotated_by=annotated_by,
        created_at=str(payload.get("created_at", "")),
        path=path,
        notes=str(payload.get("notes", "")),
    )


def load_ground_truth(path: str | Path) -> GroundTruthSet:
    """Read and validate a ground-truth file. A missing file is a refusal, not empty."""
    target = Path(path)
    if not target.is_file():
        raise GroundTruthRequired(
            detail=(
                f"No ground-truth file at {target}. Annotate one from "
                "evaluation/labels/template.jsonl; nothing here will invent labels."
            )
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GroundTruthSchemaError(f"{target} is not valid JSON: {exc}") from None
    return parse_ground_truth(payload, path=str(target))


def report_header(gt: Optional[GroundTruthSet]) -> dict[str, Any]:
    """The provenance block every semantic report must carry, GT present or not."""
    if gt is None:
        return {
            "ground_truth": None,
            "banner": "NO GROUND TRUTH — NO SEMANTIC RESULT",
            "official": False,
        }
    return {"ground_truth": gt.provenance(), "banner": gt.banner, "official": gt.is_official}


__all__ = [
    "FORBIDDEN_ANNOTATORS",
    "LABEL_SOURCES",
    "LABEL_SOURCE_OFFICIAL",
    "LABEL_SOURCE_PRIVATE_DEV",
    "OFFICIAL_GT_BANNER",
    "PRIVATE_GT_BANNER",
    "SUPPORTED_TASKS",
    "GroundTruthEntry",
    "GroundTruthSchemaError",
    "GroundTruthSet",
    "load_ground_truth",
    "parse_entry",
    "parse_ground_truth",
    "report_header",
]
