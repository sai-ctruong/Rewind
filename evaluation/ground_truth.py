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

import hashlib
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
FORBIDDEN_ANNOTATORS = frozenset(
    {"system", "engine", "model", "auto", "prediction", "self", "clip", "vlm", "generated"}
)

# A template row exists to show the shape of a label. It is NOT a label, and the whole
# point of marking it is that it must never unlock a semantic evaluation. Rows carrying
# this marker are parsed, reported, and excluded from every scored set.
EXAMPLE_MARKER = "EXAMPLE_ONLY"


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
    # One interval per event, in event order: what the official scorer consumes.
    event_frame_ranges: tuple[tuple[int, int], ...] = ()
    # Every interval the annotator recorded per event, including any second one. Kept
    # so a genuinely multi-interval event is not silently reduced in the record.
    event_range_groups: tuple[tuple[tuple[int, int], ...], ...] = ()
    events: tuple[str, ...] = ()
    event_text: str = ""
    question: str = ""
    answers: tuple[str, ...] = ()
    answer_type: str = ""
    annotated_by: str = ""
    notes: str = ""

    @property
    def is_official(self) -> bool:
        return self.label_source == LABEL_SOURCE_OFFICIAL

    @property
    def is_example(self) -> bool:
        """A template row showing the shape of a label. Never a label."""
        haystack = " ".join(
            (self.query_id, self.query, self.notes, self.annotated_by, self.event_text)
        ).upper()
        return EXAMPLE_MARKER in haystack

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
        for name in ("query", "event_text", "question", "answer_type", "annotated_by", "notes"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        if self.frame_ranges:
            payload["frame_ranges"] = [list(item) for item in self.frame_ranges]
        if self.event_range_groups:
            payload["event_frame_ranges"] = [
                [list(item) for item in group] for group in self.event_range_groups
            ]
        elif self.event_frame_ranges:
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
        return iter(self.real_entries)

    def __len__(self) -> int:
        return len(self.real_entries)

    @property
    def is_official(self) -> bool:
        return self.label_source == LABEL_SOURCE_OFFICIAL

    @property
    def banner(self) -> str:
        return OFFICIAL_GT_BANNER if self.is_official else PRIVATE_GT_BANNER

    @property
    def real_entries(self) -> tuple[GroundTruthEntry, ...]:
        """Human labels only. Template rows are excluded from everything scored."""
        return tuple(item for item in self.entries if not item.is_example)

    @property
    def example_entries(self) -> tuple[GroundTruthEntry, ...]:
        return tuple(item for item in self.entries if item.is_example)

    @property
    def has_real_labels(self) -> bool:
        return bool(self.real_entries)

    def for_task(self, task: str) -> tuple[GroundTruthEntry, ...]:
        """Real labels for one task. Templates never appear here."""
        return tuple(item for item in self.real_entries if item.task == str(task).lower())

    def counts(self) -> dict[str, int]:
        return {task: len(self.for_task(task)) for task in SUPPORTED_TASKS}

    def example_counts(self) -> dict[str, int]:
        return {
            task: len([e for e in self.example_entries if e.task == task])
            for task in SUPPORTED_TASKS
        }

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
            "entries": len(self.real_entries),
            "rows_in_file": len(self.entries),
            "example_rows_ignored": len(self.example_entries),
            "counts": self.counts(),
            "example_counts": self.example_counts(),
        }

    def content_hash(self) -> str:
        """Stable hash of the REAL labels, for the experiment manifest.

        Templates are excluded, so adding or editing an example cannot make two runs
        look like they scored different label sets.
        """
        payload = json.dumps(
            [item.to_dict() for item in self.real_entries],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _event_range_groups(
    value: Any, field_name: str
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Per-event interval lists, accepting both shapes a human might write.

    The documented private-dev shape gives each event its own LIST of intervals::

        "event_frame_ranges": [[[s1, e1]], [[s2, e2], [s2b, e2b]]]

    The older flat shape gives each event exactly one interval::

        "event_frame_ranges": [[s1, e1], [s2, e2]]

    Both are accepted and neither is silently rewritten into the other: an event with
    two genuine intervals keeps both, and the official scorer is given the event's
    first interval per its own one-interval-per-event contract.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise GroundTruthSchemaError(f"{field_name} must be a list, one entry per event")
    groups: list[tuple[tuple[int, int], ...]] = []
    for position, item in enumerate(value):
        label = f"{field_name}[{position}]"
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(bound, (int, float)) for bound in item)
        ):
            groups.append(_ranges([item], label))  # flat shape: one interval
        else:
            group = _ranges(item, label)
            if not group:
                raise GroundTruthSchemaError(
                    f"{label} is empty; every event needs at least one interval"
                )
            groups.append(group)
    return tuple(groups)


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
    # Per-row provenance, checked with the same rule as the file header: a row is only
    # a label if a human says they made it.
    annotator = str(row.get("annotated_by", "")).strip()
    if annotator.lower() in FORBIDDEN_ANNOTATORS:
        raise GroundTruthSchemaError(
            f"entry {index} has annotated_by={annotator!r}: a label produced by the "
            "system being measured is circular, not evidence."
        )

    frame_ranges = _ranges(row.get("frame_ranges"), f"entry {index} frame_ranges")
    event_groups = _event_range_groups(
        row.get("event_frame_ranges"), f"entry {index} event_frame_ranges"
    )
    # The official scorer takes one interval per event; a second interval on an event is
    # kept in `event_range_groups` for the annotator's record, not quietly merged away.
    event_ranges = tuple(group[0] for group in event_groups)
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
        event_range_groups=event_groups,
        events=events,
        # `event_description` is the documented private-dev field name; `event_text` is
        # the older one. Both mean the same thing and both are accepted.
        event_text=str(row.get("event_description", row.get("event_text", ""))),
        question=str(row.get("question", "")),
        answers=answers,
        answer_type=str(row.get("answer_type", "")),
        annotated_by=str(row.get("annotated_by", "")),
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


PRIVATE_DEV_DIR = Path(__file__).resolve().parent / "private_dev"
PRIVATE_DEV_FILES = {"kis": "kis.json", "qa": "qa.json", "trake": "trake.json"}

SPLIT_DEVELOPMENT = "development"
SPLIT_HOLDOUT = "holdout"
SPLITS = (SPLIT_DEVELOPMENT, SPLIT_HOLDOUT)


def _task_from_shape(row: Mapping_types) -> Optional[str]:
    """The task a row's own fields imply, or None when they imply nothing."""
    if row.get("events") or row.get("event_frame_ranges"):
        return "trake"
    if row.get("answers") or row.get("question") or row.get("event_description"):
        return "qa"
    return None


def load_private_dev(
    directory: str | Path = PRIVATE_DEV_DIR, *, split: Optional[str] = None
) -> GroundTruthSet:
    """Load the three private-development task files as one labelled set.

    A missing directory or a file with no HUMAN rows is not an error here — that is the
    normal state before anyone has annotated anything. What it produces is an empty set,
    and an empty set is what makes the semantic evaluator refuse. Template rows are
    parsed for their shape and then excluded.
    """
    root = Path(directory)
    entries: list[GroundTruthEntry] = []
    examples: list[GroundTruthEntry] = []
    annotators: set[str] = set()
    for task, filename in PRIVATE_DEV_FILES.items():
        path = root / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GroundTruthSchemaError(f"{path} is not valid JSON: {exc}") from None
        if not isinstance(payload, Mapping_types):
            raise GroundTruthSchemaError(f"{path} must be an object with an entries list")
        source = str(payload.get("label_source", LABEL_SOURCE_PRIVATE_DEV)).strip().lower()
        if source != LABEL_SOURCE_PRIVATE_DEV:
            raise GroundTruthSchemaError(
                f"{path} declares label_source={source!r}; files under private_dev/ must "
                f"declare {LABEL_SOURCE_PRIVATE_DEV!r}."
            )
        for index, row in enumerate(payload.get("entries") or ()):
            if isinstance(row, Mapping_types):
                declared = str(row.get("task", "")).strip().lower()
                # A row's own fields say what task it is. Trusting the filename alone
                # would let a Q&A row dropped into kis.json parse as a KIS label, quietly
                # discarding its question and answers.
                shaped = _task_from_shape(row)
                if declared and declared != task:
                    raise GroundTruthSchemaError(
                        f"{path} entry {index} declares task {declared!r} but lives in "
                        f"the {task} file; one file holds one task."
                    )
                if shaped and shaped != task:
                    raise GroundTruthSchemaError(
                        f"{path} entry {index} looks like a {shaped} row (it carries "
                        f"{shaped}-only fields) but lives in the {task} file; one file "
                        "holds one task."
                    )
                row = {**row, "task": task}
            entry = parse_entry(row, label_source=source, index=index)
            if split is not None and str(row.get("split", SPLIT_DEVELOPMENT)) != split:
                continue
            (examples if entry.is_example else entries).append(entry)
            if entry.annotated_by:
                annotators.add(entry.annotated_by)

    seen: set[str] = set()
    for entry in entries:
        if entry.query_id in seen:
            raise GroundTruthSchemaError(f"duplicate query_id {entry.query_id!r}")
        seen.add(entry.query_id)

    return GroundTruthSet(
        label_source=LABEL_SOURCE_PRIVATE_DEV,
        entries=tuple(entries + examples),
        dataset="AIC 2026 local visual-development subset",
        split=split or "all",
        annotated_by=", ".join(sorted(annotators)),
        path=str(root),
    )


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
    "EXAMPLE_MARKER",
    "FORBIDDEN_ANNOTATORS",
    "PRIVATE_DEV_DIR",
    "PRIVATE_DEV_FILES",
    "SPLITS",
    "SPLIT_DEVELOPMENT",
    "SPLIT_HOLDOUT",
    "load_private_dev",
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
