"""Strict validation and deterministic inspection for the official AIC dataset."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION

from .config import AppConfig, DatasetScopeConfig
from .dataset_scope import (
    DatasetScopeError,
    excluded_video_ids,
    hash_selected_video_ids,
    normalize_scope,
    scope_payload,
    select_video_ids,
)

DATASET_INSPECTION_SCHEMA_VERSION = 2
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAP_REQUIRED_COLUMNS = ("n", "pts_time", "fps", "frame_idx")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class DatasetError(RuntimeError):
    """Base class for AIC dataset failures."""


class DatasetLayoutError(DatasetError):
    """Raised when DATA_ROOT itself is unavailable or unusable."""


class DatasetSourceError(DatasetError):
    """Raised when a dataset source cannot be read safely."""


class DatasetAlignmentError(DatasetError):
    """Raised when one video's map rows and feature vectors do not align."""

    def __init__(
        self,
        *,
        video_id: str,
        map_rows: int,
        feature_vectors: int,
        map_path: str | Path,
        feature_path: str | Path,
    ) -> None:
        self.video_id = video_id
        self.map_rows = int(map_rows)
        self.feature_vectors = int(feature_vectors)
        self.map_path = str(map_path)
        self.feature_path = str(feature_path)
        super().__init__(
            f"Dataset alignment error for video_id={video_id!r}: expected map rows "
            f"({self.map_rows}) to equal feature vectors ({self.feature_vectors}); "
            f"map={self.map_path}, features={self.feature_path}. Fix or replace the "
            "source files, then inspect the dataset again."
        )


class DatasetValidationError(DatasetError):
    """Raised when inspection proves that a dataset is unsafe to index."""

    def __init__(self, report: "DatasetInspectionReport") -> None:
        self.report = report
        codes = sorted(code for code, count in report.issue_counts.items() if count)
        summary = ", ".join(codes[:8]) or "unknown validation issue"
        super().__init__(
            f"Dataset validation failed for {report.data_root}: "
            f"{report.invalid_video_count} invalid video(s); issue codes: {summary}. "
            "Run inspect-data and fix required sources before rebuilding the cache."
        )


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    code: str
    video_id: str | None
    source: str | None
    path: str | None
    message: str
    expected: Any | None = None
    actual: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoDatasetReport:
    video_id: str
    map_rows: int = 0
    feature_vectors: int = 0
    feature_dim: int = 0
    feature_dtype: str = "unknown"
    keyframe_images: int = 0
    object_files: int = 0
    media_info_exists: bool = False
    video_exists: bool = False
    first_frame_idx: int | None = None
    last_frame_idx: int | None = None
    duplicate_frame_indices: list[int] = field(default_factory=list)
    non_monotonic_frame_indices: list[int] = field(default_factory=list)
    decreasing_frame_indices: list[int] = field(default_factory=list)
    duplicate_keyframe_ordinals: list[int] = field(default_factory=list)
    missing_keyframes: int = 0
    missing_keyframe_ordinals: list[int] = field(default_factory=list)
    missing_object_files: int = 0
    corrupt_object_files: int = 0
    feature_non_finite_count: int = 0
    feature_zero_vector_count: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["valid"] = self.valid
        return data


@dataclass
class DatasetInspectionReport:
    schema_version: int
    created_at_utc: str
    data_root: str
    strict: bool
    deep: bool
    # `video_count` counts the SELECTED videos, i.e. exactly the videos this report
    # validated. Use `discovered_video_count` for everything present under DATA_ROOT.
    video_count: int
    scope: dict[str, list[str]]
    discovered_video_count: int
    selected_video_count: int
    excluded_video_count: int
    selected_video_ids_hash: str
    excluded_video_ids_sample: list[str]
    valid_video_count: int
    invalid_video_count: int
    map_file_count: int
    feature_file_count: int
    keyframe_folder_count: int
    object_folder_count: int
    media_info_count: int
    mp4_count: int
    total_map_rows: int
    total_feature_vectors: int
    total_keyframe_images: int
    feature_dimensions: list[int]
    feature_dtypes: list[str]
    missing_sources: dict[str, list[str]]
    orphan_sources: dict[str, list[str]]
    issue_counts: dict[str, int]
    videos: list[VideoDatasetReport]
    issues: list[DatasetIssue]
    valid_for_index_build: bool
    duplicate_official_frame_idx_count: int = 0
    duplicate_internal_keyframe_id_count: int = 0
    invalid_selected_video_ids: list[str] = field(default_factory=list)
    inspection_seconds: float = 0.0

    def summary(self, *, report_path: str | None = None) -> dict[str, Any]:
        return {
            "data_root": self.data_root,
            "valid_for_index_build": self.valid_for_index_build,
            "scope": self.scope,
            "discovered_video_count": self.discovered_video_count,
            "selected_video_count": self.selected_video_count,
            "excluded_video_count": self.excluded_video_count,
            "selected_video_ids_hash": self.selected_video_ids_hash,
            "excluded_video_ids_sample": self.excluded_video_ids_sample,
            "video_count": self.video_count,
            "valid_video_count": self.valid_video_count,
            "invalid_video_count": self.invalid_video_count,
            "invalid_selected_video_ids": self.invalid_selected_video_ids[:20],
            "total_map_rows": self.total_map_rows,
            "total_feature_vectors": self.total_feature_vectors,
            "total_keyframe_images": self.total_keyframe_images,
            "feature_dimensions": self.feature_dimensions,
            "feature_dtypes": self.feature_dtypes,
            "duplicate_official_frame_idx_count": self.duplicate_official_frame_idx_count,
            "duplicate_internal_keyframe_id_count": self.duplicate_internal_keyframe_id_count,
            "issue_counts": self.issue_counts,
            "missing_required_sources": {
                key: len(self.missing_sources.get(key, ()))
                for key in ("map", "features", "keyframes")
            },
            "missing_sources": {key: len(value) for key, value in self.missing_sources.items()},
            "orphan_sources": {key: len(value) for key, value in self.orphan_sources.items()},
            "inspection_seconds": self.inspection_seconds,
            "report_path": report_path,
        }

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        if summary_only:
            return {
                "schema_version": self.schema_version,
                "created_at_utc": self.created_at_utc,
                "strict": self.strict,
                "deep": self.deep,
                **self.summary(),
            }
        data = asdict(self)
        data["videos"] = [video.to_dict() for video in self.videos]
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data


@dataclass(frozen=True)
class MediaMetadata:
    media_title: str = ""
    media_description: str = ""
    media_tags: tuple[str, ...] = ()
    media_channel: str = ""
    media_duration: float | None = None
    media_fps: float | None = None


def _issue(
    target: VideoDatasetReport,
    severity: str,
    code: str,
    *,
    source: str | None,
    path: Path | None,
    message: str,
    expected: Any = None,
    actual: Any = None,
) -> None:
    target.issues.append(
        DatasetIssue(
            severity=severity,
            code=code,
            video_id=target.video_id,
            source=source,
            path=None if path is None else str(path),
            message=message,
            expected=expected,
            actual=actual,
        )
    )


def _parse_integer(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text or not _INTEGER_RE.fullmatch(text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _source_files(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in directory.glob(f"*{suffix}") if path.is_file()}


def _source_folders(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.name: path for path in directory.iterdir() if path.is_dir()}


def _read_map_rows(path: Path, report: VideoDatasetReport) -> list[dict[str, str]] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            missing = [name for name in MAP_REQUIRED_COLUMNS if name not in columns]
            if missing:
                _issue(
                    report,
                    "error",
                    "MAP_REQUIRED_COLUMN_MISSING",
                    source="map",
                    path=path,
                    message=f"Map CSV is missing required column(s): {', '.join(missing)}.",
                    expected=list(MAP_REQUIRED_COLUMNS),
                    actual=list(columns),
                )
                return None
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        _issue(
            report,
            "error",
            "MAP_READ_ERROR",
            source="map",
            path=path,
            message=f"Cannot read map CSV: {type(exc).__name__}: {exc}",
        )
        return None


def _validate_map_rows(
    rows: Sequence[Mapping[str, str]], path: Path, report: VideoDatasetReport
) -> tuple[list[int], list[int]]:
    frame_indices: list[int] = []
    ordinals: list[int] = []
    timestamps: list[float] = []
    seen: set[int] = set()
    duplicates: set[int] = set()
    seen_ordinals: set[int] = set()
    duplicate_ordinals: set[int] = set()
    non_monotonic: list[int] = []
    decreasing: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        ordinal = _parse_integer(row.get("n"))
        if ordinal is None or ordinal <= 0:
            _issue(
                report, "error", "KEYFRAME_ORDINAL_INVALID", source="map", path=path,
                message=f"Row {row_number} has invalid keyframe ordinal n={row.get('n')!r}.",
                expected="positive integer", actual=row.get("n"),
            )
            ordinal = row_number
        elif ordinal in seen_ordinals:
            duplicate_ordinals.add(ordinal)
        seen_ordinals.add(ordinal)
        ordinals.append(ordinal)

        frame_idx = _parse_integer(row.get("frame_idx"))
        if frame_idx is None:
            _issue(
                report, "error", "FRAME_IDX_INVALID", source="map", path=path,
                message=f"Row {row_number} has non-integer frame_idx={row.get('frame_idx')!r}.",
                expected="integer >= 0", actual=row.get("frame_idx"),
            )
        else:
            if frame_idx < 0:
                _issue(
                    report, "error", "FRAME_IDX_NEGATIVE", source="map", path=path,
                    message=f"Row {row_number} has negative frame_idx={frame_idx}.",
                    expected=">= 0", actual=frame_idx,
                )
            if frame_idx in seen:
                duplicates.add(frame_idx)
            if frame_indices and frame_idx < frame_indices[-1]:
                decreasing.append(frame_idx)
            elif frame_indices and frame_idx == frame_indices[-1]:
                non_monotonic.append(frame_idx)
            seen.add(frame_idx)
            frame_indices.append(frame_idx)

        timestamp = _parse_finite_float(row.get("pts_time"))
        if timestamp is None or timestamp < 0:
            _issue(
                report, "error", "TIMESTAMP_INVALID", source="map", path=path,
                message=f"Row {row_number} has invalid pts_time={row.get('pts_time')!r}.",
                expected="finite float >= 0", actual=row.get("pts_time"),
            )
            timestamp = None
        else:
            if timestamps and timestamp < timestamps[-1]:
                _issue(
                    report, "error", "TIMESTAMP_NON_MONOTONIC", source="map", path=path,
                    message=f"Row {row_number} timestamp decreases from {timestamps[-1]} to {timestamp}.",
                    expected=f">= {timestamps[-1]}", actual=timestamp,
                )
            timestamps.append(timestamp)

        fps = _parse_finite_float(row.get("fps"))
        if fps is None or fps <= 0:
            _issue(
                report, "warning", "MAP_FPS_INVALID", source="map", path=path,
                message=f"Row {row_number} has unreliable fps={row.get('fps')!r}.",
                expected="finite float > 0", actual=row.get("fps"),
            )
        elif frame_idx is not None and frame_idx >= 0 and timestamp is not None:
            expected_timestamp = frame_idx / fps
            tolerance = max(1.0, 3.0 / fps)
            if abs(timestamp - expected_timestamp) > tolerance:
                _issue(
                    report, "warning", "TIMESTAMP_FRAME_FPS_INCONSISTENT", source="map", path=path,
                    message=(f"Row {row_number} pts_time differs from frame_idx/fps by more than "
                             f"{tolerance:.3f}s; FPS metadata may be unreliable."),
                    expected=expected_timestamp, actual=timestamp,
                )

    report.duplicate_frame_indices = sorted(duplicates)
    report.non_monotonic_frame_indices = non_monotonic
    report.decreasing_frame_indices = decreasing
    report.duplicate_keyframe_ordinals = sorted(duplicate_ordinals)
    # Evidence (artifacts/map_schema_report.json, all 873 official map files): frame_idx
    # is int(pts_time * fps) in 177,321 of 177,321 rows, so two keyframes one source
    # frame apart near t=0 legitimately truncate to the same official frame_idx. Repeats
    # are therefore official data, not corruption, and are reported as information only.
    # A strictly DECREASING frame_idx never occurs anywhere in the collection and stays a
    # hard error. Internal identity is protected separately by the keyframe ordinal.
    if duplicates:
        _issue(
            report, "info", "FRAME_IDX_DUPLICATE", source="map", path=path,
            message=(
                f"Map repeats {len(duplicates)} official frame_idx value(s): "
                f"{sorted(duplicates)[:10]}. This is valid BTC mapping; internal keyframe "
                "IDs use the keyframe ordinal and stay unique."
            ),
            expected="non-decreasing frame_idx", actual=sorted(duplicates),
        )
    if non_monotonic:
        _issue(
            report, "info", "FRAME_IDX_NON_MONOTONIC", source="map", path=path,
            message="frame_idx repeats the previous row's value; non-decreasing order is valid.",
            expected="non-decreasing", actual=non_monotonic[:10],
        )
    if decreasing:
        _issue(
            report, "error", "FRAME_IDX_DECREASING", source="map", path=path,
            message="frame_idx decreases in keyframe order; keyframe ordering is unusable.",
            expected="non-decreasing", actual=decreasing[:10],
        )
    if duplicate_ordinals:
        _issue(
            report, "error", "KEYFRAME_ORDINAL_DUPLICATE", source="map", path=path,
            message=(
                f"Map repeats keyframe ordinal n={sorted(duplicate_ordinals)[:10]}; internal "
                "keyframe IDs and keyframe image lookup would collide."
            ),
            expected="unique keyframe ordinal", actual=sorted(duplicate_ordinals),
        )
    if frame_indices:
        report.first_frame_idx = frame_indices[0]
        report.last_frame_idx = frame_indices[-1]
    return frame_indices, ordinals

def _feature_facts(
    path: Path,
    report: VideoDatasetReport,
    *,
    expected_dim: int | None,
    supported_dtypes: set[str],
    verify_finite: bool,
    verify_norms: bool,
) -> np.ndarray | None:
    try:
        features = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        _issue(
            report,
            "error",
            "FEATURE_LOAD_ERROR",
            source="features",
            path=path,
            message=f"Cannot mmap feature array: {type(exc).__name__}: {exc}",
        )
        return None
    report.feature_dtype = str(features.dtype)
    if features.ndim != 2:
        report.feature_vectors = int(features.shape[0]) if features.ndim else 0
        _issue(
            report,
            "error",
            "FEATURE_INVALID_NDIM",
            source="features",
            path=path,
            message=f"Feature array must be 2-D, got shape={features.shape}.",
            expected=2,
            actual=features.ndim,
        )
        return features
    report.feature_vectors = int(features.shape[0])
    report.feature_dim = int(features.shape[1])
    if expected_dim is not None and report.feature_dim != expected_dim:
        _issue(
            report,
            "error",
            "FEATURE_DIM_MISMATCH",
            source="features",
            path=path,
            message=f"Feature dimension {report.feature_dim} does not match expected {expected_dim}.",
            expected=expected_dim,
            actual=report.feature_dim,
        )
    if report.feature_dtype not in supported_dtypes:
        _issue(
            report,
            "error",
            "FEATURE_DTYPE_UNSUPPORTED",
            source="features",
            path=path,
            message=f"Unsupported feature dtype {report.feature_dtype!r}.",
            expected=sorted(supported_dtypes),
            actual=report.feature_dtype,
        )
    if features.dtype.kind not in {"f", "i", "u"}:
        return features
    non_finite = 0
    zero_vectors = 0
    chunk_rows = 8192
    for start in range(0, report.feature_vectors, chunk_rows):
        chunk = np.asarray(features[start : start + chunk_rows])
        if verify_finite:
            non_finite += int(chunk.size - np.count_nonzero(np.isfinite(chunk)))
        if verify_norms:
            finite_rows = np.all(np.isfinite(chunk), axis=1)
            if np.any(finite_rows):
                norms = np.linalg.norm(chunk[finite_rows].astype(np.float32, copy=False), axis=1)
                zero_vectors += int(np.count_nonzero(norms == 0))
    report.feature_non_finite_count = non_finite
    report.feature_zero_vector_count = zero_vectors
    if non_finite:
        _issue(
            report,
            "error",
            "FEATURE_NON_FINITE",
            source="features",
            path=path,
            message=f"Feature array contains {non_finite} NaN/Inf value(s).",
            expected=0,
            actual=non_finite,
        )
    if zero_vectors:
        _issue(
            report,
            "error",
            "FEATURE_ZERO_VECTOR",
            source="features",
            path=path,
            message=f"Feature array contains {zero_vectors} zero vector(s).",
            expected=0,
            actual=zero_vectors,
        )
    return features


def _find_keyframe(folder: Path, ordinal: int) -> Path | None:
    for stem in (f"{ordinal:03d}", f"{ordinal:04d}", str(ordinal)):
        for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS):
            candidate = folder / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    return None


def _validate_keyframes(
    folder: Path,
    ordinals: Sequence[int],
    report: VideoDatasetReport,
    *,
    require_images: bool,
    deep: bool,
    max_decode_checks: int | None,
) -> None:
    if not folder.is_dir():
        _issue(
            report,
            "error",
            "KEYFRAME_FOLDER_MISSING",
            source="keyframes",
            path=folder,
            message="Required keyframe folder is missing.",
            expected="directory",
            actual=None,
        )
        report.missing_keyframes = len(ordinals)
        report.missing_keyframe_ordinals = list(ordinals[:50])
        return
    image_paths = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )
    report.keyframe_images = len(image_paths)
    expected_paths: set[Path] = set()
    missing: list[int] = []
    for ordinal in ordinals:
        image_path = _find_keyframe(folder, ordinal)
        if image_path is None:
            missing.append(ordinal)
        else:
            expected_paths.add(image_path.resolve(strict=False))
    report.missing_keyframes = len(missing)
    report.missing_keyframe_ordinals = missing[:50]
    severity = "error" if require_images else "warning"
    if missing:
        _issue(
            report,
            severity,
            "KEYFRAME_IMAGE_MISSING",
            source="keyframes",
            path=folder,
            message=f"Missing {len(missing)} mapped keyframe image(s); first ordinals={missing[:10]}.",
            expected=len(ordinals),
            actual=len(ordinals) - len(missing),
        )
    orphan = [path.name for path in image_paths if path.resolve(strict=False) not in expected_paths]
    if orphan:
        _issue(
            report,
            "warning",
            "KEYFRAME_IMAGE_ORPHAN",
            source="keyframes",
            path=folder,
            message=f"Found {len(orphan)} image(s) without a matching map ordinal.",
            expected=0,
            actual=orphan[:10],
        )
    if len(image_paths) != len(ordinals):
        _issue(
            report,
            severity,
            "KEYFRAME_COUNT_MISMATCH",
            source="keyframes",
            path=folder,
            message="Keyframe image count does not match map row count.",
            expected=len(ordinals),
            actual=len(image_paths),
        )
    if deep:
        from PIL import Image, UnidentifiedImageError

        checks = image_paths if max_decode_checks is None else image_paths[:max_decode_checks]
        corrupt: list[str] = []
        for image_path in checks:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError):
                corrupt.append(image_path.name)
        if corrupt:
            _issue(
                report,
                "error",
                "KEYFRAME_IMAGE_CORRUPT",
                source="keyframes",
                path=folder,
                message=f"Cannot decode {len(corrupt)} checked image(s).",
                expected="decodable image",
                actual=corrupt[:10],
            )


def _validate_object_payload(data: Any, path: Path, report: VideoDatasetReport) -> None:
    if not isinstance(data, Mapping):
        _issue(
            report,
            "error",
            "OBJECT_SCHEMA_INVALID",
            source="objects",
            path=path,
            message="Object JSON root must be an object.",
            expected="object",
            actual=type(data).__name__,
        )
        return
    labels = data.get("detection_class_entities") or data.get("detection_class_names") or []
    scores = data.get("detection_scores") or []
    boxes = data.get("detection_boxes") or []
    if not isinstance(labels, list) or not isinstance(scores, list) or not isinstance(boxes, list):
        _issue(
            report,
            "error",
            "OBJECT_SCHEMA_INVALID",
            source="objects",
            path=path,
            message="Object labels, scores, and boxes must be arrays.",
        )
        return
    if scores and len(scores) != len(labels):
        _issue(
            report,
            "error",
            "OBJECT_SCHEMA_INVALID",
            source="objects",
            path=path,
            message="Object score count does not match label count.",
            expected=len(labels),
            actual=len(scores),
        )
    if boxes and len(boxes) != len(labels):
        _issue(
            report,
            "error",
            "OBJECT_SCHEMA_INVALID",
            source="objects",
            path=path,
            message="Object box count does not match label count.",
            expected=len(labels),
            actual=len(boxes),
        )
    for value in scores:
        score = _parse_finite_float(value)
        if score is None or not 0 <= score <= 1:
            _issue(
                report,
                "error",
                "OBJECT_SCORE_INVALID",
                source="objects",
                path=path,
                message=f"Object confidence must be finite in [0, 1], got {value!r}.",
                expected="finite value in [0, 1]",
                actual=value,
            )
            break
    for box in boxes:
        values = [_parse_finite_float(value) for value in box] if isinstance(box, list) else []
        malformed = len(values) != 4 or any(value is None for value in values)
        if not malformed:
            ymin, xmin, ymax, xmax = values
            malformed = ymin > ymax or xmin > xmax
        if malformed:
            _issue(
                report,
                "error",
                "OBJECT_BOX_INVALID",
                source="objects",
                path=path,
                message=f"Object box must be finite [ymin, xmin, ymax, xmax] with ordered corners, got {box!r}.",
                expected="four finite coordinates with ymin <= ymax and xmin <= xmax",
                actual=box,
            )
            break


def _validate_objects(
    folder: Path,
    ordinals: Sequence[int],
    report: VideoDatasetReport,
    *,
    enabled: bool,
    strict_objects: bool,
) -> None:
    object_paths = sorted(folder.glob("*.json"), key=lambda path: path.name) if folder.is_dir() else []
    report.object_files = len(object_paths)
    if not enabled:
        return
    missing: list[int] = []
    for ordinal in ordinals:
        path = folder / f"{ordinal:03d}.json"
        if not path.is_file():
            missing.append(ordinal)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            report.corrupt_object_files += 1
            _issue(
                report,
                "error" if strict_objects else "warning",
                "OBJECT_JSON_CORRUPT",
                source="objects",
                path=path,
                message=f"Cannot parse object JSON: {type(exc).__name__}: {exc}",
            )
            continue
        before = len(report.issues)
        _validate_object_payload(data, path, report)
        if not strict_objects:
            for index in range(before, len(report.issues)):
                issue = report.issues[index]
                if issue.source == "objects" and issue.severity == "error":
                    report.issues[index] = DatasetIssue(**{**issue.to_dict(), "severity": "warning"})
    report.missing_object_files = len(missing)
    if missing:
        _issue(
            report,
            "error" if strict_objects else "warning",
            "OBJECT_FILE_MISSING",
            source="objects",
            path=folder,
            message=f"Missing {len(missing)} object JSON file(s); first ordinals={missing[:10]}.",
            expected=len(ordinals),
            actual=len(ordinals) - len(missing),
        )


def read_media_metadata(path: str | Path) -> MediaMetadata:
    path = Path(path)
    if not path.is_file():
        return MediaMetadata()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetSourceError(f"Cannot parse media-info JSON {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise DatasetSourceError(f"Media-info JSON root must be an object: {path}")
    tags = data.get("keywords", ())
    if isinstance(tags, str):
        media_tags = (tags.strip(),) if tags.strip() else ()
    elif isinstance(tags, list):
        media_tags = tuple(str(item).strip() for item in tags if str(item).strip())
    else:
        media_tags = ()
    duration = _parse_finite_float(data.get("length", data.get("duration")))
    fps = _parse_finite_float(data.get("fps"))
    return MediaMetadata(
        media_title=str(data.get("title") or "").strip(),
        media_description=str(data.get("description") or "").strip(),
        media_tags=media_tags,
        media_channel=str(data.get("author") or data.get("channel") or "").strip(),
        media_duration=duration,
        media_fps=fps,
    )


def _validate_media(path: Path, report: VideoDatasetReport) -> None:
    report.media_info_exists = path.is_file()
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(
            report,
            "warning",
            "MEDIA_JSON_CORRUPT",
            source="media-info",
            path=path,
            message=f"Cannot parse media-info JSON: {type(exc).__name__}: {exc}",
        )
        return
    if not isinstance(data, Mapping):
        _issue(
            report,
            "warning",
            "MEDIA_SCHEMA_INVALID",
            source="media-info",
            path=path,
            message="Media-info JSON root must be an object.",
            expected="object",
            actual=type(data).__name__,
        )
        return
    tags = data.get("keywords")
    if tags is not None and not isinstance(tags, (str, list)):
        _issue(
            report,
            "warning",
            "MEDIA_SCHEMA_INVALID",
            source="media-info",
            path=path,
            message="Media keywords must be a string or array.",
            expected="string or array",
            actual=type(tags).__name__,
        )
    duration_value = data.get("length", data.get("duration"))
    if duration_value is not None:
        duration = _parse_finite_float(duration_value)
        if duration is None or duration < 0:
            _issue(
                report,
                "warning",
                "MEDIA_DURATION_INVALID",
                source="media-info",
                path=path,
                message=f"Media duration is invalid: {duration_value!r}.",
                expected="finite value >= 0",
                actual=duration_value,
            )
    if data.get("fps") is not None:
        fps = _parse_finite_float(data.get("fps"))
        if fps is None or fps <= 0:
            _issue(
                report,
                "warning",
                "MEDIA_FPS_INVALID",
                source="media-info",
                path=path,
                message=f"Media fps is invalid: {data.get('fps')!r}.",
                expected="finite value > 0",
                actual=data.get("fps"),
            )


def inspect_aic_dataset(
    data_root: Path,
    *,
    app_config: AppConfig,
    strict: bool = False,
    deep: bool = False,
    limit_videos: int | None = None,
    scope: DatasetScopeConfig | None = None,
) -> DatasetInspectionReport:
    import time

    started = time.perf_counter()
    root = Path(data_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise DatasetLayoutError(
            f"AIC DATA_ROOT does not exist or is not a directory: {root}. "
            "Provide a directory containing map-keyframes, clip-features-32, and keyframes."
        )
    map_files = _source_files(root / "map-keyframes", ".csv")
    feature_files = _source_files(root / "clip-features-32", ".npy")
    keyframe_folders = _source_folders(root / "keyframes")
    object_folders = _source_folders(root / "objects")
    media_files = _source_files(root / "media-info", ".json")
    video_files = _source_files(root / "video", ".mp4")
    required_ids = set(map_files) | set(feature_files) | set(keyframe_folders)
    # DISCOVERED = everything under DATA_ROOT. SELECTED = what the scope keeps, and the
    # only domain that is validated: sources outside the scope can be absent without
    # making the selected dataset invalid.
    discovered_ids = sorted(required_ids | set(object_folders) | set(media_files) | set(video_files))
    effective_scope = normalize_scope(scope if scope is not None else app_config.dataset.scope)
    all_ids = list(select_video_ids(discovered_ids, effective_scope))
    if limit_videos is not None:
        all_ids = all_ids[: max(0, int(limit_videos))]
    selected = set(all_ids)
    selected_required_ids = selected & required_ids
    excluded_ids = excluded_video_ids(discovered_ids, all_ids)
    validation = app_config.dataset.validation
    expected_dim = validation.expected_feature_dim or app_config.encoder.feature_dim
    supported_dtypes = set(str(item) for item in validation.supported_feature_dtypes)
    video_reports: list[VideoDatasetReport] = []
    missing_sources = {"map": [], "features": [], "keyframes": [], "objects": [], "media_info": [], "video": []}
    orphan_sources = {"map": [], "features": [], "keyframes": [], "objects": [], "media_info": [], "video": []}

    for video_id in all_ids:
        report = VideoDatasetReport(video_id=video_id)
        map_path = map_files.get(video_id)
        feature_path = feature_files.get(video_id)
        keyframe_folder = keyframe_folders.get(video_id)
        is_required_id = video_id in required_ids
        if map_path is None and is_required_id:
            missing_sources["map"].append(video_id)
            missing_path = root / "map-keyframes" / f"{video_id}.csv"
            _issue(report, "error", "MAP_MISSING", source="map", path=missing_path, message="Required map CSV is missing.", expected="existing CSV", actual=None)
            _issue(report, "error", "REQUIRED_SOURCE_MISSING", source="map", path=missing_path, message="A required map source is missing.", expected="map", actual=None)
        if feature_path is None and is_required_id:
            missing_sources["features"].append(video_id)
            missing_path = root / "clip-features-32" / f"{video_id}.npy"
            _issue(report, "error", "FEATURE_MISSING", source="features", path=missing_path, message="Required CLIP feature file is missing.", expected="existing NPY", actual=None)
            _issue(report, "error", "REQUIRED_SOURCE_MISSING", source="features", path=missing_path, message="A required feature source is missing.", expected="features", actual=None)
        if keyframe_folder is None and is_required_id:
            missing_sources["keyframes"].append(video_id)
            missing_path = root / "keyframes" / video_id
            _issue(report, "error", "REQUIRED_SOURCE_MISSING", source="keyframes", path=missing_path, message="A required keyframe source is missing.", expected="keyframe folder", actual=None)
        if video_id not in object_folders and is_required_id:
            missing_sources["objects"].append(video_id)
        if video_id not in media_files and is_required_id:
            missing_sources["media_info"].append(video_id)
            _issue(report, "info", "MEDIA_INFO_MISSING", source="media-info", path=root / "media-info" / f"{video_id}.json", message="Optional media-info is missing.")
        if video_id not in video_files and is_required_id:
            missing_sources["video"].append(video_id)

        if map_path is not None and feature_path is None:
            orphan_sources["map"].append(video_id)
            _issue(report, "warning", "ORPHAN_MAP", source="map", path=map_path, message="Map has no matching feature file.")
        if feature_path is not None and map_path is None:
            orphan_sources["features"].append(video_id)
            _issue(report, "warning", "ORPHAN_FEATURE", source="features", path=feature_path, message="Feature file has no matching map.")
        if keyframe_folder is not None and map_path is None:
            orphan_sources["keyframes"].append(video_id)
            _issue(report, "warning", "ORPHAN_KEYFRAME_FOLDER", source="keyframes", path=keyframe_folder, message="Keyframe folder has no matching map.")
        if video_id in object_folders and video_id not in map_files:
            orphan_sources["objects"].append(video_id)
            _issue(report, "warning", "ORPHAN_OBJECT_FOLDER", source="objects", path=object_folders[video_id], message="Object folder has no matching map.")
        if video_id in media_files and video_id not in map_files:
            orphan_sources["media_info"].append(video_id)
            _issue(report, "warning", "ORPHAN_MEDIA_INFO", source="media-info", path=media_files[video_id], message="Media-info file has no matching map.")
        if video_id in video_files and video_id not in map_files:
            orphan_sources["video"].append(video_id)
            _issue(report, "warning", "ORPHAN_VIDEO", source="video", path=video_files[video_id], message="MP4 has no matching map.")

        rows: list[dict[str, str]] | None = None
        ordinals: list[int] = []
        if map_path is not None:
            rows = _read_map_rows(map_path, report)
            if rows is not None:
                report.map_rows = len(rows)
                _, ordinals = _validate_map_rows(rows, map_path, report)
        if feature_path is not None:
            _feature_facts(
                feature_path,
                report,
                expected_dim=int(expected_dim) if expected_dim is not None else None,
                supported_dtypes=supported_dtypes,
                verify_finite=validation.verify_feature_finite,
                verify_norms=validation.verify_feature_norms,
            )
        if rows is not None and feature_path is not None and report.map_rows != report.feature_vectors:
            _issue(
                report,
                "error",
                "MAP_FEATURE_COUNT_MISMATCH",
                source="alignment",
                path=map_path,
                message=(
                    f"Map rows ({report.map_rows}) do not equal feature vectors "
                    f"({report.feature_vectors}); this video cannot be indexed."
                ),
                expected=report.map_rows,
                actual=report.feature_vectors,
            )
            _issue(
                report,
                "error",
                "FEATURE_COUNT_MISMATCH",
                source="features",
                path=feature_path,
                message="Feature vector count does not match map row count.",
                expected=report.map_rows,
                actual=report.feature_vectors,
            )
        if is_required_id:
            _validate_keyframes(
                keyframe_folder or root / "keyframes" / video_id,
                ordinals,
                report,
                require_images=validation.require_keyframe_images,
                deep=deep or validation.verify_image_decode,
                max_decode_checks=validation.max_image_decode_checks_per_video,
            )
        _validate_objects(
            object_folders.get(video_id, root / "objects" / video_id),
            ordinals,
            report,
            enabled=app_config.dataset.load_objects,
            strict_objects=validation.strict_objects,
        )
        _validate_media(media_files.get(video_id, root / "media-info" / f"{video_id}.json"), report)
        report.video_exists = video_id in video_files
        video_reports.append(report)

    dimensions = sorted({report.feature_dim for report in video_reports if report.feature_dim})
    dtypes = sorted({report.feature_dtype for report in video_reports if report.feature_dtype != "unknown"})
    if len(dimensions) > 1:
        for report in video_reports:
            if report.feature_dim:
                _issue(
                    report,
                    "error",
                    "FEATURE_DIM_MISMATCH",
                    source="features",
                    path=feature_files.get(report.video_id),
                    message=f"Collection contains multiple feature dimensions: {dimensions}.",
                    expected="one consistent dimension",
                    actual=dimensions,
                )
    issues = [issue for report in video_reports for issue in report.issues]
    issue_counts: dict[str, int] = {}
    for issue in issues:
        issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
    valid_count = sum(1 for report in video_reports if report.valid)
    # Internal keyframe IDs are `{video_id}/kf_{ordinal:06d}`, so they can only collide
    # when a map repeats a keyframe ordinal. Official frame_idx repeats do not.
    duplicate_internal_ids = sum(len(item.duplicate_keyframe_ordinals) for item in video_reports)
    report = DatasetInspectionReport(
        schema_version=DATASET_INSPECTION_SCHEMA_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        data_root=root.as_posix(),
        strict=bool(strict),
        deep=bool(deep),
        video_count=len(video_reports),
        scope=scope_payload(effective_scope),
        discovered_video_count=len(discovered_ids),
        selected_video_count=len(all_ids),
        excluded_video_count=len(excluded_ids),
        selected_video_ids_hash=hash_selected_video_ids(all_ids),
        excluded_video_ids_sample=list(excluded_ids[:10]),
        valid_video_count=valid_count,
        invalid_video_count=len(video_reports) - valid_count,
        map_file_count=len(selected & set(map_files)),
        feature_file_count=len(selected & set(feature_files)),
        keyframe_folder_count=len(selected & set(keyframe_folders)),
        object_folder_count=len(selected & set(object_folders)),
        media_info_count=len(selected & set(media_files)),
        mp4_count=len(selected & set(video_files)),
        total_map_rows=sum(item.map_rows for item in video_reports),
        total_feature_vectors=sum(item.feature_vectors for item in video_reports),
        total_keyframe_images=sum(item.keyframe_images for item in video_reports),
        feature_dimensions=dimensions,
        feature_dtypes=dtypes,
        missing_sources={key: values for key, values in missing_sources.items() if values},
        orphan_sources={key: values for key, values in orphan_sources.items() if values},
        issue_counts=dict(sorted(issue_counts.items())),
        videos=video_reports,
        issues=issues,
        valid_for_index_build=bool(selected_required_ids) and valid_count == len(video_reports),
        duplicate_official_frame_idx_count=sum(
            len(item.duplicate_frame_indices) for item in video_reports
        ),
        duplicate_internal_keyframe_id_count=duplicate_internal_ids,
        invalid_selected_video_ids=[item.video_id for item in video_reports if not item.valid],
        inspection_seconds=time.perf_counter() - started,
    )
    if strict and not report.valid_for_index_build:
        raise DatasetValidationError(report)
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError("Dataset reports must not contain ndarray values")
    return value


def write_dataset_report(
    report: DatasetInspectionReport,
    path: str | Path,
    *,
    summary_only: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    payload = _json_safe(report.to_dict(summary_only=summary_only))
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def first_alignment_error(report: DatasetInspectionReport) -> DatasetAlignmentError | None:
    for video in report.videos:
        if any(issue.code == "MAP_FEATURE_COUNT_MISMATCH" for issue in video.issues):
            root = Path(report.data_root)
            return DatasetAlignmentError(
                video_id=video.video_id,
                map_rows=video.map_rows,
                feature_vectors=video.feature_vectors,
                map_path=root / "map-keyframes" / f"{video.video_id}.csv",
                feature_path=root / "clip-features-32" / f"{video.video_id}.npy",
            )
    return None


__all__ = [
    "DATASET_INSPECTION_SCHEMA_VERSION",
    "DatasetAlignmentError",
    "DatasetError",
    "DatasetInspectionReport",
    "DatasetIssue",
    "DatasetLayoutError",
    "DatasetScopeError",
    "DatasetSourceError",
    "DatasetValidationError",
    "MediaMetadata",
    "VideoDatasetReport",
    "first_alignment_error",
    "inspect_aic_dataset",
    "read_media_metadata",
    "write_dataset_report",
]
