"""Strict loader for the official AIC 2026 preliminary-round data layout."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ingestion.build_index import IndexConfig, KeyframeIndex, l2_normalize, tokenize
from ingestion.build_records import searchable_text
from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION, KeyframeRecord, RawKeyframe
from retrieval.video_engine import VideoIndexEntry

from .config import AppConfig, DatasetScopeConfig
from .dataset_scope import (
    excluded_video_ids,
    hash_selected_video_ids,
    normalize_scope,
    select_video_ids,
)
from .dataset_validation import (
    DatasetAlignmentError,
    DatasetInspectionReport,
    DatasetLayoutError,
    DatasetSourceError,
    DatasetValidationError,
    MediaMetadata,
    first_alignment_error,
    inspect_aic_dataset,
    read_media_metadata,
)


@dataclass(frozen=True)
class AICDataPaths:
    root: Path
    features_dir: Path
    keyframes_dir: Path
    map_dir: Path
    objects_dir: Path
    media_info_dir: Path
    video_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "AICDataPaths":
        root = Path(root)
        return cls(
            root=root,
            features_dir=root / "clip-features-32",
            keyframes_dir=root / "keyframes",
            map_dir=root / "map-keyframes",
            objects_dir=root / "objects",
            media_info_dir=root / "media-info",
            video_dir=root / "video",
        )

    def validate(self) -> None:
        if not self.root.is_dir():
            raise DatasetLayoutError(
                f"AIC DATA_ROOT does not exist or is not a directory: {self.root}"
            )
        required = (self.features_dir, self.keyframes_dir, self.map_dir)
        missing = [path for path in required if not path.is_dir()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise DatasetLayoutError(f"Missing required AIC data directories: {joined}")


@dataclass(frozen=True)
class AICDatasetStats:
    videos: int
    frames: int
    missing_keyframes: int
    missing_objects: int
    missing_videos: int
    feature_dim: int
    feature_dtype: str = "unknown"
    dataset_validated: bool = False
    dataset_report_path: str | None = None
    invalid_video_count: int = 0
    record_schema_version: int = AIC_RECORD_SCHEMA_VERSION
    scope_include_patterns: tuple[str, ...] = ("*",)
    scope_exclude_patterns: tuple[str, ...] = ()
    discovered_videos: int = 0
    excluded_videos: int = 0
    selected_video_ids_hash: str = ""


def _read_objects(
    path: Path, score_threshold: float, max_objects: int
) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    names = (
        data.get("detection_class_entities")
        or data.get("detection_class_names")
        or []
    )
    scores = data.get("detection_scores") or []
    boxes = data.get("detection_boxes") or []
    if (
        not isinstance(names, list)
        or not isinstance(scores, list)
        or not isinstance(boxes, list)
    ):
        return [], []
    detections: list[dict] = []
    for index, name in enumerate(names):
        if not name:
            continue
        confidence = 1.0
        if index < len(scores):
            try:
                confidence = float(scores[index])
            except (TypeError, ValueError):
                continue
        if not math.isfinite(confidence) or confidence < score_threshold:
            continue
        label = " ".join(str(name).replace("_", " ").casefold().split())
        box = boxes[index] if index < len(boxes) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            box = None
        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "bounding_box": None if box is None else list(box),
            }
        )
    detections.sort(key=lambda item: (-item["confidence"], item["label"]))
    detections = detections[:max_objects]
    labels = list(dict.fromkeys(item["label"] for item in detections))
    return labels, detections


def _find_keyframe(paths: AICDataPaths, video_id: str, ordinal: int) -> Optional[Path]:
    """Locate a keyframe image by KEYFRAME ORDINAL, never by official frame_idx.

    Official packages name images after the 1-based map row (`001.jpg`, `002.jpg`, ...),
    which is the same position as the CLIP feature row.
    """
    folder = paths.keyframes_dir / video_id
    for stem in (f"{ordinal:03d}", f"{ordinal:04d}", str(ordinal)):
        for extension in (".jpg", ".jpeg", ".png"):
            candidate = folder / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    return None


def _internal_id(video_id: str, keyframe_ordinal: int) -> str:
    """Internal, globally unique keyframe identity.

    Built from the keyframe ordinal, NOT from the official `frame_idx`: 192 official
    videos repeat a `frame_idx`, so an ID derived from it would collide and silently
    drop keyframes. The official submission value is carried separately on
    `RawKeyframe.frame_idx` and must never be parsed back out of this string.
    """
    return f"{video_id}/kf_{int(keyframe_ordinal):06d}"


def _metadata_search_text(metadata: MediaMetadata) -> str:
    fields = (
        ("media_title", metadata.media_title),
        ("media_description", metadata.media_description),
        ("media_tags", " ".join(metadata.media_tags)),
        ("media_channel", metadata.media_channel),
    )
    return "\n".join(f"{source}: {value}" for source, value in fields if value)


class AICDatasetLoader:
    """Build an index only after the selected source data passes strict inspection."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        object_score_threshold: float = 0.2,
        max_objects_per_frame: int = 12,
        load_objects: bool = True,
        include_media_text: bool = True,
        verify_keyframes: bool = True,
        index_kind: str = "flat",
        app_config: AppConfig | None = None,
        scope: DatasetScopeConfig | None = None,
    ):
        self.paths = AICDataPaths.from_root(data_root)
        self.scope = normalize_scope(
            scope
            if scope is not None
            else (app_config.dataset.scope if app_config is not None else None)
        )
        self.object_score_threshold = object_score_threshold
        self.max_objects_per_frame = max_objects_per_frame
        self.load_objects = load_objects
        self.include_media_text = include_media_text
        self.verify_keyframes = verify_keyframes
        self.index_kind = index_kind
        self.app_config = app_config
        self.last_report: DatasetInspectionReport | None = None

    def video_ids(self) -> list[str]:
        """Video IDs the configured scope selects (the dataset this loader builds)."""
        return list(select_video_ids(self.discovered_video_ids(), self.scope))

    def discovered_video_ids(self) -> list[str]:
        """Every video ID with a CLIP feature file under DATA_ROOT, scope ignored."""
        self.paths.validate()
        return sorted(path.stem for path in self.paths.features_dir.glob("*.npy"))

    def _inspection_config(self, video_ids: Sequence[str]) -> AppConfig:
        base = self.app_config or AppConfig()
        feature_dim = 0
        if self.app_config is None:
            for video_id in video_ids:
                path = self.paths.features_dir / f"{video_id}.npy"
                if not path.is_file():
                    continue
                try:
                    shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
                except (OSError, ValueError, EOFError):
                    continue
                if len(shape) == 2:
                    feature_dim = int(shape[1])
                    break
        validation = base.dataset.validation
        if self.app_config is None:
            validation = replace(
                validation,
                strict_alignment=True,
                require_keyframe_images=True,
                expected_feature_dim=feature_dim or None,
            )
        if not validation.strict_alignment or not validation.require_keyframe_images:
            raise DatasetSourceError(
                "Index builds require strict_alignment=true and require_keyframe_images=true; "
                "non-strict behavior is available only through inspect-data."
            )
        dataset = replace(
            base.dataset,
            root=str(self.paths.root),
            load_objects=self.load_objects,
            include_media_text=self.include_media_text,
            verify_keyframes=self.verify_keyframes,
            index_kind=self.index_kind,
            validation=validation,
        )
        encoder = replace(base.encoder, feature_dim=feature_dim or base.encoder.feature_dim)
        return replace(base, dataset=dataset, encoder=encoder)

    def build_entry(
        self,
        video_ids: Optional[Sequence[str]] = None,
        *,
        dataset_id: str = "__aic2026__",
        limit_videos: Optional[int] = None,
        limit_frames_per_video: Optional[int] = None,
    ) -> tuple[VideoIndexEntry, AICDatasetStats]:
        self.paths.validate()
        discovered_video_ids = self.discovered_video_ids()
        # Scope first, so inspection and record creation share one selection.
        selected_video_ids = list(select_video_ids(discovered_video_ids, self.scope))
        requested_video_ids = (
            list(video_ids) if video_ids is not None else selected_video_ids
        )
        if limit_videos is not None:
            requested_video_ids = requested_video_ids[: max(0, int(limit_videos))]
        if not requested_video_ids:
            raise DatasetSourceError(f"No AIC feature files were selected from {self.paths.root}")

        inspection_limit = limit_videos if video_ids is None else None
        inspection_scope = (
            self.scope
            if video_ids is None
            else DatasetScopeConfig(include_patterns=tuple(requested_video_ids))
        )
        try:
            report = inspect_aic_dataset(
                self.paths.root,
                app_config=self._inspection_config(requested_video_ids),
                strict=True,
                deep=False,
                limit_videos=inspection_limit,
                scope=inspection_scope,
            )
        except DatasetValidationError as exc:
            report = exc.report
            self.last_report = report
            alignment_error = first_alignment_error(report)
            if alignment_error is not None:
                raise alignment_error from exc
            raise
        self.last_report = report

        report_video_ids = {item.video_id for item in report.videos}
        missing_requested = [
            video_id
            for video_id in requested_video_ids
            if video_id not in report_video_ids
        ]
        if missing_requested:
            raise DatasetSourceError(
                "Requested video IDs were not found during dataset inspection: "
                + ", ".join(missing_requested[:10])
            )

        raws: dict[str, RawKeyframe] = {}
        records: list[KeyframeRecord] = []
        feature_dim = 0
        feature_dtype = "unknown"
        for video_id in requested_video_ids:
            feature_path = self.paths.features_dir / f"{video_id}.npy"
            map_path = self.paths.map_dir / f"{video_id}.csv"
            try:
                features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                rows = self._read_map_rows(map_path)
            except (OSError, UnicodeError, csv.Error, ValueError, EOFError) as exc:
                raise DatasetSourceError(f"Cannot load validated sources for {video_id}: {exc}") from exc
            if len(rows) != int(features.shape[0]):
                raise DatasetAlignmentError(
                    video_id=video_id,
                    map_rows=len(rows),
                    feature_vectors=int(features.shape[0]),
                    map_path=map_path,
                    feature_path=feature_path,
                )
            feature_dim = int(features.shape[1])
            feature_dtype = str(features.dtype)
            frame_count = len(rows)
            if (
                limit_frames_per_video is not None
                and frame_count > int(limit_frames_per_video)
            ):
                frame_count = max(0, int(limit_frames_per_video))

            metadata = MediaMetadata()
            if self.include_media_text:
                try:
                    metadata = read_media_metadata(
                        self.paths.media_info_dir / f"{video_id}.json"
                    )
                except DatasetSourceError:
                    metadata = MediaMetadata()
            source_video = self.paths.video_dir / f"{video_id}.mp4"
            source_video_str = str(source_video) if source_video.is_file() else None

            # INVARIANT: feature row i <-> map row i <-> keyframe ordinal of row i <->
            # keyframe image of that ordinal. The CLIP feature array is ordered by
            # keyframe, while AIC submissions use the official source frame_idx read
            # out of the same row. The two must never be substituted for each other.
            for index in range(frame_count):
                row = rows[index]
                ordinal = int(row["n"])
                frame_idx = int(row["frame_idx"])
                timestamp = float(row["pts_time"])
                keyframe_id = _internal_id(video_id, ordinal)
                image_path = _find_keyframe(self.paths, video_id, ordinal)
                object_path = self.paths.objects_dir / video_id / f"{ordinal:03d}.json"
                if self.load_objects:
                    objects, object_detections = _read_objects(
                        object_path,
                        self.object_score_threshold,
                        self.max_objects_per_frame,
                    )
                else:
                    objects, object_detections = [], []
                if keyframe_id in raws:
                    raise DatasetSourceError(
                        f"Internal keyframe ID collision for {keyframe_id!r}: map keyframe "
                        f"ordinal n={ordinal} is repeated in {map_path}."
                    )
                raws[keyframe_id] = RawKeyframe(
                    id=keyframe_id,
                    video_id=video_id,
                    timestamp=timestamp,
                    image_path=None if image_path is None else str(image_path),
                    objects=objects,
                    object_detections=object_detections,
                    source_video=source_video_str,
                    frame_idx=frame_idx,
                    keyframe_ordinal=ordinal,
                )
                records.append(
                    KeyframeRecord(
                        id=keyframe_id,
                        video_id=video_id,
                        timestamp=timestamp,
                        clip_embedding=np.asarray(features[index], dtype=np.float32),
                        objects=objects,
                        media_title=metadata.media_title,
                        media_description=metadata.media_description,
                        media_tags=metadata.media_tags,
                        media_channel=metadata.media_channel,
                        media_duration=metadata.media_duration,
                        media_fps=metadata.media_fps,
                    )
                )

        if not records:
            raise DatasetSourceError(f"No AIC records were loaded from {self.paths.root}")
        index = _build_aic_index(records, kind=self.index_kind)
        caption_by_id: dict[str, str] = {}
        for record in records:
            metadata_text = _metadata_search_text(
                MediaMetadata(
                    media_title=record.media_title,
                    media_description=record.media_description,
                    media_tags=record.media_tags,
                    media_channel=record.media_channel,
                    media_duration=record.media_duration,
                    media_fps=record.media_fps,
                )
            )
            if metadata_text:
                caption_by_id[record.id] = metadata_text
        entry = VideoIndexEntry(
            video_id=dataset_id,
            index=index,
            raws=raws,
            num_sampled=len(records),
            num_indexed=len(records),
            caption_by_id=caption_by_id,
        )
        stats = AICDatasetStats(
            videos=len(requested_video_ids),
            frames=len(records),
            missing_keyframes=sum(item.missing_keyframes for item in report.videos),
            missing_objects=sum(item.missing_object_files for item in report.videos),
            missing_videos=len(report.missing_sources.get("video", ())),
            feature_dim=feature_dim,
            feature_dtype=feature_dtype,
            dataset_validated=True,
            invalid_video_count=report.invalid_video_count,
            scope_include_patterns=tuple(report.scope["include_patterns"]),
            scope_exclude_patterns=tuple(report.scope["exclude_patterns"]),
            discovered_videos=len(discovered_video_ids),
            excluded_videos=len(excluded_video_ids(discovered_video_ids, requested_video_ids)),
            selected_video_ids_hash=hash_selected_video_ids(requested_video_ids),
        )
        return entry, stats

    @staticmethod
    def _read_map_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))


def _build_aic_index(records: list[KeyframeRecord], *, kind: str = "flat") -> KeyframeIndex:
    """Build an exact flat index by default; retain HNSW for experiments."""
    if kind == "hnsw":
        return KeyframeIndex.build(records)
    if kind != "flat":
        raise ValueError(f"Unknown AIC index kind: {kind}")
    import faiss
    from rank_bm25 import BM25Okapi

    index_data = KeyframeIndex(
        ids=[record.id for record in records],
        video_ids=[record.video_id for record in records],
        timestamps=[record.timestamp for record in records],
        objects=[list(record.objects) for record in records],
        config=IndexConfig(),
    )
    matrix = l2_normalize(
        np.stack([record.clip_embedding for record in records]).astype(np.float32)
    )
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    index_data._clip_index = index
    corpus = [tokenize(searchable_text(record)) for record in records]
    index_data._bm25 = BM25Okapi([tokens if tokens else ["empty"] for tokens in corpus])
    return index_data


def official_frame_id(entry: VideoIndexEntry, keyframe_id: str) -> str:
    """Official AIC submission frame ID for an internal keyframe ID.

    Always read from the stored `frame_idx`. The internal ID encodes the keyframe
    ordinal, so parsing it would emit the ordinal as the submission frame and be wrong
    for every keyframe.
    """
    raw = entry.raws.get(keyframe_id)
    if raw is None:
        raise KeyError(f"Unknown internal keyframe ID: {keyframe_id!r}")
    if raw.frame_idx is None:
        raise ValueError(
            f"Keyframe {keyframe_id!r} has no official frame_idx; it cannot be submitted."
        )
    return str(raw.frame_idx)


def iter_official_rows(
    entry: VideoIndexEntry, keyframe_ids: Iterable[str]
) -> Iterable[tuple[str, str]]:
    for keyframe_id in keyframe_ids:
        raw = entry.raws.get(keyframe_id)
        if raw is None:
            continue
        yield raw.video_id, official_frame_id(entry, keyframe_id)
