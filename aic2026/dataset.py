"""Loader for the official AIC 2026 preliminary-round data layout.

Expected DATA_ROOT layout:
- clip-features-32/{video_id}.npy
- map-keyframes/{video_id}.csv with n, pts_time, fps, frame_idx
- keyframes/{video_id}/{n:03d}.jpg or compatible numbered jpg/png
- objects/{video_id}/{n:03d}.json
- media-info/{video_id}.json
- video/{video_id}.mp4 (optional for display fallback)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ingestion.build_index import IndexConfig, KeyframeIndex, l2_normalize, tokenize
from ingestion.build_records import searchable_text
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.video_engine import VideoIndexEntry


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
        missing = [
            p for p in (
                self.features_dir,
                self.keyframes_dir,
                self.map_dir,
                self.objects_dir,
                self.media_info_dir,
            )
            if not p.exists()
        ]
        if missing:
            joined = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Missing AIC data directories: {joined}")


@dataclass(frozen=True)
class AICDatasetStats:
    videos: int
    frames: int
    missing_keyframes: int
    missing_objects: int
    missing_videos: int
    feature_dim: int


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_media_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    parts: list[str] = []
    for key in ("title", "description"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    kws = data.get("keywords")
    if isinstance(kws, list):
        parts.append(" ".join(str(k) for k in kws if k))
    elif isinstance(kws, str):
        parts.append(kws)
    return "\n".join(parts)


def _read_objects(path: Path, score_threshold: float, max_objects: int) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    names = data.get("detection_class_entities") or data.get("detection_class_names") or []
    scores = data.get("detection_scores") or []
    boxes = data.get("detection_boxes") or []
    detections: list[dict] = []
    for i, name in enumerate(names):
        if not name:
            continue
        confidence = _as_float(scores[i], 1.0) if i < len(scores) else 1.0
        if confidence < score_threshold:
            continue
        label = " ".join(str(name).replace("_", " ").casefold().split())
        box = boxes[i] if i < len(boxes) else None
        detections.append({
            "label": label,
            "confidence": confidence,
            "bounding_box": list(box) if isinstance(box, (list, tuple)) else None,
        })
    detections.sort(key=lambda item: (-item["confidence"], item["label"]))
    detections = detections[:max_objects]
    labels = list(dict.fromkeys(item["label"] for item in detections))
    return labels, detections

def _find_keyframe(paths: AICDataPaths, video_id: str, n: int) -> Optional[Path]:
    folder = paths.keyframes_dir / video_id
    for stem in (f"{n:03d}", f"{n:04d}", str(n)):
        for ext in (".jpg", ".jpeg", ".png"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _internal_id(video_id: str, frame_idx: int) -> str:
    return f"{video_id}/{frame_idx}"


class AICDatasetLoader:
    """Build a VideoIndexEntry directly from official AIC keyframes/features."""

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
    ):
        self.paths = AICDataPaths.from_root(data_root)
        self.object_score_threshold = object_score_threshold
        self.max_objects_per_frame = max_objects_per_frame
        self.load_objects = load_objects
        self.include_media_text = include_media_text
        self.verify_keyframes = verify_keyframes
        self.index_kind = index_kind

    def video_ids(self) -> list[str]:
        self.paths.validate()
        return sorted(p.stem for p in self.paths.features_dir.glob("*.npy"))

    def build_entry(
        self,
        video_ids: Optional[Sequence[str]] = None,
        *,
        dataset_id: str = "__aic2026__",
        limit_videos: Optional[int] = None,
        limit_frames_per_video: Optional[int] = None,
    ) -> tuple[VideoIndexEntry, AICDatasetStats]:
        self.paths.validate()
        vids = list(video_ids) if video_ids is not None else self.video_ids()
        if limit_videos is not None:
            vids = vids[:limit_videos]
        raws: dict[str, RawKeyframe] = {}
        records: list[KeyframeRecord] = []
        missing_keyframes = 0
        missing_objects = 0
        missing_videos = 0
        feature_dim = 0

        for video_id in vids:
            feature_path = self.paths.features_dir / f"{video_id}.npy"
            map_path = self.paths.map_dir / f"{video_id}.csv"
            if not feature_path.exists() or not map_path.exists():
                continue
            features = np.load(feature_path, mmap_mode="r")
            feature_dim = int(features.shape[1]) if len(features.shape) == 2 else feature_dim
            rows = self._read_map_rows(map_path)
            if limit_frames_per_video is not None:
                rows = rows[:limit_frames_per_video]
            n = min(len(rows), int(features.shape[0]))
            media_text = _read_media_text(self.paths.media_info_dir / f"{video_id}.json") if self.include_media_text else ""
            source_video = self.paths.video_dir / f"{video_id}.mp4"
            if not source_video.exists():
                missing_videos += 1
                source_video_str = None
            else:
                source_video_str = str(source_video)

            for i in range(n):
                row = rows[i]
                ordinal = _as_int(row.get("n"), i + 1)
                frame_idx = _as_int(row.get("frame_idx"), ordinal)
                timestamp = _as_float(row.get("pts_time"), 0.0)
                kid = _internal_id(video_id, frame_idx)
                if self.verify_keyframes:
                    image_path = _find_keyframe(self.paths, video_id, ordinal)
                    if image_path is None:
                        missing_keyframes += 1
                else:
                    image_path = self.paths.keyframes_dir / video_id / f"{ordinal:03d}.jpg"
                object_path = self.paths.objects_dir / video_id / f"{ordinal:03d}.json"
                if self.load_objects:
                    objects, object_detections = _read_objects(
                        object_path,
                        self.object_score_threshold,
                        self.max_objects_per_frame,
                    )
                    if not object_path.exists():
                        missing_objects += 1
                else:
                    objects = []
                    object_detections = []
                raw = RawKeyframe(
                    id=kid,
                    video_id=video_id,
                    timestamp=timestamp,
                    image_path=str(image_path) if image_path is not None else None,
                    objects=objects,
                    object_detections=object_detections,
                    source_video=source_video_str,
                    frame_idx=frame_idx,
                )
                raws[kid] = raw
                records.append(KeyframeRecord(
                    id=kid,
                    video_id=video_id,
                    timestamp=timestamp,
                    clip_embedding=np.asarray(features[i], dtype=np.float32),
                    objects=objects,
                    llm_caption=media_text,
                ))

        if not records:
            raise RuntimeError(f"No AIC records were loaded from {self.paths.root}")
        index = _build_aic_index(records, kind=self.index_kind)
        entry = VideoIndexEntry(
            video_id=dataset_id,
            index=index,
            raws=raws,
            num_sampled=len(records),
            num_indexed=len(records),
            caption_by_id={r.id: r.llm_caption for r in records if r.llm_caption},
        )
        stats = AICDatasetStats(
            videos=len(vids),
            frames=len(records),
            missing_keyframes=missing_keyframes,
            missing_objects=missing_objects,
            missing_videos=missing_videos,
            feature_dim=feature_dim,
        )
        return entry, stats

    @staticmethod
    def _read_map_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))




def _build_aic_index(records: list[KeyframeRecord], *, kind: str = "flat") -> KeyframeIndex:
    """Build a KeyframeIndex for AIC features.

    The default is exact Faiss IndexFlatIP. It has no approximation loss and builds
    quickly for the AIC preliminary batch; HNSW is still available for experiments.
    """
    if kind == "hnsw":
        return KeyframeIndex.build(records)
    if kind != "flat":
        raise ValueError(f"Unknown AIC index kind: {kind}")
    import faiss
    from rank_bm25 import BM25Okapi

    idx = KeyframeIndex(
        ids=[r.id for r in records],
        video_ids=[r.video_id for r in records],
        timestamps=[r.timestamp for r in records],
        objects=[list(r.objects) for r in records],
        config=IndexConfig(),
    )
    mat = l2_normalize(np.stack([r.clip_embedding for r in records]).astype(np.float32))
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    idx._clip_index = index
    corpus = [tokenize(searchable_text(r)) for r in records]
    idx._bm25 = BM25Okapi([toks if toks else ["empty"] for toks in corpus])
    return idx

def official_frame_id(entry: VideoIndexEntry, keyframe_id: str) -> str:
    raw = entry.raws.get(keyframe_id)
    if raw is None or raw.frame_idx is None:
        return keyframe_id.rsplit("/", 1)[-1]
    return str(raw.frame_idx)


def iter_official_rows(entry: VideoIndexEntry, keyframe_ids: Iterable[str]) -> Iterable[tuple[str, str]]:
    for kid in keyframe_ids:
        raw = entry.raws.get(kid)
        if raw is None:
            continue
        yield raw.video_id, official_frame_id(entry, kid)