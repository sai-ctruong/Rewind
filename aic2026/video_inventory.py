"""Inventory of the original AIC videos and their supporting data.

Official AIC documentation makes the **video** the competition data; keyframes,
objects, CLIP features, and metadata are *supporting* data. This module keeps that
distinction visible: it discovers what MP4s actually exist under `DATA_ROOT`, and
reports, per video, which supporting sources accompany them.

It deliberately depends on nothing else in `aic2026`, so that both scope resolution
and the diagnostic tooling can use it without an import cycle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

VIDEO_DIRNAME = "video"
MAP_DIRNAME = "map-keyframes"
FEATURE_DIRNAME = "clip-features-32"
KEYFRAME_DIRNAME = "keyframes"
OBJECT_DIRNAME = "objects"
MEDIA_INFO_DIRNAME = "media-info"


@dataclass(frozen=True)
class VideoFile:
    video_id: str
    path: str
    size_bytes: int
    readable: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoSupport:
    """Which sources exist for one canonical video ID."""

    video_id: str
    video: bool
    map: bool
    clip: bool
    keyframe_jpeg: bool
    objects: bool
    media_info: bool

    @property
    def retrieval_supported(self) -> bool:
        """map + CLIP is what the official-feature global index actually needs."""
        return self.map and self.clip

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["retrieval_supported"] = self.retrieval_supported
        return data


@dataclass
class VideoInventory:
    data_root: str
    video_root: str
    videos: list[VideoFile] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.video_id for item in self.videos}))

    @property
    def readable_video_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.video_id for item in self.videos if item.readable}))

    @property
    def collections(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for video_id in self.video_ids:
            counts[collection_of(video_id)] = counts.get(collection_of(video_id), 0) + 1
        return dict(sorted(counts.items()))

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.videos)

    def to_dict(self) -> dict[str, Any]:
        unreadable = [item.to_dict() for item in self.videos if not item.readable]
        return {
            "data_root": self.data_root,
            "video_root": self.video_root,
            "video_count": len(self.video_ids),
            "file_count": len(self.videos),
            "collections": self.collections,
            "video_ids": list(self.video_ids),
            "duplicate_ids": list(self.duplicate_ids),
            "unreadable": unreadable,
            "unreadable_count": len(unreadable),
            "total_bytes": self.total_bytes,
            "files": [item.to_dict() for item in self.videos],
        }


def collection_of(video_id: str) -> str:
    """`L21_V006` -> `L21`. Falls back to the whole ID when there is no separator."""
    head, separator, _ = str(video_id).partition("_")
    return head if separator else str(video_id)


def _probe_readable(path: Path) -> tuple[bool, str | None]:
    """Open the container and read one frame; never decode the whole video."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency
        return True, f"readability not probed: {exc}"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, "OpenCV cannot open the container"
        ok, frame = capture.read()
        if not ok or frame is None:
            return False, "OpenCV opened the container but decoded no frame"
    finally:
        capture.release()
    return True, None


def discover_videos(data_root: str | Path, *, probe_readable: bool = False) -> VideoInventory:
    """List the MP4s present under `DATA_ROOT/video`.

    `probe_readable` opens each container and decodes one frame. It is off by default
    because scope resolution only needs to know which files exist.
    """
    root = Path(data_root).expanduser().resolve(strict=False)
    video_root = root / VIDEO_DIRNAME
    inventory = VideoInventory(data_root=root.as_posix(), video_root=video_root.as_posix())
    if not video_root.is_dir():
        return inventory
    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    for path in sorted(video_root.rglob("*.mp4"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        video_id = path.stem
        if video_id in seen:
            duplicates.add(video_id)
        else:
            seen[video_id] = path.as_posix()
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        readable, error = (True, None)
        if probe_readable:
            readable, error = _probe_readable(path)
        inventory.videos.append(
            VideoFile(
                video_id=video_id,
                path=path.as_posix(),
                size_bytes=size,
                readable=readable,
                error=error,
            )
        )
    inventory.duplicate_ids = sorted(duplicates)
    return inventory


def video_path_for(data_root: str | Path, video_id: str) -> Path:
    return Path(data_root) / VIDEO_DIRNAME / f"{video_id}.mp4"


def _stems(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob(f"*{suffix}") if path.is_file()}


def _folder_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.name for path in directory.iterdir() if path.is_dir()}


def _folder_has_files(directory: Path) -> bool:
    return directory.is_dir() and any(path.is_file() for path in directory.iterdir())


def support_coverage(
    data_root: str | Path, *, video_ids: Iterable[str] | None = None
) -> list[VideoSupport]:
    """Per-video availability of every source, for the given IDs or for all discovered."""
    root = Path(data_root).expanduser().resolve(strict=False)
    videos = _stems(root / VIDEO_DIRNAME, ".mp4")
    maps = _stems(root / MAP_DIRNAME, ".csv")
    features = _stems(root / FEATURE_DIRNAME, ".npy")
    keyframes = _folder_names(root / KEYFRAME_DIRNAME)
    objects = _folder_names(root / OBJECT_DIRNAME)
    media = _stems(root / MEDIA_INFO_DIRNAME, ".json")
    ids = (
        sorted({str(item) for item in video_ids})
        if video_ids is not None
        else sorted(videos | maps | features | keyframes | objects | media)
    )
    return [
        VideoSupport(
            video_id=video_id,
            video=video_id in videos,
            map=video_id in maps,
            clip=video_id in features,
            # An empty keyframes folder is not usable JPEG coverage.
            keyframe_jpeg=video_id in keyframes
            and _folder_has_files(root / KEYFRAME_DIRNAME / video_id),
            objects=video_id in objects,
            media_info=video_id in media,
        )
        for video_id in ids
    ]


def summarize_coverage(coverage: Iterable[VideoSupport]) -> dict[str, int]:
    items = list(coverage)
    return {
        "total": len(items),
        "video": sum(1 for item in items if item.video),
        "video_map_clip": sum(1 for item in items if item.video and item.retrieval_supported),
        "video_map_clip_jpeg": sum(
            1 for item in items if item.video and item.retrieval_supported and item.keyframe_jpeg
        ),
        "video_needing_mp4_fallback": sum(
            1
            for item in items
            if item.video and item.retrieval_supported and not item.keyframe_jpeg
        ),
        "video_without_map": sum(1 for item in items if item.video and not item.map),
        "video_without_clip": sum(1 for item in items if item.video and not item.clip),
        "map_clip_without_video": sum(
            1 for item in items if item.retrieval_supported and not item.video
        ),
    }


def retrieval_ready_video_ids(data_root: str | Path) -> tuple[str, ...]:
    """Every canonical video ID with map + CLIP support, MP4 present or not.

    This is the resolution behind scope mode `retrieval_ready`, and it is the honest
    definition of what the global coarse index can search: the official CLIP features
    and the keyframe map are what retrieval consumes. Requiring an MP4 as well — which
    is what `existing_videos` does — silently deletes retrieval coverage for videos
    whose supporting data is complete, in exchange for a preview capability that
    retrieval never uses.
    """
    return tuple(
        item.video_id for item in support_coverage(data_root) if item.retrieval_supported
    )


def existing_video_ids_with_retrieval_support(data_root: str | Path) -> tuple[str, ...]:
    """The video-backed development set: MP4s present INTERSECT map + CLIP present.

    This is the resolution behind scope mode `existing_videos`. It is computed from
    disk every time, so downloading another video changes the answer — and therefore
    the selected-IDs hash and the cache fingerprint.
    """
    return tuple(
        item.video_id
        for item in support_coverage(data_root)
        if item.video and item.retrieval_supported
    )


__all__ = [
    "VideoFile",
    "VideoInventory",
    "VideoSupport",
    "collection_of",
    "discover_videos",
    "existing_video_ids_with_retrieval_support",
    "retrieval_ready_video_ids",
    "summarize_coverage",
    "support_coverage",
    "video_path_for",
]
