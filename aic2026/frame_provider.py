"""On-demand visual frames for mapped AIC keyframes.

Official AIC documentation makes the **video** the competition data and the keyframe
JPEGs supporting data. This module encodes that priority: it serves a BTC keyframe
JPEG when one exists, and otherwise decodes the exact mapped frame from the original
MP4. Global retrieval never depends on it — it runs on the official CLIP features —
so a failure here degrades visualization, not search.

Three rules the implementation must never break:

1. The official `frame_idx` is the submission identifier and is never changed by what
   the decoder does. `decoded_frame_idx` records where the decoder actually landed,
   separately.
2. Derived frames are written only under the configured artifacts cache directory,
   never into `data/keyframes/`. They are disposable, and must never be mistaken for
   BTC-provided keyframes.
3. Nothing is invented. A missing image is reported as unavailable; a neighbouring
   keyframe is never substituted.

Decoding is lazy: nothing here runs until a caller actually asks for a frame, so a
Top-100 search does no video I/O.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .video_inventory import video_path_for

DEFAULT_FRAME_CACHE_DIR = Path("artifacts") / "video_frame_cache"
JPEG_QUALITY = 90
# Reading forward is cheaper than re-seeking for small gaps; past this many frames a
# fresh seek wins. Only affects speed, never which frame is returned.
SEQUENTIAL_READ_LIMIT = 24

SOURCE_KEYFRAME_JPEG = "keyframe_jpeg"
SOURCE_VIDEO_DECODE = "video_decode"
SOURCE_NONE = "none"


class MappedKeyframe(Protocol):
    """What a frame request needs; `RawKeyframe` satisfies this structurally."""

    id: str
    video_id: str
    timestamp: float
    image_path: str | None
    source_video: str | None
    frame_idx: int | None
    keyframe_ordinal: int | None


@dataclass(frozen=True)
class VideoMetadata:
    """Container facts needed to plan a bounded sample window."""

    fps: float
    frame_count: int
    duration_s: float

    @property
    def usable(self) -> bool:
        return self.fps > 0.0 and self.frame_count > 0


@dataclass(frozen=True)
class DecodedFrame:
    """One frame decoded from an MP4, kept in memory as a BGR array.

    `requested_frame_idx` is what the sample plan asked for and is the identity used
    everywhere downstream; `decoded_frame_idx` records where the decoder actually
    landed, and the two are reported separately for the same reason as in `FrameResult`.
    """

    requested_frame_idx: int
    decoded_frame_idx: int
    timestamp: float
    image: Any


@dataclass(frozen=True)
class FrameResult:
    """Outcome of one frame request. `available` is false rather than raising."""

    video_id: str
    source: str
    image_bytes: bytes | None = None
    frame_idx: int | None = None
    timestamp: float | None = None
    requested_frame_idx: int | None = None
    decoded_frame_idx: int | None = None
    seek_method: str | None = None
    cache_hit: bool = False
    cache_path: str | None = None
    warning: str | None = None

    @property
    def available(self) -> bool:
        return self.image_bytes is not None

    def to_dict(self, *, include_bytes: bool = False) -> dict[str, Any]:
        data = {
            "available": self.available,
            "video_id": self.video_id,
            "source": self.source,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "requested_frame_idx": self.requested_frame_idx,
            "decoded_frame_idx": self.decoded_frame_idx,
            "seek_method": self.seek_method,
            "cache_hit": self.cache_hit,
            "warning": self.warning,
        }
        if include_bytes:
            data["image_bytes"] = self.image_bytes
        return data


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DerivedFrameCache:
    """Disk cache of frames decoded from MP4s.

    Keyed by video ID and the requested official `frame_idx`, so the key is
    deterministic and readable. Staleness is detected from the source MP4's size and
    mtime, recorded in a sidecar next to the image: re-downloading a video invalidates
    its frames without a global purge.

    This cache is a disposable visual artifact. It is deliberately NOT part of the
    retrieval cache manifest and never gates an index build.
    """

    def __init__(self, cache_dir: str | Path | None = None, *, enabled: bool = True):
        self.cache_dir = Path(cache_dir or DEFAULT_FRAME_CACHE_DIR)
        self.enabled = bool(enabled)

    def key(self, video_id: str, frame_idx: int) -> str:
        return f"{video_id}/frame_{int(frame_idx):08d}"

    def path_for(self, video_id: str, frame_idx: int) -> Path:
        return self.cache_dir / video_id / f"frame_{int(frame_idx):08d}.jpg"

    @staticmethod
    def _source_stamp(source_video: str | Path | None) -> str:
        if source_video is None:
            return "no-source"
        try:
            stat = Path(source_video).stat()
        except OSError:
            return "missing-source"
        return f"{int(stat.st_size)}:{int(stat.st_mtime_ns)}"

    def load(self, video_id: str, frame_idx: int, source_video: str | Path | None) -> bytes | None:
        if not self.enabled:
            return None
        path = self.path_for(video_id, frame_idx)
        sidecar = path.with_suffix(".source")
        if not path.is_file():
            return None
        try:
            stamp = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if stamp != self._source_stamp(source_video):
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def store(
        self, video_id: str, frame_idx: int, payload: bytes, source_video: str | Path | None
    ) -> Path | None:
        if not self.enabled:
            return None
        path = self.path_for(video_id, frame_idx)
        try:
            _atomic_write_bytes(path, payload)
            _atomic_write_bytes(
                path.with_suffix(".source"), self._source_stamp(source_video).encode("utf-8")
            )
        except OSError:
            return None
        return path


class FrameProvider:
    """Resolve a mapped keyframe to displayable JPEG bytes."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        *,
        cache: DerivedFrameCache | None = None,
        cache_dir: str | Path | None = None,
        cache_enabled: bool = True,
    ):
        self.data_root = None if data_root is None else Path(data_root)
        self.cache = cache or DerivedFrameCache(cache_dir, enabled=cache_enabled)
        # Container properties only (fps, frame count): no pixels, tiny, and re-read
        # whenever the file's size or mtime changes.
        self._metadata: dict[str, tuple[str, VideoMetadata]] = {}

    # ------------------------------------------------------------------ resolution

    def _source_video(self, record: MappedKeyframe) -> Path | None:
        source = getattr(record, "source_video", None)
        if source:
            path = Path(source)
            if path.is_file():
                return path
        if self.data_root is not None:
            path = video_path_for(self.data_root, record.video_id)
            if path.is_file():
                return path
        return None

    def video_path(self, video_id: str) -> Path | None:
        """Locate the original MP4 for a video ID under the ACTIVE data root."""
        if self.data_root is None:
            return None
        path = video_path_for(self.data_root, str(video_id))
        return path if path.is_file() else None

    @staticmethod
    def _keyframe_jpeg(record: MappedKeyframe) -> Path | None:
        image_path = getattr(record, "image_path", None)
        if not image_path:
            return None
        path = Path(image_path)
        return path if path.is_file() else None

    def describe(self, record: MappedKeyframe) -> dict[str, Any]:
        """Cheap availability facts for a record: no decoding, no file reads."""
        jpeg = self._keyframe_jpeg(record)
        video = self._source_video(record)
        if jpeg is not None:
            source = SOURCE_KEYFRAME_JPEG
        elif video is not None:
            source = SOURCE_VIDEO_DECODE
        else:
            source = SOURCE_NONE
        return {
            "image_available": source != SOURCE_NONE,
            "image_source": source,
            "keyframe_jpeg_available": jpeg is not None,
            "video_available": video is not None,
        }

    # --------------------------------------------------------------------- frames

    def get_frame(
        self,
        record: MappedKeyframe,
        *,
        prefer_keyframe_jpeg: bool = True,
    ) -> FrameResult:
        """Return JPEG bytes for one mapped keyframe.

        Priority: BTC keyframe JPEG, then MP4 decode, then an explicit unavailable
        result. `prefer_keyframe_jpeg=False` forces the MP4 path, which is how the
        video fallback is exercised without touching official data.
        """
        video_id = str(record.video_id)
        frame_idx = getattr(record, "frame_idx", None)
        timestamp = float(getattr(record, "timestamp", 0.0) or 0.0)
        requested = None if frame_idx is None else int(frame_idx)

        if prefer_keyframe_jpeg:
            jpeg = self._keyframe_jpeg(record)
            if jpeg is not None:
                try:
                    payload = jpeg.read_bytes()
                except OSError as exc:
                    payload = None
                    jpeg_warning = f"Cannot read keyframe JPEG {jpeg}: {exc}"
                else:
                    jpeg_warning = None
                if payload:
                    return FrameResult(
                        video_id=video_id,
                        source=SOURCE_KEYFRAME_JPEG,
                        image_bytes=payload,
                        frame_idx=requested,
                        timestamp=timestamp,
                        requested_frame_idx=requested,
                    )
                return self._decode(record, requested, timestamp, warning=jpeg_warning)

        return self._decode(record, requested, timestamp)

    # ------------------------------------------------------ low-level video access

    def video_metadata(
        self, video_id: str, *, source_video: str | Path | None = None
    ) -> VideoMetadata | None:
        """FPS and frame count for one video, or None when it cannot be opened.

        Needed to clamp a local sample window to the real video, because AIC videos do
        not all share one frame rate. Cached per file identity: opening a container to
        read two properties is cheap but not free, and a refinement request asks for
        the same video several times.
        """
        source = Path(source_video) if source_video else self.video_path(video_id)
        if source is None or not source.is_file():
            return None
        try:
            stat = source.stat()
            stamp = f"{source}|{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return None
        cached = self._metadata.get(str(video_id))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        metadata = _read_video_metadata(source)
        if metadata is not None:
            self._metadata[str(video_id)] = (stamp, metadata)
        return metadata

    def decode_frames(
        self,
        video_id: str,
        frame_indices: Sequence[int],
        *,
        source_video: str | Path | None = None,
    ) -> tuple[list[DecodedFrame], str | None]:
        """Decode several frames of one video with a SINGLE capture.

        This is the shared video-access primitive for local refinement. Opening the
        container once and reading forward between nearby targets is what keeps a
        bounded window affordable; one open-and-seek per frame would not be.

        Returns the frames that could be decoded plus an optional warning. It never
        raises for a missing or unreadable video: a visualization/refinement failure
        must never fail a retrieval request.
        """
        source = Path(source_video) if source_video else self.video_path(video_id)
        if source is None or not source.is_file():
            return [], f"No original MP4 is available for video {video_id!r}."
        return _decode_frames_bgr(source, frame_indices)

    def get_video_frame(
        self,
        video_id: str,
        *,
        frame_idx: int | None = None,
        timestamp: float | None = None,
        source_video: str | Path | None = None,
    ) -> FrameResult:
        """Decode ONE arbitrary frame of a video, by index or by timestamp.

        The keyframe-record API (`get_frame`) stays exactly as it was; this is the
        addition Phase 5 needs, because a refined frame is generally not a mapped
        keyframe and therefore has no record.
        """
        source = Path(source_video) if source_video else self.video_path(video_id)
        if source is None or not source.is_file():
            return FrameResult(
                video_id=str(video_id),
                source=SOURCE_NONE,
                frame_idx=frame_idx,
                timestamp=timestamp,
                requested_frame_idx=frame_idx,
                warning=f"No original MP4 is available for video {video_id!r}.",
            )
        record = _AdHocFrame(
            id=f"{video_id}/decoded",
            video_id=str(video_id),
            timestamp=float(timestamp or 0.0),
            image_path=None,
            source_video=str(source),
            frame_idx=None if frame_idx is None else int(frame_idx),
            keyframe_ordinal=None,
        )
        return self._decode(record, record.frame_idx, record.timestamp)

    def _decode(
        self,
        record: MappedKeyframe,
        requested: int | None,
        timestamp: float,
        *,
        warning: str | None = None,
    ) -> FrameResult:
        video_id = str(record.video_id)
        source_video = self._source_video(record)
        if source_video is None:
            return FrameResult(
                video_id=video_id,
                source=SOURCE_NONE,
                frame_idx=requested,
                timestamp=timestamp,
                requested_frame_idx=requested,
                warning=_join(
                    warning,
                    "No visual source: neither a BTC keyframe JPEG nor the original MP4 "
                    "is available for this keyframe.",
                ),
            )

        cache_key_frame = requested if requested is not None else int(round(timestamp * 1000))
        cached = self.cache.load(video_id, cache_key_frame, source_video)
        if cached is not None:
            return FrameResult(
                video_id=video_id,
                source=SOURCE_VIDEO_DECODE,
                image_bytes=cached,
                frame_idx=requested,
                timestamp=timestamp,
                requested_frame_idx=requested,
                decoded_frame_idx=requested,
                seek_method="cache",
                cache_hit=True,
                cache_path=str(self.cache.path_for(video_id, cache_key_frame)),
                warning=warning,
            )

        payload, decoded_idx, seek_method, decode_warning = _decode_frame_jpeg(
            source_video, requested, timestamp
        )
        if payload is None:
            return FrameResult(
                video_id=video_id,
                source=SOURCE_NONE,
                frame_idx=requested,
                timestamp=timestamp,
                requested_frame_idx=requested,
                seek_method=seek_method,
                warning=_join(warning, decode_warning),
            )
        cache_path = self.cache.store(video_id, cache_key_frame, payload, source_video)
        return FrameResult(
            video_id=video_id,
            source=SOURCE_VIDEO_DECODE,
            image_bytes=payload,
            # The official submission frame is what was REQUESTED; decoder behaviour is
            # reported separately and never rewrites it.
            frame_idx=requested,
            timestamp=timestamp,
            requested_frame_idx=requested,
            decoded_frame_idx=decoded_idx,
            seek_method=seek_method,
            cache_path=None if cache_path is None else str(cache_path),
            warning=_join(warning, decode_warning),
        )


@dataclass(frozen=True)
class _AdHocFrame:
    """A `MappedKeyframe` for a frame that has no BTC map row."""

    id: str
    video_id: str
    timestamp: float
    image_path: str | None
    source_video: str | None
    frame_idx: int | None
    keyframe_ordinal: int | None


def _join(*parts: str | None) -> str | None:
    kept = [part for part in parts if part]
    return " ".join(kept) if kept else None


def _read_video_metadata(source_video: Path) -> VideoMetadata | None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV is a project dependency
        return None
    capture = cv2.VideoCapture(str(source_video))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:  # pragma: no cover - defensive: containers vary
        return None
    finally:
        capture.release()
    if not (fps > 0.0):
        return None
    duration = float(frame_count) / fps if frame_count > 0 else 0.0
    return VideoMetadata(fps=fps, frame_count=max(0, frame_count), duration_s=duration)


def _decode_frames_bgr(
    source_video: Path, frame_indices: Sequence[int]
) -> tuple[list[DecodedFrame], str | None]:
    """Decode the requested frame indices from one capture, in ascending order.

    Deterministic: the same request against the same file yields the same frames in the
    same order, regardless of the order the caller listed them in.
    """
    wanted = sorted({int(index) for index in frame_indices if int(index) >= 0})
    if not wanted:
        return [], None
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency
        return [], f"OpenCV is unavailable: {exc}"

    capture = cv2.VideoCapture(str(source_video))
    decoded: list[DecodedFrame] = []
    failures = 0
    try:
        if not capture.isOpened():
            return [], f"OpenCV cannot open {source_video.name}."
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        position: int | None = None  # index the next read() would return
        for target in wanted:
            if position is None or target < position or target - position > SEQUENTIAL_READ_LIMIT:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                position = int(target)
            while position < target:
                ok, _ = capture.read()
                if not ok:
                    break
                position += 1
            if position != target:
                failures += 1
                position = None
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                position = None
                continue
            landed = position
            position += 1
            decoded.append(
                DecodedFrame(
                    requested_frame_idx=int(target),
                    decoded_frame_idx=int(landed),
                    timestamp=(float(target) / fps) if fps > 0 else 0.0,
                    image=frame,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive: corrupt containers vary
        return decoded, f"Decoding {source_video.name} failed: {type(exc).__name__}: {exc}"
    finally:
        capture.release()
    warning = (
        f"{failures} of {len(wanted)} requested frames could not be decoded from "
        f"{source_video.name}."
        if failures
        else None
    )
    return decoded, warning


def _decode_frame_jpeg(
    source_video: Path, frame_idx: int | None, timestamp: float
) -> tuple[bytes | None, int | None, str, str | None]:
    """Decode one frame with OpenCV, preferring an exact frame-index seek.

    Frame-index seek was verified exact against the real L21 MP4s (requested == landed
    at the first, second, middle, and last mapped rows). Timestamp seek is kept as a
    fallback for containers where the index seek fails.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency
        return None, None, "none", f"OpenCV is unavailable: {exc}"

    capture = cv2.VideoCapture(str(source_video))
    try:
        if not capture.isOpened():
            return None, None, "none", f"OpenCV cannot open {source_video}."
        attempts: list[tuple[str, float, int]] = []
        if frame_idx is not None and frame_idx >= 0:
            attempts.append(("frame_index", cv2.CAP_PROP_POS_FRAMES, int(frame_idx)))
        attempts.append(("timestamp", cv2.CAP_PROP_POS_MSEC, int(max(0.0, timestamp) * 1000)))
        last_warning: str | None = None
        for method, prop, value in attempts:
            capture.set(prop, value)
            ok, frame = capture.read()
            if not ok or frame is None:
                last_warning = f"{method} seek returned no frame from {source_video.name}."
                continue
            landed = capture.get(cv2.CAP_PROP_POS_FRAMES)
            decoded_idx = None if landed is None else int(landed) - 1
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not encoded:
                last_warning = f"Decoded frame from {source_video.name} could not be JPEG encoded."
                continue
            warning = None
            if frame_idx is not None and decoded_idx is not None and decoded_idx != frame_idx:
                warning = (
                    f"Decoder landed on frame {decoded_idx} for requested official frame "
                    f"{frame_idx}; the submission frame_idx is unchanged."
                )
            return buffer.tobytes(), decoded_idx, method, warning
        return None, None, "none", last_warning or f"No frame could be decoded from {source_video.name}."
    except Exception as exc:  # pragma: no cover - defensive: corrupt containers vary
        return None, None, "none", f"Decoding {source_video.name} failed: {type(exc).__name__}: {exc}"
    finally:
        capture.release()


__all__ = [
    "DEFAULT_FRAME_CACHE_DIR",
    "SEQUENTIAL_READ_LIMIT",
    "SOURCE_KEYFRAME_JPEG",
    "SOURCE_NONE",
    "SOURCE_VIDEO_DECODE",
    "DecodedFrame",
    "DerivedFrameCache",
    "FrameProvider",
    "FrameResult",
    "MappedKeyframe",
    "VideoMetadata",
]
