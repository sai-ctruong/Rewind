"""Phase 3.2: on-demand visual frames, with the original MP4 as the fallback source.

Fixtures are tiny generated MP4s written with OpenCV; nothing here needs the network,
a model download, or the real dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.frame_provider import (
    SOURCE_KEYFRAME_JPEG,
    SOURCE_NONE,
    SOURCE_VIDEO_DECODE,
    DerivedFrameCache,
    FrameProvider,
)
from ingestion.schemas import RawKeyframe

cv2 = pytest.importorskip("cv2", reason="OpenCV is required to generate MP4 fixtures")

FPS = 10.0
FRAME_COUNT = 30
WIDTH = 64
HEIGHT = 48
VIDEO_ID = "L21_V028"


def write_video(path: Path, *, frame_count: int = FRAME_COUNT) -> Path:
    """Write a short MP4 whose blue channel encodes the frame index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():  # pragma: no cover - depends on local codec support
        pytest.skip("No MP4 encoder is available in this OpenCV build")
    try:
        for index in range(frame_count):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 8) % 256
            frame[:, :, 1] = 40
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:  # pragma: no cover
        pytest.skip("OpenCV produced no usable MP4 in this environment")
    return path


def write_jpeg(path: Path, *, red: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 2] = red
    assert cv2.imwrite(str(path), frame)
    return path


def make_record(
    *,
    keyframe_id: str = f"{VIDEO_ID}/kf_000002",
    ordinal: int = 2,
    frame_idx: int = 12,
    timestamp: float | None = None,
    image_path: Path | None = None,
    source_video: Path | None = None,
) -> RawKeyframe:
    return RawKeyframe(
        id=keyframe_id,
        video_id=VIDEO_ID,
        timestamp=frame_idx / FPS if timestamp is None else timestamp,
        image_path=None if image_path is None else str(image_path),
        source_video=None if source_video is None else str(source_video),
        frame_idx=frame_idx,
        keyframe_ordinal=ordinal,
    )


def provider(tmp_path: Path, *, data_root: Path | None = None) -> FrameProvider:
    return FrameProvider(data_root, cache_dir=tmp_path / "artifacts" / "video_frame_cache")


def decode_red_blue(payload: bytes) -> tuple[int, int]:
    array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert array is not None
    blue, _green, red = (int(array[0, 0, channel]) for channel in range(3))
    return red, blue


# ------------------------------------------------------------------- priority


def test_existing_keyframe_jpeg_is_preferred(tmp_path) -> None:
    jpeg = write_jpeg(tmp_path / "data" / "keyframes" / VIDEO_ID / "002.jpg")
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(image_path=jpeg, source_video=video)
    result = provider(tmp_path).get_frame(record)
    assert result.available
    assert result.source == SOURCE_KEYFRAME_JPEG
    red, _blue = decode_red_blue(result.image_bytes)
    assert red > 150, "served the JPEG, not a decoded video frame"


def test_missing_keyframe_jpeg_falls_back_to_the_video(tmp_path) -> None:
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(image_path=None, source_video=video)
    result = provider(tmp_path).get_frame(record)
    assert result.available
    assert result.source == SOURCE_VIDEO_DECODE
    assert result.seek_method in {"frame_index", "timestamp"}


def test_prefer_keyframe_jpeg_false_forces_the_video_path(tmp_path) -> None:
    jpeg = write_jpeg(tmp_path / "data" / "keyframes" / VIDEO_ID / "002.jpg")
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(image_path=jpeg, source_video=video)
    result = provider(tmp_path).get_frame(record, prefer_keyframe_jpeg=False)
    assert result.source == SOURCE_VIDEO_DECODE
    red, blue = decode_red_blue(result.image_bytes)
    assert red < 100 and blue > 0, "decoded the video frame, not the JPEG"
    # The JPEG on disk is untouched.
    assert jpeg.is_file()


def test_video_can_be_found_from_data_root_without_source_video(tmp_path) -> None:
    write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(image_path=None, source_video=None)
    result = provider(tmp_path, data_root=tmp_path / "data").get_frame(record)
    assert result.available and result.source == SOURCE_VIDEO_DECODE


def test_no_visual_source_returns_an_explicit_unavailable_result(tmp_path) -> None:
    record = make_record(image_path=None, source_video=None)
    result = provider(tmp_path).get_frame(record)
    assert not result.available
    assert result.image_bytes is None
    assert result.source == SOURCE_NONE
    assert "No visual source" in result.warning
    # The official identifiers survive the failure.
    assert result.frame_idx == 12 and result.requested_frame_idx == 12


def test_corrupt_video_gives_a_structured_error_not_an_exception(tmp_path) -> None:
    broken = tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"this is not an mp4 container")
    record = make_record(image_path=None, source_video=broken)
    result = provider(tmp_path).get_frame(record)
    assert not result.available
    assert result.source == SOURCE_NONE
    assert result.warning


# ------------------------------------------------------- official identifiers


def test_decoded_frame_never_changes_the_official_frame_idx(tmp_path) -> None:
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(frame_idx=17, ordinal=3, source_video=video)
    result = provider(tmp_path).get_frame(record)
    assert result.available
    assert result.frame_idx == 17
    assert result.requested_frame_idx == 17
    # Where the decoder landed is reported separately and never overwrites the above.
    assert result.decoded_frame_idx is not None


def test_frame_index_seek_lands_on_the_requested_frame(tmp_path) -> None:
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    result = provider(tmp_path).get_frame(make_record(frame_idx=20, source_video=video))
    assert result.seek_method == "frame_index"
    assert result.decoded_frame_idx == 20
    # mp4v is lossy, so compare within a tolerance far tighter than the 8-per-frame
    # step: this still fails if the decoder landed on a neighbouring frame.
    _red, blue = decode_red_blue(result.image_bytes)
    assert abs(blue - (20 * 8) % 256) <= 6, "decoded the frame that was actually requested"


# -------------------------------------------------------------- derived cache


def test_derived_frames_are_never_written_into_official_data(tmp_path) -> None:
    data_root = tmp_path / "data"
    video = write_video(data_root / "video" / f"{VIDEO_ID}.mp4")
    keyframes_dir = data_root / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    before = sorted(path.as_posix() for path in keyframes_dir.rglob("*"))
    result = provider(tmp_path, data_root=data_root).get_frame(
        make_record(image_path=None, source_video=video)
    )
    assert result.available
    assert sorted(path.as_posix() for path in keyframes_dir.rglob("*")) == before
    cache_root = tmp_path / "artifacts" / "video_frame_cache"
    written = list(cache_root.rglob("*.jpg"))
    assert written, "the derived frame should live under artifacts/"
    for path in written:
        assert data_root not in path.parents


def test_derived_frame_cache_is_reused_on_the_second_request(tmp_path) -> None:
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(image_path=None, source_video=video)
    shared = provider(tmp_path)
    first = shared.get_frame(record)
    second = shared.get_frame(record)
    assert first.available and second.available
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.seek_method == "cache"
    assert first.image_bytes == second.image_bytes


def test_cache_key_is_deterministic() -> None:
    cache = DerivedFrameCache("artifacts/video_frame_cache")
    assert cache.key("L21_V028", 12) == cache.key("L21_V028", 12) == "L21_V028/frame_00000012"
    assert cache.key("L21_V028", 12) != cache.key("L21_V029", 12)
    assert cache.key("L21_V028", 12) != cache.key("L21_V028", 13)
    assert cache.path_for("L21_V028", 12).name == "frame_00000012.jpg"


def test_replacing_the_source_video_invalidates_its_cached_frames(tmp_path) -> None:
    path = tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4"
    write_video(path)
    record = make_record(image_path=None, source_video=path)
    shared = provider(tmp_path)
    assert shared.get_frame(record).available
    assert shared.get_frame(record).cache_hit
    # A re-download changes size/mtime, so the stale derived frame is not served.
    write_video(path, frame_count=FRAME_COUNT + 5)
    assert shared.get_frame(record).cache_hit is False


def test_cache_can_be_disabled(tmp_path) -> None:
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    record = make_record(image_path=None, source_video=video)
    shared = FrameProvider(cache_dir=tmp_path / "artifacts" / "frames", cache_enabled=False)
    assert shared.get_frame(record).available
    assert shared.get_frame(record).cache_hit is False
    assert not list((tmp_path / "artifacts" / "frames").rglob("*.jpg"))


# ------------------------------------------------------------------- describe


def test_describe_reports_availability_without_decoding(tmp_path) -> None:
    data_root = tmp_path / "data"
    jpeg = write_jpeg(data_root / "keyframes" / VIDEO_ID / "002.jpg")
    video = write_video(data_root / "video" / f"{VIDEO_ID}.mp4")
    shared = provider(tmp_path, data_root=data_root)

    with_jpeg = shared.describe(make_record(image_path=jpeg, source_video=video))
    assert with_jpeg == {
        "image_available": True,
        "image_source": SOURCE_KEYFRAME_JPEG,
        "keyframe_jpeg_available": True,
        "video_available": True,
    }

    without_jpeg = shared.describe(make_record(image_path=None, source_video=video))
    assert without_jpeg["image_source"] == SOURCE_VIDEO_DECODE
    assert without_jpeg["image_available"] is True

    nothing = FrameProvider().describe(make_record(image_path=None, source_video=None))
    assert nothing["image_source"] == SOURCE_NONE
    assert nothing["image_available"] is False

    # describe() must not have populated the derived cache.
    assert not list((tmp_path / "artifacts").rglob("*.jpg"))


def test_result_dict_hides_bytes_by_default(tmp_path) -> None:
    video = write_video(tmp_path / "data" / "video" / f"{VIDEO_ID}.mp4")
    result = provider(tmp_path).get_frame(make_record(image_path=None, source_video=video))
    payload = result.to_dict()
    assert "image_bytes" not in payload
    assert payload["available"] is True and payload["source"] == SOURCE_VIDEO_DECODE
    assert result.to_dict(include_bytes=True)["image_bytes"] == result.image_bytes
