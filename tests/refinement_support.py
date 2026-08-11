"""Offline test doubles and synthetic MP4s for Phase 5 local refinement.

Everything here is deterministic and needs no model, no network, and no AIC data. The
synthetic videos encode a frame's index in its pixels, so a fake scorer can be told to
prefer a specific frame and the test can assert that local search actually found it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from aic2026.frame_scorer import ScorerStatus
from ingestion.build_index import KeyframeIndex
from ingestion.schemas import RawKeyframe
from retrieval.video_engine import VideoIndexEntry

# Frame i is painted with blue = BASE + STEP * i. mp4v is lossy, so the step is wide
# enough that a decoded frame still rounds back to its own index.
PIXEL_BASE = 10
PIXEL_STEP = 8
# Beyond this the encoding would saturate at 255 and several frames would decode to the
# same index, which would silently make a sampling test meaningless.
MAX_ENCODABLE_FRAMES = (255 - PIXEL_BASE) // PIXEL_STEP + 1


def frame_value(frame_idx: int) -> int:
    value = PIXEL_BASE + PIXEL_STEP * int(frame_idx)
    if value > 255:
        raise ValueError(
            f"frame {frame_idx} cannot be encoded; synthetic videos are limited to "
            f"{MAX_ENCODABLE_FRAMES} frames."
        )
    return value


def recover_frame_idx(image: np.ndarray) -> int:
    """Invert `frame_value` from a decoded frame's blue channel (OpenCV is BGR)."""
    blue = float(np.asarray(image)[:, :, 0].mean())
    return int(round((blue - PIXEL_BASE) / PIXEL_STEP))


def write_synthetic_video(
    path: Path, *, frames: int = 31, fps: float = 10.0, size: tuple[int, int] = (64, 48)
) -> Path:
    """Write an MP4 whose every frame carries its own index in the blue channel."""
    if frames > MAX_ENCODABLE_FRAMES:
        raise ValueError(
            f"{frames} frames cannot be encoded distinctly; the limit is "
            f"{MAX_ENCODABLE_FRAMES}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        for index in range(frames):
            image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            image[:, :, 0] = frame_value(index)
            writer.write(image)
    finally:
        writer.release()
    return path


class FakeFrameScorer:
    """A scorer that prefers one known frame index. Deterministic, no model.

    It reads the frame index back out of the pixels, so it exercises the real decode
    path: if sampling returned the wrong frames, the scores would not line up.
    """

    def __init__(self, target_frame_idx: int, *, fail: bool = False, non_finite: bool = False):
        self.target_frame_idx = int(target_frame_idx)
        self.fail = bool(fail)
        self.non_finite = bool(non_finite)
        self.prepare_calls = 0
        self.score_calls = 0
        self.batch_sizes: list[int] = []
        self.queries: list[str] = []

    def prepare_query(self, query: str):
        self.prepare_calls += 1
        self.queries.append(str(query))
        if self.fail == "prepare":
            raise RuntimeError("fake scorer cannot prepare a query")
        return {"query": str(query)}

    def score_frames(self, prepared_query, frames: Sequence[np.ndarray]) -> list[float]:
        self.score_calls += 1
        self.batch_sizes.append(len(frames))
        if self.fail is True:
            raise RuntimeError("fake scorer failed")
        scores = []
        for image in frames:
            index = recover_frame_idx(image)
            scores.append(1.0 - 0.01 * abs(index - self.target_frame_idx))
        if self.non_finite and scores:
            scores[0] = float("nan")
        return scores

    def status(self, *, initialize: bool = False) -> ScorerStatus:
        return ScorerStatus(
            backend="fake",
            model_name="fake-deterministic",
            device="cpu",
            state="ready",
            available=True,
            production_ready=False,
            warning="Test scorer; never used in production.",
        )


class UnavailableScorer:
    """A scorer whose model cannot be loaded, as a real one may be offline."""

    def __init__(self, message: str = "model weights are unavailable"):
        self.message = message
        self.prepare_calls = 0

    def prepare_query(self, query: str):
        self.prepare_calls += 1
        raise RuntimeError(self.message)

    def score_frames(self, prepared_query, frames):  # pragma: no cover - never reached
        raise AssertionError("score_frames must not be called when the query failed")

    def status(self, *, initialize: bool = False) -> ScorerStatus:
        return ScorerStatus(
            backend="fake",
            model_name="unavailable",
            device="cpu",
            state="unavailable",
            available=False,
            fallback_reason=self.message,
        )


def make_entry(raws: dict[str, RawKeyframe]) -> VideoIndexEntry:
    index = KeyframeIndex(
        ids=list(raws),
        video_ids=[raws[key].video_id for key in raws],
        timestamps=[raws[key].timestamp for key in raws],
        objects=[[] for _ in raws],
    )
    return VideoIndexEntry("dataset", index, raws, len(raws), len(raws))


__all__ = [
    "MAX_ENCODABLE_FRAMES",
    "PIXEL_BASE",
    "PIXEL_STEP",
    "FakeFrameScorer",
    "UnavailableScorer",
    "frame_value",
    "make_entry",
    "recover_frame_idx",
    "write_synthetic_video",
]

