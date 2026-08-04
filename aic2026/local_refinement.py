"""Bounded local frame refinement with a keyframe-only fallback."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from retrieval.video_engine import VideoIndexEntry


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    uncertainty_only: bool = True
    top_hypotheses: int = 10
    window_before_s: float = 4.0
    window_after_s: float = 4.0
    fine_fps: float = 4.0
    max_frames: int = 32
    batch_size: int = 16
    margin_threshold: float = 0.03
    cache_size_mb: int = 256


@dataclass(frozen=True)
class RefinementResult:
    selected_frame_id: str
    selected_timestamp: float
    confidence: float
    refinement_mode: str
    evidence_frame_ids: tuple[str, ...]
    decoded_frame_count: int
    warning: str | None = None


class LocalFrameRefiner:
    def __init__(self, config: RefinementConfig | None = None):
        self.config = config or RefinementConfig()
        self._cache: OrderedDict[tuple, list[tuple[int, float, np.ndarray]]] = OrderedDict()
        self._cache_bytes = 0

    def _keyframe_fallback(self, entry: VideoIndexEntry, keyframe_id: str) -> RefinementResult:
        raw = entry.raws[keyframe_id]
        neighbors = sorted(
            (item for item in entry.raws.values() if item.video_id == raw.video_id),
            key=lambda item: item.timestamp,
        )
        position = next(i for i, item in enumerate(neighbors) if item.id == keyframe_id)
        evidence = neighbors[max(0, position - 2): position + 3]
        return RefinementResult(
            str(raw.frame_idx), raw.timestamp, 0.0, "keyframe_only",
            tuple(str(item.frame_idx) for item in evidence), 0,
            "Original MP4 or a compatible visual frame scorer is unavailable; frame accuracy is limited to mapped keyframes.",
        )

    def refine(
        self,
        entry: VideoIndexEntry,
        keyframe_id: str,
        frame_scorer: Callable[[Sequence[np.ndarray]], Sequence[float]] | None = None,
    ) -> RefinementResult:
        raw = entry.raws[keyframe_id]
        if not raw.source_video or frame_scorer is None:
            return self._keyframe_fallback(entry, keyframe_id)
        import cv2

        low = max(0.0, raw.timestamp - self.config.window_before_s)
        high = raw.timestamp + self.config.window_after_s
        cache_key = (raw.source_video, round(low, 3), round(high, 3), self.config.fine_fps, self.config.max_frames)
        decoded = self._cache.get(cache_key)
        if decoded is None:
            cap = cv2.VideoCapture(raw.source_video)
            if not cap.isOpened():
                return self._keyframe_fallback(entry, keyframe_id)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            step = max(1, round(fps / max(0.1, self.config.fine_fps)))
            start_frame, end_frame = int(low * fps), int(high * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            decoded = []
            frame_no = start_frame
            try:
                while frame_no <= end_frame and len(decoded) < self.config.max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if (frame_no - start_frame) % step == 0:
                        decoded.append((frame_no, frame_no / fps, frame))
                    frame_no += 1
            finally:
                cap.release()
            self._cache[cache_key] = decoded
            self._cache_bytes += sum(frame.nbytes for _, _, frame in decoded)
            limit = self.config.cache_size_mb * 1024 * 1024
            while self._cache and self._cache_bytes > limit:
                _, removed = self._cache.popitem(last=False)
                self._cache_bytes -= sum(frame.nbytes for _, _, frame in removed)
        if not decoded:
            return self._keyframe_fallback(entry, keyframe_id)
        scores = np.asarray(frame_scorer([frame for _, _, frame in decoded]), dtype=np.float32)
        if scores.shape != (len(decoded),) or not np.isfinite(scores).all():
            raise ValueError("Frame scorer must return one finite score per decoded frame.")
        best = int(np.argmax(scores))
        frame_no, timestamp, _ = decoded[best]
        margin = float(scores[best] - np.partition(scores, -2)[-2]) if len(scores) > 1 else float(scores[best])
        return RefinementResult(str(frame_no), timestamp, max(0.0, min(1.0, margin)), "mp4", tuple(str(n) for n, _, _ in decoded), len(decoded))
