"""Offline fixtures and fake backends for Phase 6 grounded Q&A.

The central fixture is a two-video dataset whose videos are visually unambiguous: one is
red, one is blue. A fake visual backend answers from the pixels it is handed, so if an
answer ever crossed from one video to another the test would see "red" attached to the
blue video. That is the whole point.

Nothing here needs a network, an API key, or a downloaded model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from aic2026.config import app_config_from_dict
from aic2026.qa import (
    ANSWER_STATUS_ANSWERED,
    BACKEND_STATE_NOT_AVAILABLE,
    BACKEND_STATE_READY,
    QAAnswererStatus,
    QAAnswerResult,
    QABackendUnavailable,
    QAEvidenceBundle,
    canonical_answer_type,
    normalize_answer,
)

FPS = 10.0
FRAMES_PER_VIDEO = 31
FRAME_IDS = (5, 15, 25)
# (video_id, BGR colour, the word a visual backend should read off its frames)
VIDEO_COLOURS = {
    "L21_V001": ((0, 0, 220), "red"),
    "L21_V002": ((220, 0, 0), "blue"),
    "L21_V003": ((0, 200, 0), "green"),
}


def colour_of_frame(image: np.ndarray) -> str:
    """Read a video's colour back out of one decoded frame (OpenCV BGR order)."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        blue, green, red = (float(array[:, :, i].mean()) for i in range(3))
    else:  # pragma: no cover - defensive
        return "unknown"
    dominant = max((red, "red"), (green, "green"), (blue, "blue"))
    return dominant[1] if dominant[0] > 40 else "unknown"


def colour_of_jpeg(payload: bytes) -> str:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return "unknown" if decoded is None else colour_of_frame(decoded)


def write_colour_video(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    try:
        for _ in range(FRAMES_PER_VIDEO):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :] = colour
            writer.write(frame)
    finally:
        writer.release()
    return path


def make_qa_root(
    root: Path,
    *,
    videos: Sequence[str] = ("L21_V001", "L21_V002"),
    jpeg_videos: Sequence[str] | None = None,
    captions: dict[str, str] | None = None,
) -> Path:
    """A miniature AIC root: map CSVs, CLIP features, MP4s, and optional BTC JPEGs.

    `jpeg_videos` defaults to the first video only, so one video is JPEG-backed and the
    other must fall back to its MP4 -- both visual paths are exercised.
    """
    jpeg_videos = list(videos[:1] if jpeg_videos is None else jpeg_videos)
    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    for position, video_id in enumerate(videos):
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate(FRAME_IDS, start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        # Distinct unit vectors so the two videos are separable but both retrievable.
        angle = 0.3 + 0.4 * position
        features = np.array(
            [
                [np.cos(angle), np.sin(angle)],
                [np.cos(angle + 0.05), np.sin(angle + 0.05)],
                [np.cos(angle + 0.10), np.sin(angle + 0.10)],
            ],
            dtype=np.float32,
        )
        np.save(root / "clip-features-32" / f"{video_id}.npy", features)
        colour = VIDEO_COLOURS.get(video_id, ((128, 128, 128), "grey"))[0]
        write_colour_video(root / "video" / f"{video_id}.mp4", colour)
        if video_id in jpeg_videos:
            folder = root / "keyframes" / video_id
            folder.mkdir(parents=True, exist_ok=True)
            for ordinal in range(1, len(FRAME_IDS) + 1):
                Image.new("RGB", (16, 16), (colour[2], colour[1], colour[0])).save(
                    folder / f"{ordinal:03d}.jpg"
                )
    if captions:
        # media-info supplies the only text signal the mock backend can reason over.
        media = root / "media-info"
        media.mkdir(parents=True, exist_ok=True)
        import json

        for video_id, text in captions.items():
            (media / f"{video_id}.json").write_text(
                json.dumps({"title": text, "description": text}), encoding="utf-8"
            )
    return root


def make_qa_config(root: Path, cache_dir: Path, frame_cache_dir: Path, **qa):
    settings = {
        "top_video_hypotheses": 3,
        "frame_hypotheses_per_video": 2,
        "evidence_frame_count": 3,
        "evidence_temporal_diversity_s": 0.4,
        "max_answers": 100,
        "backend": {"type": "mock"},
    }
    settings.update(qa)
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir),
                    "frame_cache_dir": str(frame_cache_dir),
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
                "ranking": {"min_frame_gap": 0, "final_top_k": 100},
                "refinement": {"mode": "disabled"},
                "qa": settings,
            }
        }
    )


class FakeVisualQAAnswerer:
    """A visual backend whose answer is read off the pixels it was given.

    It reports `visual_capable=True` and refuses to answer without images, so it can
    only produce "red" for the red video and "blue" for the blue one. An answer appearing
    on the wrong video therefore proves cross-video contamination.
    """

    backend_type = "fake_visual"

    def __init__(self, *, supports_multi_image: bool = True, fail_for: Sequence[str] = ()):
        self.supports_multi_image = bool(supports_multi_image)
        self.fail_for = set(fail_for)
        self.calls: list[str] = []
        self.image_counts: list[int] = []

    def answer(
        self,
        question: str,
        evidence: QAEvidenceBundle,
        *,
        expected_answer_type: str | None = None,
    ) -> QAAnswerResult:
        self.calls.append(evidence.video_id)
        if evidence.video_id in self.fail_for:
            raise QABackendUnavailable(f"fake backend refuses {evidence.video_id}")
        frames = evidence.visual_frames
        if not frames:
            raise QABackendUnavailable("no visual evidence")
        self.image_counts.append(len(frames))
        colours = {colour_of_jpeg(frame.image_bytes) for frame in frames}
        kind = canonical_answer_type(expected_answer_type or evidence.expected_answer_type)
        if kind == "number":
            text = str(len(frames))
        elif kind == "boolean":
            text = "yes" if colours else "no"
        else:
            text = sorted(colours)[0]
        return QAAnswerResult(
            video_id=evidence.video_id,
            answer=text,
            normalized_answer=normalize_answer(text, expected_type=kind),
            status=ANSWER_STATUS_ANSWERED,
            backend_type=self.backend_type,
            visual=True,
            used_evidence_ids=tuple(frame.evidence_id for frame in frames),
        )

    def status(self) -> QAAnswererStatus:
        return QAAnswererStatus(
            backend_type=self.backend_type,
            state=BACKEND_STATE_READY,
            visual_capable=True,
            supports_multi_image=self.supports_multi_image,
            production_ready=False,
            model_name="fake-visual",
            device="cpu",
            warning="Test backend; never used in production.",
        )


class ScriptedQAAnswerer:
    """Returns a fixed answer per video, for normalization and policy tests."""

    backend_type = "scripted"

    def __init__(self, answers: dict[str, str], *, visual: bool = True, default: str = ""):
        self.answers = dict(answers)
        self.default = default
        self.visual = bool(visual)
        self.calls: list[str] = []

    def answer(
        self,
        question: str,
        evidence: QAEvidenceBundle,
        *,
        expected_answer_type: str | None = None,
    ) -> QAAnswerResult:
        self.calls.append(evidence.video_id)
        text = self.answers.get(evidence.video_id, self.default)
        kind = canonical_answer_type(expected_answer_type or evidence.expected_answer_type)
        return QAAnswerResult(
            video_id=evidence.video_id,
            answer=text,
            normalized_answer=normalize_answer(text, expected_type=kind),
            status=ANSWER_STATUS_ANSWERED,
            backend_type=self.backend_type,
            visual=self.visual,
            used_evidence_ids=tuple(frame.evidence_id for frame in evidence.frames),
        )

    def status(self) -> QAAnswererStatus:
        return QAAnswererStatus(
            backend_type=self.backend_type,
            state=BACKEND_STATE_READY,
            visual_capable=self.visual,
            supports_multi_image=True,
            production_ready=False,
            model_name="scripted",
            device="cpu",
        )


class BrokenQAAnswerer:
    """A backend that always fails. Must never yield a fabricated answer."""

    backend_type = "broken"

    def answer(self, question, evidence, *, expected_answer_type=None):
        raise RuntimeError("backend exploded")

    def status(self) -> QAAnswererStatus:
        return QAAnswererStatus(
            backend_type=self.backend_type,
            state=BACKEND_STATE_NOT_AVAILABLE,
            visual_capable=False,
            model_name="broken",
            device="cpu",
            fallback_reason="always fails",
        )


class FakeLocalVlm:
    """Stands in for a locally hosted multimodal model. Never downloads anything."""

    def __init__(self, reply: str = "two"):
        self.reply = reply
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def generate(self, images, prompt: str) -> str:
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        return self.reply


__all__ = [
    "FPS",
    "FRAME_IDS",
    "FRAMES_PER_VIDEO",
    "VIDEO_COLOURS",
    "BrokenQAAnswerer",
    "FakeLocalVlm",
    "FakeVisualQAAnswerer",
    "ScriptedQAAnswerer",
    "colour_of_frame",
    "colour_of_jpeg",
    "make_qa_config",
    "make_qa_root",
    "write_colour_video",
]
