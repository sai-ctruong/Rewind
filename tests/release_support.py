"""Shared fixtures for the Phase 11 release tests.

Everything here is offline and deterministic: a two-frame synthetic dataset, a config
built in memory, and duck-typed stand-ins for an engine. No model is downloaded, no
network call is made, and no ground truth is invented.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from PIL import Image

from aic2026.config import app_config_from_dict


class TinyTextEncoder:
    """Two-dimensional deterministic text encoder; stands in for CLIP's text tower."""

    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def make_data(root, video_ids=("L01_V001",)):
    """A minimal but structurally complete AIC data root."""
    for relative in ("clip-features-32", "map-keyframes", "media-info", "video", "objects"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        (root / "keyframes" / video_id).mkdir(parents=True, exist_ok=True)
        (root / "objects" / video_id).mkdir(parents=True, exist_ok=True)
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,30,100\n2,1.0,30,130\n", encoding="utf-8"
        )
        np.save(
            root / "clip-features-32" / f"{video_id}.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        for ordinal in (1, 2):
            Image.new("RGB", (8, 8), (ordinal * 64, 0, 0)).save(
                root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
            )
            (root / "objects" / video_id / f"{ordinal:03d}.json").write_text(
                "{}", encoding="utf-8"
            )
        (root / "media-info" / f"{video_id}.json").write_text("{}", encoding="utf-8")
    return root


def make_retrieval_only_video(root, video_id: str) -> None:
    """A video with map + CLIP + objects + metadata and NO pixels at all.

    This is the shape of 844 of the 873 videos in the real collection: complete
    supporting data, no local MP4 and no keyframe JPEGs. Retrieval must reach it; every
    visual capability must report itself unavailable rather than pretend.
    """
    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    (root / "media-info").mkdir(parents=True, exist_ok=True)
    (root / "map-keyframes" / f"{video_id}.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30,100\n2,1.0,30,130\n", encoding="utf-8"
    )
    np.save(
        root / "clip-features-32" / f"{video_id}.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    (root / "media-info" / f"{video_id}.json").write_text("{}", encoding="utf-8")
    # Deliberately absent: video/<id>.mp4 and keyframes/<id>/.


def make_config(root, cache, **overrides):
    data: dict[str, Any] = {
        "aic2026": {
            "runtime": {"production_mode": False, "seed": 42, "device": "auto"},
            "dataset": {
                "root": str(root),
                "cache_dir": str(cache),
                "load_objects": False,
                "include_media_text": False,
                "verify_keyframes": False,
                "index_kind": "flat",
            },
            "cache": {
                "allow_stale_cache": False,
                "validate_data_signature": True,
                "code_version_policy": "warn",
            },
            "encoder": {
                "type": "auto_clip",
                "model_name": "openai/clip-vit-base-patch32",
                "feature_dim": 2,
                "normalize": True,
                "batch_size": 2,
            },
            "ranking": {"final_top_k": 25},
            "evaluation": {"ks": [1, 5, 20]},
        }
    }
    for section, values in overrides.items():
        data["aic2026"].setdefault(section, {}).update(values)
    return app_config_from_dict(data)


def build_engine(tmp_path, **overrides):
    """Build a real (tiny) engine plus its config, with no model download."""
    from aic2026.engine import AICCompetitionEngine

    root = make_data(tmp_path / "data")
    cache = tmp_path / "cache"
    config = make_config(root, cache, **overrides)
    engine, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    return engine, config, load


def channel_status(
    *,
    clip: bool = True,
    optional: bool = False,
    reason: str = "no records",
) -> dict[str, dict[str, Any]]:
    """A `channel_status()` payload with controllable availability."""
    names = ("clip", "bm25", "objects", "metadata", "ocr", "asr", "caption")
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        usable = clip if name == "clip" else optional
        out[name] = {
            "name": name,
            "enabled": True,
            "available": usable,
            "usable": usable,
            "records": 10 if usable else 0,
            "reason": "" if usable else reason,
        }
    return out


def qa_status(*, visual: bool = False, backend_type: str = "mock") -> dict[str, Any]:
    return {
        "backend": {
            "backend_type": backend_type,
            "state": "ready" if visual else "mock",
            "visual_capable": visual,
            "production_ready": visual,
        }
    }


class FakeEngine:
    """Duck-typed engine for readiness branches that need no real index."""

    def __init__(
        self,
        app_config,
        *,
        video_ids=("L01_V001",),
        frames: int = 2,
        channels: Optional[dict] = None,
        qa: Optional[dict] = None,
        frame_provider: Any = object(),
    ) -> None:
        self.app_config = app_config
        self.entry = SimpleNamespace(
            raws={
                f"{video_id}/kf_{i:06d}": SimpleNamespace(video_id=video_id)
                for video_id in video_ids
                for i in range(1, frames + 1)
            },
            num_indexed=frames * len(video_ids),
        )
        self._channels = channels if channels is not None else channel_status()
        self._qa = qa if qa is not None else qa_status()
        self.frame_provider = frame_provider

    def channel_status(self) -> dict:
        return self._channels

    def qa_status(self) -> dict:
        return self._qa

    def query_cache_status(self) -> dict:
        return {
            "query_embeddings": {"max_entries": 256, "entries": 0, "hits": 0, "misses": 0},
            "persisted": False,
        }

    def encoder_status(self, *, initialize: bool = False) -> dict:
        return {"state": "ready", "ready": True, "backend": "fake"}


def kis_prediction(video_id: str, frame_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        video_id=video_id, frame_id=frame_id, score=1.0, refinement=None, qa=None, trake=None
    )


def trake_prediction(video_id: str, frames) -> SimpleNamespace:
    return SimpleNamespace(
        video_id=video_id,
        frame_id=frames[0],
        score=1.0,
        refinement=None,
        qa=None,
        trake={
            "steps": [
                {"submission_frame_idx": frame, "visual_frame_idx": None} for frame in frames
            ]
        },
    )
