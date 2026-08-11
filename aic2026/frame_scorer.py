"""Query-conditioned visual scoring of decoded video frames.

Phase 5 needs to answer one question for a densely sampled local window: *which of
these frames best matches the query text?* That is a visual question, so it needs the
image tower — the coarse index only stores BTC's precomputed frame vectors and cannot
score a frame that was never in the index.

`LocalFrameRefiner` deliberately knows nothing about CLIP. It talks to a `FrameScorer`:

    prepared = scorer.prepare_query(query)      # once per refinement request
    scores   = scorer.score_frames(prepared, frames)   # batched, many frames at once

so a fake deterministic scorer can drive the whole algorithm in tests, and a different
visual backend can be dropped in later without touching the refinement logic.

`CLIPFrameScorer` is the production implementation. It reuses the shared checkpoint
from `aic2026/clip_backend.py`, so the text tower used for retrieval and the image
tower used for refinement are the same weights in the same embedding space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .clip_backend import DEFAULT_CLIP_MODEL, CLIPBackendError, get_clip_backend

SCORER_STATE_READY = "ready"
SCORER_STATE_NOT_LOADED = "not_loaded"
SCORER_STATE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ScorerStatus:
    """What the application can say about the visual scorer without loading it."""

    backend: str
    model_name: str
    device: str
    state: str = SCORER_STATE_NOT_LOADED
    available: bool = False
    production_ready: bool = False
    fallback_reason: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class FrameScorer(Protocol):
    """Scores decoded frames against one prepared query.

    `frames` are BGR uint8 arrays exactly as OpenCV decodes them; an implementation
    that needs RGB converts internally. Scores are plain floats, higher is better, and
    every returned score must be finite.
    """

    def prepare_query(self, query: str) -> Any: ...

    def score_frames(self, prepared_query: Any, frames: Sequence[np.ndarray]) -> Sequence[float]: ...

    def status(self, *, initialize: bool = False) -> ScorerStatus: ...


def validate_scores(values: Sequence[float], expected: int) -> tuple[float, ...]:
    """Reject a scorer result that cannot safely enter ranking.

    Non-finite values are refused rather than coerced: silently mapping NaN to 0 would
    place a broken frame in the middle of the ranking instead of surfacing the fault.
    """
    scores = np.asarray(list(values), dtype=np.float32)
    if scores.shape != (int(expected),):
        raise ValueError(
            f"Frame scorer returned {scores.shape} scores for {int(expected)} frames."
        )
    if not np.isfinite(scores).all():
        raise ValueError("Frame scorer returned a non-finite score.")
    return tuple(float(value) for value in scores)


class CLIPFrameScorer:
    """Cosine similarity between a CLIP text embedding and CLIP frame embeddings.

    Lazy in every sense: the checkpoint loads on the first `prepare_query` call, so
    building an engine, answering `/health`, or running a search with refinement
    disabled never touches the model. Once loaded it is reused for the lifetime of the
    owning engine (and shared with the text encoder through `clip_backend`).
    """

    backend_name = "clip"

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        device: Optional[str] = None,
        batch_size: int = 8,
        expected_dim: Optional[int] = None,
        local_files_only: bool = False,
    ):
        self.model_name = str(model_name)
        self.requested_device = device or "auto"
        self.batch_size = max(1, int(batch_size))
        self.expected_dim = None if expected_dim is None else int(expected_dim)
        self.local_files_only = bool(local_files_only)
        self._backend = None
        self._failure: Optional[str] = None

    # ------------------------------------------------------------------- loading

    def _load(self):
        if self._backend is not None:
            return self._backend
        if self._failure is not None:
            raise CLIPBackendError(self._failure)
        backend = get_clip_backend(
            self.model_name,
            device=self.requested_device,
            local_files_only=self.local_files_only,
        )
        try:
            backend.ensure_loaded()
            if self.expected_dim is not None:
                backend.require_projection_dim(self.expected_dim)
        except CLIPBackendError as exc:
            self._failure = str(exc)
            raise
        self._backend = backend
        return backend

    # ------------------------------------------------------------------- scoring

    def prepare_query(self, query: str) -> np.ndarray:
        """Embed the query once; the same vector scores every sampled frame."""
        backend = self._load()
        matrix = backend.encode_text([str(query)], batch_size=1)
        return np.ascontiguousarray(matrix[0], dtype=np.float32)

    def score_frames(
        self, prepared_query: Any, frames: Sequence[np.ndarray]
    ) -> tuple[float, ...]:
        items = list(frames)
        if not items:
            return ()
        backend = self._load()
        query = np.asarray(prepared_query, dtype=np.float32).reshape(-1)
        images = [_to_rgb(frame) for frame in items]
        embeddings = backend.encode_images(images, batch_size=self.batch_size)
        if embeddings.shape[1] != query.shape[0]:
            raise CLIPBackendError(
                f"Image embedding dim {embeddings.shape[1]} does not match the query "
                f"embedding dim {query.shape[0]}."
            )
        # Both sides are L2-normalized by the backend, so the dot product IS the cosine
        # similarity, in [-1, 1]. It is not a probability and is never treated as one.
        return validate_scores(embeddings @ query, len(items))

    def status(self, *, initialize: bool = False) -> ScorerStatus:
        if initialize and self._backend is None and self._failure is None:
            try:
                self._load()
            except CLIPBackendError:
                pass
        if self._backend is not None:
            return ScorerStatus(
                backend=self.backend_name,
                model_name=self.model_name,
                device=str(self._backend.device or self.requested_device),
                state=SCORER_STATE_READY,
                available=True,
                production_ready=True,
            )
        if self._failure is not None:
            return ScorerStatus(
                backend=self.backend_name,
                model_name=self.model_name,
                device=str(self.requested_device),
                state=SCORER_STATE_UNAVAILABLE,
                available=False,
                production_ready=False,
                fallback_reason=self._failure,
                warning="Local refinement is skipped; coarse retrieval is unaffected.",
            )
        return ScorerStatus(
            backend=self.backend_name,
            model_name=self.model_name,
            device=str(self.requested_device),
            state=SCORER_STATE_NOT_LOADED,
            available=False,
            production_ready=False,
            warning="Visual scorer has not been initialized yet.",
        )


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """OpenCV hands back BGR; CLIP's processor expects RGB."""
    array = np.asarray(frame)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Cannot score a frame with shape {array.shape}.")
    return np.ascontiguousarray(array[:, :, 2::-1].astype(np.uint8))


def build_frame_scorer(
    scorer_type: str,
    *,
    model_name: str = DEFAULT_CLIP_MODEL,
    device: Optional[str] = None,
    batch_size: int = 8,
    expected_dim: Optional[int] = None,
) -> FrameScorer:
    """Construct the configured scorer. There is no production fake.

    A test double is injected by the test, never selected by configuration, so a
    misconfigured deployment cannot end up 'refining' against meaningless scores.
    """
    kind = str(scorer_type or "").strip().lower()
    if kind == "clip":
        return CLIPFrameScorer(
            model_name,
            device=device,
            batch_size=batch_size,
            expected_dim=expected_dim,
        )
    raise ValueError(
        f"Unsupported refinement.scorer_type {scorer_type!r}; the only production "
        "visual scorer is 'clip'."
    )


__all__ = [
    "SCORER_STATE_NOT_LOADED",
    "SCORER_STATE_READY",
    "SCORER_STATE_UNAVAILABLE",
    "CLIPFrameScorer",
    "FrameScorer",
    "ScorerStatus",
    "build_frame_scorer",
    "validate_scores",
]
