"""Text encoders compatible with the AIC CLIP image feature space."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ingestion.embed_clip import deterministic_unit_vector

from .clip_backend import get_clip_backend


FALLBACK_WARNING = "Fallback encoder - results are not meaningful for accuracy"


@dataclass(frozen=True)
class EncoderStatus:
    encoder_name: str
    encoder_type: str
    device: str
    feature_dim: int
    production_ready: bool
    fallback_reason: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class TextQueryEncoder(Protocol):
    def encode_text(self, text: str) -> np.ndarray: ...
    def encode_batch(self, texts: Sequence[str]) -> np.ndarray: ...
    def status(self) -> EncoderStatus: ...


def _validate_embeddings(values, expected_dim: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Text embedding shape {matrix.shape} does not match AIC feature dimension {expected_dim}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Text embeddings contain NaN or Inf values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Text encoder returned a zero-length embedding.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class HashingTextEncoder:
    """Deterministic test/smoke encoder; never a production retrieval model."""

    def __init__(self, dim: int = 512, *, production_mode: bool = False):
        if production_mode:
            raise RuntimeError(
                "HashingTextEncoder is disabled in production mode. Install with: "
                ".venv\\Scripts\\pip.exe install -r requirements-full.txt"
            )
        self.dim = int(dim)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        values = [deterministic_unit_vector(str(text), self.dim, salt="aic-text") for text in texts]
        return _validate_embeddings(values, self.dim)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def status(self) -> EncoderStatus:
        return EncoderStatus(
            encoder_name="deterministic-hashing",
            encoder_type="hashing_fallback",
            device="cpu",
            feature_dim=self.dim,
            production_ready=False,
            fallback_reason="Real CLIP encoder was not selected or could not be loaded.",
            warning=FALLBACK_WARNING,
        )


class CLIPTextEncoder:
    """OpenAI CLIP ViT-B/32 text tower matching 512-dimensional BTC features.

    Phase 5 added query-conditioned image scoring, which needs the image tower of the
    *same* checkpoint. Rather than loading `CLIPModel` twice, this encoder now takes its
    model from `aic2026.clip_backend`, which hands the identical instance to
    `CLIPFrameScorer`. Behaviour is unchanged: the same model, the same offline-first
    policy, the same projection-dimension check, and the same public attributes.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: Optional[str] = None,
        feature_dim: int = 512,
        batch_size: int = 32,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.feature_dim = int(feature_dim)
        self.batch_size = max(1, int(batch_size))
        backend = get_clip_backend(
            model_name, device=device, local_files_only=local_files_only
        )
        backend.ensure_loaded()
        if int(backend.projection_dim or 0) != self.feature_dim:
            raise ValueError(
                f"Model {model_name!r} outputs {backend.projection_dim} dimensions, "
                f"but the AIC index requires {self.feature_dim}."
            )
        self._backend = backend
        self.device = str(backend.device)
        self.processor = backend.processor
        self.model = backend.model

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - integration
        clean = [str(text) for text in texts]
        if not clean:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        unique = list(dict.fromkeys(clean))
        unique_matrix = _validate_embeddings(
            self._backend.encode_text(unique, batch_size=self.batch_size), self.feature_dim
        )
        row_by_text = {text: row for text, row in zip(unique, unique_matrix)}
        return np.ascontiguousarray([row_by_text[text] for text in clean], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def status(self) -> EncoderStatus:
        return EncoderStatus(self.model_name, "clip", self.device, self.feature_dim, True)


class AutoCLIPTextEncoder:
    """Lazy real encoder with an explicit, policy-controlled smoke fallback."""

    def __init__(
        self,
        *,
        feature_dim: int = 512,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
        production_mode: bool = False,
        allow_hashing_fallback: bool = True,
        local_files_only: bool = False,
    ):
        self.feature_dim = int(feature_dim)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.production_mode = bool(production_mode)
        self.allow_hashing_fallback = bool(allow_hashing_fallback)
        self.local_files_only = bool(local_files_only)
        self._encoder: Optional[TextQueryEncoder] = None
        self._failure: Optional[str] = None

    def _load(self) -> TextQueryEncoder:
        if self._encoder is not None:
            return self._encoder
        if self._failure is None:
            cache_exc = None
            try:
                self._encoder = CLIPTextEncoder(
                    self.model_name,
                    device=self.device,
                    feature_dim=self.feature_dim,
                    batch_size=self.batch_size,
                    local_files_only=True,
                )
            except Exception as exc:
                cache_exc = exc
            if self._encoder is None and not self.local_files_only:
                try:
                    self._encoder = CLIPTextEncoder(
                        self.model_name,
                        device=self.device,
                        feature_dim=self.feature_dim,
                        batch_size=self.batch_size,
                        local_files_only=False,
                    )
                except Exception as online_exc:
                    self._failure = (
                        f"Real CLIP encoder unavailable: {online_exc}. Install dependencies with: "
                        ".venv\\Scripts\\pip.exe install -r requirements-full.txt"
                    )
            elif self._encoder is None:
                self._failure = f"Real CLIP encoder unavailable in local cache: {cache_exc}"
        if self._encoder is None:
            if self.production_mode or not self.allow_hashing_fallback:
                raise RuntimeError(self._failure)
            self._encoder = HashingTextEncoder(self.feature_dim)
        return self._encoder

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        return _validate_embeddings(self._load().encode_batch(texts), self.feature_dim)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def status(self, *, initialize: bool = False) -> EncoderStatus:
        if initialize:
            self._load()
        if self._encoder is not None:
            status = self._encoder.status()
            if self._failure and status.encoder_type == "hashing_fallback":
                return EncoderStatus(**{**status.to_dict(), "fallback_reason": self._failure})
            return status
        return EncoderStatus(
            self.model_name, "clip_pending", self.device or "auto", self.feature_dim, False,
            fallback_reason=self._failure, warning="Encoder has not been initialized yet."
        )


TransformersClipTextEncoder = CLIPTextEncoder
AutoClipTextEncoder = AutoCLIPTextEncoder


def encoder_status(encoder: object, feature_dim: int) -> EncoderStatus:
    status_fn = getattr(encoder, "status", None)
    if callable(status_fn):
        return status_fn()
    return EncoderStatus(
        type(encoder).__name__, "injected_test_encoder", "cpu", feature_dim, False,
        warning="Injected encoder has no production status contract."
    )


def encode_many(encoder: object, texts: Sequence[str], expected_dim: int) -> np.ndarray:
    batch_fn = getattr(encoder, "encode_batch", None)
    if callable(batch_fn):
        return _validate_embeddings(batch_fn(texts), expected_dim)
    single_fn = getattr(encoder, "encode_text", None)
    if not callable(single_fn):
        raise TypeError("Text encoder must implement encode_text or encode_batch.")
    return _validate_embeddings([single_fn(text) for text in texts], expected_dim)
