"""One CLIP model instance shared by the text tower and the image tower.

`transformers.CLIPModel` already contains **both** towers. Before Phase 5 only the
text tower was used (`aic2026/text_encoder.py`); Phase 5 adds query-conditioned image
scoring, which needs the image tower of the *same* model — the two embedding spaces
are only comparable if they come from one checkpoint.

Loading `openai/clip-vit-base-patch32` twice would mean roughly 600 MB of duplicated
weights for no benefit, so this module owns the checkpoint and hands the same instance
to both callers. Sharing is keyed on `(model_name, device)`: two components asking for
the same model on the same device get the same object, and a component asking for a
different model or device gets its own.

Nothing here imports torch or transformers at module import time; a process that never
scores an image never pays for them.
"""
from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

_REGISTRY: dict[tuple[str, str], "CLIPBackend"] = {}
_REGISTRY_LOCK = threading.RLock()


class CLIPBackendError(RuntimeError):
    """Raised when the CLIP checkpoint cannot be loaded or produces bad output."""


def resolve_device(requested: Optional[str]) -> str:
    """Turn ``auto``/``None`` into a concrete device, refusing an impossible one."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is an optional dependency
        raise CLIPBackendError(
            "CLIP requires torch and transformers. Install with: "
            ".venv\\Scripts\\pip.exe install -r requirements-full.txt"
        ) from exc
    wanted = (requested or "auto").strip() or "auto"
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        raise CLIPBackendError(f"CUDA device {wanted!r} requested but CUDA is unavailable.")
    return wanted


def _feature_tensor(features: Any):
    """Unwrap whatever `get_*_features` returned into a plain tensor.

    transformers 4.x returns the projected embedding directly; transformers 5.x wraps
    it in a `BaseModelOutputWithPooling` whose `pooler_output` is that same projected
    embedding. Both shapes are accepted so the encoder does not break on an upgrade.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    if isinstance(features, (tuple, list)):
        return features[0]
    return features


def _normalize_rows(matrix: np.ndarray, *, what: str) -> np.ndarray:
    """L2-normalize, refusing non-finite or zero-length vectors.

    A NaN embedding would silently poison every downstream cosine similarity, so it is
    rejected at the boundary rather than propagated into ranking.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if not np.isfinite(values).all():
        raise CLIPBackendError(f"CLIP produced non-finite {what} embeddings.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise CLIPBackendError(f"CLIP produced a zero-length {what} embedding.")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class CLIPBackend:
    """A loaded CLIP checkpoint plus its processor, usable from both towers.

    Construction is cheap: the weights load on the first `ensure_loaded()` call, so a
    backend can be created (and reported in `/health`) without paying for the model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        self.model_name = str(model_name)
        self.requested_device = device
        self.local_files_only = bool(local_files_only)
        self.device: Optional[str] = None
        self.model: Any = None
        self.processor: Any = None
        self.projection_dim: Optional[int] = None
        self.loaded_offline: Optional[bool] = None
        self._torch: Any = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------- loading

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self) -> "CLIPBackend":
        """Load the checkpoint once. Concurrent callers wait rather than double-load."""
        if self.model is not None:
            return self
        with self._lock:
            if self.model is not None:
                return self
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise CLIPBackendError(
                    "CLIP requires torch and transformers. Install with: "
                    ".venv\\Scripts\\pip.exe install -r requirements-full.txt"
                ) from exc
            device = resolve_device(self.requested_device)
            # Prefer the local cache so an offline machine never blocks on a download,
            # then fall back to the hub only when the caller allows it.
            attempts = [True] if self.local_files_only else [True, False]
            last_error: Exception | None = None
            for offline in attempts:
                try:
                    processor = CLIPProcessor.from_pretrained(
                        self.model_name, local_files_only=offline
                    )
                    model = CLIPModel.from_pretrained(
                        self.model_name, local_files_only=offline
                    ).to(device)
                except Exception as exc:  # noqa: BLE001 - reported as CLIPBackendError
                    last_error = exc
                    continue
                model.eval()  # deterministic: no dropout, and no gradients anywhere
                self._torch = torch
                self.device = device
                self.processor = processor
                self.model = model
                self.projection_dim = int(getattr(model.config, "projection_dim", 0) or 0)
                self.loaded_offline = offline
                return self
            raise CLIPBackendError(
                f"Cannot load CLIP checkpoint {self.model_name!r} on device {device!r}: "
                f"{type(last_error).__name__}: {last_error}"
            )

    def require_projection_dim(self, expected: int) -> None:
        """Refuse a checkpoint whose embedding space is not the AIC feature space."""
        self.ensure_loaded()
        if int(self.projection_dim or 0) != int(expected):
            raise CLIPBackendError(
                f"Model {self.model_name!r} outputs {self.projection_dim} dimensions, "
                f"but the AIC index requires {int(expected)}."
            )

    # ------------------------------------------------------------------ encoding

    def encode_text(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        self.ensure_loaded()
        clean = [str(text) for text in texts]
        if not clean:
            return np.empty((0, int(self.projection_dim or 0)), dtype=np.float32)
        torch = self._torch
        chunks: list[np.ndarray] = []
        for start in range(0, len(clean), max(1, int(batch_size))):
            inputs = self.processor(
                text=clean[start:start + max(1, int(batch_size))],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.inference_mode():
                features = _feature_tensor(self.model.get_text_features(**inputs))
            chunks.append(features.detach().float().cpu().numpy())
        return _normalize_rows(np.concatenate(chunks, axis=0), what="text")

    def encode_images(self, images: Sequence[Any], *, batch_size: int = 8) -> np.ndarray:
        """Embed RGB images (`PIL.Image` or HxWx3 uint8 arrays) in batches.

        Batched on purpose: one forward pass per batch, never one per frame.
        """
        self.ensure_loaded()
        items = list(images)
        if not items:
            return np.empty((0, int(self.projection_dim or 0)), dtype=np.float32)
        torch = self._torch
        chunks: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for start in range(0, len(items), step):
            inputs = self.processor(
                images=items[start:start + step], return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                features = _feature_tensor(self.model.get_image_features(**inputs))
            chunks.append(features.detach().float().cpu().numpy())
        return _normalize_rows(np.concatenate(chunks, axis=0), what="image")

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device or (self.requested_device or "auto"),
            "loaded": self.is_loaded,
            "projection_dim": self.projection_dim,
            "loaded_offline": self.loaded_offline,
        }


def get_clip_backend(
    model_name: str = DEFAULT_CLIP_MODEL,
    *,
    device: Optional[str] = None,
    local_files_only: bool = False,
) -> CLIPBackend:
    """Return the process-wide backend for this checkpoint and device.

    Keyed on the *requested* device string so that `auto` and an explicit `cpu` can
    resolve to one entry only after torch is importable; resolution is deliberately not
    forced here, because `/health` must be answerable without importing torch.
    """
    key = (str(model_name), str(device or "auto"))
    with _REGISTRY_LOCK:
        backend = _REGISTRY.get(key)
        if backend is None:
            backend = CLIPBackend(
                model_name, device=device, local_files_only=local_files_only
            )
            _REGISTRY[key] = backend
        return backend


def reset_clip_backends() -> None:
    """Drop cached backends. For tests only; never called by request handling."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


__all__ = [
    "DEFAULT_CLIP_MODEL",
    "CLIPBackend",
    "CLIPBackendError",
    "get_clip_backend",
    "reset_clip_backends",
    "resolve_device",
]
