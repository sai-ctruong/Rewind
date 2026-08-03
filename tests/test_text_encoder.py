from __future__ import annotations

import numpy as np
import pytest

import aic2026.text_encoder as encmod
from aic2026.text_encoder import (
    AutoCLIPTextEncoder,
    CLIPTextEncoder,
    HashingTextEncoder,
    encode_many,
)


def test_hashing_encoder_is_normalized_finite_and_deterministic() -> None:
    encoder = HashingTextEncoder(512)
    first = encoder.encode_batch(["a red shirt", "a blue car"])
    second = encoder.encode_batch(["a red shirt", "a blue car"])
    assert first.shape == (2, 512)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.array_equal(first, second)
    cosine = float(first[0] @ first[1])
    assert -1.0001 <= cosine <= 1.0001


def test_hashing_encoder_is_rejected_in_production() -> None:
    with pytest.raises(RuntimeError, match="disabled in production"):
        HashingTextEncoder(512, production_mode=True)


def test_auto_encoder_does_not_silently_fallback_in_production(monkeypatch) -> None:
    class MissingClip:
        def __init__(self, *args, **kwargs):
            raise ImportError("missing test dependency")

    monkeypatch.setattr(encmod, "CLIPTextEncoder", MissingClip)
    encoder = AutoCLIPTextEncoder(feature_dim=512, production_mode=True)
    with pytest.raises(RuntimeError, match="requirements-full.txt"):
        encoder.encode_text("query")


def test_auto_encoder_fallback_has_explicit_warning(monkeypatch) -> None:
    class MissingClip:
        def __init__(self, *args, **kwargs):
            raise ImportError("missing test dependency")

    monkeypatch.setattr(encmod, "CLIPTextEncoder", MissingClip)
    encoder = AutoCLIPTextEncoder(feature_dim=512, allow_hashing_fallback=True)
    assert encoder.encode_text("query").shape == (512,)
    status = encoder.status()
    assert status.encoder_type == "hashing_fallback"
    assert status.production_ready is False
    assert status.warning and "not meaningful" in status.warning
    assert status.fallback_reason and "missing test dependency" in status.fallback_reason


def test_embedding_validation_rejects_wrong_dimension_and_nonfinite() -> None:
    class BadDimension:
        def encode_text(self, text):
            return np.ones(3, dtype=np.float32)

    class NonFinite:
        def encode_text(self, text):
            return np.array([1.0, np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="does not match"):
        encode_many(BadDimension(), ["x"], 2)
    with pytest.raises(ValueError, match="NaN or Inf"):
        encode_many(NonFinite(), ["x"], 2)


@pytest.mark.integration
def test_real_clip_encoder_matches_aic_dimension_when_cached() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        encoder = CLIPTextEncoder(local_files_only=True)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"CLIP model is not cached locally: {exc}")
    values = encoder.encode_batch(["a person", "a person"])
    assert values.shape == (2, 512)
    assert np.array_equal(values[0], values[1])
    assert encoder.status().production_ready is True
