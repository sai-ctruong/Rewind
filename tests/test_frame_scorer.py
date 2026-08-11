"""FrameScorer contract, the CLIP implementation, and its safety checks.

No test here downloads a model. The CLIP scorer is exercised only through paths that
are lazy by construction, plus one opt-in integration test that skips when the
checkpoint is not already cached locally.
"""
from __future__ import annotations

import numpy as np
import pytest

from aic2026.clip_backend import (
    CLIPBackend,
    CLIPBackendError,
    _feature_tensor,
    _normalize_rows,
    get_clip_backend,
    reset_clip_backends,
)
from aic2026.frame_scorer import (
    SCORER_STATE_NOT_LOADED,
    SCORER_STATE_UNAVAILABLE,
    CLIPFrameScorer,
    FrameScorer,
    ScorerStatus,
    _to_rgb,
    build_frame_scorer,
    validate_scores,
)
from aic2026.local_refinement import (
    LocalFrameRefiner,
    LocalRefinementRequest,
    RefinementCandidate,
    RefinementConfig,
)
from aic2026.frame_provider import FrameProvider
from tests.refinement_support import FakeFrameScorer, write_synthetic_video


# --------------------------------------------------------------------- interface


def test_fake_scorer_satisfies_the_frame_scorer_protocol() -> None:
    scorer = FakeFrameScorer(target_frame_idx=5)
    assert isinstance(scorer, FrameScorer)
    prepared = scorer.prepare_query("a person walking")
    frames = [np.full((8, 8, 3), 10 + 8 * i, dtype=np.uint8) for i in range(3)]
    scores = scorer.score_frames(prepared, frames)
    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_fake_scorer_is_deterministic() -> None:
    frames = [np.full((8, 8, 3), 10 + 8 * i, dtype=np.uint8) for i in range(4)]
    first = FakeFrameScorer(target_frame_idx=2).score_frames(None, frames)
    second = FakeFrameScorer(target_frame_idx=2).score_frames(None, frames)
    assert list(first) == list(second)
    # The nearest frame to the target wins, and the ordering is a pure function.
    assert int(np.argmax(first)) == 2


def test_clip_scorer_is_a_frame_scorer_without_loading_anything() -> None:
    scorer = CLIPFrameScorer("openai/clip-vit-base-patch32", device="cpu")
    assert isinstance(scorer, FrameScorer)
    status = scorer.status()
    assert status.state == SCORER_STATE_NOT_LOADED
    assert status.available is False
    assert scorer._backend is None


# ------------------------------------------------------------------- validation


def test_validate_scores_rejects_non_finite_and_wrong_length() -> None:
    assert validate_scores([0.2, -0.5], 2) == (pytest.approx(0.2), pytest.approx(-0.5))
    with pytest.raises(ValueError, match="non-finite"):
        validate_scores([0.1, float("nan")], 2)
    with pytest.raises(ValueError, match="non-finite"):
        validate_scores([float("inf")], 1)
    with pytest.raises(ValueError, match="scores for"):
        validate_scores([0.1], 2)


def test_backend_rejects_non_finite_or_zero_length_embeddings() -> None:
    good = _normalize_rows(np.array([[3.0, 4.0]], dtype=np.float32), what="image")
    assert np.allclose(np.linalg.norm(good, axis=1), 1.0)
    with pytest.raises(CLIPBackendError, match="non-finite"):
        _normalize_rows(np.array([[np.nan, 1.0]], dtype=np.float32), what="image")
    with pytest.raises(CLIPBackendError, match="zero-length"):
        _normalize_rows(np.zeros((1, 4), dtype=np.float32), what="text")


def test_feature_tensor_unwraps_both_transformers_shapes() -> None:
    class Wrapped:
        pooler_output = "projected"

    assert _feature_tensor(Wrapped()) == "projected"
    assert _feature_tensor(("first", "second")) == "first"
    assert _feature_tensor("plain") == "plain"


def test_bgr_frames_are_converted_to_rgb_for_the_image_tower() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 0] = 200  # blue in OpenCV order
    rgb = _to_rgb(bgr)
    assert rgb[0, 0, 2] == 200 and rgb[0, 0, 0] == 0
    grey = _to_rgb(np.full((2, 2), 40, dtype=np.uint8))
    assert grey.shape == (2, 2, 3)
    with pytest.raises(ValueError):
        _to_rgb(np.zeros((2, 2, 1), dtype=np.uint8))


# ------------------------------------------------------------- batching and reuse


def test_query_is_prepared_once_per_refinement_request(tmp_path) -> None:
    video = write_synthetic_video(tmp_path / "video" / "V1.mp4", frames=31, fps=10.0)
    scorer = FakeFrameScorer(target_frame_idx=15)
    refiner = LocalFrameRefiner(
        RefinementConfig(mode="always", candidate_budget=3, window_before_s=1.0,
                         window_after_s=1.0, fine_fps=5.0, max_frames=6),
        frame_provider=FrameProvider(tmp_path),
        scorer=scorer,
    )
    candidates = tuple(
        RefinementCandidate(
            keyframe_id=f"V1/kf_{i:06d}",
            video_id="V1",
            coarse_frame_idx=frame,
            timestamp=frame / 10.0,
            coarse_score=1.0 - 0.001 * i,
            source_video=str(video),
        )
        for i, frame in enumerate((4, 15, 27))
    )
    result = refiner.refine(LocalRefinementRequest("a query", candidates))
    assert result.applied
    # One text embedding for the whole request, no matter how many candidates.
    assert scorer.prepare_calls == 1
    assert scorer.queries == ["a query"]


def test_frames_are_scored_in_one_batched_call(tmp_path) -> None:
    video = write_synthetic_video(tmp_path / "video" / "V1.mp4", frames=31, fps=10.0)
    scorer = FakeFrameScorer(target_frame_idx=15)
    refiner = LocalFrameRefiner(
        RefinementConfig(mode="always", candidate_budget=3, window_before_s=1.0,
                         window_after_s=1.0, fine_fps=5.0, max_frames=5),
        frame_provider=FrameProvider(tmp_path),
        scorer=scorer,
    )
    candidates = tuple(
        RefinementCandidate(f"V1/kf_{i}", "V1", frame, frame / 10.0, 1.0 - 0.001 * i, str(video))
        for i, frame in enumerate((4, 15, 27))
    )
    result = refiner.refine(LocalRefinementRequest("q", candidates))
    total_frames = sum(item.frames_decoded for item in result.refinements)
    # One inference call covering every sampled frame of every candidate — never one
    # call per frame.
    assert scorer.score_calls == 1
    assert scorer.batch_sizes == [total_frames]
    assert total_frames > 3


def test_build_frame_scorer_only_accepts_the_real_backend() -> None:
    scorer = build_frame_scorer("clip", model_name="openai/clip-vit-base-patch32", device="cpu")
    assert isinstance(scorer, CLIPFrameScorer)
    assert scorer.status().state == SCORER_STATE_NOT_LOADED
    with pytest.raises(ValueError, match="only production visual scorer"):
        build_frame_scorer("fake")


def test_scorer_status_reports_an_unavailable_model_without_raising() -> None:
    scorer = CLIPFrameScorer("definitely/not-a-real-model", device="cpu", local_files_only=True)
    status = scorer.status(initialize=True)
    assert status.state == SCORER_STATE_UNAVAILABLE
    assert status.available is False
    assert status.fallback_reason


def test_one_backend_is_shared_per_model_and_device() -> None:
    reset_clip_backends()
    try:
        first = get_clip_backend("openai/clip-vit-base-patch32", device="cpu")
        second = get_clip_backend("openai/clip-vit-base-patch32", device="cpu")
        other = get_clip_backend("openai/clip-vit-base-patch32", device="cuda")
        assert first is second, "the text tower and the image tower must share weights"
        assert first is not other
        assert first.is_loaded is False, "registering a backend must not load it"
    finally:
        reset_clip_backends()


def test_backend_describe_is_answerable_before_loading() -> None:
    backend = CLIPBackend("openai/clip-vit-base-patch32", device="cpu")
    described = backend.describe()
    assert described["loaded"] is False
    assert described["model_name"] == "openai/clip-vit-base-patch32"
    assert described["projection_dim"] is None


def test_scorer_status_helper_shape() -> None:
    status = ScorerStatus(backend="clip", model_name="m", device="cpu").to_dict()
    assert set(status) == {
        "backend", "model_name", "device", "state", "available",
        "production_ready", "fallback_reason", "warning",
    }


# ------------------------------------------------------------------- integration


@pytest.mark.integration
def test_real_clip_scorer_prefers_the_matching_image_when_cached() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    scorer = CLIPFrameScorer("openai/clip-vit-base-patch32", device="cpu", expected_dim=512)
    try:
        prepared = scorer.prepare_query("a solid red image")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CLIP model is not available locally: {exc}")
    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[:, :, 2] = 220  # BGR: red
    blue = np.zeros((64, 64, 3), dtype=np.uint8)
    blue[:, :, 0] = 220
    scores = scorer.score_frames(prepared, [red, blue])
    assert len(scores) == 2
    assert all(-1.0 <= value <= 1.0 for value in scores)
    assert scores[0] > scores[1]
    assert scorer.status().state == "ready"

