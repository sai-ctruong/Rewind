"""Bounded local sampling and frame selection over real (synthetic) MP4s.

These prove the algorithm: that the window is bounded and clamped, that the coarse
frame is always examined, and that an earlier, later, or unchanged frame is chosen when
the scorer says so. They prove nothing about AIC retrieval quality: there is no ground
truth in this repository.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.frame_provider import FrameProvider
from aic2026.local_refinement import (
    FRAME_OUTPUT_DECODED_FRAME,
    REASON_DECODE_FAILED,
    REASON_METADATA_UNAVAILABLE,
    REASON_REFINED,
    REASON_SCORER_FAILED,
    REASON_SCORER_UNAVAILABLE,
    REASON_VIDEO_UNAVAILABLE,
    LocalFrameRefiner,
    LocalRefinementRequest,
    RefinementCandidate,
    RefinementConfig,
    build_sample_plan,
)
from tests.refinement_support import FakeFrameScorer, UnavailableScorer, write_synthetic_video

FPS = 10.0
# Every synthetic frame encodes its own index in its pixels, which caps the length; see
# tests/refinement_support.py. COARSE sits far enough from both ends that a +-1s window
# is never clamped by accident.
FRAMES = 31
COARSE = 15


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    return write_synthetic_video(tmp_path / "video" / "V1.mp4", frames=FRAMES, fps=FPS)


def config(**overrides) -> RefinementConfig:
    base = dict(
        mode="always",
        candidate_budget=3,
        window_before_s=1.0,
        window_after_s=1.0,
        fine_fps=5.0,
        max_frames=12,
    )
    base.update(overrides)
    return RefinementConfig(**base)


def refiner_for(root: Path, scorer, **overrides) -> LocalFrameRefiner:
    return LocalFrameRefiner(
        config(**overrides), frame_provider=FrameProvider(root), scorer=scorer
    )


def candidate(frame_idx: int, *, score: float = 0.9, source: Path | None = None, video_id: str = "V1"):
    return RefinementCandidate(
        keyframe_id=f"{video_id}/kf_{frame_idx:06d}",
        video_id=video_id,
        coarse_frame_idx=frame_idx,
        timestamp=frame_idx / FPS,
        coarse_score=score,
        source_video=None if source is None else str(source),
    )


# ------------------------------------------------------------------- sampling


def test_sample_window_is_bounded_by_the_configured_window() -> None:
    plan, start, end = build_sample_plan(300, 25.0, 10_000, config(window_before_s=2.0, window_after_s=3.0, fine_fps=5.0, max_frames=64))
    assert min(plan) >= 300 - 50
    assert max(plan) <= 300 + 75
    assert start == pytest.approx(10.0)
    assert end == pytest.approx(15.0)


def test_coarse_frame_is_always_in_the_plan() -> None:
    for coarse in (0, 1, 37, 999):
        plan, _, _ = build_sample_plan(coarse, 25.0, 10_000, config())
        assert coarse in plan, f"coarse frame {coarse} must always be sampled"


def test_plan_has_no_duplicate_frame_requests() -> None:
    plan, _, _ = build_sample_plan(500, 30.0, 10_000, config(fine_fps=1000.0, max_frames=40))
    assert len(plan) == len(set(plan))


def test_max_sampled_frames_is_a_hard_cap() -> None:
    plan, _, _ = build_sample_plan(500, 30.0, 10_000, config(window_before_s=30.0, window_after_s=30.0, fine_fps=30.0, max_frames=9))
    assert len(plan) == 9


def test_window_clamps_at_the_beginning_of_the_video() -> None:
    plan, start, _ = build_sample_plan(3, 25.0, 10_000, config(window_before_s=5.0, window_after_s=1.0))
    assert min(plan) >= 0
    assert start == pytest.approx(0.0)


def test_window_clamps_at_the_end_of_the_video() -> None:
    plan, _, end = build_sample_plan(95, 10.0, 100, config(window_before_s=5.0, window_after_s=5.0))
    assert max(plan) <= 99
    assert end == pytest.approx(9.9)


def test_sample_plan_is_deterministic_and_ordered() -> None:
    first, _, _ = build_sample_plan(200, 24.0, 5_000, config())
    second, _, _ = build_sample_plan(200, 24.0, 5_000, config())
    assert first == second
    assert list(first) == sorted(first)


def test_sample_plan_uses_this_video_fps_not_a_constant() -> None:
    slow, _, _ = build_sample_plan(100, 10.0, 5_000, config(fine_fps=5.0, max_frames=64))
    fast, _, _ = build_sample_plan(100, 50.0, 5_000, config(fine_fps=5.0, max_frames=64))
    # One second of video is 10 frames at 10 fps and 50 frames at 50 fps.
    assert max(slow) - min(slow) == 20
    assert max(fast) - min(fast) == 100


def test_zero_fps_yields_no_plan_rather_than_a_guess() -> None:
    assert build_sample_plan(10, 0.0, 100, config()) == ((), 0.0, 0.0)


# ----------------------------------------------------------------- refinement


def test_coarse_frame_stays_selected_when_it_is_the_strongest(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(target_frame_idx=COARSE))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.applied and item.reason == REASON_REFINED
    assert item.best_visual_frame_idx == COARSE
    assert item.best_is_coarse_frame is True
    assert item.score_gain == pytest.approx(0.0)
    assert item.selected_offset_frames == 0


def test_an_earlier_local_frame_can_win(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(target_frame_idx=COARSE - 4))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.best_visual_frame_idx is not None and item.best_visual_frame_idx < COARSE
    assert item.best_is_coarse_frame is False
    assert item.score_gain > 0
    assert item.selected_offset_seconds < 0


def test_a_later_local_frame_can_win(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(target_frame_idx=COARSE + 4))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.best_visual_frame_idx is not None and item.best_visual_frame_idx > COARSE
    assert item.best_is_coarse_frame is False
    assert item.selected_offset_seconds > 0


def test_missing_mp4_preserves_the_coarse_candidate(tmp_path: Path) -> None:
    refiner = refiner_for(tmp_path, FakeFrameScorer(target_frame_idx=COARSE))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE),)))
    item = result.refinements[0]
    assert item.applied is False
    assert item.reason in {REASON_VIDEO_UNAVAILABLE, REASON_METADATA_UNAVAILABLE}
    assert item.coarse_official_frame_idx == COARSE
    assert item.submission_frame_idx == COARSE
    assert item.refined_score == pytest.approx(item.coarse_score)
    assert result.diagnostics["decode_failures"] == 1


def test_unreadable_video_preserves_the_coarse_candidate(tmp_path: Path) -> None:
    broken = tmp_path / "video" / "V1.mp4"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not an mp4 at all")
    refiner = refiner_for(tmp_path, FakeFrameScorer(target_frame_idx=COARSE))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=broken),)))
    item = result.refinements[0]
    assert item.applied is False
    assert item.refined_score == pytest.approx(item.coarse_score)
    assert item.submission_frame_idx == COARSE


def test_scorer_unavailable_preserves_the_coarse_candidate(video: Path) -> None:
    refiner = refiner_for(video.parents[1], UnavailableScorer())
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.applied is False
    assert item.reason == REASON_SCORER_UNAVAILABLE
    assert item.refined_score == pytest.approx(item.coarse_score)
    assert result.warnings and "unavailable" in result.warnings[0].lower()
    assert result.diagnostics["scorer_failures"] == 1


def test_scorer_failure_preserves_the_coarse_candidate(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(COARSE, fail=True))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.applied is False
    assert item.reason == REASON_SCORER_FAILED
    assert item.submission_frame_idx == COARSE


def test_a_required_scorer_fails_loudly_instead_of_pretending(video: Path) -> None:
    refiner = refiner_for(video.parents[1], UnavailableScorer(), scorer_required=True)
    with pytest.raises(RuntimeError, match="scorer_required"):
        refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))


def test_non_finite_scorer_values_never_enter_ranking(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(COARSE, non_finite=True))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.applied is False
    assert item.reason == REASON_SCORER_FAILED
    assert item.refined_score == pytest.approx(item.coarse_score)


def test_score_gain_is_recorded_against_the_coarse_frame(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(target_frame_idx=COARSE + 4))
    item = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),))).refinements[0]
    assert item.coarse_visual_score is not None and item.best_visual_score is not None
    assert item.score_gain == pytest.approx(item.best_visual_score - item.coarse_visual_score)
    assert item.refined_score == pytest.approx(item.coarse_score + 0.10 * item.score_gain)


def test_refined_frame_differs_from_the_submission_frame_under_preserve_coarse(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(target_frame_idx=COARSE + 4))
    item = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),))).refinements[0]
    assert item.best_visual_frame_idx != item.coarse_official_frame_idx
    # The official submission frame is the coarse mapped frame_idx, unchanged.
    assert item.submission_frame_idx == COARSE


def test_decoded_frame_policy_is_available_but_never_the_default(video: Path) -> None:
    assert RefinementConfig().frame_output_policy == "preserve_coarse"
    refiner = refiner_for(
        video.parents[1],
        FakeFrameScorer(target_frame_idx=COARSE + 4),
        frame_output_policy=FRAME_OUTPUT_DECODED_FRAME,
    )
    item = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),))).refinements[0]
    assert item.submission_frame_idx == item.best_visual_frame_idx != COARSE


def test_decoding_never_exceeds_the_frame_budget(video: Path) -> None:
    refiner = refiner_for(
        video.parents[1], FakeFrameScorer(COARSE), window_before_s=30.0, window_after_s=30.0,
        fine_fps=FPS, max_frames=7,
    )
    item = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),))).refinements[0]
    assert item.sampled_frame_count == 7
    assert item.frames_decoded <= 7


def test_a_candidate_without_an_official_frame_idx_is_skipped(video: Path) -> None:
    unmapped = RefinementCandidate("V1/kf_x", "V1", None, 3.0, 0.9, str(video))
    refiner = refiner_for(video.parents[1], FakeFrameScorer(COARSE))
    item = refiner.refine(LocalRefinementRequest("q", (unmapped,))).refinements[0]
    assert item.applied is False
    assert item.submission_frame_idx is None


def test_timings_are_reported_separately(video: Path) -> None:
    refiner = refiner_for(video.parents[1], FakeFrameScorer(target_frame_idx=COARSE))
    result = refiner.refine(LocalRefinementRequest("q", (candidate(COARSE, source=video),)))
    item = result.refinements[0]
    assert item.decode_ms >= 0 and item.inference_ms >= 0
    assert item.total_ms == pytest.approx(item.decode_ms + item.inference_ms, abs=0.01)
    assert result.diagnostics["refinement_ms"] >= 0
    assert REASON_DECODE_FAILED not in {r.reason for r in result.refinements}
