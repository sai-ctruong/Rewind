"""Trigger policy, candidate budget, and region deduplication.

These decide *whether* and *how much* video gets decoded, so they are what keeps
refinement affordable. The thresholds exercised here are the configured defaults; no
threshold in this repository is tuned against retrieval accuracy, because there is no
AIC ground truth to tune against.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.frame_provider import FrameProvider
from aic2026.local_refinement import (
    MODE_ALWAYS,
    MODE_DISABLED,
    MODE_UNCERTAINTY,
    REASON_ALWAYS,
    REASON_DISABLED,
    REASON_MARGIN_ABOVE_THRESHOLD,
    REASON_MARGIN_BELOW_THRESHOLD,
    REASON_NO_CANDIDATES,
    REASON_SINGLE_REGION,
    LocalFrameRefiner,
    LocalRefinementRequest,
    RefinementCandidate,
    RefinementConfig,
    aggregate_diagnostics,
    decide_refinement,
    merge_candidate_regions,
)
from tests.refinement_support import FakeFrameScorer, write_synthetic_video

FPS = 10.0


class CountingFrameProvider(FrameProvider):
    """A provider that records every decode request, to prove what was NOT decoded."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_calls: list[tuple[str, tuple[int, ...]]] = []

    def decode_frames(self, video_id, frame_indices, *, source_video=None):
        self.decode_calls.append((str(video_id), tuple(int(i) for i in frame_indices)))
        return super().decode_frames(video_id, frame_indices, source_video=source_video)


def candidate(video_id: str, frame_idx: int, score: float, source: Path | None = None):
    return RefinementCandidate(
        keyframe_id=f"{video_id}/kf_{frame_idx:06d}",
        video_id=video_id,
        coarse_frame_idx=frame_idx,
        timestamp=frame_idx / FPS,
        coarse_score=score,
        source_video=None if source is None else str(source),
    )


@pytest.fixture()
def videos(tmp_path: Path) -> dict[str, Path]:
    return {
        name: write_synthetic_video(tmp_path / "video" / f"{name}.mp4", frames=31, fps=FPS)
        for name in ("V1", "V2", "V3", "V4", "V5", "V6")
    }


def refiner(tmp_path: Path, scorer, **overrides) -> LocalFrameRefiner:
    base = dict(window_before_s=0.5, window_after_s=0.5, fine_fps=5.0, max_frames=4)
    base.update(overrides)
    return LocalFrameRefiner(
        RefinementConfig(**base),
        frame_provider=CountingFrameProvider(tmp_path),
        scorer=scorer,
    )


# ------------------------------------------------------------------------ modes


def test_disabled_mode_never_decodes_anything(tmp_path: Path, videos) -> None:
    scorer = FakeFrameScorer(target_frame_idx=15)
    engine = refiner(tmp_path, scorer, mode=MODE_DISABLED)
    result = engine.refine(
        LocalRefinementRequest("q", (candidate("V1", 15, 0.9, videos["V1"]),))
    )
    assert result.decision.triggered is False
    assert result.decision.reason == REASON_DISABLED
    assert result.refinements == ()
    assert engine.frame_provider.decode_calls == []
    assert scorer.prepare_calls == 0, "a disabled refiner must not even embed the query"


def test_enabled_false_is_equivalent_to_disabled(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(15), enabled=False, mode=MODE_ALWAYS)
    result = engine.refine(
        LocalRefinementRequest("q", (candidate("V1", 15, 0.9, videos["V1"]),))
    )
    assert result.decision.reason == REASON_DISABLED
    assert engine.frame_provider.decode_calls == []


def test_always_mode_refines_up_to_the_candidate_budget(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(15), mode=MODE_ALWAYS, candidate_budget=2)
    candidates = tuple(
        candidate(name, 15, 0.9 - 0.1 * i, videos[name])
        for i, name in enumerate(("V1", "V2", "V3", "V4"))
    )
    result = engine.refine(LocalRefinementRequest("q", candidates))
    assert result.decision.triggered is True
    assert result.decision.reason == REASON_ALWAYS
    # Four clearly separated candidates, but only two windows are ever opened.
    assert len(engine.frame_provider.decode_calls) == 2
    assert [item.video_id for item in result.refinements] == ["V1", "V2"]
    assert all(item.applied for item in result.refinements)


def test_uncertainty_mode_triggers_when_the_margin_is_small(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(15), mode=MODE_UNCERTAINTY, margin_threshold=0.05)
    candidates = (
        candidate("V1", 15, 0.900, videos["V1"]),
        candidate("V2", 15, 0.890, videos["V2"]),  # relative margin 0.011 < 0.05
    )
    result = engine.refine(LocalRefinementRequest("q", candidates))
    assert result.decision.triggered is True
    assert result.decision.reason == REASON_MARGIN_BELOW_THRESHOLD
    assert result.decision.margin == pytest.approx(0.010, abs=1e-6)
    assert result.decision.relative_margin == pytest.approx(0.010 / 0.900, abs=1e-6)
    assert result.decision.threshold == pytest.approx(0.05)
    assert engine.frame_provider.decode_calls


def test_uncertainty_mode_skips_when_the_margin_is_large(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(15), mode=MODE_UNCERTAINTY, margin_threshold=0.05)
    candidates = (
        candidate("V1", 15, 0.900, videos["V1"]),
        candidate("V2", 15, 0.400, videos["V2"]),  # relative margin 0.56 > 0.05
    )
    result = engine.refine(LocalRefinementRequest("q", candidates))
    assert result.decision.triggered is False
    assert result.decision.reason == REASON_MARGIN_ABOVE_THRESHOLD
    assert result.decision.regions_selected == 0
    assert engine.frame_provider.decode_calls == [], "a confident ranking decodes nothing"


def test_a_single_region_counts_as_unconfirmed_and_triggers(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(15), mode=MODE_UNCERTAINTY)
    result = engine.refine(
        LocalRefinementRequest("q", (candidate("V1", 15, 0.9, videos["V1"]),))
    )
    assert result.decision.triggered is True
    assert result.decision.reason == REASON_SINGLE_REGION


def test_no_candidates_means_no_decision_to_make() -> None:
    decision = decide_refinement((), RefinementConfig(mode=MODE_ALWAYS), considered=0)
    assert decision.triggered is False
    assert decision.reason == REASON_NO_CANDIDATES


def test_trigger_reason_and_counts_are_always_recorded(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(15), mode=MODE_ALWAYS, candidate_budget=2)
    candidates = tuple(
        candidate(name, 15, 0.9 - 0.1 * i, videos[name]) for i, name in enumerate(("V1", "V2", "V3"))
    )
    result = engine.refine(LocalRefinementRequest("q", candidates))
    payload = result.to_dict()["decision"]
    assert payload["mode"] == MODE_ALWAYS
    assert payload["reason"] == REASON_ALWAYS
    assert payload["candidates_considered"] == 3
    assert payload["regions_found"] == 3
    assert payload["regions_selected"] == 2
    assert result.diagnostics["trigger_reason"] == REASON_ALWAYS


# --------------------------------------------------------------- region merging


def test_nearby_candidates_of_one_video_become_a_single_region() -> None:
    candidates = (
        candidate("V1", 100, 0.90),   # t = 10.0s
        candidate("V1", 105, 0.85),   # t = 10.5s -> same region
        candidate("V1", 400, 0.80),   # t = 40.0s -> separate region
        candidate("V2", 101, 0.88),   # different video -> separate region
    )
    regions = merge_candidate_regions(candidates, merge_s=1.0)
    assert len(regions) == 3
    anchors = [(item.anchor.video_id, item.anchor.coarse_frame_idx) for item in regions]
    assert anchors == [("V1", 100), ("V2", 101), ("V1", 400)]
    # The merged candidate is recorded, not discarded.
    assert regions[0].members == ("V1/kf_000105",)


def test_region_merging_keeps_the_highest_scoring_anchor() -> None:
    regions = merge_candidate_regions(
        (candidate("V1", 100, 0.50), candidate("V1", 103, 0.95)), merge_s=1.0
    )
    assert len(regions) == 1
    assert regions[0].anchor.coarse_frame_idx == 103
    assert regions[0].members == ("V1/kf_000100",)


def test_zero_merge_window_keeps_every_candidate_separate() -> None:
    regions = merge_candidate_regions(
        (candidate("V1", 100, 0.90), candidate("V1", 101, 0.85)), merge_s=0.0
    )
    assert len(regions) == 2


def test_dedup_prevents_the_budget_being_spent_twice_on_one_region(tmp_path: Path, videos) -> None:
    engine = refiner(
        tmp_path, FakeFrameScorer(15), mode=MODE_ALWAYS, candidate_budget=2, region_merge_s=1.0
    )
    candidates = (
        candidate("V1", 15, 0.90, videos["V1"]),
        candidate("V1", 16, 0.89, videos["V1"]),  # 0.1s away: same region
        candidate("V2", 15, 0.88, videos["V2"]),
    )
    result = engine.refine(LocalRefinementRequest("q", candidates))
    decoded_videos = [call[0] for call in engine.frame_provider.decode_calls]
    assert decoded_videos == ["V1", "V2"], "the second V1 candidate must not open a window"
    assert result.refinements[0].merged_keyframe_ids == ("V1/kf_000016",)


def test_only_top_hypotheses_are_considered_for_regions(tmp_path: Path, videos) -> None:
    engine = refiner(
        tmp_path, FakeFrameScorer(15), mode=MODE_ALWAYS, top_hypotheses=2, candidate_budget=5
    )
    candidates = tuple(
        candidate(name, 15, 0.9 - 0.1 * i, videos[name])
        for i, name in enumerate(("V1", "V2", "V3", "V4"))
    )
    result = engine.refine(LocalRefinementRequest("q", candidates))
    assert result.decision.candidates_considered == 2
    assert len(engine.frame_provider.decode_calls) == 2


# -------------------------------------------------------------------- rollups


def test_aggregate_diagnostics_are_structural_not_accuracy(tmp_path: Path, videos) -> None:
    engine = refiner(tmp_path, FakeFrameScorer(target_frame_idx=17), mode=MODE_ALWAYS)
    runs = [
        engine.refine(LocalRefinementRequest(f"q{i}", (candidate("V1", 15, 0.9, videos["V1"]),)))
        for i in range(3)
    ]
    rollup = aggregate_diagnostics(runs)
    assert rollup["searches"] == 3
    assert rollup["trigger_rate"] == pytest.approx(1.0)
    assert rollup["candidates_refined_total"] == 3
    assert rollup["refinement_ms_p50"] >= 0
    assert rollup["refinement_ms_p95"] >= 0
    assert 0.0 <= rollup["fraction_best_differs_from_coarse"] <= 1.0
    assert "accuracy" in rollup["note"]
    # None of these keys may masquerade as a retrieval quality metric.
    assert not {"precision", "recall", "map", "accuracy"} & set(rollup)
    assert aggregate_diagnostics([]) == {"searches": 0}
