"""R1 stage 2/3: staged sampling under the same hard frame budget as the fixed plan.

The mechanism claim is narrow and testable without any ground truth: an easy case stops
early and spends fewer frames, an ambiguous case uses more stages, and neither can ever
exceed the budget or score a frame twice. Whether stopping early costs quality is a
different question that no test here pretends to answer.
"""
from __future__ import annotations

import pytest

from aic2026.progressive_refinement import (
    STOP_BUDGET,
    STOP_CONFIDENT,
    STOP_FAILED,
    STOP_NO_FRAMES,
    STOP_STAGES_DONE,
    progressive_sample,
    sparse_plan,
    zoom_plan,
)

STAGES = (8, 8, 16)
BUDGET = 32


def scorer(peak: int, *, width: float = 4.0, flat: bool = False):
    """A synthetic scorer with a single peak, or a flat surface when `flat`."""
    calls: list[list[int]] = []

    def score_frames(indices):
        indices = list(indices)
        calls.append(indices)
        if flat:
            return {index: 0.5 for index in indices}
        return {index: 1.0 / (1.0 + abs(index - peak) / width) for index in indices}

    score_frames.calls = calls  # type: ignore[attr-defined]
    return score_frames


# ----------------------------------------------------------------------- plans


def test_sparse_plan_spans_the_window_and_keeps_the_anchor() -> None:
    plan = sparse_plan(anchor=50, low=0, high=100, count=5)
    assert 50 in plan
    assert min(plan) >= 0 and max(plan) <= 100
    assert len(plan) == 5
    assert plan == sorted(plan)


def test_sparse_plan_of_one_is_the_anchor() -> None:
    assert sparse_plan(anchor=7, low=0, high=100, count=1) == [7]


def test_zoom_plan_packs_around_its_centre() -> None:
    plan = zoom_plan(center=50, low=0, high=100, count=5, step=2)
    assert 50 in plan
    assert max(plan) - min(plan) <= 8
    assert len(set(plan)) == len(plan)


def test_plans_respect_video_bounds() -> None:
    assert all(0 <= index <= 10 for index in sparse_plan(5, 0, 10, 20))
    assert all(0 <= index <= 10 for index in zoom_plan(0, 0, 10, 8, step=3))


# ------------------------------------------------------------------ mechanism


def test_a_confident_peak_stops_early_and_saves_budget() -> None:
    score = scorer(peak=50, width=0.5)
    result = progressive_sample(
        anchor=50, low=0, high=100, budget=BUDGET, stage_frames=STAGES,
        stop_margin=0.15, score_frames=score, fps=25.0,
    )
    assert result.stop_reason == STOP_CONFIDENT
    assert result.stages_entered == 1
    assert result.frames_scored < BUDGET
    assert result.budget_saved > 0
    assert result.best_index == 50


def test_an_ambiguous_case_uses_more_stages_but_stays_bounded() -> None:
    score = scorer(peak=50, width=1000.0)  # nearly flat: no separation
    result = progressive_sample(
        anchor=50, low=0, high=100, budget=BUDGET, stage_frames=STAGES,
        stop_margin=0.5, score_frames=score, fps=25.0,
    )
    assert result.stages_entered > 1
    assert result.frames_scored <= BUDGET
    assert result.stop_reason in {STOP_BUDGET, STOP_STAGES_DONE}


def test_the_hard_budget_is_never_exceeded() -> None:
    for budget in (1, 5, 12, 32, 64):
        result = progressive_sample(
            anchor=50, low=0, high=1000, budget=budget, stage_frames=(100, 100, 100),
            stop_margin=1.1, score_frames=scorer(peak=500, flat=True), fps=25.0,
        )
        assert result.frames_scored <= budget


def test_no_frame_is_ever_scored_twice() -> None:
    score = scorer(peak=50, width=1000.0)
    result = progressive_sample(
        anchor=50, low=0, high=100, budget=BUDGET, stage_frames=STAGES,
        stop_margin=1.1, score_frames=score, fps=25.0,
    )
    seen: list[int] = []
    for call in score.calls:  # type: ignore[attr-defined]
        seen.extend(call)
    assert len(seen) == len(set(seen))
    assert len(result.scored_by_index) == len(set(seen))


def test_the_coarse_frame_is_always_in_the_first_stage() -> None:
    score = scorer(peak=90)
    progressive_sample(
        anchor=50, low=0, high=100, budget=BUDGET, stage_frames=STAGES,
        stop_margin=1.1, score_frames=score, fps=25.0,
    )
    assert 50 in score.calls[0]  # type: ignore[attr-defined]


def test_later_stages_zoom_towards_the_current_peak() -> None:
    score = scorer(peak=90, width=1000.0)
    progressive_sample(
        anchor=10, low=0, high=100, budget=BUDGET, stage_frames=STAGES,
        stop_margin=1.1, score_frames=score, fps=25.0,
    )
    calls = score.calls  # type: ignore[attr-defined]
    assert len(calls) >= 2
    first_spread = max(calls[0]) - min(calls[0])
    second_spread = max(calls[1]) - min(calls[1])
    assert second_spread <= first_spread


def test_an_empty_window_reports_no_frames() -> None:
    result = progressive_sample(
        anchor=0, low=10, high=0, budget=8, stage_frames=(4,),
        stop_margin=0.2, score_frames=scorer(0), fps=25.0,
    )
    assert result.stop_reason == STOP_NO_FRAMES
    assert result.frames_scored == 0


def test_zero_budget_scores_nothing() -> None:
    result = progressive_sample(
        anchor=5, low=0, high=10, budget=0, stage_frames=(4,),
        stop_margin=0.2, score_frames=scorer(5), fps=25.0,
    )
    assert result.frames_scored == 0 and result.stop_reason == STOP_NO_FRAMES


def test_a_scorer_failure_falls_back_instead_of_raising() -> None:
    def broken(indices):
        raise RuntimeError("decode failed")

    result = progressive_sample(
        anchor=5, low=0, high=100, budget=16, stage_frames=(8, 8),
        stop_margin=0.2, score_frames=broken, fps=25.0,
    )
    assert result.stop_reason == STOP_FAILED
    assert result.frames_scored == 0


def test_result_reports_stages_and_savings() -> None:
    result = progressive_sample(
        anchor=50, low=0, high=100, budget=BUDGET, stage_frames=STAGES,
        stop_margin=0.15, score_frames=scorer(peak=50, width=0.5), fps=25.0,
    )
    payload = result.to_dict()
    assert payload["budget"] == BUDGET
    assert payload["frames_scored"] + payload["budget_saved"] == BUDGET
    assert payload["stages"][0]["stage"] == "stage_A"
    assert "whether it costs quality is unknown" in payload["note"]


def test_fixed_and_progressive_share_the_same_ceiling() -> None:
    """Matched-budget comparison: the adaptive sampler may never outspend the fixed one."""
    fixed_budget = 32
    result = progressive_sample(
        anchor=50, low=0, high=10_000, budget=fixed_budget, stage_frames=(64, 64, 64),
        stop_margin=1.1, score_frames=scorer(peak=5000, flat=True), fps=25.0,
    )
    assert result.frames_scored <= fixed_budget
