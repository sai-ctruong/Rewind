"""Phase 11: semantic metrics refuse to run without official ground truth.

R@k and Final Score are semantic claims. This repository holds no AIC ground truth, so
any number produced for them here would be manufactured. The guard makes that a loud
refusal rather than a quiet zero, while leaving every structural diagnostic available.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.metrics import (
    FrameRange,
    GroundTruthRequired,
    KISGroundTruth,
    QAGroundTruth,
    RankedAnswer,
    TRAKEGroundTruth,
    evaluate_query,
    final_score_from_r_scores,
    is_structurally_valid_trake_row,
    ranked_r_scores,
    require_ground_truth,
    trake_r_score,
)
from evaluation.official_eval import evaluate_labels, require_labels

ROOT = Path(__file__).resolve().parent.parent

KIS_PREDICTIONS = [RankedAnswer("L01_V001", ("120",)), RankedAnswer("L01_V001", ("400",))]


# ------------------------------------------------------------------- the refusal


def test_require_ground_truth_refuses_none() -> None:
    with pytest.raises(GroundTruthRequired) as excinfo:
        require_ground_truth(None, "kis")
    assert excinfo.value.error_code == "GROUND_TRUTH_REQUIRED"
    assert "kis" in str(excinfo.value)


def test_refusal_explains_what_is_still_available() -> None:
    with pytest.raises(GroundTruthRequired) as excinfo:
        require_ground_truth(None)
    assert "Structural diagnostics are available" in str(excinfo.value)


@pytest.mark.parametrize(
    "empty",
    [
        KISGroundTruth("L01_V001", ()),
        QAGroundTruth("L01_V001", (), ("red",)),
        TRAKEGroundTruth("L01_V001", ()),
    ],
)
def test_ground_truth_without_frame_ranges_is_refused(empty) -> None:
    """An empty label object is not ground truth; it just looks like one."""
    with pytest.raises(GroundTruthRequired):
        require_ground_truth(empty, "kis")


@pytest.mark.parametrize("task", ["kis", "qa", "trake"])
def test_evaluate_query_refuses_without_ground_truth(task) -> None:
    with pytest.raises(GroundTruthRequired):
        evaluate_query(task, KIS_PREDICTIONS, None)


def test_guard_is_a_runtime_error_so_it_cannot_be_swallowed_as_a_value() -> None:
    assert issubclass(GroundTruthRequired, RuntimeError)
    assert not issubclass(GroundTruthRequired, (ValueError, KeyError))


def test_benchmark_over_no_labels_refuses_instead_of_reporting_zeros() -> None:
    with pytest.raises(GroundTruthRequired) as excinfo:
        evaluate_labels([], lambda label: [])
    assert "nothing to score" in str(excinfo.value)


def test_missing_label_file_asks_for_annotation() -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        require_labels(ROOT / "evaluation" / "labels" / "definitely_absent.jsonl")
    assert "ground-truth" in str(excinfo.value)


# --------------------------------------------------- real ground truth still works


def test_real_ground_truth_scores_normally() -> None:
    gt = KISGroundTruth("L01_V001", (FrameRange(100, 150),))
    report = evaluate_query("kis", KIS_PREDICTIONS, gt)
    assert report["R@1"] == 1.0
    assert report["Final Score"] == pytest.approx(1.0)


def test_scoring_a_miss_is_zero_not_a_refusal() -> None:
    gt = KISGroundTruth("L01_V001", (FrameRange(900, 950),))
    assert evaluate_query("kis", KIS_PREDICTIONS, gt)["R@100"] == 0.0


def test_trake_length_mismatch_scores_zero_rather_than_partial_credit() -> None:
    gt = TRAKEGroundTruth("L01_V001", (FrameRange(0, 10), FrameRange(20, 30), FrameRange(40, 50)))
    short_row = RankedAnswer("L01_V001", ("5", "25"))
    assert is_structurally_valid_trake_row(short_row.frame_ids, 3) is False
    assert trake_r_score(short_row, gt) == 0.0
    full_row = RankedAnswer("L01_V001", ("5", "25", "45"))
    assert trake_r_score(full_row, gt) == pytest.approx(1.0)


# --------------------------------------------- structural diagnostics need no labels


def test_structural_helpers_work_without_any_ground_truth() -> None:
    assert is_structurally_valid_trake_row(("1", "2", "3"), 3) is True
    assert is_structurally_valid_trake_row(("1", "2"), 3) is False
    assert final_score_from_r_scores([1.0, 0.0])["R@1"] == 1.0


def test_ranked_r_scores_still_requires_a_gt_object() -> None:
    """The low-level scorer is not a back door: it needs a real gt to do anything."""
    with pytest.raises(AttributeError):
        ranked_r_scores("kis", KIS_PREDICTIONS, None)


# ------------------------------------------------------- no AIC labels are shipped


def test_repository_ships_no_aic_ground_truth() -> None:
    """If this ever fails, real labels arrived and the guard's premise changed."""
    labels_dir = ROOT / "evaluation" / "labels"
    shipped = [p.name for p in labels_dir.glob("*") if p.is_file()]
    assert shipped == ["template.jsonl"], shipped


def test_local_demo_labels_are_not_aic_videos() -> None:
    """`evaluation/labels.json` annotates the bundled demo clips, not AIC videos.

    AIC video ids look like `L21_V001`. Anything in this file matching that shape would
    be a self-authored label masquerading as official ground truth.
    """
    import re

    rows = json.loads((ROOT / "evaluation" / "labels.json").read_text(encoding="utf-8"))
    for row in rows:
        assert not re.fullmatch(r"L\d{2}_V\d{3}", str(row.get("video_id", "")))
