"""R0: labels must declare where they came from, and may never be self-generated.

The failure mode this guards against is not malice, it is drift: a development label set
gets quoted six weeks later as an AIC result. So provenance is mandatory, the banner is
part of the report, and a file annotated by the system itself is refused outright.
"""
from __future__ import annotations

import json

import pytest

from aic2026.metrics import GroundTruthRequired, evaluate_query
from evaluation.ground_truth import (
    LABEL_SOURCE_OFFICIAL,
    LABEL_SOURCE_PRIVATE_DEV,
    OFFICIAL_GT_BANNER,
    PRIVATE_GT_BANNER,
    GroundTruthSchemaError,
    load_ground_truth,
    parse_ground_truth,
    report_header,
)

KIS_ROW = {
    "query_id": "kis_1",
    "task": "kis",
    "query": "a person walking",
    "video_id": "L21_V001",
    "frame_ranges": [[100, 200]],
}
QA_ROW = {
    "query_id": "qa_1",
    "task": "qa",
    "event_text": "a vehicle",
    "question": "what colour?",
    "video_id": "L21_V002",
    "frame_ranges": [[10, 20]],
    "answers": ["red"],
}
TRAKE_ROW = {
    "query_id": "trake_1",
    "task": "trake",
    "events": ["a", "b"],
    "video_id": "L21_V003",
    "event_frame_ranges": [[10, 20], [30, 40]],
}


def document(**overrides):
    payload = {
        "label_source": LABEL_SOURCE_PRIVATE_DEV,
        "dataset": "local",
        "split": "dev",
        "annotated_by": "a human",
        "entries": [KIS_ROW, QA_ROW, TRAKE_ROW],
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------- provenance


def test_label_source_is_mandatory() -> None:
    payload = document()
    payload.pop("label_source")
    with pytest.raises(GroundTruthSchemaError) as excinfo:
        parse_ground_truth(payload)
    assert "label_source" in str(excinfo.value)


def test_unknown_label_source_is_refused() -> None:
    with pytest.raises(GroundTruthSchemaError):
        parse_ground_truth(document(label_source="probably_official"))


def test_private_labels_carry_the_non_official_banner() -> None:
    gt = parse_ground_truth(document())
    assert gt.label_source == LABEL_SOURCE_PRIVATE_DEV
    assert gt.is_official is False
    assert gt.banner == PRIVATE_GT_BANNER
    assert "NOT OFFICIAL" in gt.banner


def test_official_labels_are_marked_official() -> None:
    gt = parse_ground_truth(document(label_source=LABEL_SOURCE_OFFICIAL))
    assert gt.is_official is True and gt.banner == OFFICIAL_GT_BANNER


def test_report_header_repeats_the_provenance() -> None:
    header = report_header(parse_ground_truth(document()))
    assert header["official"] is False
    assert header["banner"] == PRIVATE_GT_BANNER
    assert header["ground_truth"]["counts"] == {"kis": 1, "qa": 1, "trake": 1}


def test_report_header_without_ground_truth_says_so() -> None:
    header = report_header(None)
    assert header["ground_truth"] is None
    assert "NO SEMANTIC RESULT" in header["banner"]


@pytest.mark.parametrize("annotator", ["system", "model", "auto", "prediction", "Self"])
def test_self_generated_labels_are_refused(annotator) -> None:
    """A label produced by the system being measured is circular, not evidence."""
    with pytest.raises(GroundTruthSchemaError) as excinfo:
        parse_ground_truth(document(annotated_by=annotator))
    assert "circular" in str(excinfo.value)


# ----------------------------------------------------------------------- schema


def test_every_task_parses_into_its_metric_object() -> None:
    gt = parse_ground_truth(document())
    kis, qa, trake = gt.for_task("kis")[0], gt.for_task("qa")[0], gt.for_task("trake")[0]
    assert kis.to_metric_gt().ranges[0].contains(150)
    assert qa.to_metric_gt().answers == ("red",)
    assert len(trake.to_metric_gt().event_ranges) == 2


def test_kis_label_without_a_range_is_refused() -> None:
    row = dict(KIS_ROW)
    row.pop("frame_ranges")
    with pytest.raises(GroundTruthSchemaError, match="frame range"):
        parse_ground_truth(document(entries=[row]))


def test_qa_label_without_an_answer_is_refused() -> None:
    row = dict(QA_ROW)
    row["answers"] = []
    with pytest.raises(GroundTruthSchemaError, match="answer"):
        parse_ground_truth(document(entries=[row]))


def test_trake_events_and_ranges_must_correspond_one_to_one() -> None:
    row = dict(TRAKE_ROW)
    row["events"] = ["a", "b", "c"]
    with pytest.raises(GroundTruthSchemaError, match="one to one"):
        parse_ground_truth(document(entries=[row]))


def test_reversed_or_negative_ranges_are_refused() -> None:
    for ranges in ([[200, 100]], [[-5, 10]]):
        row = dict(KIS_ROW)
        row["frame_ranges"] = ranges
        with pytest.raises(GroundTruthSchemaError):
            parse_ground_truth(document(entries=[row]))


def test_duplicate_query_ids_are_refused() -> None:
    with pytest.raises(GroundTruthSchemaError, match="duplicate"):
        parse_ground_truth(document(entries=[KIS_ROW, KIS_ROW]))


def test_empty_entry_list_is_refused() -> None:
    with pytest.raises(GroundTruthSchemaError, match="non-empty"):
        parse_ground_truth(document(entries=[]))


def test_unknown_task_is_refused() -> None:
    row = dict(KIS_ROW)
    row["task"] = "avs"
    with pytest.raises(GroundTruthSchemaError, match="task"):
        parse_ground_truth(document(entries=[row]))


# ----------------------------------------------------------------- file loading


def test_missing_file_raises_ground_truth_required(tmp_path) -> None:
    with pytest.raises(GroundTruthRequired) as excinfo:
        load_ground_truth(tmp_path / "absent.json")
    assert excinfo.value.error_code == "GROUND_TRUTH_REQUIRED"
    assert "invent" in str(excinfo.value)


def test_valid_file_round_trips(tmp_path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(document()), encoding="utf-8")
    gt = load_ground_truth(path)
    assert len(gt) == 3
    assert gt.path == str(path)
    assert gt.provenance()["entries"] == 3


def test_shipped_template_is_a_valid_private_dev_file() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    payload = json.loads(
        (root / "evaluation" / "labels" / "private_dev_template.json").read_text(encoding="utf-8")
    )
    gt = parse_ground_truth(payload)
    assert gt.is_official is False
    assert gt.counts() == {"kis": 1, "qa": 1, "trake": 1}


# ---------------------------------------------------- guard still refuses with no GT


def test_semantic_evaluation_still_refuses_without_ground_truth() -> None:
    with pytest.raises(GroundTruthRequired):
        evaluate_query("kis", [], None)


def test_supplying_private_gt_scores_but_stays_labelled() -> None:
    """Private labels may be scored. The result is simply never called an AIC score."""
    from aic2026.metrics import RankedAnswer

    gt = parse_ground_truth(document())
    entry = gt.for_task("kis")[0]
    report = evaluate_query("kis", [RankedAnswer("L21_V001", ("150",))], entry.to_metric_gt())
    assert report["R@1"] == 1.0
    assert report_header(gt)["official"] is False
