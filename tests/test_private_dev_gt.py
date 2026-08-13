"""Private development ground truth: parses when real, refuses when not.

The failure this guards against is a quiet one — a template row, or a row the system
wrote about itself, being counted as evidence and producing a confident number. Every
test here is about the difference between a label and something that merely looks like one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.metrics import TOP_KS, GroundTruthRequired, RankedAnswer, evaluate_query
from evaluation.experiment_manifest import (
    EXPERIMENTS,
    EXPERIMENT_B0_CLEAN,
    EXPERIMENT_B0_RELEASE,
    EXPERIMENT_R1_ADAPTIVE,
    build_manifest,
    comparable,
    hash_query_set,
)
from evaluation.ground_truth import (
    EXAMPLE_MARKER,
    LABEL_SOURCE_PRIVATE_DEV,
    PRIVATE_DEV_DIR,
    PRIVATE_GT_BANNER,
    GroundTruthSchemaError,
    load_private_dev,
    parse_ground_truth,
    report_header,
)
from tests.release_support import make_config, make_data

ROOT = Path(__file__).resolve().parent.parent

KIS = {
    "query_id": "kis_0001",
    "query": "a person pushes a bicycle across a crossing",
    "video_id": "L21_V004",
    "frame_ranges": [[1200, 1260]],
    "label_source": "private_dev",
    "annotated_by": "a human",
    "notes": "",
}
QA = {
    "query_id": "qa_0001",
    "event_description": "a vehicle stops at the intersection",
    "question": "What colour is the vehicle?",
    "video_id": "L21_V002",
    "frame_ranges": [[300, 360]],
    "answers": ["red", "đỏ"],
    "answer_type": "color",
    "label_source": "private_dev",
    "annotated_by": "a human",
}
TRAKE = {
    "query_id": "trake_0001",
    "events": ["a person approaches", "the person enters", "the vehicle moves away"],
    "video_id": "L21_V003",
    "event_frame_ranges": [[[100, 108]], [[210, 219]], [[330, 352]]],
    "label_source": "private_dev",
    "annotated_by": "a human",
}


def write_set(tmp_path: Path, **files) -> Path:
    directory = tmp_path / "private_dev"
    directory.mkdir(parents=True, exist_ok=True)
    for task, entries in files.items():
        (directory / f"{task}.json").write_text(
            json.dumps({"label_source": LABEL_SOURCE_PRIVATE_DEV, "entries": entries}),
            encoding="utf-8",
        )
    return directory


# ------------------------------------------------------------------- 6, 7, 8 parse


def test_private_kis_gt_parses(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    entry = gt.for_task("kis")[0]
    assert entry.query_id == "kis_0001"
    assert entry.frame_ranges == ((1200, 1260),)
    assert entry.to_metric_gt().ranges[0].contains(1230)
    assert gt.has_real_labels is True


def test_private_qa_gt_parses(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, qa=[QA]))
    entry = gt.for_task("qa")[0]
    assert entry.event_text == "a vehicle stops at the intersection"
    assert entry.answer_type == "color"
    assert entry.to_metric_gt().answers == ("red", "đỏ")


def test_private_trake_gt_parses_per_event_interval_lists(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, trake=[TRAKE]))
    entry = gt.for_task("trake")[0]
    assert len(entry.events) == 3
    assert entry.event_frame_ranges == ((100, 108), (210, 219), (330, 352))
    assert len(entry.event_range_groups) == 3
    assert entry.to_metric_gt().event_ranges[1].contains(215)


def test_the_older_flat_trake_shape_still_parses(tmp_path) -> None:
    flat = {**TRAKE, "event_frame_ranges": [[100, 108], [210, 219], [330, 352]]}
    entry = load_private_dev(write_set(tmp_path, trake=[flat])).for_task("trake")[0]
    assert entry.event_frame_ranges == ((100, 108), (210, 219), (330, 352))


def test_a_second_interval_on_an_event_is_kept_not_merged(tmp_path) -> None:
    """A genuinely recurring event keeps both intervals in the record."""
    row = {**TRAKE, "event_frame_ranges": [[[100, 108], [900, 910]], [[210, 219]], [[330, 352]]]}
    entry = load_private_dev(write_set(tmp_path, trake=[row])).for_task("trake")[0]
    assert entry.event_range_groups[0] == ((100, 108), (900, 910))
    # The official scorer takes one interval per event.
    assert entry.event_frame_ranges[0] == (100, 108)


def test_short_intervals_are_preserved_exactly(tmp_path) -> None:
    """Official TRAKE intervals are often under ten frames; nothing widens them."""
    row = {**TRAKE, "event_frame_ranges": [[[100, 103]], [[210, 210]], [[330, 335]]]}
    entry = load_private_dev(write_set(tmp_path, trake=[row])).for_task("trake")[0]
    assert entry.event_frame_ranges == ((100, 103), (210, 210), (330, 335))


# --------------------------------------------------------------- 9-12 rejections


@pytest.mark.parametrize("annotator", ["system", "model", "auto", "clip", "vlm", "generated"])
def test_model_generated_annotation_is_rejected(tmp_path, annotator) -> None:
    row = {**KIS, "annotated_by": annotator}
    with pytest.raises(GroundTruthSchemaError, match="circular"):
        load_private_dev(write_set(tmp_path, kis=[row]))


@pytest.mark.parametrize(
    "ranges",
    [[[300, 100]], [[-5, 10]], [], [[1, 2, 3]]],
)
def test_malformed_frame_interval_is_rejected(tmp_path, ranges) -> None:
    row = {**KIS, "frame_ranges": ranges}
    with pytest.raises(GroundTruthSchemaError):
        load_private_dev(write_set(tmp_path, kis=[row]))


def test_mismatched_trake_event_count_is_rejected(tmp_path) -> None:
    row = {**TRAKE, "event_frame_ranges": [[[100, 108]], [[210, 219]]]}
    with pytest.raises(GroundTruthSchemaError, match="one to one"):
        load_private_dev(write_set(tmp_path, trake=[row]))


def test_empty_qa_answers_are_rejected(tmp_path) -> None:
    with pytest.raises(GroundTruthSchemaError, match="answer"):
        load_private_dev(write_set(tmp_path, qa=[{**QA, "answers": []}]))


def test_an_empty_interval_list_for_an_event_is_rejected(tmp_path) -> None:
    row = {**TRAKE, "event_frame_ranges": [[], [[210, 219]], [[330, 352]]]}
    with pytest.raises(GroundTruthSchemaError):
        load_private_dev(write_set(tmp_path, trake=[row]))


def test_a_file_in_the_wrong_task_slot_is_rejected(tmp_path) -> None:
    with pytest.raises(GroundTruthSchemaError, match="one file holds one task"):
        load_private_dev(write_set(tmp_path, kis=[QA]))


def test_an_official_label_source_is_refused_under_private_dev(tmp_path) -> None:
    directory = tmp_path / "private_dev"
    directory.mkdir(parents=True)
    (directory / "kis.json").write_text(
        json.dumps({"label_source": "official", "entries": [KIS]}), encoding="utf-8"
    )
    with pytest.raises(GroundTruthSchemaError, match="must declare"):
        load_private_dev(directory)


def test_duplicate_query_ids_are_rejected(tmp_path) -> None:
    with pytest.raises(GroundTruthSchemaError, match="duplicate"):
        load_private_dev(write_set(tmp_path, kis=[KIS, KIS]))


# ------------------------------------------------- 13 templates unlock nothing


def test_template_rows_are_not_labels(tmp_path) -> None:
    template = {**KIS, "query_id": f"{EXAMPLE_MARKER}_kis", "annotated_by": EXAMPLE_MARKER}
    gt = load_private_dev(write_set(tmp_path, kis=[template]))
    assert gt.has_real_labels is False
    assert gt.for_task("kis") == ()
    assert len(gt.example_entries) == 1
    assert len(gt) == 0


def test_the_shipped_templates_unlock_nothing() -> None:
    gt = load_private_dev(PRIVATE_DEV_DIR)
    assert gt.has_real_labels is False
    assert gt.counts() == {"kis": 0, "qa": 0, "trake": 0}
    assert sum(gt.example_counts().values()) == 3


def test_the_runner_refuses_when_only_templates_exist() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_private_gt", ROOT / "tools" / "evaluate_private_gt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(GroundTruthRequired) as excinfo:
        module.require_labels(load_private_dev(PRIVATE_DEV_DIR))
    assert excinfo.value.error_code == "GROUND_TRUTH_REQUIRED"
    assert "template row" in str(excinfo.value)


def test_a_real_label_beside_a_template_does_unlock(tmp_path) -> None:
    template = {**KIS, "query_id": f"{EXAMPLE_MARKER}_kis", "annotated_by": EXAMPLE_MARKER}
    gt = load_private_dev(write_set(tmp_path, kis=[template, KIS]))
    assert gt.has_real_labels is True
    assert [entry.query_id for entry in gt.for_task("kis")] == ["kis_0001"]


# ------------------------------------------------------------------ 14 banner


def test_private_reports_are_labelled_not_official(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    header = report_header(gt)
    assert gt.is_official is False
    assert header["banner"] == PRIVATE_GT_BANNER
    assert "NOT OFFICIAL AIC SCORE" in header["banner"]
    assert header["ground_truth"]["label_source"] == LABEL_SOURCE_PRIVATE_DEV


def test_the_readme_states_the_warning() -> None:
    text = (PRIVATE_DEV_DIR / "README.md").read_text(encoding="utf-8")
    assert "NOT OFFICIAL AIC SCORE" in text


# ------------------------------------------------------- 15, 16 official metrics


def test_private_scoring_uses_the_official_cutoffs() -> None:
    assert tuple(sorted(TOP_KS)) == (1, 5, 20, 50, 100)


def test_private_scoring_uses_the_official_formula(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    entry = gt.for_task("kis")[0]
    report = evaluate_query("kis", [RankedAnswer("L21_V004", ("1230",))], entry.to_metric_gt())
    assert set(report) == {f"R@{k}" for k in TOP_KS} | {"Final Score"}
    assert report["Final Score"] == pytest.approx(
        sum(report[f"R@{k}"] for k in TOP_KS) / len(TOP_KS)
    )
    assert report["Final Score"] == 1.0


def test_a_wrong_video_scores_zero_even_with_the_right_frame(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    entry = gt.for_task("kis")[0]
    report = evaluate_query("kis", [RankedAnswer("L21_V999", ("1230",))], entry.to_metric_gt())
    assert report["Final Score"] == 0.0


def test_trake_scores_the_fraction_of_events_hit(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, trake=[TRAKE]))
    entry = gt.for_task("trake")[0]
    two_of_three = RankedAnswer("L21_V003", ("104", "215", "999"))
    report = evaluate_query("trake", [two_of_three], entry.to_metric_gt())
    assert report["R@1"] == pytest.approx(2 / 3)


# ------------------------------------------------------------- 17 manifest hashes


def test_manifest_records_the_full_run_identity(tmp_path) -> None:
    gt = load_private_dev(write_set(tmp_path, kis=[KIS], qa=[QA]))
    config = make_config(make_data(tmp_path / "data"), tmp_path / "cache")
    manifest = build_manifest(config, name=EXPERIMENT_B0_CLEAN, gt=gt)
    for field in (
        "git_commit", "config_hash", "scope_mode", "model", "compute_budget",
        "ground_truth", "ground_truth_hash", "query_set_hash", "b0_release_commit",
    ):
        assert field in manifest
    assert manifest["experiment"] == EXPERIMENT_B0_CLEAN
    assert manifest["official_ground_truth"] is False
    assert manifest["compute_budget"]["qa_max_vlm_calls_per_query"] >= 1


def test_manifest_hashes_change_with_the_labels(tmp_path) -> None:
    config = make_config(make_data(tmp_path / "data"), tmp_path / "cache")
    one = load_private_dev(write_set(tmp_path / "a", kis=[KIS]))
    two = load_private_dev(write_set(tmp_path / "b", kis=[KIS, {**KIS, "query_id": "kis_0002"}]))
    assert one.content_hash() != two.content_hash()
    assert (
        build_manifest(config, name="x", gt=one)["ground_truth_hash"]
        != build_manifest(config, name="x", gt=two)["ground_truth_hash"]
    )


def test_a_template_edit_does_not_change_the_label_hash(tmp_path) -> None:
    """Templates are excluded, so editing one cannot look like a changed label set."""
    template = {**KIS, "query_id": f"{EXAMPLE_MARKER}_a", "annotated_by": EXAMPLE_MARKER}
    edited = {**template, "query": "a completely different example sentence"}
    first = load_private_dev(write_set(tmp_path / "a", kis=[KIS, template]))
    second = load_private_dev(write_set(tmp_path / "b", kis=[KIS, edited]))
    assert first.content_hash() == second.content_hash()


def test_query_set_hash_is_order_stable() -> None:
    assert hash_query_set(["a", "b"]) == hash_query_set(["a", "b"])
    assert hash_query_set(["a", "b"]) != hash_query_set(["b", "a"])


def full_manifest(config, name, gt, *, fingerprint="fp", videos=("L21_V001",)):
    """A manifest with every discriminator recorded, as a real run produces."""
    from types import SimpleNamespace

    return build_manifest(
        config,
        name=name,
        gt=gt,
        load=SimpleNamespace(cache_fingerprint=fingerprint),
        selected_video_ids=list(videos),
    )


def test_runs_on_different_labels_are_not_comparable(tmp_path) -> None:
    config = make_config(make_data(tmp_path / "data"), tmp_path / "cache")
    one = full_manifest(config, "a", load_private_dev(write_set(tmp_path / "a", kis=[KIS])))
    two = full_manifest(
        config, "b", load_private_dev(write_set(tmp_path / "b", kis=[{**KIS, "query_id": "z"}]))
    )
    ok, problems = comparable(one, two)
    assert ok is False
    assert any("different labels" in item for item in problems)


def test_identical_runs_are_comparable(tmp_path) -> None:
    config = make_config(make_data(tmp_path / "data"), tmp_path / "cache")
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    ok, problems = comparable(
        full_manifest(config, "a", gt), full_manifest(config, "b", gt)
    )
    assert ok is True and problems == []


def test_an_unrecorded_discriminator_is_not_treated_as_identical(tmp_path) -> None:
    """Absent is not the same as equal: two runs that never recorded their index are
    not thereby proven to have used the same one."""
    config = make_config(make_data(tmp_path / "data"), tmp_path / "cache")
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    bare = build_manifest(config, name="a", gt=gt)  # no load, no selected IDs
    ok, problems = comparable(bare, bare)
    assert ok is False
    assert any("unverifiable" in item for item in problems)
    assert any("cache_fingerprint" in item for item in problems)


def test_the_two_scopes_are_never_comparable(tmp_path) -> None:
    """A 29-video run and an 873-video run must never be compared silently."""
    root = make_data(tmp_path / "data")
    visual = make_config(root, tmp_path / "a", dataset={"scope": {"mode": "existing_videos"}})
    full = make_config(root, tmp_path / "b", dataset={"scope": {"mode": "retrieval_ready"}})
    gt = load_private_dev(write_set(tmp_path, kis=[KIS]))
    ok, problems = comparable(
        full_manifest(visual, "visual", gt, fingerprint="fp29", videos=("L21_V001",)),
        full_manifest(full, "full", gt, fingerprint="fp873", videos=("L21_V001", "L22_V001")),
    )
    assert ok is False
    assert any("different dataset scope" in item for item in problems)
    assert any("different index" in item for item in problems)
    assert any("different dataset selection" in item for item in problems)


def test_the_three_named_experiments_exist() -> None:
    assert EXPERIMENTS == (EXPERIMENT_B0_RELEASE, EXPERIMENT_B0_CLEAN, EXPERIMENT_R1_ADAPTIVE)


# ------------------------------------------------------------------ split policy


def test_splits_load_independently(tmp_path) -> None:
    dev = {**KIS, "query_id": "kis_dev", "split": "development"}
    holdout = {**KIS, "query_id": "kis_holdout", "split": "holdout"}
    directory = write_set(tmp_path, kis=[dev, holdout])
    assert [e.query_id for e in load_private_dev(directory, split="development")] == ["kis_dev"]
    assert [e.query_id for e in load_private_dev(directory, split="holdout")] == ["kis_holdout"]
    assert len(load_private_dev(directory)) == 2


def test_a_row_without_a_split_is_development(tmp_path) -> None:
    directory = write_set(tmp_path, kis=[KIS])
    assert len(load_private_dev(directory, split="development")) == 1
    assert len(load_private_dev(directory, split="holdout")) == 0


# ------------------------------------------------------------- annotation helper


@pytest.fixture()
def helper(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "annotate_private_gt", ROOT / "tools" / "annotate_private_gt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PRIVATE_DEV_DIR", write_set(tmp_path, kis=[], qa=[], trake=[]))
    return module


def test_helper_maps_a_timestamp_to_an_official_frame(tmp_path, helper) -> None:
    """The helper reads map-keyframes; it never interpolates a frame that is not mapped."""
    root = tmp_path / "data"
    (root / "map-keyframes").mkdir(parents=True)
    (root / "map-keyframes" / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30,0\n2,10.0,30,300\n3,20.0,30,600\n", encoding="utf-8"
    )
    result = helper.frame_at(root, "L21_V001", 9.0)
    assert result["nearest_mapped_frame_idx"] == 300
    assert result["nearest_mapped_pts_time"] == 10.0
    back = helper.time_at(root, "L21_V001", 590)
    assert back["nearest_mapped_frame_idx"] == 600


def test_helper_only_offers_videos_with_a_local_mp4(tmp_path, helper) -> None:
    root = tmp_path / "data"
    (root / "video").mkdir(parents=True)
    (root / "video" / "L21_V001.mp4").write_bytes(b"x")
    assert helper.annotatable_videos(root) == ["L21_V001"]


def test_helper_appends_a_label_and_refuses_a_duplicate_id(helper) -> None:
    entry = {
        "query_id": "kis_0001",
        "video_id": "L21_V001",
        "label_source": "private_dev",
        "annotated_by": "a human",
        "split": "development",
        "query": "a person walks",
        "frame_ranges": [[300, 360]],
    }
    helper.append_entry("kis", entry)
    gt = load_private_dev(helper.PRIVATE_DEV_DIR)
    assert [item.query_id for item in gt.for_task("kis")] == ["kis_0001"]
    with pytest.raises(SystemExit, match="already exists"):
        helper.append_entry("kis", entry)


def test_helper_never_writes_a_model_annotator(helper) -> None:
    """Even by hand, the helper's own guard refuses an empty or absent human name."""
    import argparse

    args = argparse.Namespace(
        query_id="x", video="L21_V001", annotated_by="  ", split="development", notes=""
    )
    with pytest.raises(SystemExit, match="must name a human"):
        helper.base_entry(args, "kis")


def test_helper_source_does_not_import_the_engine() -> None:
    """Research tooling: it reads map-keyframes and MP4s, and runs no model."""
    text = (ROOT / "tools" / "annotate_private_gt.py").read_text(encoding="utf-8")
    for forbidden in ("AICCompetitionEngine", "encode_query", "score_frames", "requests", "urllib"):
        assert forbidden not in text
