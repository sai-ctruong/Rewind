"""TRAKE output shape at the engine, metric, export, and HTTP boundaries.

Every emitted row must carry exactly one frame per query event. These tests check that
at each place a row can escape: `search_trake`, the R-score metric, the CSV writer, and
the `/api/video/temporal` payload.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ui.app as appmod
from aic2026.config import ConfigError, app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.metrics import (
    FrameRange,
    RankedAnswer,
    SubmissionStructureError,
    TRAKEGroundTruth,
    is_structurally_valid_trake_row,
    trake_r_score,
    write_submission,
)
from aic2026.text_encoder import HashingTextEncoder
from aic2026.trake import METHOD_BEAM_DP, TrakeStructureError

FPS = 10.0
FRAME_IDS = (5, 15, 25, 35, 45, 55)
VIDEOS = ("L21_V001", "L21_V002")


def make_root(root: Path) -> Path:
    """A small AIC-shaped root with enough keyframes per video for several events."""
    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    for position, video_id in enumerate(VIDEOS):
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate(FRAME_IDS, start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        angle = 0.2 + 0.5 * position
        features = np.array(
            [[np.cos(angle + 0.03 * i), np.sin(angle + 0.03 * i)] for i in range(len(FRAME_IDS))],
            dtype=np.float32,
        )
        np.save(root / "clip-features-32" / f"{video_id}.npy", features)
        folder = root / "keyframes" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        for ordinal in range(1, len(FRAME_IDS) + 1):
            Image.new("RGB", (16, 16), (40 * position, 30, 20)).save(folder / f"{ordinal:03d}.jpg")
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path, **trake):
    settings = {"min_gap_s": 0.0, "top_video_hypotheses": 5, "per_event_top_k": 20}
    settings.update(trake)
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir),
                    "frame_cache_dir": str(frame_cache_dir),
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
                "ranking": {"min_frame_gap": 0, "final_top_k": 100},
                "refinement": {"mode": "disabled"},
                "qa": {"backend": {"type": "mock"}},
                "trake": settings,
            }
        }
    )


def build(tmp_path: Path, **trake):
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames", **trake)
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
    )
    return engine


# ------------------------------------------------------------------- the engine


@pytest.mark.parametrize("event_count", [2, 3, 4])
def test_every_returned_row_has_one_frame_per_event(tmp_path: Path, event_count: int) -> None:
    engine = build(tmp_path)
    events = [f"event number {i}" for i in range(event_count)]
    outcome = engine.search_trake_detailed(events, max_results=20)
    assert outcome.predictions, "the fixture should align at least one video"
    for prediction in outcome.predictions:
        assert len(prediction.event_frame_ids) == event_count
        # The submission row is video_id followed by exactly one frame per event.
        assert len(prediction.row()) == event_count + 1
        assert prediction.row()[0] == prediction.video_id
    assert outcome.diagnostics["event_count"] == event_count
    assert outcome.diagnostics["wrong_event_count_prediction_count"] == 0


def test_engine_structural_summary_is_all_zero(tmp_path: Path) -> None:
    engine = build(tmp_path)
    outcome = engine.search_trake_detailed(["a", "b", "c"], max_results=20)
    summary = outcome.structural_summary()
    assert summary["malformed_prediction_count"] == 0
    assert summary["wrong_event_count_prediction_count"] == 0
    assert summary["cross_video_step_count"] == 0


def test_matches_stay_one_to_one_with_events(tmp_path: Path) -> None:
    engine = build(tmp_path)
    events = ["first thing", "second thing", "third thing"]
    predictions, matches = engine.search_trake(events, max_results=10)
    assert matches
    for match in matches:
        assert len(match.steps) == len(events)
        # Step i carries event i's text, so a UI zipping them cannot shift labels.
        assert [step.event for step in match.steps] == events
    assert len(predictions) == len(matches)


def test_trake_provenance_is_attached_to_predictions(tmp_path: Path) -> None:
    engine = build(tmp_path)
    outcome = engine.search_trake_detailed(["a", "b", "c"], max_results=5)
    payload = outcome.predictions[0].trake
    assert payload["event_count"] == 3
    assert len(payload["frame_ids"]) == 3
    assert len(payload["steps"]) == 3
    assert payload["method"] == METHOD_BEAM_DP
    assert payload["missing_event_indices"] == []
    assert [item["event_index"] for item in payload["steps"]] == [0, 1, 2]
    # Submission frames are the coarse official mapped frames; no decoded frame appears.
    assert all(item["visual_frame_idx"] is None for item in payload["steps"])
    assert all(
        item["submission_frame_idx"] == item["coarse_official_frame_idx"]
        for item in payload["steps"]
    )


def test_final_top_k_is_respected(tmp_path: Path) -> None:
    engine = build(tmp_path)
    outcome = engine.search_trake_detailed(["a", "b"], max_results=1)
    assert len(outcome.predictions) <= 1


def test_fewer_than_two_events_is_rejected(tmp_path: Path) -> None:
    engine = build(tmp_path)
    with pytest.raises(ValueError, match="at least two ordered events"):
        engine.search_trake(["only one"])


def test_no_complete_alignment_returns_an_empty_result(tmp_path: Path) -> None:
    engine = build(tmp_path)
    # No candidates at all: the result must be empty, not a set of short rows.
    engine.search_candidates = lambda *args, **kwargs: []
    outcome = engine.search_trake_detailed(["a", "b", "c"], max_results=10)
    assert outcome.predictions == []
    assert outcome.matches == []
    assert outcome.diagnostics["returned_complete_predictions"] == 0
    assert outcome.structural_summary()["malformed_prediction_count"] == 0


def test_refinement_is_off_by_default_and_reported_honestly(tmp_path: Path) -> None:
    engine = build(tmp_path)
    outcome = engine.search_trake_detailed(["a", "b"], refine_window_s=6.0, max_results=5)
    # Phase 8 wires refine_window_s into the local window, but refinement itself is
    # opt-in, so a default query still does no video work and says so.
    assert outcome.diagnostics["refinement_applied"] is False
    assert outcome.diagnostics["refinement_status"] == "disabled"
    assert outcome.diagnostics["refine_window_s_requested"] == 6.0
    assert outcome.diagnostics["frames_decoded"] == 0
    other = engine.search_trake_detailed(["a", "b"], refine_window_s=None, max_results=5)
    assert [p.row() for p in outcome.predictions] == [p.row() for p in other.predictions]


# ---------------------------------------------------------------------- metrics


def test_metric_refuses_a_row_with_the_wrong_number_of_frames() -> None:
    gt = TRAKEGroundTruth("V", tuple(FrameRange(i, i) for i in (1, 2, 3, 4)))
    # A short row used to be silently zipped and scored as partially correct.
    assert trake_r_score(RankedAnswer("V", ("1", "2", "3")), gt) == 0.0
    assert trake_r_score(RankedAnswer("V", ("1", "2", "3", "4", "5")), gt) == 0.0
    assert trake_r_score(RankedAnswer("V", ("1", "2", "3", "4")), gt) == pytest.approx(1.0)
    assert trake_r_score(RankedAnswer("V", ("1", "2", "30", "4")), gt) == pytest.approx(0.75)


def test_structural_validity_helper() -> None:
    assert is_structurally_valid_trake_row(("1", "2", "3"), 3) is True
    assert is_structurally_valid_trake_row(("1", "2"), 3) is False
    assert is_structurally_valid_trake_row((), 0) is False


def test_metric_still_rejects_the_wrong_video() -> None:
    gt = TRAKEGroundTruth("V", (FrameRange(1, 1), FrameRange(2, 2)))
    assert trake_r_score(RankedAnswer("OTHER", ("1", "2")), gt) == 0.0


# ----------------------------------------------------------------------- export


def test_writer_refuses_a_malformed_trake_row(tmp_path: Path) -> None:
    rows = [["V", "1", "2", "3"], ["W", "1", "2"]]
    with pytest.raises(SubmissionStructureError, match="Row 1 has 3 columns"):
        write_submission(rows, tmp_path / "trake.csv", require_row_length=4)
    assert not (tmp_path / "trake.csv").exists(), "nothing may be written on refusal"


def test_writer_accepts_well_formed_rows(tmp_path: Path) -> None:
    rows = [["V", "1", "2", "3"], ["W", "4", "5", "6"]]
    path = write_submission(rows, tmp_path / "trake.csv", require_row_length=4)
    assert path.read_text(encoding="utf-8").strip().splitlines() == ["V,1,2,3", "W,4,5,6"]


def test_writer_without_a_length_requirement_is_unchanged(tmp_path: Path) -> None:
    path = write_submission([["V", "1"], ["W", "2", "3"]], tmp_path / "kis.csv")
    assert path.is_file()


# ------------------------------------------------------------------------ HTTP


@pytest.fixture()
def client(tmp_path: Path):
    root = make_root(tmp_path / "data")
    config = make_config(root, tmp_path / "cache", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    http = app.test_client()
    assert http.post("/api/video/index_folder", json={"path": str(root)}).status_code == 200
    return http


def test_temporal_payload_is_event_preserving(client) -> None:
    events = ["a person appears", "the person moves", "the person leaves"]
    body = client.post("/api/video/temporal", json={"events": events, "max_results": 5}).get_json()
    assert body["event_count"] == 3
    assert body["count"] >= 1
    for chain in body["matches"]:
        assert chain["event_count"] == 3
        assert len(chain["frame_ids"]) == 3
        assert len(chain["steps"]) == 3
        assert chain["missing_event_indices"] == []
        assert chain["method"] == METHOD_BEAM_DP
        # Each step names its own event and its own index: no positional zipping.
        assert [step["event_index"] for step in chain["steps"]] == [0, 1, 2]
        assert [step["event"] for step in chain["steps"]] == events
        assert [step["event_label"] for step in chain["steps"]] == ["Event 1", "Event 2", "Event 3"]
    for row in body["predictions"]:
        assert len(row) == 4
    assert body["structural"]["malformed_prediction_count"] == 0
    assert body["structural"]["cross_video_step_count"] == 0


def test_temporal_payload_reports_refinement_honestly(client) -> None:
    body = client.post(
        "/api/video/temporal", json={"events": ["a", "b"], "refine_window": 6.0}
    ).get_json()
    assert body["refinement"]["applied"] is False
    assert body["refinement"]["status"] == "disabled"
    assert body["diagnostics"]["refinement_applied"] is False
    assert body["diagnostics"]["refine_window_s_requested"] == 6.0


def test_temporal_payload_carries_the_runtime_generation_and_no_paths(client, tmp_path) -> None:
    body = client.post("/api/video/temporal", json={"events": ["a", "b", "c"]}).get_json()
    assert body["generation"] >= 1
    for chain in body["matches"]:
        for step in chain["steps"]:
            assert step["generation"] == body["generation"]
            assert step["image"].startswith("/api/")
            assert f"generation={body['generation']}" in step["image"]
    assert str(tmp_path) not in str(body)


def test_submission_save_refuses_a_short_trake_row(client) -> None:
    good = client.post(
        "/api/submission/save",
        json={"task": "trake", "event_count": 3, "rows": [["V", "1", "2", "3"]]},
    )
    assert good.status_code == 200
    bad = client.post(
        "/api/submission/save",
        json={"task": "trake", "event_count": 3, "rows": [["V", "1", "2"]]},
    )
    assert bad.status_code == 422
    assert bad.get_json()["error_code"] == "MALFORMED_SUBMISSION_ROW"


def test_trake_alignment_method_config_rejects_exact_dp() -> None:
    assert app_config_from_dict(
        {"aic2026": {"trake": {"alignment_method": "beam_pruned_dp"}}}
    ).trake.alignment_method == "beam_pruned_dp"
    with pytest.raises(ConfigError, match="not exact DP"):
        app_config_from_dict({"aic2026": {"trake": {"alignment_method": "exact_dp"}}})
