"""Phase 11: the final smoke runner's contract.

The smoke exists to prove the pipeline runs end to end and emits well-formed
submissions. These tests pin what it may and may not say: a real serialize -> read back
-> validate -> reserialize round trip, an export that is refused when the rows are not
submittable, and no accuracy vocabulary anywhere in the output.

Nothing here runs the real dataset; the runner's pieces are exercised directly so the
suite stays offline and fast.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "final_smoke_queries.json"
RUNNER = ROOT / "tools" / "run_competition_smoke.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_competition_smoke", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load_runner()


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ fixture


def test_fixture_covers_all_three_tasks(fixture) -> None:
    assert len(fixture["kis"]) >= 5
    assert len(fixture["qa"]) >= 3
    assert len(fixture["trake"]) >= 3


def test_fixture_includes_vietnamese_queries(fixture) -> None:
    """Vietnamese must be exercised end to end, not only in the normalizer's unit tests."""
    accented = [q for q in fixture["kis"] if any(ord(ch) > 127 for ch in q)]
    assert len(accented) >= 2


def test_every_trake_entry_is_a_multi_event_sequence(fixture) -> None:
    for events in fixture["trake"]:
        assert len(events) >= 2
        assert all(isinstance(event, str) and event.strip() for event in events)


def test_qa_entries_declare_an_expected_answer_type(fixture) -> None:
    types = {item["expected_answer_type"] for item in fixture["qa"]}
    assert types <= {"color", "number", "yes/no", "text", "auto"}
    for item in fixture["qa"]:
        assert item["event"].strip() and item["question"].strip()


def test_fixture_carries_no_labels_or_expected_results(fixture) -> None:
    """The fixture is an integration-stability harness, not a graded benchmark."""
    text = json.dumps(fixture, ensure_ascii=False).lower()
    for forbidden in ("expected_answer\"", "expected_frame", "gold", "ground_truth", "correct_"):
        assert forbidden not in text
    assert "NO expected results" in fixture["description"]
    for item in fixture["qa"]:
        assert set(item) == {"event", "question", "expected_answer_type"}


# ----------------------------------------------------------------------- summarize


def test_summarize_reports_nothing_for_no_samples(runner) -> None:
    assert runner.summarize([]) == {"count": 0}


def test_summarize_withholds_percentiles_until_they_mean_something(runner) -> None:
    few = runner.summarize([10.0, 20.0, 30.0])
    assert few["count"] == 3 and "p95_ms" not in few
    many = runner.summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert many["p50_ms"] == 30.0 and many["p95_ms"] == 50.0
    assert many["min_ms"] == 10.0 and many["max_ms"] == 50.0


# ------------------------------------------------------------------ round tripping


def kis_prediction(video_id: str, frame_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        video_id=video_id, frame_id=frame_id, refinement=None, qa=None, trake=None
    )


def qa_prediction(video_id: str, frame_id: int, answer: str, *, visual: bool) -> SimpleNamespace:
    return SimpleNamespace(
        video_id=video_id,
        frame_id=frame_id,
        answer=answer,
        refinement=None,
        trake=None,
        qa={
            "submission_frame_idx": frame_id,
            "best_visual_frame_idx": None,
            "answer_status": "answered",
            "backend_visual": visual,
        },
    )


def trake_prediction(video_id: str, frames) -> SimpleNamespace:
    return SimpleNamespace(
        video_id=video_id,
        frame_id=frames[0],
        refinement=None,
        qa=None,
        trake={
            "steps": [
                {"submission_frame_idx": frame, "visual_frame_idx": None} for frame in frames
            ]
        },
    )


def test_kis_roundtrip_is_byte_identical(runner, tmp_path) -> None:
    predictions = [kis_prediction("L01_V001", 100 + i * 10) for i in range(5)]
    result = runner.roundtrip("kis", predictions, tmp_path, "kis")
    assert result["preflight_valid"] is True
    assert result["exported"] is True
    assert result["revalidated_valid"] is True
    assert result["rows"] == result["csv_rows"] == result["revalidated_rows"] == 5
    assert result["byte_identical_roundtrip"] is True


def test_roundtrip_writes_a_validation_report_beside_the_csv(runner, tmp_path) -> None:
    runner.roundtrip("kis", [kis_prediction("L01_V001", 100)], tmp_path, "kis")
    assert (tmp_path / "kis.csv").is_file()
    reports = list(tmp_path.glob("*.json"))
    assert reports, "a validation report must accompany every exported submission"
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["source"] == "final_smoke" and payload["task"] == "kis"
    assert payload["valid"] is True


def test_roundtrip_leaves_no_temporary_reserialized_file(runner, tmp_path) -> None:
    runner.roundtrip("kis", [kis_prediction("L01_V001", 100)], tmp_path, "kis")
    assert not (tmp_path / "kis.roundtrip.csv").exists()


def test_trake_roundtrip_preserves_every_event_column(runner, tmp_path) -> None:
    predictions = [trake_prediction("L01_V001", [10, 20, 30]), trake_prediction("L02_V001", [11, 21, 31])]
    result = runner.roundtrip("trake", predictions, tmp_path, "trake", event_count=3)
    assert result["exported"] is True and result["byte_identical_roundtrip"] is True
    rows = [line.split(",") for line in (tmp_path / "trake.csv").read_text(encoding="utf-8").splitlines()]
    assert all(len(row) == 4 for row in rows)


def test_trake_roundtrip_refuses_a_row_missing_an_event(runner, tmp_path) -> None:
    result = runner.roundtrip(
        "trake", [trake_prediction("L01_V001", [10, 20])], tmp_path, "trake", event_count=3
    )
    assert result["preflight_valid"] is False
    assert result["exported"] is False
    assert not (tmp_path / "trake.csv").exists()


def test_mock_backend_answers_are_refused_rather_than_exported(runner, tmp_path) -> None:
    """A non-visual backend cannot answer a question about a video. The refusal is correct."""
    predictions = [qa_prediction("L01_V001", 100, "red", visual=False)]
    result = runner.roundtrip("qa", predictions, tmp_path, "qa")
    assert result["preflight_valid"] is False
    assert result["exported"] is False
    assert "QA_NON_SUBMITTABLE_STATUS" in result["errors"]
    assert not (tmp_path / "qa.csv").exists()


def test_overlong_answers_are_refused(runner, tmp_path) -> None:
    """A backend that echoes a video description is not producing an answer."""
    predictions = [qa_prediction("L01_V001", 100, "x" * 4000, visual=True)]
    result = runner.roundtrip("qa", predictions, tmp_path, "qa")
    assert result["preflight_valid"] is False
    assert "QA_ANSWER_TOO_LONG" in result["errors"]


def test_a_visual_backend_answer_does_export(runner, tmp_path) -> None:
    predictions = [qa_prediction("L01_V001", 100, "red", visual=True)]
    result = runner.roundtrip("qa", predictions, tmp_path, "qa")
    assert result["exported"] is True and result["byte_identical_roundtrip"] is True
    assert (tmp_path / "qa.csv").read_text(encoding="utf-8").strip() == "L01_V001,100,red"


# --------------------------------------------------------------------- report shape


def minimal_report() -> dict:
    return {
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "config": "configs/competition.yaml",
        "fixture": "final_smoke_queries.json",
        "disclaimer": "Structural integration smoke. No AIC ground truth exists.",
        "system_profile": {"project_version": "0.11.0-aic2026", "git_commit": "abc1234"},
        "readiness": {"status": "READY_WITH_WARNINGS"},
        "timing": {"startup_ms": 1.0},
        "structural_invariants": {"malformed_prediction_count": 0},
        "tasks": {
            "kis": {
                "runs": [
                    {
                        "query": "a person <walking>",
                        "results": 100,
                        "candidate_union_size": 500,
                        "channels_contributing": ["clip"],
                        "top": [["L01_V001", 100]],
                        "total_ms": 12.0,
                    }
                ],
                "submission": {"exported": True},
            },
            "qa": {
                "note": "No production visual Q&A backend is available.",
                "runs": [
                    {
                        "question": "What color?",
                        "expected_answer_type": "color",
                        "answered": 8,
                        "hypotheses": 8,
                        "backend": "mock",
                        "backend_visual_capable": False,
                        "cross_video_answer_copy_count": 0,
                        "total_ms": 5.0,
                    }
                ],
                "submission": {"exported": False},
            },
            "trake": {
                "runs": [
                    {
                        "events": ["a", "b", "c"],
                        "videos_with_full_event_coverage": 4,
                        "returned": 30,
                        "expansion_triggered": True,
                        "events_expanded": [0],
                        "structural": {"malformed_prediction_count": 0},
                        "total_ms": 900.0,
                    }
                ],
                "submission": {"exported": True},
            },
        },
    }


def test_html_report_renders_and_escapes_queries(runner) -> None:
    html = runner.render_html(minimal_report())
    assert html.lower().startswith("<!doctype html>")
    assert "&lt;walking&gt;" in html
    assert "<walking>" not in html


def test_html_report_states_it_is_structural_only(runner) -> None:
    html = runner.render_html(minimal_report())
    assert "Structural only" in html
    assert "No AIC ground truth exists" in html
    assert "NON-VISUAL" in html


def test_runner_source_makes_no_quality_claim() -> None:
    text = RUNNER.read_text(encoding="utf-8").lower()
    for forbidden in ("sota", "state of the art", "outperform", "improves recall", "r@1"):
        assert forbidden not in text
    # The word "accuracy" may appear only where it is being disclaimed.
    assert text.count("accuracy") == text.count("no accuracy is") > 0


def test_generated_smoke_summary_conforms_when_present() -> None:
    """If a real smoke has been run in this checkout, its output must match the contract."""
    summary = ROOT / "artifacts" / "final_release_smoke" / "summary.json"
    if not summary.is_file():
        pytest.skip("no smoke run in this checkout")
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert set(report) >= {
        "generated_at_utc",
        "config",
        "disclaimer",
        "system_profile",
        "readiness",
        "cache",
        "timing",
        "tasks",
        "memory",
        "structural_invariants",
    }
    assert set(report["tasks"]) == {"kis", "qa", "trake"}
    assert "no result is labelled correct" in report["disclaimer"]
    for key in ("malformed_prediction_count", "wrong_event_count_prediction_count",
                "cross_video_step_count", "unordered_submission_sequence_count",
                "cross_video_answer_copy_count",
                "answer_without_matching_evidence_video_count"):
        assert report["structural_invariants"][key] == 0, key
