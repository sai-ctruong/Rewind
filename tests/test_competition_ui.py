from pathlib import Path

import ui.app as appmod


def test_competition_ui_exposes_status_and_manual_controls() -> None:
    html = (Path(appmod.__file__).parent / "index.html").read_text(encoding="utf-8")
    for token in (
        'id="qa-event"', 'id="qa-type"', 'id="qa-confidence"', 'id="qa-correction"',
        'id="trake-status"', 'id="gv-dataset"', 'id="gv-objects" type="checkbox" checked', "Load Dataset",
        'id="kis-display" type="hidden" value="100"',
        "function compactAnswer(value, limit = 100)",
        "KIS - t", "Q&amp;A - t", "TRAKE - chu",
        "Copy video_id, frame", "Show nearby keyframes",
        'id="trake-nearby"', "Copy sequence", "query_hints", "event_hints",
        'retrieval_query_mode: "evidence"',
        # The edit control is now explicitly row-scoped and names the SUBMISSION frame.
        "Manual SUBMISSION frame edit (this row only)",
        "score_breakdown", "mp4", "encoder_type",
    ):
        assert token in html
    for token in (
        'data-view="evaluation"',
        'id="view-evaluation"',
        'id="evaluation-labels"',
        "/api/evaluation/run",
        "/api/evaluation/status",
        'id="gv-select"',
        'id="gv-load"',
        'id="gv-folder"',
        'id="gv-folder-btn"',
        'id="gv-save"',
        'id="gv-asr"',
        'id="gv-caption"',
        'id="kis-refine"',
        'id="kis-refine-status"',
        'id="qa-window"',
        'id="kis-preflight"',
        'id="qa-preflight"',
        'id="trake-preflight"',
        'id="kis-save"',
        'id="qa-save"',
        'id="trake-save"',
        "Video/index",
        "Dataset Root",
        "Local refinement",
        "Save CSV",
        "Show top",
        "Per event K",
        "Expected answer type",
    ):
        assert token not in html


def test_ui_never_mutates_rows_by_matching_frame_values() -> None:
    """The Phase 0 bug: an edit that rewrote every matching value across every task."""
    html = (Path(appmod.__file__).parent / "index.html").read_text(encoding="utf-8")
    # The old implementation looped over every task and replaced matching values.
    assert "Object.keys(state.rows).forEach" not in html
    assert 'String(value) === old' not in html
    # Edits and export now address a backend result batch by id.
    assert "/api/results/" in html
    assert "/api/submission/preflight" in html
    assert "row_id" in html


def test_evaluation_requires_loaded_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(appmod, "DATA_ROOT", tmp_path / "data")
    app = appmod.create_app()
    app.testing = True
    client = app.test_client()
    assert client.get("/api/evaluation/status").status_code == 200
    assert client.post("/api/evaluation/run", json={"labels": "missing.jsonl"}).status_code == 400
