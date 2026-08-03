from pathlib import Path

import ui.app as appmod


def test_competition_ui_exposes_status_manual_controls_and_background_eval() -> None:
    html = (Path(appmod.__file__).parent / "index.html").read_text(encoding="utf-8")
    for token in (
        'id="qa-event"', 'id="qa-type"', 'id="qa-confidence"', 'id="qa-correction"',
        'id="trake-status"', 'id="view-evaluation"', 'id="evaluation-labels"',
        "/api/evaluation/run", "/api/evaluation/status", "Manual frame adjustment",
        "score_breakdown", "mp4", "encoder_type",
    ):
        assert token in html


def test_evaluation_requires_loaded_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(appmod, "DATA_ROOT", tmp_path / "data")
    app = appmod.create_app()
    app.testing = True
    client = app.test_client()
    assert client.get("/api/evaluation/status").status_code == 200
    assert client.post("/api/evaluation/run", json={"labels": "missing.jsonl"}).status_code == 400
