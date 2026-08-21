"""Backend/UI smoke tests for the AIC 2026 competition mode."""
from __future__ import annotations

from pathlib import Path

import pytest

import ui.app as appmod

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(appmod, "AIC_CACHE_DIR", tmp_path / "index")
    monkeypatch.setattr(appmod, "SUBMISSION_DIR", tmp_path / "submissions")
    app = appmod.create_app()
    app.testing = True
    return app.test_client()


def test_home_serves_competition_html(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "<!doctype html>" in html.lower()
    assert "AIC 2026 Rewind" in html
    assert "KIS - t" in html and "Q&amp;A - t" in html and "TRAKE - chu" in html


def test_health_reports_competition_mode(client) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["mode"] == "aic2026"
    assert body["dataset_size"] == 0 and body["videos"] == 0


def test_video_list_endpoint_uses_data_video(client) -> None:
    r = client.get("/api/video/list")
    assert r.status_code == 200
    body = r.get_json()
    assert "videos" in body and "indexed" in body and "folder" in body
    assert body["folder"].endswith("data")


@pytest.mark.parametrize("path,method", [
    ("/api/video/search_image", "post"),
    ("/api/video/explore?video=x", "get"),
    ("/api/video/similar/x", "get"),
    ("/api/filter/start", "post"),
    ("/api/filter/refine", "post"),
    ("/api/filter/reset", "post"),
    ("/api/agent/ask", "post"),
    ("/api/agent/reset", "post"),
    ("/api/kisc/start", "post"),
    ("/api/kisc/respond", "post"),
])
def test_legacy_endpoints_are_gone(client, path: str, method: str) -> None:
    call = getattr(client, method)
    assert call(path, json={}).status_code == 404


def test_aic_index_without_data_errors(client) -> None:
    r = client.post("/api/video/index", json={"video": "__aic2026__"})
    assert r.status_code == 400



def test_search_requires_loaded_index(client) -> None:
    r = client.post("/api/video/search", json={"video": "x", "query": "street"})
    assert r.status_code == 400


def test_search_requires_query(client) -> None:
    r = client.post("/api/video/search", json={"video": "x", "query": "  "})
    assert r.status_code == 400


def test_vqa_requires_loaded_index(client) -> None:
    r = client.post("/api/video/vqa", json={"video": "x", "question": "who?"})
    assert r.status_code == 400


def test_temporal_requires_loaded_index_first(client) -> None:
    r = client.post("/api/video/temporal", json={"video": "x", "events": ["a", "b"]})
    assert r.status_code == 400


def test_neighbors_and_frame_requires_loaded_index(client) -> None:
    assert client.get("/api/video/neighbors/nope").status_code == 400
    assert client.get("/api/video/frame/nope").status_code == 400


def test_progress_endpoint(client) -> None:
    r = client.get("/api/video/progress")
    assert r.status_code == 200
    b = r.get_json()
    assert "active" in b and "count" in b and "fps" in b


def test_save_without_index_errors(client) -> None:
    r = client.post("/api/video/save", json={})
    assert r.status_code == 400


def test_index_folder_validation(client, tmp_path) -> None:
    assert client.post("/api/video/index_folder", json={"path": "  "}).status_code == 400
    assert client.post("/api/video/index_folder", json={"path": str(tmp_path / "missing")}).status_code == 400
    assert client.post("/api/video/index_folder", json={"path": str(tmp_path)}).status_code == 400


def test_index_html_wires_only_competition_apis() -> None:
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for hook in (
        "/api/health",
        "/api/video/list",
        "/api/video/index",
        "/api/video/progress",
        "/api/video/search",
        "/api/video/vqa",
        "/api/video/temporal",
        "/api/video/neighbors/",
        "/api/submission/save",
    ):
        assert hook in html
    for removed in (
        "/api/video/search_image",
        "/api/video/explore",
        "/api/video/similar",
        "/api/filter/",
        "/api/agent/",
        "view-agent",
        "view-kisc",
        "video-sketch",
    ):
        assert removed not in html


def test_global_index_controls_and_three_views_exist() -> None:
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for gid in ("gv-bar", "gv-dataset", "gv-objects"):
        assert f'id="{gid}"' in html
    for hidden in ("gv-ocr", "gv-select", "gv-load", "gv-folder-btn", "gv-save", "gv-asr", "gv-caption"):
        assert f'id="{hidden}"' not in html
    for sid in ("kis-reset", "qa-reset", "trake-reset"):
        assert f'id="{sid}"' in html
    for hidden in ("kis-preflight", "qa-preflight", "trake-preflight", "kis-save", "qa-save", "trake-save"):
        assert f'id="{hidden}"' not in html
    for view in ("kis", "qa", "trake"):
        assert f'id="view-{view}"' in html
        assert f'data-view="{view}"' in html
    assert "function currentVideo()" in html
