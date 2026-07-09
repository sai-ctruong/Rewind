"""Unit test Phase 10 — web UI backend (CLAUDE.md Mục 8).

DoD Phase 10: "Demo được toàn bộ luồng end-to-end cho người ngoài xem". Ta kiểm backend
Flask (ui/app.py) nối ĐÚNG pipeline thật cho cả 4 luồng: trang chủ, KISC hội thoại,
tìm kiếm, VQA. Frontend index.html là trang tĩnh nên chỉ kiểm sự tồn tại + gắn kết API.

Chạy offline bằng Flask test client (không mở cổng mạng).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ui.app import create_app

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


@pytest.fixture()
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_home_serves_full_html(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "<!doctype html>" in html.lower()
    assert "Trợ lý Truy xuất Đa phương tiện" in html


def test_health(client) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["dataset_size"] > 0


# ------------------------------- KISC end-to-end ------------------------------
def test_kisc_flow_converges(client) -> None:
    r1 = client.post("/api/kisc/start",
                     json={"query": "Tôi gặp một người bạn cũ vào tuần trước."})
    s1 = r1.get_json()
    assert not s1["finished"]           # còn mơ hồ -> hỏi tiếp
    assert s1["count"] > 5
    assert "?" in s1["response"]         # có câu hỏi làm rõ

    r2 = client.post("/api/kisc/respond",
                     json={"answer": "Ở quán cà phê ngoài trời, anh ấy mặc áo xanh dương."})
    s2 = r2.get_json()
    assert s2["finished"]
    assert s2["count"] <= 5
    assert s2["top"][0]["id"] == "kf_0000"   # hội tụ đúng ground-truth
    # filter tích luỹ có mặt các slot đã trích.
    assert s2["filters"].get("clothing_color") == "blue"


def test_kisc_respond_without_start_errors(client) -> None:
    r = client.post("/api/kisc/respond", json={"answer": "gì đó"})
    assert r.status_code == 400


def test_kisc_start_empty_query_errors(client) -> None:
    r = client.post("/api/kisc/start", json={"query": "   "})
    assert r.status_code == 400


# ------------------------------- Search --------------------------------------
def test_search_returns_ranked_results(client) -> None:
    r = client.post("/api/search",
                    json={"query": "người đàn ông áo xanh dương quán cà phê"})
    body = r.get_json()
    assert body["count"] > 0
    first = body["results"][0]
    assert {"id", "video_id", "timestamp", "score", "attributes", "caption"} <= set(first)


def test_search_empty_errors(client) -> None:
    assert client.post("/api/search", json={"query": ""}).status_code == 400


# ------------------------------- VQA -----------------------------------------
def test_vqa_counts_candles(client) -> None:
    r = client.post("/api/vqa", json={"question": "Có bao nhiêu ngọn nến trên bánh?"})
    body = r.get_json()
    assert body["value"] == 5


def test_vqa_identifies_gift_giver(client) -> None:
    r = client.post("/api/vqa", json={"question": "Ai là người tặng quà?"})
    assert "người đàn ông" in r.get_json()["answer"].lower()


# ------------------------------- Frontend asset ------------------------------
def test_index_html_exists_and_wires_apis() -> None:
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for hook in ("/api/health", "/api/kisc/start", "/api/search", "/api/vqa"):
        assert hook in html
    # Có bản mô phỏng dự phòng cho chế độ Artifact (không backend).
    assert "simSession" in html and "data-theme" in html
