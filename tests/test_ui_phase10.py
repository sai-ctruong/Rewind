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


# ------------------------------- Video (thật) --------------------------------
def test_video_list_endpoint(client) -> None:
    r = client.get("/api/video/list")
    assert r.status_code == 200
    body = r.get_json()
    assert "videos" in body and "indexed" in body and "folder" in body
    assert isinstance(body["videos"], list)


def test_video_index_missing_file_errors(client) -> None:
    r = client.post("/api/video/index", json={"video": "khong_ton_tai.mp4"})
    assert r.status_code == 404


def test_video_search_without_index_errors(client) -> None:
    r = client.post("/api/video/search", json={"video": "chua_nap", "query": "test"})
    assert r.status_code == 400


def test_video_temporal_without_index_errors(client) -> None:
    # Chưa nạp video -> 400 (logic chuỗi thứ tự đã test kỹ ở test_video_engine).
    r = client.post("/api/video/temporal",
                    json={"video": "chua_nap", "events": ["a", "b"]})
    assert r.status_code == 400


def test_video_search_image_without_index_errors(client) -> None:
    # Q1: chưa nạp video -> 400 (logic tìm-bằng-ảnh test ở test_video_engine).
    r = client.post("/api/video/search_image", data={"video": "chua_nap"})
    assert r.status_code == 400


def test_video_neighbors_missing_errors(client) -> None:
    # D1: frame không tồn tại -> 404.
    r = client.get("/api/video/neighbors/khong/co")
    assert r.status_code == 404


def test_video_progress_endpoint(client) -> None:
    # A6: endpoint tiến độ luôn trả trạng thái (rảnh -> active False).
    r = client.get("/api/video/progress")
    assert r.status_code == 200
    b = r.get_json()
    assert "active" in b and "count" in b and "fps" in b


def test_video_frame_missing_errors(client) -> None:
    r = client.get("/api/video/frame/khong/co")
    assert r.status_code == 404


def test_index_folder_missing_path_errors(client) -> None:
    r = client.post("/api/video/index_folder", json={"path": "  "})
    assert r.status_code == 400


def test_index_folder_invalid_dir_errors(client, tmp_path) -> None:
    r = client.post("/api/video/index_folder",
                    json={"path": str(tmp_path / "khong_ton_tai")})
    assert r.status_code == 400


def test_index_folder_empty_dir_errors(client, tmp_path) -> None:
    r = client.post("/api/video/index_folder", json={"path": str(tmp_path)})
    assert r.status_code == 404   # thư mục hợp lệ nhưng không có video


# ------------------------------- Frontend asset ------------------------------
def test_index_html_exists_and_wires_apis() -> None:
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for hook in ("/api/health", "/api/kisc/start", "/api/search", "/api/vqa",
                 "/api/video/list", "/api/video/search", "/api/video/index_folder",
                 "/api/video/temporal", "/api/video/search_image",
                 "/api/video/neighbors/"):
        assert hook in html
    # Có bản mô phỏng dự phòng cho chế độ Artifact (không backend).
    assert "simSession" in html and "data-theme" in html
