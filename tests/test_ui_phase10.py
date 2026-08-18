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
def client(tmp_path):
    # index_dir rỗng -> test không bị index đã lưu trên đĩa làm nhiễu
    app = create_app(index_dir=tmp_path / "index")
    app.testing = True
    return app.test_client()


def test_home_serves_full_html(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "<!doctype html>" in html.lower()
    assert "Trợ lý Truy xuất Đa phương tiện" in html


def test_health_reports_real_indexed_numbers(client) -> None:
    """Health phải báo số liệu THẬT (video đã index + keyframe thật), không phải kích
    thước một dataset tổng hợp. Chưa nạp video -> 0 (trung thực), không phải 200 giả."""
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["dataset_size"] == 0 and body["videos"] == 0


def test_synthetic_kisc_endpoints_are_gone(client) -> None:
    """Đã gỡ dataset lifelog tổng hợp + API không còn UI nào gọi. Thuật toán hội thoại
    theo thuộc tính vẫn được phủ ở tests/test_dialogue_integration_phase8.py."""
    assert client.post("/api/kisc/start", json={"query": "x"}).status_code == 404
    assert client.post("/api/kisc/respond", json={"answer": "x"}).status_code == 404
    assert client.post("/api/search", json={"query": "x"}).status_code == 404


# ------------------------------- VQA (trên video thật) ------------------------
# Suy luận VQA (đếm nến, ai tặng quà) được phủ ở tests/test_vqa_phase7.py; cầu nối
# vào index video thật ở tests/test_vqa_on_video.py. Ở đây chỉ kiểm guard HTTP.
def test_video_vqa_requires_loaded_video(client) -> None:
    r = client.post("/api/video/vqa", json={"video": "chua_nap", "question": "mấy người?"})
    assert r.status_code == 400
    assert "Nạp video" in r.get_json()["error"]


def test_video_vqa_requires_question(client) -> None:
    r = client.post("/api/video/vqa", json={"video": "x", "question": "  "})
    assert r.status_code == 400


def test_old_canned_vqa_endpoint_is_gone(client) -> None:
    """Bộ keyframe 'sinh nhật' cứng + /api/vqa đã gỡ: tab VQA giờ chạy video thật."""
    assert client.post("/api/vqa", json={"question": "x"}).status_code == 404


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


def test_video_explore_without_index_errors(client) -> None:
    r = client.get("/api/video/explore?video=chua_nap")
    assert r.status_code == 400


def test_video_similar_missing_errors(client) -> None:
    r = client.get("/api/video/similar/khong/co")
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


# --------------------- Bộ lọc ảnh hội thoại (/api/filter/*) -------------------
# Luồng thu hẹp ĐẦY ĐỦ được test ở tests/test_image_filter.py (engine mock, offline).
# Ở đây chỉ kiểm tầng HTTP: guard khi chưa nạp video / chưa có phiên, và reset.
def test_filter_start_requires_loaded_video(client) -> None:
    r = client.post("/api/filter/start", json={"video": "chua_nap", "query": "phố"})
    assert r.status_code == 400
    assert "Nạp video" in r.get_json()["error"]


def test_filter_refine_without_session_errors(client) -> None:
    r = client.post("/api/filter/refine", json={"text": "áo trắng"})
    assert r.status_code == 400


def test_filter_reset_ok_even_without_session(client) -> None:
    assert client.post("/api/filter/reset", json={}).get_json()["ok"] is True


# ----------------------------- Agent (/api/agent/*) ---------------------------
# Vòng lặp Agent được test đầy đủ ở tests/test_search_agent.py (engine mock, offline).
# Ở đây kiểm tầng HTTP: guard khi chưa nạp video / thiếu câu, và reset.
def test_agent_ask_requires_loaded_video(client) -> None:
    r = client.post("/api/agent/ask", json={"video": "chua_nap", "query": "phố"})
    assert r.status_code == 400
    assert "Nạp video" in r.get_json()["error"]


def test_agent_reset_ok(client) -> None:
    assert client.post("/api/agent/reset", json={}).get_json()["ok"] is True


# ------------------------------- Frontend asset ------------------------------
def test_index_html_exists_and_wires_apis() -> None:
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for hook in ("/api/health", "/api/video/vqa", "/api/video/list", "/api/video/search",
                 "/api/video/index_folder", "/api/video/temporal",
                 "/api/video/search_image", "/api/video/neighbors/",
                 "/api/filter/start", "/api/filter/refine", "/api/filter/reset",
                 "/api/agent/ask", "/api/agent/reset"):
        assert hook in html
    assert "simVqa" not in html and "VQA_FRAMES" not in html


def test_video_picker_is_global_not_per_tab() -> None:
    """Nạp video MỘT LẦN, mọi tab dùng chung: chỉ còn 1 bộ chọn/nạp (thanh #gv-bar),
    không còn dropdown + nút nạp riêng ở từng tab."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    # Thanh chung có đủ: chọn · nạp · thư mục · lưu index · tuỳ chọn LÚC NẠP.
    for gid in ("gv-bar", "gv-select", "gv-load", "gv-folder-btn", "gv-save",
                "gv-ocr", "gv-asr", "gv-caption"):
        assert f'id="{gid}"' in html, gid
    # KHÔNG còn điều khiển nạp riêng ở bất kỳ tab nào.
    for key in ("kis", "avs", "video", "temporal", "kisc", "agent", "vqa"):
        for ctl in ("select", "load", "folder", "save", "ocr", "asr", "caption"):
            assert f'id="{key}-{ctl}"' not in html, f"{key}-{ctl} lẽ ra đã gỡ"
    # Rerank là tuỳ chọn LÚC TÌM -> vẫn ở lại từng tab.
    for key in ("kis", "avs", "video"):
        assert f'id="{key}-rerank"' in html
    # Mọi tab đọc chung một nguồn + được báo khi đổi video.
    assert "function currentVideo()" in html and "onVideoChange(" in html


def test_agent_tab_shows_tool_trace_and_memory() -> None:
    """Tab Agent phải phơi đúng thứ làm nên 'smart path': trace tool đã gọi, trí nhớ
    phiên, đáp án của Reader, và panel Agent chủ động hỏi lại."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    assert 'data-view="agent"' in html and 'id="view-agent"' in html
    assert 'id="agent-trace"' in html and "renderTrace" in html   # Agent đã gọi tool nào
    assert 'id="agent-mem"' in html                               # trí nhớ xuyên lượt
    assert 'id="agent-answer"' in html                            # đáp án Reader
    assert 'id="agent-clarify"' in html                           # hỏi lại khi mơ hồ
    assert 'id="agent-chains"' in html and "renderChains" in html # nhánh temporal
    assert "data-theme" in html
    # Q1/Q2: ô tải ảnh + canvas phác hoạ (đều tìm qua /api/video/search_image).
    assert 'id="video-imgfile"' in html and 'id="video-sketch"' in html


def test_kisc_tab_is_image_filter_not_text_list() -> None:
    """Tab KISC phải hiện LƯỚI ẢNH thật (thu hẹp dần), không còn danh sách text."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="kisc-grid"' in html and 'id="kisc-pick"' in html   # lưới ảnh + panel chọn
    # Thẻ ảnh dựng từ URL server trả về (it.image -> /api/video/frame/<id>).
    assert "frameCard" in html and "img.src = it.image" in html
    assert 'id="kisc-shrink"' in html                              # hiện "20 → 8"
    # Bảng ứng viên dạng text cũ và bản mô phỏng KISC offline đã bị gỡ.
    assert 'id="cands"' not in html and "simSession" not in html
