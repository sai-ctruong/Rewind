"""Unit test cho retrieval/agent_tools — G1 Tool Registry (Action Space cho Agent).

Tái dùng ColorMockEncoder (màu = ngữ nghĩa) như test_video_engine: dựng 1 video 3 cảnh
(đỏ/lá/dương), index bằng engine mock, rồi kiểm mọi tool trong registry CHẠY THẬT trên
engine đó và trả ToolResult chuẩn hoá. Không cần model thật / API key (đúng pattern
ABC+Mock+Claude-lazy) — G2 Search Agent sẽ dựng trên đúng registry này.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from retrieval.agent_tools import ToolRegistry, build_registry  # noqa: E402
from retrieval.video_engine import VideoSearchEngine  # noqa: E402
from tests.test_video_engine import ColorMockEncoder, _make_video  # noqa: E402


def _jpeg(bgr, size=64) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def registry(tmp_path) -> ToolRegistry:
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return build_registry(engine, entry, images={"red": _jpeg((0, 0, 255)),
                                                  "blue": _jpeg((255, 0, 0))})


# ------------------------------- registry cơ bản ------------------------------
def test_registers_full_action_space(registry) -> None:
    names = set(registry.names())
    assert {"search", "search_temporal", "search_by_image", "search_multimodal",
            "understand", "neighbors", "search_similar", "suggest_concepts",
            "disambiguation"} <= names


def test_specs_anthropic_and_openai_shape(registry) -> None:
    a = registry.specs("anthropic")
    assert all({"name", "description", "input_schema"} <= set(t) for t in a)
    # input_schema là JSON Schema hợp lệ (object + properties)
    s = next(t for t in a if t["name"] == "search")
    assert s["input_schema"]["type"] == "object"
    assert "query" in s["input_schema"]["properties"]
    assert s["input_schema"]["required"] == ["query"]

    o = registry.specs("openai")
    assert all(t["type"] == "function" for t in o)
    assert all("parameters" in t["function"] for t in o)


def test_specs_bad_format_raises(registry) -> None:
    with pytest.raises(ValueError):
        registry.specs("cohere")


# ------------------------------- call: đường thành công -----------------------
def test_call_search_returns_normalized(registry) -> None:
    res = registry.call("search", query="cảnh màu đỏ", top_k=3)
    assert res.ok and res.tool == "search"
    assert res.items and all(
        {"keyframe_id", "video_id", "timestamp", "score"} <= set(it)
        for it in res.items)
    # màu đỏ -> keyframe cảnh đỏ (timestamp ~[0,1)) đứng đầu (ngữ nghĩa mock đúng)
    assert res.items[0]["timestamp"] < 1.0


def test_call_search_by_image(registry) -> None:
    res = registry.call("search_by_image", image_ref="blue", top_k=3)
    assert res.ok and res.items
    assert res.items[0]["timestamp"] >= 2.0  # cảnh xanh dương ~[2,3)


def test_call_understand_routes_temporal(registry) -> None:
    res = registry.call("understand", query="cảnh đỏ trước khi cảnh xanh dương")
    assert res.ok
    assert res.meta["has_temporal"] is True
    assert res.meta["events"] and len(res.meta["events"]) >= 2


def test_call_search_temporal(registry) -> None:
    res = registry.call("search_temporal", events=["cảnh đỏ", "cảnh xanh dương"])
    assert res.ok and res.tool == "search_temporal"
    # có thể rỗng nếu không có tổ hợp, nhưng đúng thứ tự thì mỗi item có steps tăng dần
    for m in res.items:
        ts = [s["timestamp"] for s in m["steps"]]
        assert ts == sorted(ts)


def test_call_neighbors_and_similar(registry) -> None:
    first = registry.call("search", query="cảnh màu đỏ", top_k=1).items[0]
    kid = first["keyframe_id"]
    nb = registry.call("neighbors", frame_id=kid, before=2, after=2)
    assert nb.ok
    sim = registry.call("search_similar", keyframe_id=kid, top_k=3)
    assert sim.ok
    assert kid not in [it["keyframe_id"] for it in sim.items]  # loại chính nó


def test_call_disambiguation_confident_when_clear(registry) -> None:
    # top-1 nổi trội -> confident, không cần hỏi lại
    cands = [{"keyframe_id": "a", "score": 0.9}, {"keyframe_id": "b", "score": 0.1},
             {"keyframe_id": "c", "score": 0.05}, {"keyframe_id": "d", "score": 0.02},
             {"keyframe_id": "e", "score": 0.01}]
    res = registry.call("disambiguation", candidates=cands, k=3)
    assert res.ok and res.meta["confident"] is True and res.items == []


# ------------------------------- call: đường lỗi (self-reflect) ---------------
def test_call_unknown_tool_returns_error_not_raise(registry) -> None:
    res = registry.call("teleport", x=1)
    assert res.ok is False and res.error and "teleport" in res.error


def test_call_bad_image_ref_returns_error(registry) -> None:
    res = registry.call("search_by_image", image_ref="khong_ton_tai")
    assert res.ok is False and "image_ref" in res.error


def test_call_missing_required_param_returns_error(registry) -> None:
    res = registry.call("search")  # thiếu query
    assert res.ok is False and res.error
