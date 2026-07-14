"""Unit test cho retrieval/search_agent — G2 Search Agent (orchestrator loop).

Tái dùng engine mock màu (ColorMockEncoder) như G1. Kiểm:
  - MockPlanner ĐỊNH TUYẾN đúng: chữ -> search, thứ tự thời gian -> search_temporal,
    ảnh -> search_by_image, chữ+ảnh -> search_multimodal; luôn understand trước.
  - ClaudePlanner lái đúng tool-use loop của Anthropic bằng FAKE client (offline,
    không cần API key / mạng): tool_use -> thực thi -> tool_result -> text cuối.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from retrieval.search_agent import (  # noqa: E402
    ClaudePlanner, MockPlanner, SearchAgent)
from retrieval.video_engine import VideoSearchEngine  # noqa: E402
from tests.test_video_engine import ColorMockEncoder, _make_video  # noqa: E402


def _jpeg(bgr, size=64) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def agent(tmp_path) -> SearchAgent:
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return SearchAgent(engine, entry)  # mặc định MockPlanner


# ------------------------------- MockPlanner định tuyến -----------------------
def test_text_query_routes_to_search(agent) -> None:
    run = agent.run("cảnh màu đỏ")
    assert run.tools_used()[0] == "understand"          # luôn hiểu trước
    assert "search" in run.tools_used()
    assert run.meta["route"] == "search"
    assert run.results and run.results[0]["timestamp"] < 1.0  # cảnh đỏ đầu video


def test_temporal_query_routes_to_search_temporal(agent) -> None:
    run = agent.run("cảnh đỏ trước khi cảnh xanh dương")
    assert run.meta["route"] == "search_temporal"
    assert "search_temporal" in run.tools_used()
    # nhánh temporal KHÔNG gọi disambiguation
    assert "disambiguation" not in run.tools_used()


def test_image_query_routes_to_search_by_image(agent) -> None:
    run = agent.run("", images={"q": _jpeg((255, 0, 0))})  # chỉ ảnh (xanh dương)
    assert run.meta["route"] == "search_by_image"
    assert run.results and run.results[0]["timestamp"] >= 2.0


def test_text_plus_image_routes_multimodal(agent) -> None:
    run = agent.run("cảnh màu đỏ", images={"q": _jpeg((0, 0, 255))})
    assert run.meta["route"] == "search_multimodal"
    assert "search_multimodal" in run.tools_used()


def test_text_route_runs_disambiguation_step(agent) -> None:
    run = agent.run("cảnh màu đỏ")
    assert "disambiguation" in run.tools_used()
    # ít keyframe + top-1 rõ -> tự tin, không gắn cờ hỏi lại
    assert "need_clarification" not in run.meta


def test_run_records_finish_step(agent) -> None:
    run = agent.run("cảnh màu đỏ")
    assert run.steps[-1].action.kind == "finish"
    assert run.steps[-1].result is None


# ------------------------------- ClaudePlanner loop (fake client) -------------
class _FakeClaude:
    """Giả lập anthropic client: trả lần lượt các response đã kịch bản hoá."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs  # để assert tools/messages được truyền
        resp = self._scripted[self.calls]
        self.calls += 1
        return resp


def _tool_use(name, input, id="tu1"):
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def test_claude_planner_tool_use_then_answer(agent) -> None:
    # Kịch bản: lượt 1 Claude gọi search; lượt 2 trả lời text.
    scripted = [
        SimpleNamespace(content=[_tool_use("search", {"query": "cảnh màu đỏ", "top_k": 3})]),
        SimpleNamespace(content=[_text("Tôi tìm thấy cảnh màu đỏ ở đầu video.")]),
    ]
    fake = _FakeClaude(scripted)
    agent.planner = ClaudePlanner(client=fake)

    run = agent.run("tìm cảnh màu đỏ")
    assert fake.calls == 2
    assert run.tools_used() == ["search"]
    assert run.answer and "đỏ" in run.answer
    assert run.results and run.results[0]["timestamp"] < 1.0   # kết quả tool thật
    # tools + messages có được đẩy cho API
    assert any(t["name"] == "search" for t in fake.captured["tools"])
    assert run.steps[-1].action.kind == "finish"


def test_claude_planner_stops_at_max_steps(agent) -> None:
    # Claude gọi tool mãi -> dừng khi hết max_steps, vẫn giữ kết quả gần nhất.
    scripted = [SimpleNamespace(content=[_tool_use("search", {"query": "đỏ"}, id=f"t{i}")])
                for i in range(10)]
    fake = _FakeClaude(scripted)
    agent.planner = ClaudePlanner(client=fake)

    run = agent.run("đỏ", max_steps=3)
    assert fake.calls == 3
    assert run.meta["stop"] == "max_steps"
    assert run.answer is None


def test_claude_planner_requires_key_without_client(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = ClaudePlanner()  # không tiêm client
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        p._get_client()


# ------------------------------- G4 Reader tích hợp ---------------------------
def test_reader_synthesizes_answer_for_mock_planner(agent) -> None:
    from retrieval.vqa_module import MockReader
    agent.reader = MockReader()                      # MockPlanner answer=None -> Reader điền
    run = agent.run("cảnh màu đỏ")
    assert run.answer and "[" in run.answer          # câu trả lời grounded có trích dẫn
    assert run.meta.get("cited_frame_ids")           # id được trích dẫn


def test_reader_does_not_override_claude_answer(agent) -> None:
    from retrieval.vqa_module import MockReader
    scripted = [
        SimpleNamespace(content=[_tool_use("search", {"query": "cảnh màu đỏ"})]),
        SimpleNamespace(content=[_text("Câu trả lời của Claude.")]),
    ]
    agent.planner = ClaudePlanner(client=_FakeClaude(scripted))
    agent.reader = MockReader()
    run = agent.run("cảnh màu đỏ")
    assert run.answer == "Câu trả lời của Claude."   # Reader KHÔNG ghi đè
