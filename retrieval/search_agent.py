"""G2 — Search Agent: vòng lặp điều phối (observe → reason → act) trên Action Space.

BỐI CẢNH (Slide Buổi 3):
    Đây là "bộ não" đặt TRÊN G1 (ToolRegistry). Thay vì chạy một pipeline CỨNG
    (coarse → rerank cố định), Agent nhận truy vấn rồi TỰ QUYẾT chuỗi công cụ cần gọi:
      - VideoQA STAR (slide 30): Planner luân phiên gọi tool thời gian / không gian tuỳ
        trạng thái quan sát được.
      - MemoriEase 3.0 (slide 31): agent hội thoại chọn hành động (trích lọc → tìm →
        rerank → hỏi lại) dựa trên quan sát nhiều lượt.
    Chu trình chuẩn của một Agent: OBSERVE (kết quả tool) → REASON (chọn bước kế) →
    ACT (gọi tool) → lặp cho tới khi FINISH.

ĐỊNH VỊ (rất quan trọng, xem TASKS.md Mục 9):
    Agent là "smart path" cho query KHÓ / hội thoại — KHÔNG thay pipeline "fast path"
    (`engine.search`) vốn vẫn là mặc định nhanh. Ở đây Agent chỉ ĐIỀU PHỐI các tool đã
    có, không tự cài thuật toán retrieval mới.

THIẾT KẾ (ABC + Mock + Claude-lazy — CLAUDE.md Mục 1.5):
  - `Planner` (ABC): bộ ra quyết định. Nhận query + ảnh + registry + hàm `record`
    (ghi lại mỗi bước để truy vết), tự lái vòng lặp, trả `AgentResult`.
  - `MockPlanner`: LUẬT tất định, CHẠY OFFLINE không cần API key — đủ để test/đo và
    demo "hệ có bộ não" ngay bây giờ. Luật: understand (định tuyến) → chọn đúng nhánh
    tìm (temporal / ảnh / đa phương thức / chữ) → nếu là tìm-chữ và còn mơ hồ thì
    disambiguation → finish.
  - `ClaudePlanner`: tool-use loop THẬT của Anthropic (function-calling), bật khi có
    `ANTHROPIC_API_KEY`. Cho tiêm `client` để test vòng lặp offline không cần mạng.

  Vì sao tách `Planner.run()` tự lái vòng lặp (thay vì SearchAgent gọi `next_action`
  từng bước)? Vì ClaudePlanner cần GIỮ TRẠNG THÁI hội thoại (messages) xuyên các lượt
  gọi tool trong CÙNG một truy vấn — để LLM thấy được quan sát trước đó mà suy luận
  tiếp. Đưa cả vòng lặp vào planner là cách tự nhiên nhất để giữ trạng thái đó; còn
  việc ghi-truy-vết-bước dùng chung qua callback `record` nên SearchAgent vẫn gói được
  kết quả thống nhất.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from retrieval.agent_tools import ToolRegistry, ToolResult, build_registry

# ----------------------------------------------------------------------------- types


@dataclass
class AgentAction:
    """Một hành động Agent quyết định: gọi tool, hoặc kết thúc."""

    kind: str                              # "call" | "finish"
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    rationale: str = ""                    # vì sao chọn bước này (để truy vết/giải thích)


@dataclass
class AgentStep:
    """Một bước đã THỰC THI: hành động + kết quả tool (None nếu là finish)."""

    action: AgentAction
    result: Optional[ToolResult] = None


@dataclass
class AgentResult:
    """Kết quả cuối planner trả về (trước khi SearchAgent gói thành AgentRun)."""

    results: list[dict] = field(default_factory=list)
    answer: Optional[str] = None           # câu trả lời NL (để trống cho tới G4 Reader)
    meta: dict = field(default_factory=dict)


@dataclass
class AgentRun:
    """Toàn bộ một lần chạy Agent — đủ để hiển thị 'Agent đã suy nghĩ/làm gì'."""

    query: str
    steps: list[AgentStep]
    results: list[dict]
    answer: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def tools_used(self) -> list[str]:
        return [s.action.tool for s in self.steps if s.action.kind == "call"]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tools_used": self.tools_used(),
            "results": self.results,
            "answer": self.answer,
            "meta": self.meta,
            "steps": [{"action": s.action.__dict__,
                       "result": (None if s.result is None else s.result.to_dict())}
                      for s in self.steps],
        }


Recorder = Callable[[AgentAction, Optional[ToolResult]], None]


# ----------------------------------------------------------------------------- planner ABC


class Planner(ABC):
    """Bộ ra quyết định của Agent. Tự lái vòng lặp, gọi `registry.call`, `record` mỗi bước."""

    @abstractmethod
    def run(self, query: str, images: dict[str, bytes], registry: ToolRegistry,
            record: Recorder, max_steps: int) -> AgentResult:
        ...


# ----------------------------------------------------------------------------- MockPlanner


class MockPlanner(Planner):
    """Planner LUẬT — offline, tất định. Đủ để chạy/đo/demo mà không cần API key.

    Luật (một state machine nhỏ, tối đa ~4 bước):
      1. `understand`  -> định tuyến theo cấu trúc (có thứ tự thời gian? có ảnh?).
      2. Chọn ĐÚNG nhánh tìm:
           - có thứ tự thời gian  -> search_temporal(events)
           - có cả chữ + ảnh      -> search_multimodal
           - chỉ có ảnh           -> search_by_image
           - còn lại              -> search (chữ)
      3. Nếu nhánh là tìm-chữ/đa-phương-thức và có kết quả -> `disambiguation` để BIẾT
         còn mơ hồ không (đúng vòng phản hồi KISC). Không tự bịa rerank (tránh nạp VLM
         nặng offline) — rerank để ClaudePlanner/route rõ ràng quyết định.
      4. finish, trả kết quả nhánh tìm + cờ need_clarification (nếu mơ hồ).
    """

    def run(self, query: str, images: dict[str, bytes], registry: ToolRegistry,
            record: Recorder, max_steps: int = 6) -> AgentResult:
        # -- bước 1: hiểu + định tuyến ------------------------------------
        u = registry.call("understand", query=query)
        record(AgentAction("call", "understand", {"query": query},
                           "phan tich cau truy van de dinh tuyen"), u)
        events = u.meta.get("events") if u.ok else None
        has_temporal = bool(u.meta.get("has_temporal")) if u.ok else False
        img_ref = next(iter(images), None)

        # -- bước 2: chọn nhánh tìm ---------------------------------------
        if has_temporal and events:
            act = AgentAction("call", "search_temporal", {"events": events},
                              "cau co rang buoc THU TU thoi gian -> tim chuoi")
            r = registry.call("search_temporal", events=events)
        elif img_ref and query.strip():
            act = AgentAction("call", "search_multimodal",
                              {"query": query, "image_ref": img_ref},
                              "co ca CHU va ANH -> ket hop da phuong thuc")
            r = registry.call("search_multimodal", query=query, image_ref=img_ref)
        elif img_ref:
            act = AgentAction("call", "search_by_image", {"image_ref": img_ref},
                              "chi co ANH -> image-to-video")
            r = registry.call("search_by_image", image_ref=img_ref)
        else:
            act = AgentAction("call", "search", {"query": query, "rerank": False},
                              "mo ta bang CHU -> tim thuong")
            r = registry.call("search", query=query, rerank=False)
        record(act, r)

        results = r.items if r.ok else []
        meta: dict[str, Any] = {"route": act.tool}

        # -- bước 3: kiểm mơ hồ (chỉ cho nhánh tìm-chữ có candidate) -------
        if act.tool in ("search", "search_multimodal") and results:
            cands = [{"keyframe_id": it["keyframe_id"],
                      "score": it.get("score") or 0.0} for it in results]
            d = registry.call("disambiguation", candidates=cands)
            record(AgentAction("call", "disambiguation",
                               {"candidates": f"<{len(cands)} ket qua>"},
                               "con mo ho khong? co can hoi lai?"), d)
            if d.ok and not d.meta.get("confident", True):
                meta["need_clarification"] = [it["keyframe_id"] for it in d.items]

        record(AgentAction("finish", rationale="da co ket qua, ket thuc"), None)
        return AgentResult(results=results, answer=None, meta=meta)


# ----------------------------------------------------------------------------- ClaudePlanner


class ClaudePlanner(Planner):
    """Planner THẬT dùng Anthropic tool-use loop (function-calling).

    Mỗi vòng: gửi hội thoại + `tools` (registry.specs) cho Claude; nếu Claude trả
    `tool_use` -> thực thi tool, đẩy `tool_result` (JSON đã chuẩn hoá) vào hội thoại rồi
    lặp; nếu Claude trả text thuần -> đó là câu trả lời cuối (mầm cho G4 Reader). Cho
    tiêm `client` để test vòng lặp offline; mặc định tạo `anthropic.Anthropic()` (đọc
    ANTHROPIC_API_KEY từ môi trường)."""

    DEFAULT_SYSTEM = (
        "Ban la tro ly TIM KIEM VIDEO. Dung cac cong cu duoc cap de tim keyframe dung "
        "y nguoi dung. Goi 'understand' truoc de dinh tuyen; dung 'search_temporal' khi "
        "co rang buoc thu tu thoi gian; 'search_by_image'/'search_multimodal' khi co anh; "
        "'disambiguation' khi ket qua con mo ho. Khi da du tu tin, tra loi ngan gon bang "
        "tieng Viet kem cac keyframe_id lien quan."
    )

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1024,
                 system: Optional[str] = None, client: Any = None):
        self.model = model
        self.max_tokens = max_tokens
        self.system = system or self.DEFAULT_SYSTEM
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ClaudePlanner can ANTHROPIC_API_KEY. Dat bien moi truong hoac dung "
                    "MockPlanner (offline).")
            try:
                import anthropic  # lazy — chỉ cần khi chạy thật
            except ImportError as e:  # pragma: no cover
                raise ImportError("Chua cai 'anthropic'. pip install anthropic") from e
            self._client = anthropic.Anthropic()
        return self._client

    def _initial_content(self, query: str, images: dict[str, bytes]) -> str:
        img_note = (f" (co {len(images)} anh truy van, tham chieu qua image_ref: "
                    f"{list(images)})") if images else ""
        return f"Truy van: {query}{img_note}"

    def run(self, query: str, images: dict[str, bytes], registry: ToolRegistry,
            record: Recorder, max_steps: int = 6) -> AgentResult:
        client = self._get_client()
        tools = registry.specs("anthropic")
        messages: list[dict] = [
            {"role": "user", "content": self._initial_content(query, images)}]
        last_results: list[dict] = []

        for _ in range(max_steps):
            resp = client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=self.system,
                tools=tools, messages=messages)
            blocks = list(resp.content)
            tool_uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]

            if not tool_uses:  # Claude trả text -> câu trả lời cuối
                answer = "".join(getattr(b, "text", "") for b in blocks
                                 if getattr(b, "type", None) == "text").strip()
                record(AgentAction("finish", rationale="claude tra loi cuoi"), None)
                return AgentResult(results=last_results, answer=answer or None,
                                   meta={"stop": "answer"})

            # Ghi lại lượt assistant (kèm tool_use) đúng định dạng API để lặp tiếp.
            messages.append({"role": "assistant", "content": blocks})
            tool_result_blocks = []
            for tu in tool_uses:
                args = dict(tu.input)
                obs = registry.call(tu.name, **args)
                record(AgentAction("call", tu.name, args, "claude chon tool"), obs)
                if obs.ok and obs.items:
                    last_results = obs.items
                tool_result_blocks.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": json.dumps(obs.to_dict(), ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_result_blocks})

        record(AgentAction("finish", rationale="het so buoc cho phep"), None)
        return AgentResult(results=last_results, answer=None,
                           meta={"stop": "max_steps"})


# ----------------------------------------------------------------------------- SearchAgent


class SearchAgent:
    """Bộ điều phối: dựng Action Space cho (engine, entry) rồi giao Planner tự lái.

    Mặc định MockPlanner (offline). Bơm ClaudePlanner khi có API key để dùng bộ não thật."""

    def __init__(self, engine, entry, planner: Optional[Planner] = None):
        self.engine = engine
        self.entry = entry
        self.planner: Planner = planner or MockPlanner()

    def run(self, query: str, images: Optional[dict[str, bytes]] = None,
            max_steps: int = 6) -> AgentRun:
        registry = build_registry(self.engine, self.entry, images or {})
        steps: list[AgentStep] = []

        def record(action: AgentAction, result: Optional[ToolResult]) -> None:
            steps.append(AgentStep(action, result))

        res = self.planner.run(query, images or {}, registry, record, max_steps)
        return AgentRun(query=query, steps=steps, results=res.results,
                        answer=res.answer, meta=res.meta)
