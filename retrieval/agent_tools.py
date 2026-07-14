"""G1 — Tool Registry: hình thức hoá "Action Space" của Agent (Slide Buổi 3).

BỐI CẢNH (Buổi 3 — "Kiến trúc Agentic AI & LLM trong tìm kiếm"):
    Một AI Agent = bộ não LLM (reasoning) + một ACTION SPACE (tập công cụ gọi được).
    Trong VideoQA STAR (slide 30) Planner luân phiên gọi Tool *Thời gian* (cắt đoạn,
    chọn keyframe) và Tool *Không gian* (phát hiện vật thể, OCR, phóng to); trong
    MemoriEase 3.0 (slide 31) agent trích bộ lọc metadata → tính lại vector (Rocchio)
    → Rerank → Reader. Điểm CHUNG: mỗi năng lực phải là một CÔNG CỤ KHAI BÁO ĐƯỢC để
    LLM biết "có gì để gọi" và "gọi với tham số nào".

VÌ SAO CÓ FILE NÀY (G1, nền cho G2 Search Agent):
    Project đã có đủ mọi năng lực rời trong `VideoSearchEngine` (search, temporal, ảnh,
    đa phương thức, understand, neighbors, similar, gợi ý concept, disambiguation) nhưng
    chúng chỉ là method Python — LLM KHÔNG "nhìn thấy" được. File này bọc chúng thành các
    `Tool` có:
      - name + description (tiếng Việt, nói RÕ KHI NÀO nên dùng) → LLM chọn đúng tool;
      - parameters dạng JSON Schema → LLM sinh đúng tham số (function-calling);
      - hàm `fn` đã BIND sẵn engine + entry → LLM chỉ cần cấp tham số ngữ nghĩa
        (query, top_k…), KHÔNG phải cầm các object Python (VideoIndexEntry, bytes ảnh).

QUYẾT ĐỊNH THIẾT KẾ:
  1. BIND engine+entry vào registry (không để LLM truyền). LLM không thể (và không nên)
     serialize một index vào JSON — nó chỉ điều phối bằng tham số ngữ nghĩa.
  2. Ảnh truy vấn tham chiếu qua `image_ref` (khoá vào dict `images` do registry giữ),
     KHÔNG nhét bytes vào schema — bytes không thuộc "action space" của LLM.
  3. Mọi tool trả `ToolResult` ĐÃ CHUẨN HOÁ (list[dict] JSON-friendly) để G2/Reader
     tiêu thụ thống nhất, và `call()` NUỐT lỗi thành `ToolResult(ok=False, error=…)` —
     để Agent có thể "self-reflect" trên lỗi tool (Buổi 3) thay vì sập cả vòng lặp.
  4. `specs(fmt="anthropic"|"openai")` xuất đúng định dạng function-calling của 2 SDK
     phổ biến → nối thẳng vào ClaudePlanner (G2) khi có `ANTHROPIC_API_KEY`, đồng thời
     MockPlanner offline vẫn dùng chung registry này (không cần key) — đúng pattern
     ABC+Mock+Claude-lazy của CLAUDE.md Mục 1.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

# ----------------------------------------------------------------------------- types


@dataclass
class ToolResult:
    """Kết quả CHUẨN HOÁ của một lần gọi tool — JSON-friendly cho LLM/Reader.

    `items` là danh sách bản ghi đồng nhất (mỗi tool tự quyết field, nhưng luôn là dict
    thuần); `meta` chứa thông tin phụ (vd query đã parse, còn-mơ-hồ hay không). Khi lỗi:
    ok=False + error, items rỗng — Agent đọc để tự điều chỉnh (self-reflect)."""

    ok: bool
    tool: str
    items: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "tool": self.tool, "items": self.items,
                "meta": self.meta, "error": self.error}


@dataclass
class Tool:
    """Một công cụ trong Action Space: đủ metadata để LLM chọn + gọi qua function-calling."""

    name: str
    description: str
    parameters: dict                       # JSON Schema: {"type":"object","properties":…,"required":…}
    fn: Callable[..., ToolResult]          # đã bind engine+entry; nhận **kwargs ngữ nghĩa

    def to_anthropic(self) -> dict:
        """Định dạng tool của Anthropic Messages API (input_schema)."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.parameters}

    def to_openai(self) -> dict:
        """Định dạng function-calling của OpenAI (function.parameters)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


# --------------------------------------------------------------------- normalizers


def _obj(x: Any, *names: str) -> Any:
    for n in names:
        if isinstance(x, dict) and n in x:
            return x[n]
        if hasattr(x, n):
            return getattr(x, n)
    return None


def norm_candidates(cands: Sequence[Any]) -> list[dict]:
    """Candidate (coarse/rerank) -> [{keyframe_id, video_id, timestamp, score}]."""
    out: list[dict] = []
    for c in cands:
        out.append({
            "keyframe_id": _obj(c, "keyframe_id"),
            "video_id": _obj(c, "video_id"),
            "timestamp": _obj(c, "timestamp"),
            "score": (None if _obj(c, "score") is None else float(_obj(c, "score"))),
        })
    return out


def norm_raws(raws: Sequence[Any]) -> list[dict]:
    """RawKeyframe -> [{keyframe_id, video_id, timestamp}] (frame lân cận / explore)."""
    return [{"keyframe_id": _obj(r, "id"), "video_id": _obj(r, "video_id"),
             "timestamp": _obj(r, "timestamp")} for r in raws]


def norm_temporal(matches: Sequence[Any]) -> list[dict]:
    """TemporalMatch -> [{video_id, total_score, steps:[{event,keyframe_id,timestamp}]}]."""
    out: list[dict] = []
    for m in matches:
        out.append({
            "video_id": _obj(m, "video_id"),
            "total_score": float(_obj(m, "total_score") or 0.0),
            "steps": [{"event": s.event, "keyframe_id": s.keyframe_id,
                       "timestamp": s.timestamp} for s in _obj(m, "steps")],
        })
    return out


# ---------------------------------------------------------------------- registry


class ToolRegistry:
    """Tập Tool đã BIND vào (engine, entry) — Action Space cụ thể cho MỘT index.

    Dùng:
        reg = ToolRegistry(engine, entry, images={"q": img_bytes})
        reg.names()                      # ['search', 'search_temporal', ...]
        reg.specs("anthropic")           # đưa vào tools=[...] của Claude
        res = reg.call("search", query="mèo trên ghế", top_k=5)
        res.items                        # [{keyframe_id, video_id, timestamp, score}, ...]
    """

    def __init__(self, engine, entry, images: Optional[dict[str, bytes]] = None):
        self.engine = engine
        self.entry = entry
        self.images: dict[str, bytes] = dict(images or {})
        self._tools: dict[str, Tool] = {}
        self._register_all()

    # -- quản lý ------------------------------------------------------------
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool khong ton tai: {name!r}. Co: {self.names()}")
        return self._tools[name]

    def specs(self, fmt: str = "anthropic") -> list[dict]:
        """Xuất tất cả tool theo định dạng function-calling cho LLM."""
        if fmt == "anthropic":
            return [t.to_anthropic() for t in self._tools.values()]
        if fmt == "openai":
            return [t.to_openai() for t in self._tools.values()]
        raise ValueError(f"fmt khong ho tro: {fmt!r} (dung 'anthropic'|'openai').")

    def call(self, name: str, **kwargs) -> ToolResult:
        """Dispatch + NUỐT lỗi thành ToolResult(ok=False) để Agent tự phản tư, không sập.

        Đây là ranh giới an toàn của vòng lặp Agent (G2): một tool gọi sai tham số / VLM
        lỗi / id không tồn tại đều biến thành phản hồi có cấu trúc, LLM đọc `error` và thử
        lại thay vì ném exception phá cả phiên."""
        try:
            tool = self.get(name)
        except KeyError as e:
            return ToolResult(ok=False, tool=name, error=str(e))
        try:
            return tool.fn(**kwargs)
        except Exception as e:  # pragma: no cover - đường lỗi, test bằng tham số sai
            return ToolResult(ok=False, tool=name, error=f"{type(e).__name__}: {e}")

    def _resolve_image(self, image_ref: str) -> bytes:
        if image_ref not in self.images:
            raise KeyError(
                f"image_ref {image_ref!r} chua duoc cap. Co: {list(self.images)}")
        return self.images[image_ref]

    # -- đăng ký toàn bộ Action Space --------------------------------------
    def _register_all(self) -> None:
        eng, entry = self.engine, self.entry

        # 1) search — công cụ chủ lực: câu chữ -> keyframe (2 tầng, rerank tuỳ chọn)
        def _search(query: str, top_k: int = 8, rerank: bool = False) -> ToolResult:
            cands = eng.search(entry, query, top_k=top_k, rerank=rerank)
            return ToolResult(ok=True, tool="search",
                              items=norm_candidates(cands),
                              meta={"query": query, "rerank": rerank})

        self.register(Tool(
            name="search",
            description=(
                "Tim keyframe theo MO TA BANG CHU (tieng Viet). Cong cu mac dinh cho hau "
                "het truy van. Dat rerank=true khi can DO CHINH XAC TOP-1 cao (bai KIS) — "
                "cham hon nhung hieu to hop tu/ngu canh tot hon."),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Cau mo ta canh can tim."},
                    "top_k": {"type": "integer", "description": "So ket qua tra ve.",
                              "default": 8},
                    "rerank": {"type": "boolean",
                               "description": "Bat VLM rerank (chinh xac hon, cham hon).",
                               "default": False},
                },
                "required": ["query"],
            },
            fn=_search,
        ))

        # 2) search_temporal — chuỗi sự kiện ĐÚNG THỨ TỰ ("A trước B")
        def _temporal(events: Sequence[str], per_event_k: int = 20,
                      max_results: int = 50) -> ToolResult:
            matches = eng.search_temporal(entry, list(events),
                                          per_event_k=per_event_k,
                                          max_results=max_results)
            return ToolResult(ok=True, tool="search_temporal",
                              items=norm_temporal(matches),
                              meta={"events": list(events)})

        self.register(Tool(
            name="search_temporal",
            description=(
                "Tim CHUOI su kien dung THU TU thoi gian trong cung video (vd 'coi mu "
                "TRUOC KHI vao phong'). Dung khi truy van co rang buoc thu tu A->B->C. "
                "events la danh sach cau mo ta theo dung thu tu mong muon (>=2)."),
            parameters={
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "string"},
                               "description": "Cac canh theo dung thu tu thoi gian."},
                    "per_event_k": {"type": "integer", "default": 20},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": ["events"],
            },
            fn=_temporal,
        ))

        # 3) search_by_image — image-to-video (thị giác thuần)
        def _by_image(image_ref: str, top_k: int = 8) -> ToolResult:
            img = self._resolve_image(image_ref)
            cands = eng.search_by_image(entry, img, top_k=top_k)
            return ToolResult(ok=True, tool="search_by_image",
                              items=norm_candidates(cands),
                              meta={"image_ref": image_ref})

        self.register(Tool(
            name="search_by_image",
            description=(
                "Tim keyframe GIONG mot ANH MAU (image-to-video), chi dung tin hieu thi "
                "giac. Dung khi nguoi dung dua ANH thay vi mo ta bang chu. image_ref la "
                "khoa tro toi anh da duoc cap cho phien."),
            parameters={
                "type": "object",
                "properties": {
                    "image_ref": {"type": "string",
                                  "description": "Khoa anh truy van (vd 'query')."},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["image_ref"],
            },
            fn=_by_image,
        ))

        # 4) search_multimodal — CHỮ + ẢNH trộn ở mức vector
        def _multimodal(query: str, image_ref: str, text_weight: float = 0.5,
                        top_k: int = 8, rerank: bool = False) -> ToolResult:
            img = self._resolve_image(image_ref)
            cands = eng.search_multimodal(entry, query, img, text_weight=text_weight,
                                          top_k=top_k, rerank=rerank)
            return ToolResult(ok=True, tool="search_multimodal",
                              items=norm_candidates(cands),
                              meta={"query": query, "image_ref": image_ref,
                                    "text_weight": text_weight})

        self.register(Tool(
            name="search_multimodal",
            description=(
                "Ket hop CAU CHU + ANH MAU cung luc (vd 'giong anh nay nhung mau do'). "
                "text_weight trong [0,1]: cao => nghieng ve chu, thap => nghieng ve anh."),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "image_ref": {"type": "string"},
                    "text_weight": {"type": "number", "default": 0.5},
                    "top_k": {"type": "integer", "default": 8},
                    "rerank": {"type": "boolean", "default": False},
                },
                "required": ["query", "image_ref"],
            },
            fn=_multimodal,
        ))

        # 5) understand — parse câu -> StructuredQuery (định tuyến)
        def _understand(query: str) -> ToolResult:
            st = eng.understand(query)
            events = eng.temporal_events(query)
            parsed = {
                "objects": list(getattr(st, "objects", []) or []),
                "actions": list(getattr(st, "actions", []) or []),
                "location": getattr(st, "location", None),
                "attributes": dict(getattr(st, "attributes", {}) or {}),
                "time_constraint": getattr(st, "time_constraint", None),
                "temporal_order": getattr(st, "temporal_order", None),
                "query_type": getattr(st, "query_type", None),
            }
            return ToolResult(ok=True, tool="understand", items=[parsed],
                              meta={"has_temporal": events is not None,
                                    "events": events})

        self.register(Tool(
            name="understand",
            description=(
                "Phan tich cau truy van thanh cau truc (objects/actions/location/thoi "
                "gian/thu tu). Goi TRUOC de DINH TUYEN: neu meta.has_temporal=true thi "
                "nen dung search_temporal voi meta.events; neu khong thi search thuong."),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=_understand,
        ))

        # 6) neighbors — frame lân cận cùng video (Video Browser)
        def _neighbors(frame_id: str, before: int = 4, after: int = 4) -> ToolResult:
            raws = eng.neighbors(entry, frame_id, before=before, after=after)
            return ToolResult(ok=True, tool="neighbors", items=norm_raws(raws),
                              meta={"frame_id": frame_id})

        self.register(Tool(
            name="neighbors",
            description=(
                "Lay cac keyframe LAN CAN (truoc/sau) cung video quanh mot frame — de "
                "dinh vi boi canh timeline tu mot ket qua. frame_id la keyframe_id goc."),
            parameters={
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string"},
                    "before": {"type": "integer", "default": 4},
                    "after": {"type": "integer", "default": 4},
                },
                "required": ["frame_id"],
            },
            fn=_neighbors,
        ))

        # 7) search_similar — "giống ảnh kết quả này" (dùng embedding đã lưu)
        def _similar(keyframe_id: str, top_k: int = 8) -> ToolResult:
            cands = eng.search_similar(entry, keyframe_id, top_k=top_k)
            return ToolResult(ok=True, tool="search_similar",
                              items=norm_candidates(cands),
                              meta={"keyframe_id": keyframe_id})

        self.register(Tool(
            name="search_similar",
            description=(
                "Tim keyframe GIONG mot keyframe da co trong ket qua (bam 1 anh -> ra "
                "cac canh tuong tu), dung thang embedding da luu, khong encode lai."),
            parameters={
                "type": "object",
                "properties": {
                    "keyframe_id": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["keyframe_id"],
            },
            fn=_similar,
        ))

        # 8) suggest_concepts — gợi ý từ khoá thu hẹp/mở rộng (khám phá/khai phá)
        def _suggest(candidate_ids: Sequence[str], query: str,
                     top_n: int = 8) -> ToolResult:
            words = eng.suggest_concepts(entry, list(candidate_ids), query, top_n=top_n)
            return ToolResult(ok=True, tool="suggest_concepts",
                              items=[{"concept": w} for w in words],
                              meta={"query": query})

        self.register(Tool(
            name="suggest_concepts",
            description=(
                "Tu top-K ket qua hien tai, goi y CAC CONCEPT (tu khoa) hay xuat hien de "
                "nguoi dung THEM VAO truy van nham thu hep/mo rong huong tim. candidate_ids "
                "la danh sach keyframe_id cua ket qua vua roi."),
            parameters={
                "type": "object",
                "properties": {
                    "candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "query": {"type": "string"},
                    "top_n": {"type": "integer", "default": 8},
                },
                "required": ["candidate_ids", "query"],
            },
            fn=_suggest,
        ))

        # 9) disambiguation — khi còn mơ hồ, chọn K ứng viên đa dạng để hỏi lại (KISC)
        def _disambig(candidates: Sequence[dict], k: int = 4,
                      score_gap: float = 0.02) -> ToolResult:
            objs = [SimpleNamespace(keyframe_id=c["keyframe_id"],
                                    score=float(c.get("score") or 0.0))
                    for c in candidates]
            picks = eng.disambiguation(entry, objs, k=k, score_gap=score_gap)
            return ToolResult(ok=True, tool="disambiguation",
                              items=([] if picks is None
                                     else [{"keyframe_id": i} for i in picks]),
                              meta={"confident": picks is None})

        self.register(Tool(
            name="disambiguation",
            description=(
                "Khi ket qua CON MO HO (nhieu ung vien, top-1 khong noi troi), chon K "
                "keyframe DA DANG de hoi nguoi dung 'cai nao dung y ban?'. Tra items rong "
                "va meta.confident=true neu da du tu tin (khong can hoi lai). candidates "
                "la list {keyframe_id, score} cua ket qua vua roi."),
            parameters={
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "keyframe_id": {"type": "string"},
                                "score": {"type": "number"},
                            },
                            "required": ["keyframe_id"],
                        },
                    },
                    "k": {"type": "integer", "default": 4},
                    "score_gap": {"type": "number", "default": 0.02},
                },
                "required": ["candidates"],
            },
            fn=_disambig,
        ))


def build_registry(engine, entry, images: Optional[dict[str, bytes]] = None) -> ToolRegistry:
    """Tạo Action Space (ToolRegistry) cho một (engine, entry). Điểm vào cho G2."""
    return ToolRegistry(engine, entry, images=images)
