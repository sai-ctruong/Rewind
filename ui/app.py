"""Web UI của Rewind — backend Flask nối thẳng pipeline THẬT trên video người dùng nạp.

  - Tìm kiếm video (chữ / ảnh / sketch / đa phương thức / chuỗi thời gian), rerank VLM,
    relevance feedback, gợi ý concept, duyệt lân cận  ->  /api/video/*
  - Bộ lọc ẢNH hội thoại (thu hẹp dần)               ->  /api/filter/*
  - Hỏi–đáp VQA (demo trên bộ keyframe mẫu)          ->  /api/vqa

MỌI thứ chạy trên index của video THẬT (video_state["videos"]) — không còn dataset
lifelog tổng hợp. Demo hội thoại theo THUỘC TÍNH (Information Gain) nằm ở
`python -m kisc_module.demo` và `retrieval/kisc_real_demo.py`, không phơi qua HTTP nữa.

Frontend (ui/index.html) là 1 trang tĩnh tự gọi các API này.

Chạy:  python -m ui.app   rồi mở http://127.0.0.1:5000
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

from ingestion import model_cache  # noqa: F401 -- đặt HF_HOME (ổ D) sớm nhất
from ingestion.schemas import KeyframeRecord
from retrieval.vqa_module import VqaModule

_UI_DIR = Path(__file__).resolve().parent
_ROOT = _UI_DIR.parent
VIDEO_DIR = _ROOT / "data" / "videos"          # nơi bỏ file video của người dùng
FRAMES_DIR = _ROOT / "artifacts" / "frames"    # nơi lưu keyframe trích ra
INDEX_DIR = _ROOT / "artifacts" / "index"      # nơi lưu index (A2 — nạp lại, không embed lại)


def build_vqa_records() -> list[KeyframeRecord]:
    def rec(kf_id, ts, cap, objs):
        return KeyframeRecord(id=kf_id, video_id="party", timestamp=ts,
                              clip_embedding=np.zeros(4, dtype=np.float32),
                              objects=objs, llm_caption=cap)
    return [
        rec("p/0", 0.0, "Mọi người quây quần quanh bàn tiệc sinh nhật.", ["người"]),
        rec("p/1", 5.0, "Một chiếc bánh sinh nhật với 5 ngọn nến đang cháy.", ["bánh", "nến"]),
        rec("p/2", 12.0, "Người đàn ông áo xanh đang tặng quà cho cô gái.", ["người", "quà"]),
        rec("p/3", 18.0, "Cô gái mở hộp quà và mỉm cười.", ["quà"]),
    ]


def create_app() -> Flask:
    app = Flask(__name__)
    vqa_records = build_vqa_records()
    vqa = VqaModule()

    @app.get("/")
    def home():
        # index.html là NỘI DUNG TRANG (không có <html>/<head>) để cũng publish được
        # làm Artifact; ở đây bọc vào skeleton đầy đủ để phục vụ như 1 trang web.
        body = (_UI_DIR / "index.html").read_text(encoding="utf-8")
        return (
            "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Trợ lý Truy xuất Đa phương tiện · AIC 2026</title></head><body>"
            f"{body}</body></html>"
        )

    @app.get("/api/health")
    def health():
        """Trạng thái backend + số liệu THẬT: bao nhiêu video đã index, tổng bao nhiêu
        keyframe. (Trước đây trả kích thước một dataset lifelog TỔNG HỢP -> con số hiện
        trên UI không liên quan gì tới video người dùng nạp.)"""
        vids = video_state["videos"]
        return jsonify(ok=True, videos=len(vids),
                       dataset_size=sum(e.num_indexed for e in vids.values()))

    @app.post("/api/vqa")
    def vqa_answer():
        question = (request.json or {}).get("question", "").strip()
        if not question:
            return jsonify(error="Nhập câu hỏi."), 400
        ans = vqa.answer(question, vqa_records)
        return jsonify(
            answer=ans.answer, value=ans.value, reasoning=ans.reasoning,
            used_frame_ids=ans.used_frame_ids,
        )

    # ===================== TÌM KIẾM TRÊN VIDEO THẬT (SigLIP) ==================
    # VideoSearchEngine (retrieval/video_engine.py): ensemble SigLIP2 + SigLIP
    # multilingual, query prompt ensemble, lấy mẫu dày + dedup ngữ nghĩa.
    # Model nạp LƯỜI (lần index đầu); mỗi video index xong được cache.
    from retrieval.video_engine import VideoIndexEntry, VideoSearchEngine

    video_state: dict = {"engine": VideoSearchEngine(), "videos": {},
                         "progress": {"active": False, "count": 0, "fps": 0.0,
                                      "elapsed": 0.0, "label": ""}}

    def _progress_cb(label: str):
        """Trả callback engine gọi khi index -> cập nhật video_state['progress'] để
        endpoint /api/video/progress đọc (A6). Đánh dấu active; caller tự tắt ở finally."""
        p = video_state["progress"]
        p.update(active=True, count=0, fps=0.0, elapsed=0.0, label=label)
        def cb(n, elapsed):
            p.update(count=n, fps=round(n / max(elapsed, 1e-6), 1),
                     elapsed=round(elapsed, 1), active=True, label=label)
        return cb

    def _progress_done():
        video_state["progress"]["active"] = False

    def _safe_name(vid: str) -> str:
        """Tên thư mục an toàn cho video_id (đề phòng ký tự lạ)."""
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in vid)

    # Tự NẠP LẠI các index đã lưu ở lần chạy trước (A2) — không phải embed lại.
    if INDEX_DIR.exists():
        for d in sorted(INDEX_DIR.iterdir()):
            if (d / "entry.pkl").exists():
                try:
                    entry = VideoIndexEntry.load(d)
                    video_state["videos"][entry.video_id] = entry
                    print(f"[ui] Đã nạp index từ đĩa: {entry.video_id} "
                          f"({entry.num_indexed} keyframe)")
                except Exception as e:  # pragma: no cover - index hỏng thì bỏ qua
                    print(f"[ui] Bỏ qua index hỏng {d}: {e!r}")

    def _index_opts(data: dict) -> dict:
        """Tuỳ chọn nạp từ request: chỉ còn bật/tắt OCR. LUÔN quét TOÀN BỘ video —
        engine dùng mặc định sample_every_s=1s và max_frames=None (không giới hạn),
        không cho người dùng chọn giây/số frame nữa (sẽ scale lên cả dataset lớn)."""
        o: dict = {}
        if "ocr" in data:
            o["enable_ocr"] = bool(data["ocr"])
        if "asr" in data:
            o["enable_asr"] = bool(data["asr"])
        if "caption" in data:
            o["enable_caption"] = bool(data["caption"])
        return o

    @app.get("/api/video/progress")
    def video_progress():
        """A6: tiến độ index hiện tại (UI poll khi đang nạp) — count + fps + elapsed."""
        return jsonify(video_state["progress"])

    @app.get("/api/video/list")
    def video_list():
        vids = (sorted(p.name for p in VIDEO_DIR.glob("*.mp4")) if VIDEO_DIR.exists()
                else [])
        return jsonify(videos=vids, indexed=sorted(video_state["videos"].keys()),
                       folder=str(VIDEO_DIR))

    @app.post("/api/video/index")
    def video_index():
        name = (request.json or {}).get("video", "")

        # --- Chế độ TOÀN BỘ DATASET: index mọi .mp4 trong data/videos/ vào 1 index.
        if name == "__dataset__":
            if "__dataset__" in video_state["videos"]:
                entry = video_state["videos"]["__dataset__"]
                return jsonify(video="__dataset__", cached=True, frames=entry.num_indexed)
            vids = sorted(VIDEO_DIR.glob("*.mp4")) if VIDEO_DIR.exists() else []
            if not vids:
                return jsonify(error="Không có video nào trong data/videos/"), 404
            try:
                entry = video_state["engine"].index_dataset(
                    vids, FRAMES_DIR, progress_cb=_progress_cb("toàn bộ dataset"),
                    **_index_opts(request.json or {}))
            except RuntimeError as e:
                return jsonify(error=str(e)), 400
            finally:
                _progress_done()
            video_state["videos"]["__dataset__"] = entry
            return jsonify(video="__dataset__", frames=entry.num_indexed,
                           sampled=entry.num_sampled, videos=len(vids))

        path = VIDEO_DIR / name
        if not name or not path.exists():
            return jsonify(error=f"Không thấy video: {name}"), 404
        # Đã index rồi -> trả cache ngay (dùng lại giữa các tab KIS/AVS/Video).
        if path.stem in video_state["videos"]:
            entry = video_state["videos"][path.stem]
            return jsonify(video=path.stem, cached=True, frames=entry.num_indexed)
        try:
            entry = video_state["engine"].index_video(
                path, FRAMES_DIR, progress_cb=_progress_cb(name),
                **_index_opts(request.json or {}))
        except RuntimeError as e:
            return jsonify(error=str(e)), 400
        finally:
            _progress_done()
        video_state["videos"][entry.video_id] = entry
        return jsonify(video=entry.video_id, frames=entry.num_indexed,
                       sampled=entry.num_sampled)

    # Đuôi file video chấp nhận khi quét thư mục dataset.
    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg"}

    @app.post("/api/video/index_folder")
    def video_index_folder():
        """Index TOÀN BỘ video trong một THƯ MỤC BẤT KỲ trên máy (đệ quy) vào 1 index
        dataset chung -> tìm xuyên suốt. Không cần chép video vào data/videos/."""
        raw_path = (request.json or {}).get("path", "").strip().strip('"')
        if not raw_path:
            return jsonify(error="Nhập đường dẫn thư mục dataset."), 400
        folder = Path(raw_path)
        if not folder.is_dir():
            return jsonify(error=f"Không phải thư mục hợp lệ: {raw_path}"), 400
        vids = sorted(p for p in folder.rglob("*")
                      if p.suffix.lower() in VIDEO_EXTS and p.is_file())
        if not vids:
            return jsonify(error=f"Không tìm thấy video trong: {raw_path}"), 404
        try:
            entry = video_state["engine"].index_dataset(
                vids, FRAMES_DIR, progress_cb=_progress_cb(f"{len(vids)} video"),
                **_index_opts(request.json or {}))
        except RuntimeError as e:
            return jsonify(error=str(e)), 400
        finally:
            _progress_done()
        video_state["videos"]["__dataset__"] = entry
        video_state["dataset_folder"] = str(folder)
        return jsonify(video="__dataset__", frames=entry.num_indexed,
                       sampled=entry.num_sampled, videos=len(vids), folder=str(folder))

    @app.post("/api/video/search")
    def video_search():
        data = request.json or {}
        vid, query = data.get("video", ""), (data.get("query", "") or "").strip()
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp. Bấm 'Nạp video' trước."), 400
        if not query:
            return jsonify(error="Nhập câu mô tả cần tìm."), 400
        rerank = bool(data.get("rerank", False))
        pos = [i for i in (data.get("positive") or []) if i]
        neg = [i for i in (data.get("negative") or []) if i]
        topk = int(data.get("topk", 6))
        eng = video_state["engine"]

        # Q4: HIỂU CÂU (không khi đang lọc theo phản hồi F2). Parse -> hiện cấu trúc +
        # TỰ ĐỊNH TUYẾN sang tìm chuỗi nếu câu có thứ tự thời gian ("A trước khi B").
        parsed = None
        if not (pos or neg) and data.get("understand", True):
            st = eng.understand(query)
            parsed = {"objects": st.objects, "actions": st.actions,
                      "location": st.location, "attributes": st.attributes,
                      "time_constraint": st.time_constraint,
                      "temporal_order": st.temporal_order, "query_type": st.query_type}
            events = eng.temporal_events(query)
            if events:  # có thứ tự thời gian -> trả CHUỖI thay vì danh sách phẳng
                matches = eng.search_temporal(entry, events, max_results=topk)
                chains = [{
                    "video_id": m.video_id, "total_score": round(float(m.total_score), 3),
                    "steps": [{"event": events[i], "timestamp": round(s.timestamp, 1),
                               "image": f"/api/video/frame/{s.keyframe_id}"}
                              for i, s in enumerate(m.steps)],
                } for m in matches]
                return jsonify(video=vid, query=query, parsed=parsed,
                               temporal=chains, events=events, results=[])

        if pos or neg:  # F2: relevance feedback (Rocchio) — dịch truy vấn theo phản hồi
            cands = eng.search_with_feedback(
                entry, query, positive_ids=pos, negative_ids=neg, top_k=topk, rerank=rerank)
        else:
            cands = eng.search(entry, query, top_k=topk, rerank=rerank)
        results = [{"id": c.keyframe_id, "timestamp": round(c.timestamp, 1),
                    "score": round(float(c.score), 3), "video_id": c.video_id,
                    "explanation": getattr(c, "explanation", None),
                    "ocr": entry.ocr_by_id.get(c.keyframe_id),
                    "asr": entry.asr_by_id.get(c.keyframe_id),
                    "caption": entry.caption_by_id.get(c.keyframe_id),
                    "image": f"/api/video/frame/{c.keyframe_id}"} for c in cands]
        # F3: gợi ý concept liên quan từ top kết quả (khám phá/khai phá).
        suggestions = eng.suggest_concepts(
            entry, [c.keyframe_id for c in cands], query)
        # F1: nếu kết quả còn mơ hồ -> chọn vài ảnh ĐA DẠNG để CHỦ ĐỘNG hỏi lại.
        disambig = []
        if not (pos or neg):
            ids = eng.disambiguation(entry, cands)
            if ids:
                by_id = {c.keyframe_id: c for c in cands}
                disambig = [{"id": i, "timestamp": round(by_id[i].timestamp, 1),
                             "video_id": by_id[i].video_id,
                             "image": f"/api/video/frame/{i}"} for i in ids if i in by_id]
        return jsonify(video=vid, query=query, reranked=rerank, parsed=parsed,
                       results=results, suggestions=suggestions, disambiguation=disambig)

    @app.post("/api/video/search_image")
    def video_search_image():
        """Q1: tìm bằng ẢNH mẫu. Nhận file ảnh (multipart 'image') + 'video' (form).
        SigLIP encode ảnh -> search thị giác thuần."""
        vid = request.form.get("video", "")
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp. Bấm 'Nạp video' trước."), 400
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="Chưa chọn ảnh mẫu."), 400
        data = f.read()
        if not data:
            return jsonify(error="Ảnh rỗng."), 400
        text = (request.form.get("query") or "").strip()
        topk = int(request.form.get("topk", 8))
        try:
            if text:  # Q3: có CẢ chữ -> multimodal (trộn vector chữ + ảnh)
                cands = video_state["engine"].search_multimodal(
                    entry, text, data, top_k=topk)
            else:      # chỉ ảnh
                cands = video_state["engine"].search_by_image(entry, data, top_k=topk)
        except Exception as e:  # pragma: no cover
            return jsonify(error=f"Lỗi tìm bằng ảnh: {e}"), 500
        results = [{"id": c.keyframe_id, "timestamp": round(c.timestamp, 1),
                    "score": round(float(c.score), 3), "video_id": c.video_id,
                    "caption": entry.caption_by_id.get(c.keyframe_id),
                    "ocr": entry.ocr_by_id.get(c.keyframe_id),
                    "image": f"/api/video/frame/{c.keyframe_id}"} for c in cands]
        return jsonify(video=vid, count=len(results), results=results)

    @app.post("/api/video/temporal")
    def video_temporal():
        """B4: tìm chuỗi 'cảnh A TRƯỚC cảnh B (…)'. `events` = danh sách câu mô tả cảnh
        theo thứ tự thời gian. Trả các tổ hợp cùng video, timestamp tăng dần."""
        data = request.json or {}
        vid = data.get("video", "")
        events = [e.strip() for e in (data.get("events") or []) if e and e.strip()]
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp. Bấm 'Nạp video' trước."), 400
        if len(events) < 2:
            return jsonify(error="Nhập ít nhất 2 cảnh (mỗi dòng 1 cảnh) theo thứ tự."), 400
        matches = video_state["engine"].search_temporal(
            entry, events, per_event_k=int(data.get("per_event_k", 20)),
            max_results=int(data.get("max_results", 20)))
        out = [{
            "video_id": m.video_id, "total_score": round(float(m.total_score), 3),
            "steps": [{"event": events[i], "timestamp": round(s.timestamp, 1),
                       "image": f"/api/video/frame/{s.keyframe_id}"}
                      for i, s in enumerate(m.steps)],
        } for m in matches]
        return jsonify(video=vid, events=events, count=len(out), matches=out)

    @app.post("/api/video/save")
    def video_save():
        """Lưu MỌI index đang có ra đĩa (A2) — lần sau mở app tự nạp lại, không embed lại."""
        saved = []
        for vid, entry in video_state["videos"].items():
            try:
                entry.save(INDEX_DIR / _safe_name(vid))
                saved.append(vid)
            except Exception as e:  # pragma: no cover
                return jsonify(error=f"Lỗi lưu {vid}: {e}"), 500
        if not saved:
            return jsonify(error="Chưa có index nào để lưu. Nạp video trước."), 400
        return jsonify(saved=saved, dir=str(INDEX_DIR))

    @app.get("/api/video/explore")
    def video_explore():
        """D2: mẫu keyframe đa dạng khắp video để 'khám phá' khi chưa biết bắt đầu."""
        vid = request.args.get("video", "")
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp."), 400
        raws = video_state["engine"].explore(
            entry, per_video=int(request.args.get("per_video", 3)),
            limit=int(request.args.get("limit", 30)))
        return jsonify(frames=[{"id": r.id, "timestamp": round(r.timestamp, 1),
                                "video_id": r.video_id,
                                "image": f"/api/video/frame/{r.id}"} for r in raws])

    @app.get("/api/video/similar/<path:frame_id>")
    def video_similar(frame_id: str):
        """D2: tìm keyframe tương tự 1 keyframe (bấm từ trang khám phá)."""
        for entry in video_state["videos"].values():
            if frame_id in entry.raws:
                cands = video_state["engine"].search_similar(
                    entry, frame_id, top_k=int(request.args.get("topk", 8)))
                return jsonify(results=[{
                    "id": c.keyframe_id, "timestamp": round(c.timestamp, 1),
                    "score": round(float(c.score), 3), "video_id": c.video_id,
                    "caption": entry.caption_by_id.get(c.keyframe_id),
                    "image": f"/api/video/frame/{c.keyframe_id}"} for c in cands])
        return jsonify(error="Không thấy keyframe."), 404

    @app.get("/api/video/neighbors/<path:frame_id>")
    def video_neighbors(frame_id: str):
        """D1: keyframe lân cận (cùng video, quanh thời điểm) để duyệt bối cảnh."""
        for entry in video_state["videos"].values():
            if frame_id in entry.raws:
                raws = video_state["engine"].neighbors(entry, frame_id)
                return jsonify(frames=[{
                    "id": r.id, "timestamp": round(r.timestamp, 1), "video_id": r.video_id,
                    "is_ref": r.id == frame_id, "image": f"/api/video/frame/{r.id}",
                } for r in raws])
        return jsonify(error="Không thấy keyframe."), 404

    # ============ BỘ LỌC ẢNH HỘI THOẠI (thu hẹp dần trên video THẬT) ============
    # Thay cho KISC-trên-dữ-liệu-tổng-hợp: mỗi lượt trả về LƯỚI ẢNH THẬT co lại dần.
    # Thu hẹp bằng 3 tín hiệu dùng đồng thời: thêm mô tả · 👍/👎 (Rocchio) · chọn ảnh
    # đại diện khi hệ hỏi lại. Logic ở retrieval/image_filter.py (test offline được).
    from retrieval.image_filter import ImageFilterSession

    filter_state: dict = {"session": None}

    def _filter_json(res) -> dict:
        """Gắn URL ảnh thật vào từng ứng viên để UI vẽ lưới ảnh."""
        d = res.to_dict()
        for it in d["results"]:
            it["image"] = f"/api/video/frame/{it['id']}"
        for it in d["disambiguation"]:
            it["image"] = f"/api/video/frame/{it['id']}"
        return d

    @app.post("/api/filter/start")
    def filter_start():
        data = request.json or {}
        vid = data.get("video", "")
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp. Bấm 'Nạp video' trước."), 400
        query = (data.get("query", "") or "").strip()
        if not query:
            return jsonify(error="Nhập mô tả để bắt đầu lọc."), 400
        sess = ImageFilterSession(video_state["engine"], entry,
                                  start_k=int(data.get("k", 20)))
        try:
            res = sess.start(query)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        filter_state["session"] = sess
        return jsonify(_filter_json(res))

    @app.post("/api/filter/refine")
    def filter_refine():
        sess = filter_state["session"]
        if sess is None:
            return jsonify(error="Chưa có phiên lọc. Hãy mô tả để bắt đầu."), 400
        data = request.json or {}
        try:
            res = sess.refine(
                text=(data.get("text") or "").strip() or None,
                positive=[i for i in (data.get("positive") or []) if i],
                negative=[i for i in (data.get("negative") or []) if i],
                pick=data.get("pick") or None,
                others=[i for i in (data.get("others") or []) if i],
            )
        except RuntimeError as e:
            return jsonify(error=str(e)), 400
        return jsonify(_filter_json(res))

    @app.post("/api/filter/reset")
    def filter_reset():
        if filter_state["session"] is not None:
            filter_state["session"].reset()
        filter_state["session"] = None
        return jsonify(ok=True)

    # ==================== AGENT (smart path — tự điều phối công cụ) ==============
    # Đưa lớp Agentic (G1–G4) lên UI: Search Agent tự quyết chuỗi tool (understand ->
    # search / search_temporal / search_with_feedback -> disambiguation), giữ trí nhớ
    # xuyên lượt (SessionMemory), và Reader tổng hợp đáp án có trích dẫn keyframe.
    # Không cần API key: mặc định MockPlanner + MockReader (tất định, offline).
    from retrieval.search_agent import SearchAgent
    from retrieval.vqa_module import MockReader

    agent_state: dict = {"agent": None, "video": None}
    AGENT_MAX_CHAINS = 8      # trần số chuỗi temporal vẽ ra UI (tổng thật vẫn báo)

    def _get_agent(entry, vid: str) -> SearchAgent:
        """Một Agent cho mỗi video (đổi video -> phiên mới, trí nhớ mới).

        Có ANTHROPIC_API_KEY -> dùng bộ não thật (ClaudePlanner + ClaudeReader);
        không có -> MockPlanner/MockReader vẫn tự định tuyến, chạy offline."""
        if agent_state["agent"] is not None and agent_state["video"] == vid:
            return agent_state["agent"]
        planner = None
        reader = MockReader()
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from retrieval.search_agent import ClaudePlanner
                from retrieval.vqa_module import ClaudeReader
                planner, reader = ClaudePlanner(), ClaudeReader()
            except Exception as e:  # pragma: no cover - thiếu SDK -> lùi về mock
                print(f"[ui] Không dùng được Claude ({e!r}); dùng MockPlanner.")
                planner, reader = None, MockReader()
        agent_state["agent"] = SearchAgent(video_state["engine"], entry,
                                           planner=planner, reader=reader)
        agent_state["video"] = vid
        return agent_state["agent"]

    def _agent_json(run) -> dict:
        """Gói AgentRun cho UI: trace các bước + kết quả (ảnh) hoặc chuỗi thời gian."""
        steps = [{
            "kind": s.action.kind, "tool": s.action.tool,
            "rationale": s.action.rationale,
            "ok": (None if s.result is None else s.result.ok),
            "n": (None if s.result is None else len(s.result.items)),
        } for s in run.steps]
        results, chains = [], []
        for it in run.results:
            if it.get("keyframe_id"):          # nhánh tìm keyframe phẳng
                results.append({**it, "id": it["keyframe_id"],
                                "image": f"/api/video/frame/{it['keyframe_id']}"})
            elif it.get("steps"):              # nhánh temporal: chuỗi cảnh đúng thứ tự
                chains.append({**it, "steps": [
                    {**s, "image": f"/api/video/frame/{s['keyframe_id']}"}
                    for s in it["steps"]]})
        meta = dict(run.meta or {})
        clar = [{"id": i, "image": f"/api/video/frame/{i}"}
                for i in (meta.get("need_clarification") or [])]
        # Temporal có thể ra hàng chục tổ hợp -> chỉ vẽ vài chuỗi đầu cho đỡ nghẽn UI,
        # nhưng VẪN báo tổng số thật (chains_total) thay vì lặng lẽ cắt bớt.
        return {"query": run.query, "answer": run.answer,
                "tools_used": run.tools_used(), "steps": steps,
                "route": meta.get("route"), "results": results,
                "chains": chains[:AGENT_MAX_CHAINS], "chains_total": len(chains),
                "clarify": clar, "cited": meta.get("cited_frame_ids") or [],
                "memory": meta.get("memory") or {}}

    @app.post("/api/agent/ask")
    def agent_ask():
        data = request.json or {}
        vid = data.get("video", "")
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp. Bấm 'Nạp video' trước."), 400
        agent = _get_agent(entry, vid)
        query = (data.get("query") or "").strip()
        if not query:   # lượt chỉ-phản-hồi -> dùng lại câu hỏi gần nhất trong trí nhớ
            recent = agent.memory.recent(1) if agent.memory else []
            query = recent[0].query if recent else ""
        if not query:
            return jsonify(error="Nhập câu để Agent tìm."), 400
        run = agent.chat(
            query,
            positive_ids=[i for i in (data.get("positive") or []) if i],
            negative_ids=[i for i in (data.get("negative") or []) if i],
        )
        return jsonify(_agent_json(run))

    @app.post("/api/agent/reset")
    def agent_reset():
        agent_state["agent"] = None
        agent_state["video"] = None
        return jsonify(ok=True)

    @app.get("/api/video/frame/<path:frame_id>")
    def video_frame(frame_id: str):
        from ingestion.schemas import load_cv2_image
        for entry in video_state["videos"].values():
            raw = entry.raws.get(frame_id)
            if not raw:
                continue
            # Ảnh giữ trong RAM (mặc định — không ghi đĩa): phục vụ thẳng từ bytes.
            if raw.image_bytes is not None:
                return send_file(BytesIO(raw.image_bytes), mimetype="image/jpeg")
            # Index nạp từ đĩa (không còn bytes) hoặc có file: dựng lại ảnh rồi encode.
            try:
                img = load_cv2_image(raw)      # đĩa hoặc decode lại từ video gốc
            except ValueError:
                img = None
            if img is not None:
                import cv2
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")
        return jsonify(error="Không thấy keyframe."), 404

    return app


app = create_app()

if __name__ == "__main__":
    # threaded=True: cho phép poll /api/video/progress SONG SONG khi 1 request index
    # đang chạy (blocking) -> progress bar mới cập nhật được (A6).
    app.run(debug=False, port=5000, threaded=True)
