"""Web UI cho hệ thống AIC 2026 (CLAUDE.md Phase 10).

Backend Flask nối thẳng pipeline THẬT:
  - Hội thoại KISC  -> KISCDialogueManager + RealKISCRetriever (Phase 8)
  - Tìm kiếm KIS/AVS -> CoarseRetriever (BM25) trên KeyframeIndex (Phase 3)
  - VQA             -> VqaModule (Phase 7)

Frontend (ui/index.html) là 1 trang tĩnh tự gọi các API này. Cùng file HTML đó cũng
chạy được ở chế độ ĐỘC LẬP (Artifact) nhờ một bản mô phỏng JS khi không có backend.

Chạy:  python -m ui.app   rồi mở http://127.0.0.1:5000
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

from ingestion import model_cache  # noqa: F401 -- đặt HF_HOME (ổ D) sớm nhất
from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord
from kisc_module.dialogue_manager import KISCDialogueManager
from kisc_module.schemas import Keyframe
from retrieval.coarse_retriever import CoarseRetriever
from retrieval.kisc_adapter import (
    ACTIVITIES,
    CLOTHING_COLORS,
    GENDERS,
    LOCATION_DESCS,
    LOCATION_TYPES,
    TARGET_ATTRS,
    TIME_PERIODS,
    RealKISCRetriever,
    build_lifelog_record,
    tags_to_attributes,
)
from retrieval.vqa_module import VqaModule

_UI_DIR = Path(__file__).resolve().parent
_ROOT = _UI_DIR.parent
VIDEO_DIR = _ROOT / "data" / "videos"          # nơi bỏ file video của người dùng
FRAMES_DIR = _ROOT / "artifacts" / "frames"    # nơi lưu keyframe trích ra
INDEX_DIR = _ROOT / "artifacts" / "index"      # nơi lưu index (A2 — nạp lại, không embed lại)


def build_dataset(n: int = 200, seed: int = 42) -> list[KeyframeRecord]:
    """Dữ liệu mẫu (đóng vai 'dữ liệu thật') + ép ground-truth vào record 0."""
    rng = np.random.default_rng(seed)
    records: list[KeyframeRecord] = []
    for i in range(n):
        attrs = {
            "location_type": str(rng.choice(LOCATION_TYPES)),
            "location_desc": str(rng.choice(LOCATION_DESCS)),
            "gender": str(rng.choice(GENDERS)),
            "clothing_color": str(rng.choice(CLOTHING_COLORS)),
            "activity": str(rng.choice(ACTIVITIES)),
            "time_period": str(rng.choice(TIME_PERIODS)),
        }
        records.append(
            build_lifelog_record(f"kf_{i:04d}", f"video_{i // 5:04d}",
                                 float(rng.integers(0, 3600)), attrs)
        )
    records[0] = build_lifelog_record(records[0].id, records[0].video_id,
                                      records[0].timestamp, dict(TARGET_ATTRS))
    return records


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
    records = build_dataset()
    index = KeyframeIndex.build(records)
    records_by_id = {r.id: r for r in records}
    coarse = CoarseRetriever(index)
    kisc_retriever = RealKISCRetriever(index)
    vqa_records = build_vqa_records()
    vqa = VqaModule()
    state = {"manager": None}  # 1 phiên hội thoại KISC cho demo cục bộ

    def kf_json(kf: Keyframe) -> dict:
        return {
            "id": kf.id, "video_id": kf.video_id, "timestamp": round(kf.timestamp, 1),
            "score": round(float(kf.score), 3), "attributes": kf.attributes,
        }

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
        return jsonify(ok=True, dataset_size=len(records))

    @app.post("/api/kisc/start")
    def kisc_start():
        query = (request.json or {}).get("query", "").strip()
        if not query:
            return jsonify(error="Vui lòng nhập mô tả ban đầu."), 400
        mgr = KISCDialogueManager(kisc_retriever, max_turns=5, max_candidates_to_stop=5)
        response = mgr.start(query)
        state["manager"] = mgr
        return jsonify(_dialogue_state(mgr, response))

    @app.post("/api/kisc/respond")
    def kisc_respond():
        mgr = state["manager"]
        if mgr is None:
            return jsonify(error="Chưa có phiên hội thoại. Hãy bắt đầu trước."), 400
        answer = (request.json or {}).get("answer", "").strip()
        response = mgr.respond(answer)
        return jsonify(_dialogue_state(mgr, response))

    def _dialogue_state(mgr, response: str) -> dict:
        top = sorted(mgr.state.candidates, key=lambda c: c.score, reverse=True)[:5]
        return {
            "response": response,
            "count": len(mgr.state.candidates),
            "finished": mgr.state.finished,
            "filters": mgr.state.filters,
            "top": [kf_json(c) for c in top],
        }

    @app.post("/api/search")
    def search():
        data = request.json or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify(error="Nhập truy vấn tìm kiếm."), 400
        # KIS lấy Top-5 (đúng 1 khoảnh khắc); AVS lấy nhiều (tất cả đoạn khớp).
        top_k = max(1, min(60, int(data.get("top_k", 24))))
        cands = coarse.search(query_text=query, top_k=top_k)
        results = []
        for c in cands:
            rec = records_by_id.get(c.keyframe_id)
            results.append({
                "id": c.keyframe_id, "video_id": c.video_id,
                "timestamp": round(c.timestamp, 1), "score": round(c.score, 4),
                "attributes": tags_to_attributes(index.objects[c.row]),
                "caption": rec.llm_caption if rec else "",
            })
        return jsonify(query=query, count=len(results), results=results)

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

    video_state: dict = {"engine": VideoSearchEngine(), "videos": {}}

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
        return o

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
                    vids, FRAMES_DIR, **_index_opts(request.json or {}))
            except RuntimeError as e:
                return jsonify(error=str(e)), 400
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
                path, FRAMES_DIR, **_index_opts(request.json or {}))
        except RuntimeError as e:
            return jsonify(error=str(e)), 400
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
                vids, FRAMES_DIR, **_index_opts(request.json or {}))
        except RuntimeError as e:
            return jsonify(error=str(e)), 400
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
        cands = video_state["engine"].search(
            entry, query, top_k=int(data.get("topk", 6)), rerank=rerank)
        results = [{"id": c.keyframe_id, "timestamp": round(c.timestamp, 1),
                    "score": round(float(c.score), 3), "video_id": c.video_id,
                    "explanation": getattr(c, "explanation", None),
                    "ocr": entry.ocr_by_id.get(c.keyframe_id),
                    "image": f"/api/video/frame/{c.keyframe_id}"} for c in cands]
        return jsonify(video=vid, query=query, reranked=rerank, results=results)

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
    app.run(debug=False, port=5000)
