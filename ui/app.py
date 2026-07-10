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

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

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
# SigLIP ĐA NGÔN NGỮ: hỗ trợ query tiếng Việt (100+ ngôn ngữ). Đổi sang bản
# "google/siglip-base-patch16-224" nếu chỉ cần tiếng Anh (nhẹ hơn chút).
VIDEO_MODEL = "google/siglip-base-patch16-256-multilingual"


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
        query = (request.json or {}).get("query", "").strip()
        if not query:
            return jsonify(error="Nhập truy vấn tìm kiếm."), 400
        cands = coarse.search(query_text=query, top_k=24)
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
    # State riêng: encoder nạp LƯỜI (chỉ khi dùng tab video -> không làm chậm khởi
    # động server); mỗi video index xong được cache để khỏi embed lại.
    video_state: dict = {"encoder": None, "videos": {}}

    def _encoder():
        if video_state["encoder"] is None:
            from ingestion.embed_siglip import SiglipEncoder
            video_state["encoder"] = SiglipEncoder(model_name=VIDEO_MODEL)
        return video_state["encoder"]

    @app.get("/api/video/list")
    def video_list():
        vids = (sorted(p.name for p in VIDEO_DIR.glob("*.mp4")) if VIDEO_DIR.exists()
                else [])
        return jsonify(videos=vids, indexed=sorted(video_state["videos"].keys()),
                       folder=str(VIDEO_DIR))

    @app.post("/api/video/index")
    def video_index():
        name = (request.json or {}).get("video", "")
        path = VIDEO_DIR / name
        if not name or not path.exists():
            return jsonify(error=f"Không thấy video: {name}"), 404
        from ingestion.video_ingest import extract_keyframes
        raws = extract_keyframes(path, FRAMES_DIR, sample_every_s=1.0, max_frames=60)
        if not raws:
            return jsonify(error="Không trích được keyframe nào."), 400
        enc = _encoder()
        records = [KeyframeRecord(id=r.id, video_id=r.video_id, timestamp=r.timestamp,
                                  clip_embedding=enc.embed(r)) for r in raws]
        index = KeyframeIndex.build(records)
        video_state["videos"][path.stem] = {"index": index,
                                             "raws": {r.id: r for r in raws}}
        return jsonify(video=path.stem, frames=len(raws))

    @app.post("/api/video/search")
    def video_search():
        data = request.json or {}
        vid, query = data.get("video", ""), (data.get("query", "") or "").strip()
        entry = video_state["videos"].get(vid)
        if entry is None:
            return jsonify(error="Video chưa được nạp. Bấm 'Nạp video' trước."), 400
        if not query:
            return jsonify(error="Nhập câu mô tả cần tìm."), 400
        qvec = _encoder().encode_text(query)
        cands = CoarseRetriever(entry["index"]).search(
            query_clip_vec=qvec, top_k=int(data.get("topk", 6)))
        results = [{"id": c.keyframe_id, "timestamp": round(c.timestamp, 1),
                    "score": round(float(c.score), 3),
                    "image": f"/api/video/frame/{c.keyframe_id}"} for c in cands]
        return jsonify(video=vid, query=query, results=results)

    @app.get("/api/video/frame/<path:frame_id>")
    def video_frame(frame_id: str):
        for entry in video_state["videos"].values():
            raw = entry["raws"].get(frame_id)
            if raw and raw.image_path and Path(raw.image_path).exists():
                return send_file(raw.image_path, mimetype="image/jpeg")
        return jsonify(error="Không thấy keyframe."), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
