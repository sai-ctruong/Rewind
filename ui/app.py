"""Flask backend for the AIC 2026 competition-oriented Rewind UI."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Thread

from flask import Flask, jsonify, request, send_file

from ingestion import model_cache  # noqa: F401 -- configure model cache early

from aic2026.config import AppConfig, config_hash, load_app_config
from aic2026.dataset import AICDataPaths, official_frame_id
from aic2026.engine import AICCompetitionEngine, MAX_PREDICTIONS
from aic2026.metrics import write_submission
from evaluation.official_eval import evaluate_labels, load_jsonl

_UI_DIR = Path(__file__).resolve().parent
_ROOT = _UI_DIR.parent
DATA_ROOT = _ROOT / "data"
AIC_CACHE_DIR = _ROOT / "artifacts" / "aic2026_index"
SUBMISSION_DIR = _ROOT / "artifacts" / "submissions"


def create_app(config_path: str | Path | None = None, app_config: AppConfig | None = None) -> Flask:
    if app_config is None:
        if config_path is None:
            config_path = _ROOT / "configs" / "settings.yaml"
            app_config = load_app_config(config_path, {"dataset": {"root": str(DATA_ROOT), "cache_dir": str(AIC_CACHE_DIR)}})
        else:
            app_config = load_app_config(config_path)
    config_path_text = str(config_path or "<in-memory>")
    app = Flask(__name__)

    state: dict = {
        "engine": None,
        "load": None,
        "config": app_config,
        "config_path": config_path_text,
        "config_hash": config_hash(app_config),
        "progress": {"active": False, "count": 0, "fps": 0.0, "elapsed": 0.0, "label": ""},
        "evaluation": {"active": False, "summary": None, "run_dir": None, "error": None},
    }

    def _config_summary() -> dict:
        cfg: AppConfig = state["config"]
        return {
            "path": state["config_path"],
            "hash": state["config_hash"],
            "production_mode": cfg.runtime.production_mode,
            "device": cfg.runtime.device,
            "encoder_type": cfg.encoder.type,
            "feature_dim": cfg.encoder.feature_dim,
            "fusion_method": cfg.fusion.method,
            "final_top_k": cfg.ranking.final_top_k,
            "refinement_enabled": cfg.refinement.enabled,
            "trake_alignment_method": cfg.trake.alignment_method,
            "qa_top_video_hypotheses": cfg.qa.top_video_hypotheses,
        }
    def _engine_or_error():
        engine = state.get("engine")
        if engine is None:
            return None, (jsonify(error="AIC index is not loaded. Click Dataset first."), 400)
        return engine, None

    def _prediction_json(pred):
        return {
            "video_id": pred.video_id,
            "frame_id": pred.frame_id,
            "id": pred.keyframe_id,
            "score": round(float(pred.score), 6),
            "answer": pred.answer,
            "event_frame_ids": pred.event_frame_ids,
            "timestamp": round(float(pred.timestamp), 3),
            "score_breakdown": pred.score_breakdown,
            "evidence": pred.evidence,
            "image": f"/api/video/frame/{pred.keyframe_id}" if pred.keyframe_id else None,
            "video_url": f"/api/video/file/{pred.video_id}" if (Path(state["config"].dataset.root) / "video" / f"{pred.video_id}.mp4").exists() else None,
        }

    def _frame_json(engine: AICCompetitionEngine, keyframe_id: str, extra: dict | None = None):
        raw = engine.entry.raws[keyframe_id]
        out = {
            "video_id": raw.video_id,
            "frame_id": official_frame_id(engine.entry, keyframe_id),
            "id": keyframe_id,
            "timestamp": round(float(raw.timestamp), 3),
            "image": f"/api/video/frame/{keyframe_id}",
        }
        if extra:
            out.update(extra)
        return out

    @app.get("/")
    def home():
        body = (_UI_DIR / "index.html").read_text(encoding="utf-8")
        return (
            "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>AIC 2026 Rewind</title></head><body>"
            f"{body}</body></html>"
        )

    @app.get("/api/health")
    def health():
        engine = state.get("engine")
        paths = AICDataPaths.from_root(state["config"].dataset.root)
        feature_count = len(list(paths.features_dir.glob("*.npy"))) if paths.features_dir.exists() else 0
        loaded = engine is not None
        encoder = None if engine is None else engine.encoder_status()
        mp4_count = len(list(paths.video_dir.glob("*.mp4"))) if paths.video_dir.exists() else 0
        return jsonify(
            ok=True,
            mode="aic2026",
            loaded=loaded,
            videos=0 if engine is None else len({r.video_id for r in engine.entry.raws.values()}),
            dataset_size=0 if engine is None else engine.entry.num_indexed,
            data_root=str(state["config"].dataset.root),
            feature_videos=feature_count,
            cache_dir=str(state["config"].dataset.cache_dir),
            mp4_available=mp4_count,
            encoder=encoder,
            warning=None if encoder is None else encoder.get("warning"),
            config=_config_summary(),
        )

    @app.get("/api/video/progress")
    def video_progress():
        return jsonify(state["progress"])

    @app.get("/api/video/list")
    def video_list():
        paths = AICDataPaths.from_root(state["config"].dataset.root)
        videos = sorted(p.stem for p in paths.features_dir.glob("*.npy")) if paths.features_dir.exists() else []
        indexed = ["__aic2026__"] if state.get("engine") is not None else []
        return jsonify(videos=videos, indexed=indexed, folder=str(state["config"].dataset.root))

    @app.post("/api/video/index")
    def video_index():
        data = request.json or {}
        rebuild = bool(data.get("rebuild", False))
        try:
            engine, load = AICCompetitionEngine.from_data_root(
                state["config"].dataset.root,
                cache_dir=state["config"].dataset.cache_dir,
                rebuild=rebuild,
                limit_videos=data.get("limit_videos"),
                limit_frames_per_video=data.get("limit_frames_per_video"),
                load_objects=bool(data["objects"]) if "objects" in data else None,
                production_mode=bool(data["production_mode"]) if "production_mode" in data else None,
                allow_hashing_fallback=bool(data["allow_hashing_fallback"]) if "allow_hashing_fallback" in data else None,
                device=data.get("device"),
                app_config=state["config"],
            )
        except Exception as exc:
            return jsonify(error=str(exc)), 400
        state["engine"] = engine
        state["load"] = load
        return jsonify(
            video="__aic2026__",
            cached=load.cache_hit,
            build_seconds=round(load.build_seconds, 3),
            frames=engine.entry.num_indexed,
            stats=None if load.stats is None else load.stats.__dict__,
        )

    @app.post("/api/video/index_folder")
    def video_index_folder():
        data = request.json or {}
        raw_path = (data.get("path") or "").strip().strip('"')
        if not raw_path:
            return jsonify(error="AIC DATA_ROOT path is required."), 400
        folder = Path(raw_path)
        if not folder.is_dir():
            return jsonify(error=f"Invalid AIC DATA_ROOT: {raw_path}"), 400
        try:
            engine, load = AICCompetitionEngine.from_data_root(
                folder,
                cache_dir=state["config"].dataset.cache_dir,
                rebuild=bool(data.get("rebuild", False)),
                load_objects=bool(data["objects"]) if "objects" in data else None,
                production_mode=bool(data["production_mode"]) if "production_mode" in data else None,
                allow_hashing_fallback=bool(data["allow_hashing_fallback"]) if "allow_hashing_fallback" in data else None,
                device=data.get("device"),
                app_config=state["config"],
            )
        except Exception as exc:
            return jsonify(error=str(exc)), 400
        state["engine"] = engine
        state["load"] = load
        return jsonify(video="__aic2026__", cached=load.cache_hit, frames=engine.entry.num_indexed, folder=str(folder))

    @app.post("/api/video/search")
    def video_search():
        engine, err = _engine_or_error()
        if err:
            return err
        data = request.json or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify(error="Query is required."), 400
        topk = max(1, min(MAX_PREDICTIONS, int(data.get("topk", 20))))
        preds = engine.search_kis(query, top_k=topk)
        return jsonify(
            task="kis",
            video="__aic2026__",
            query=query,
            count=len(preds),
            results=[_prediction_json(p) for p in preds],
            predictions=[p.row() for p in preds],
            encoder=engine.encoder_status(),
        )

    @app.post("/api/video/vqa")
    def video_vqa():
        engine, err = _engine_or_error()
        if err:
            return err
        data = request.json or {}
        question = (data.get("question") or "").strip()
        event_text = (data.get("event") or data.get("query") or "").strip()
        if not question and not event_text:
            return jsonify(error="Question or event text is required."), 400
        preds, info = engine.answer_qa(
            event_text,
            question,
            top_k=max(1, min(MAX_PREDICTIONS, int(data.get("topk", 20)))),
            window_s=float(data.get("window", 8.0)),
        )
        frames = [_frame_json(engine, fid) for fid in info.get("frame_ids", []) if fid in engine.entry.raws]
        return jsonify(
            task="qa",
            video="__aic2026__",
            question=question,
            event=event_text,
            answer=info.get("answer"),
            value=info.get("value"),
            reasoning=info.get("reasoning"),
            answer_normalized=info.get("answer_normalized"),
            grounding_score=info.get("grounding_score"),
            answer_confidence=info.get("answer_confidence"),
            warning=info.get("warning"),
            evidence_roles=info.get("evidence_roles", []),
            used_frame_ids=info.get("used_frame_ids", []),
            frames=frames,
            center_time=info.get("center_time"),
            video_id=info.get("video_id"),
            predictions=[p.row() for p in preds],
            encoder=engine.encoder_status(),
        )

    @app.post("/api/video/temporal")
    def video_temporal():
        engine, err = _engine_or_error()
        if err:
            return err
        data = request.json or {}
        events = [e.strip() for e in (data.get("events") or []) if e and e.strip()]
        if len(events) < 2:
            return jsonify(error="At least two ordered events are required."), 400
        try:
            preds, matches = engine.search_trake(
                events,
                per_event_k=int(data.get("per_event_k", 40)),
                max_results=max(1, min(MAX_PREDICTIONS, int(data.get("max_results", 20)))),
                refine_window_s=float(data.get("refine_window", 6.0)),
            )
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        out = []
        for match in matches:
            steps = []
            for i, step in enumerate(match.steps):
                steps.append(_frame_json(engine, step.keyframe_id, {"event": events[i]}))
            out.append({"video_id": match.video_id, "total_score": round(float(match.total_score), 6), "steps": steps})
        return jsonify(task="trake", video="__aic2026__", events=events, count=len(out), matches=out, predictions=[p.row() for p in preds])

    @app.get("/api/evaluation/status")
    def evaluation_status():
        return jsonify(state["evaluation"])

    @app.post("/api/evaluation/run")
    def evaluation_run():
        engine, err = _engine_or_error()
        if err:
            return err
        data = request.json or {}
        labels_path = Path((data.get("labels") or "").strip())
        if not labels_path.is_file():
            return jsonify(error="AIC JSONL ground-truth file is required."), 400
        if state["evaluation"]["active"]:
            return jsonify(error="An evaluation run is already active."), 409
        state["evaluation"] = {"active": True, "summary": None, "run_dir": None, "error": None}

        def worker():
            try:
                labels = load_jsonl(labels_path)

                def predictor(label):
                    task = label["task"].lower()
                    if task == "kis":
                        return [p.row() for p in engine.search_kis(label["query"], top_k=100)]
                    if task in {"qa", "q&a", "vqa"}:
                        predictions, _ = engine.answer_qa(
                            label.get("query", ""), label.get("question", ""), top_k=100
                        )
                        return [p.row() for p in predictions]
                    events = label.get("events") or []
                    event_text = [item.get("event", "") if isinstance(item, dict) else str(item) for item in events]
                    predictions, _ = engine.search_trake(event_text, max_results=100)
                    return [p.row() for p in predictions]

                summary, run_dir = evaluate_labels(labels, predictor, run_name="ui")
                state["evaluation"] = {
                    "active": False, "summary": summary, "run_dir": str(run_dir), "error": None
                }
            except Exception as exc:
                state["evaluation"] = {
                    "active": False, "summary": None, "run_dir": None, "error": str(exc)
                }

        Thread(target=worker, daemon=True).start()
        return jsonify(started=True), 202
    @app.post("/api/video/save")
    def video_save():
        engine, err = _engine_or_error()
        if err:
            return err
        engine.entry.save(AIC_CACHE_DIR / "entry")
        return jsonify(saved=["__aic2026__"], dir=str(AIC_CACHE_DIR))

    @app.post("/api/submission/save")
    def submission_save():
        data = request.json or {}
        rows = data.get("rows") or []
        task = data.get("task") or "submission"
        if not rows:
            return jsonify(error="No rows to save."), 400
        path = write_submission(rows, SUBMISSION_DIR / f"{task}.csv")
        return jsonify(path=str(path), rows=min(len(rows), MAX_PREDICTIONS))

    @app.get("/api/video/neighbors/<path:frame_id>")
    def video_neighbors(frame_id: str):
        engine, err = _engine_or_error()
        if err:
            return err
        if frame_id not in engine.entry.raws:
            return jsonify(error="Frame not found."), 404
        raws = sorted(
            (r for r in engine.entry.raws.values() if r.video_id == engine.entry.raws[frame_id].video_id),
            key=lambda r: r.timestamp,
        )
        idx = next(i for i, r in enumerate(raws) if r.id == frame_id)
        chosen = raws[max(0, idx - 4): idx + 5]
        return jsonify(frames=[_frame_json(engine, r.id, {"is_ref": r.id == frame_id}) for r in chosen])

    @app.get("/api/video/file/<video_id>")
    def video_file(video_id: str):
        path = Path(state["config"].dataset.root) / "video" / f"{video_id}.mp4"
        if not path.is_file():
            return jsonify(error="Original MP4 is unavailable."), 404
        return send_file(path, mimetype="video/mp4", conditional=True)
    @app.get("/api/video/frame/<path:frame_id>")
    def video_frame(frame_id: str):
        engine, err = _engine_or_error()
        if err:
            return err
        raw = engine.entry.raws.get(frame_id)
        if raw is None:
            return jsonify(error="Frame not found."), 404
        data = frame_jpeg_bytes(raw)
        if data:
            return send_file(BytesIO(data), mimetype="image/jpeg")
        return jsonify(error="Frame image unavailable."), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)