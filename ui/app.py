"""Flask backend for the AIC 2026 competition-oriented Rewind UI.

Every dataset-dependent route resolves its facts from one `RuntimeDatasetState`
snapshot taken at the start of the request. There is no module-level `DATA_ROOT` or
cache path that can disagree with the active engine: the constants below are only
defaults consulted while *constructing* the initial state, never while serving.
"""
from __future__ import annotations

from dataclasses import replace as dataclass_replace
from io import BytesIO
from pathlib import Path
from threading import Thread

from flask import Flask, jsonify, request, send_file

from ingestion import model_cache  # noqa: F401 -- configure model cache early

from aic2026.cache_manifest import (
    CacheManifestError,
    LegacyCacheError,
    StaleCacheError,
)
from aic2026.config import AppConfig, DatasetScopeConfig, load_app_config, validate_app_config
from aic2026.dataset import AICDataPaths, official_frame_id
from aic2026.dataset_scope import DatasetScopeError, scope_payload
from aic2026.dataset_validation import DatasetError, DatasetLayoutError, inspect_aic_dataset
from aic2026.engine import AICCompetitionEngine, MAX_PREDICTIONS
from aic2026.metrics import SubmissionStructureError, write_submission
from aic2026.runtime_state import (
    ACTIVE_STATE_NOT_CHANGED,
    DATA_ROOT_INVALID,
    RUNTIME_STATE_UNINITIALIZED,
    STATE_BUILD_FAILED,
    RuntimeDatasetState,
    RuntimeStateError,
    RuntimeStateManager,
    build_runtime_state,
    check_generation,
    resolve_cache_dir,
    safe_video_path,
)
from evaluation.official_eval import evaluate_labels, load_jsonl

_UI_DIR = Path(__file__).resolve().parent
_ROOT = _UI_DIR.parent

# Construction-time defaults only. Nothing below reads these while serving a request;
# the active RuntimeDatasetState is the sole authority once the app is running.
DATA_ROOT = _ROOT / "data"
AIC_CACHE_DIR = _ROOT / "artifacts" / "aic2026_index"
SUBMISSION_DIR = _ROOT / "artifacts" / "submissions"

STATE_EXTENSION_KEY = "aic_runtime_state"


def create_app(
    config_path: str | Path | None = None,
    app_config: AppConfig | None = None,
    initial_data_root: str | Path | None = None,
    initial_cache_dir: str | Path | None = None,
) -> Flask:
    """Build the app and publish an initial runtime state.

    Precedence: an explicitly supplied `app_config` wins; otherwise `config_path` is
    loaded; `initial_data_root` / `initial_cache_dir` then override the dataset roots.
    Startup never rebuilds an index — it reports cache status and waits for an
    explicit activate call.
    """
    if app_config is None:
        if config_path is None:
            config_path = _ROOT / "configs" / "settings.yaml"
            app_config = load_app_config(
                config_path,
                {"dataset": {"root": str(DATA_ROOT), "cache_dir": str(AIC_CACHE_DIR)}},
            )
        else:
            app_config = load_app_config(config_path)
    config_path_text = str(config_path or "<in-memory>")

    data_root = str(initial_data_root or app_config.dataset.root)
    cache_dir = resolve_cache_dir(app_config, data_root, explicit=initial_cache_dir)

    app = Flask(__name__)
    manager = RuntimeStateManager(
        build_runtime_state(
            app_config=app_config,
            config_path=config_path_text,
            generation=1,
            data_root=data_root,
            cache_dir=cache_dir,
        )
    )
    app.extensions[STATE_EXTENSION_KEY] = manager
    # App-level (non-dataset) settings, captured once so requests never read globals.
    submission_dir = Path(SUBMISSION_DIR)
    ui_state: dict = {
        "progress": {"active": False, "count": 0, "fps": 0.0, "elapsed": 0.0, "label": ""},
        "evaluation": {"active": False, "summary": None, "run_dir": None, "error": None},
    }

    # ------------------------------------------------------------------ helpers

    def snapshot() -> RuntimeDatasetState:
        """One state for the whole request. Never call this twice in one handler."""
        return manager.get_state()

    def _error(message: str, code: str, status: int, **extra):
        return jsonify(error=message, error_code=code, **extra), status

    def _state_error_response(exc: RuntimeStateError, *, status: int = 409):
        return _error(str(exc), exc.error_code, status)

    def _config_summary(state: RuntimeDatasetState) -> dict:
        cfg = state.app_config
        return {
            "path": state.config_path,
            "hash": state.config_hash,
            "production_mode": cfg.runtime.production_mode,
            "device": cfg.runtime.device,
            "encoder_type": cfg.encoder.type,
            "feature_dim": cfg.encoder.feature_dim,
            "dataset_scope": state.dataset_scope,
            "fusion_method": cfg.fusion.method,
            "final_top_k": cfg.ranking.final_top_k,
            "refinement_enabled": cfg.refinement.enabled,
            "refinement_mode": cfg.refinement.effective_mode,
            "refinement_frame_output_policy": cfg.refinement.frame_output_policy,
            "trake_alignment_method": cfg.trake.alignment_method,
            "qa_top_video_hypotheses": cfg.qa.top_video_hypotheses,
        }

    def _cache_error_response(exc: Exception, state: RuntimeDatasetState):
        """Cache problems are request/data states, never 500s. State is unchanged."""
        common = {"cache": state.cache_status, "active_state": state.runtime_summary()}
        if isinstance(exc, LegacyCacheError):
            return jsonify(error=str(exc), error_code="LEGACY_CACHE", **common), 409
        if isinstance(exc, StaleCacheError):
            return jsonify(error=str(exc), error_code="STALE_CACHE", **common), 409
        if isinstance(exc, CacheManifestError):
            return jsonify(error=str(exc), error_code="CORRUPT_CACHE_MANIFEST", **common), 422
        return None

    def _engine_or_error(state: RuntimeDatasetState):
        if state.engine is None:
            return None, _error(
                "AIC index is not loaded. Activate a dataset first.",
                RUNTIME_STATE_UNINITIALIZED,
                400,
            )
        return state.engine, None

    def _scope_override(base: AppConfig, data: dict) -> AppConfig:
        """Apply optional per-request scope overrides, validated like any config."""
        scope = base.dataset.scope
        changed = False
        if data.get("scope_mode"):
            scope = dataclass_replace(scope, mode=str(data["scope_mode"]))
            changed = True
        if data.get("include_patterns"):
            scope = dataclass_replace(
                scope, include_patterns=tuple(str(p) for p in data["include_patterns"])
            )
            changed = True
        if data.get("exclude_patterns") is not None:
            scope = dataclass_replace(
                scope, exclude_patterns=tuple(str(p) for p in data["exclude_patterns"])
            )
            changed = True
        if not changed:
            return base
        candidate = dataclass_replace(base, dataset=dataclass_replace(base.dataset, scope=scope))
        validate_app_config(candidate)
        return candidate

    def _activate(state: RuntimeDatasetState, data: dict, *, data_root: str, cache_dir: str | None):
        """Build a whole new state, then publish it. Failure leaves `state` active.

        Nothing on the live state is mutated before the very last step, so a partial
        switch — engine from root B, routes still on root A — cannot happen.
        """
        root_path = Path(data_root)
        if not root_path.is_dir():
            return _error(
                f"Invalid AIC data root: {data_root}",
                DATA_ROOT_INVALID,
                400,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        try:
            base_config = _scope_override(state.app_config, data)
            effective_cache_dir = resolve_cache_dir(
                base_config,
                data_root,
                explicit=cache_dir,
                scope=base_config.dataset.scope,
            )
            effective_config = dataclass_replace(
                base_config,
                dataset=dataclass_replace(
                    base_config.dataset, root=str(data_root), cache_dir=str(effective_cache_dir)
                ),
            )
            engine, load = AICCompetitionEngine.from_data_root(
                data_root,
                cache_dir=effective_cache_dir,
                rebuild=bool(data.get("rebuild", False)),
                limit_videos=data.get("limit_videos"),
                limit_frames_per_video=data.get("limit_frames_per_video"),
                load_objects=bool(data["objects"]) if "objects" in data else None,
                production_mode=bool(data["production_mode"]) if "production_mode" in data else None,
                allow_hashing_fallback=(
                    bool(data["allow_hashing_fallback"]) if "allow_hashing_fallback" in data else None
                ),
                allow_stale_cache=(
                    bool(data["allow_stale_cache"]) if "allow_stale_cache" in data else None
                ),
                device=data.get("device"),
                app_config=effective_config,
            )
            new_state = build_runtime_state(
                app_config=effective_config,
                config_path=state.config_path,
                generation=manager.next_generation(),
                data_root=data_root,
                cache_dir=effective_cache_dir,
                engine=engine,
                load=load,
            )
        except (LegacyCacheError, StaleCacheError, CacheManifestError) as exc:
            cache_error = _cache_error_response(exc, state)
            if cache_error is not None:
                return cache_error
            raise
        except DatasetLayoutError as exc:
            # The path exists but is not an AIC data root: a bad request, not bad data.
            return _error(
                str(exc),
                DATA_ROOT_INVALID,
                400,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        except (DatasetError, DatasetScopeError) as exc:
            return _error(
                str(exc),
                "DATASET_INVALID",
                422,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        except RuntimeStateError as exc:
            return _error(
                str(exc), exc.error_code, 500, active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a 400 with the reason
            return _error(
                str(exc),
                STATE_BUILD_FAILED,
                400,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )

        published = manager.replace_state(new_state)
        return jsonify(
            video="__aic2026__",
            cached=load.cache_hit,
            build_seconds=round(load.build_seconds, 3),
            frames=engine.entry.num_indexed,
            folder=published.data_root,
            stats=None if load.stats is None else load.stats.__dict__,
            cache=published.cache_status,
            manifest=None if load.cache_manifest is None else load.cache_manifest.to_dict(),
            runtime=published.runtime_summary(),
            active_state_changed=True,
        )

    def _prediction_json(state: RuntimeDatasetState, pred, available_videos: set[str]):
        raw = state.engine.entry.raws.get(pred.keyframe_id) if pred.keyframe_id else None
        visual = (
            state.frame_provider.describe(raw)
            if raw is not None
            else {"image_available": False, "image_source": "none", "video_available": False}
        )
        # Logical API URLs only; a filesystem path is never handed to the frontend. The
        # generation makes a result from a superseded dataset detectable.
        image = (
            f"/api/video/frame/{pred.keyframe_id}?generation={state.generation}"
            if pred.keyframe_id
            else None
        )
        video_url = (
            f"/api/video/file/{pred.video_id}?generation={state.generation}"
            if pred.video_id in available_videos
            else None
        )
        # The refined visual frame is EVIDENCE. `frame_id` above stays the official
        # submission frame, and the two are labelled separately because under the
        # default preserve_coarse policy they legitimately differ.
        refinement = pred.refinement
        refined_image = None
        refined_frame_idx = None
        if refinement and refinement.get("applied") and refinement.get("best_visual_frame_idx") is not None:
            refined_frame_idx = int(refinement["best_visual_frame_idx"])
            refined_image = (
                f"/api/video/decoded_frame/{pred.video_id}/{refined_frame_idx}"
                f"?generation={state.generation}"
            )
        return {
            "video_id": pred.video_id,
            "frame_id": pred.frame_id,
            "submission_frame_id": pred.frame_id,
            "id": pred.keyframe_id,
            "generation": state.generation,
            "score": round(float(pred.score), 6),
            "answer": pred.answer,
            "event_frame_ids": pred.event_frame_ids,
            "timestamp": round(float(pred.timestamp), 3),
            "score_breakdown": pred.score_breakdown,
            "evidence": pred.evidence,
            "image": image,
            "image_available": visual["image_available"],
            "image_source": visual["image_source"],
            "video_available": visual["video_available"],
            "video_url": video_url,
            "refinement": refinement,
            "refined_frame_id": refined_frame_idx,
            "refined_image": refined_image,
        }

    def _frame_json(state: RuntimeDatasetState, keyframe_id: str, extra: dict | None = None):
        raw = state.engine.entry.raws[keyframe_id]
        visual = state.frame_provider.describe(raw)
        out = {
            "video_id": raw.video_id,
            "frame_id": official_frame_id(state.engine.entry, keyframe_id),
            "id": keyframe_id,
            "generation": state.generation,
            "timestamp": round(float(raw.timestamp), 3),
            "image": f"/api/video/frame/{keyframe_id}?generation={state.generation}",
            "image_available": visual["image_available"],
            "image_source": visual["image_source"],
            "video_available": visual["video_available"],
        }
        if extra:
            out.update(extra)
        return out

    def _evidence_url(state: RuntimeDatasetState, evidence: dict) -> str | None:
        """A logical URL for one Q&A evidence frame; never a filesystem path.

        A mapped keyframe goes through the normal frame route; a refined/decoded frame
        goes through the decoded-frame route, which is explicitly NOT a submission frame.
        """
        if not evidence.get("image_available"):
            return None
        keyframe_id = evidence.get("keyframe_id")
        if keyframe_id:
            return f"/api/video/frame/{keyframe_id}?generation={state.generation}"
        frame_idx = evidence.get("frame_idx")
        if evidence.get("video_id") and frame_idx is not None:
            return (
                f"/api/video/decoded_frame/{evidence['video_id']}/{int(frame_idx)}"
                f"?generation={state.generation}"
            )
        return None

    def _available_videos(state: RuntimeDatasetState) -> set[str]:
        video_dir = Path(state.app_config.dataset.root) / "video"
        if not video_dir.is_dir():
            return set()
        return {path.stem for path in video_dir.glob("*.mp4")}

    # ------------------------------------------------------------------- routes

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
        state = snapshot()
        paths = AICDataPaths.from_root(state.app_config.dataset.root)
        feature_count = (
            len(list(paths.features_dir.glob("*.npy"))) if paths.features_dir.is_dir() else 0
        )
        engine = state.engine
        encoder = None if engine is None else engine.encoder_status()
        return jsonify(
            ok=True,
            mode="aic2026",
            loaded=state.engine_loaded,
            runtime=state.runtime_summary(),
            videos=0 if engine is None else len({r.video_id for r in engine.entry.raws.values()}),
            dataset_size=0 if engine is None else engine.entry.num_indexed,
            # Kept for the existing frontend; identical to runtime.data_root.
            data_root=str(state.app_config.dataset.root),
            cache_dir=state.cache_dir,
            feature_videos=feature_count,
            mp4_available=state.video_inventory_summary.get("available_count", 0),
            video=dict(state.video_inventory_summary),
            dataset=state.dataset_status,
            encoder=encoder,
            warning=None if encoder is None else encoder.get("warning"),
            config=_config_summary(state),
            cache=state.cache_status,
            # Configuration plus scorer state. `initialize` is deliberately not passed:
            # asking for health must never load the CLIP checkpoint.
            refinement=(
                dict(state.app_config.refinement.summary())
                if engine is None
                else engine.refinement_status()
            ),
            # Backend capability, reported truthfully and without loading any model.
            qa=None if engine is None else engine.qa_status(),
        )

    @app.get("/api/video/progress")
    def video_progress():
        return jsonify(ui_state["progress"])

    @app.get("/api/video/list")
    def video_list():
        state = snapshot()
        return jsonify(
            videos=list(state.resolved_video_ids),
            indexed=["__aic2026__"] if state.engine_loaded else [],
            folder=str(state.app_config.dataset.root),
            generation=state.generation,
            scope=state.dataset_scope,
        )

    @app.post("/api/video/index")
    def video_index():
        """Activate the dataset the active state already points at."""
        state = snapshot()
        data = request.json or {}
        return _activate(state, data, data_root=str(state.app_config.dataset.root), cache_dir=state.cache_dir)

    @app.post("/api/video/index_folder")
    def video_index_folder():
        """Switch the active dataset root, atomically."""
        state = snapshot()
        data = request.json or {}
        raw_path = (data.get("path") or "").strip().strip('"')
        if not raw_path:
            return _error(
                "AIC data root path is required.",
                DATA_ROOT_INVALID,
                400,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        cache_dir = (data.get("cache_dir") or "").strip() or None
        return _activate(state, data, data_root=raw_path, cache_dir=cache_dir)

    @app.post("/api/dataset/inspect")
    def dataset_inspect():
        """Read-only inspection. Never activates and never mutates the active state."""
        state = snapshot()
        data = request.json or {}
        raw_path = (data.get("path") or "").strip().strip('"') or str(
            state.app_config.dataset.root
        )
        if not Path(raw_path).is_dir():
            return _error(
                f"Invalid AIC data root: {raw_path}",
                DATA_ROOT_INVALID,
                400,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        try:
            config = _scope_override(state.app_config, data)
            config = dataclass_replace(
                config, dataset=dataclass_replace(config.dataset, root=str(raw_path))
            )
            report = inspect_aic_dataset(
                Path(raw_path),
                app_config=config,
                strict=False,
                deep=bool(data.get("deep", False)),
                limit_videos=data.get("limit_videos"),
            )
        except DatasetLayoutError as exc:
            return _error(
                str(exc),
                DATA_ROOT_INVALID,
                400,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        except (DatasetError, DatasetScopeError) as exc:
            return _error(
                str(exc),
                "DATASET_INVALID",
                422,
                active_state=state.runtime_summary(),
                active_state_changed=False,
            )
        return jsonify(
            inspected_root=str(Path(raw_path).resolve(strict=False).as_posix()),
            summary=report.summary(),
            active_state=state.runtime_summary(),
            active_state_changed=False,
        )

    @app.post("/api/video/search")
    def video_search():
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        data = request.json or {}
        query = (data.get("query") or "").strip()
        if not query:
            return _error("Query is required.", "QUERY_REQUIRED", 400)
        topk = max(1, min(MAX_PREDICTIONS, int(data.get("topk", 20))))
        # `refine: false` turns local refinement off for this query only, so the two
        # pipelines can be compared without restarting or editing configuration.
        refine = None if data.get("refine") is None else bool(data.get("refine"))
        outcome = engine.search_kis_detailed(query, top_k=topk, refine=refine)
        preds = outcome.predictions
        available = _available_videos(state)
        return jsonify(
            task="kis",
            video="__aic2026__",
            query=query,
            generation=state.generation,
            count=len(preds),
            results=[_prediction_json(state, p, available) for p in preds],
            predictions=[p.row() for p in preds],
            encoder=engine.encoder_status(),
            refinement={
                "enabled": engine.refinement_config.enabled,
                "mode": engine.refinement_config.effective_mode,
                "requested": refine,
                "frame_output_policy": engine.refinement_config.frame_output_policy,
                "decision": (
                    None if outcome.refinement is None else outcome.refinement.decision.to_dict()
                ),
                "scorer": engine.local_refiner.scorer_status(),
                "warnings": [] if outcome.refinement is None else list(outcome.refinement.warnings),
            },
            diagnostics=outcome.diagnostics(),
        )

    @app.post("/api/video/vqa")
    def video_vqa():
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        data = request.json or {}
        question = (data.get("question") or "").strip()
        event_text = (data.get("event") or data.get("query") or "").strip()
        if not question and not event_text:
            return _error("Question or event text is required.", "QUERY_REQUIRED", 400)
        try:
            preds, info = engine.answer_qa(
                event_text,
                question,
                top_k=max(1, min(MAX_PREDICTIONS, int(data.get("topk", 20)))),
                window_s=float(data.get("window", 8.0)),
                # Finally wired: the UI's answer-type selector reaches normalization.
                expected_answer_type=data.get("expected_answer_type"),
                use_local_refinement=(
                    None if data.get("refine") is None else bool(data.get("refine"))
                ),
            )
        except ValueError as exc:
            return _error(str(exc), "INVALID_ANSWER_TYPE", 400)
        frames = [
            _frame_json(state, fid)
            for fid in info.get("frame_ids", [])
            if fid in engine.entry.raws
        ]
        # One card per VIDEO hypothesis, each carrying only its own video's answer and
        # its own evidence. Nothing here is shared between hypotheses.
        hypotheses = [
            {
                **{k: v for k, v in item.items() if k != "evidence"},
                "evidence": [
                    {
                        **evidence,
                        "image": _evidence_url(state, evidence),
                    }
                    for evidence in item.get("evidence", [])
                ],
            }
            for item in info.get("hypotheses", [])
        ]
        return jsonify(
            task="qa",
            video="__aic2026__",
            generation=state.generation,
            question=question,
            event=event_text,
            answer=info.get("answer"),
            value=info.get("value"),
            reasoning=info.get("reasoning"),
            answer_normalized=info.get("answer_normalized"),
            answer_status=info.get("answer_status"),
            expected_answer_type=info.get("expected_answer_type"),
            grounding_score=info.get("grounding_score"),
            answer_confidence=info.get("answer_confidence"),
            answer_reliability_score=info.get("answer_reliability_score"),
            warning=info.get("warning"),
            evidence_roles=info.get("evidence_roles", []),
            used_frame_ids=info.get("used_frame_ids", []),
            frames=frames,
            center_time=info.get("center_time"),
            video_id=info.get("video_id"),
            hypotheses=hypotheses,
            backend=info.get("backend"),
            qa=engine.qa_status(),
            diagnostics=info.get("diagnostics"),
            predictions=[p.row() for p in preds],
            results=[_prediction_json(state, p, _available_videos(state)) for p in preds],
            encoder=engine.encoder_status(),
        )

    @app.post("/api/video/temporal")
    def video_temporal():
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        data = request.json or {}
        events = [e.strip() for e in (data.get("events") or []) if e and e.strip()]
        if len(events) < 2:
            return _error("At least two ordered events are required.", "QUERY_REQUIRED", 400)
        try:
            outcome = engine.search_trake_detailed(
                events,
                per_event_k=int(data.get("per_event_k", 40)),
                max_results=max(1, min(MAX_PREDICTIONS, int(data.get("max_results", 20)))),
                refine_window_s=(
                    None if data.get("refine_window") is None else float(data["refine_window"])
                ),
            )
        except ValueError as exc:
            return _error(str(exc), "QUERY_REQUIRED", 400)
        preds = outcome.predictions
        out = []
        # Every returned sequence is complete, so step i IS event i. The old code zipped
        # a compacted step list against the full event list and shifted every label after
        # a gap; that is structurally impossible now.
        for prediction, alignment in zip(preds, outcome.trake_predictions):
            steps = []
            for step in alignment.steps:
                payload = _frame_json(
                    state,
                    str(step.keyframe_id),
                    {
                        "event": step.event_text,
                        "event_index": step.event_index,
                        "event_label": f"Event {step.event_index + 1}",
                        "status": step.status,
                        "submission_frame_id": step.submission_frame_idx,
                    },
                )
                steps.append(payload)
            out.append(
                {
                    "video_id": alignment.video_id,
                    "event_count": alignment.event_count,
                    "frame_ids": list(alignment.frame_ids),
                    "alignment_status": alignment.alignment_status,
                    "recovered_event_indices": list(alignment.recovered_event_indices),
                    "missing_event_indices": list(alignment.missing_event_indices),
                    "method": alignment.method,
                    "total_score": round(float(alignment.score), 6),
                    "steps": steps,
                }
            )
        return jsonify(
            task="trake",
            video="__aic2026__",
            generation=state.generation,
            events=events,
            event_count=len(events),
            count=len(out),
            matches=out,
            predictions=[p.row() for p in preds],
            diagnostics=outcome.diagnostics,
            structural=outcome.structural_summary(),
            # Truthful: the parameter is accepted for compatibility and does nothing.
            refinement={
                "applied": False,
                "status": "not_implemented_phase_7",
                "note": "TRAKE local refinement is not implemented; refine_window has no effect.",
            },
        )

    @app.get("/api/evaluation/status")
    def evaluation_status():
        return jsonify(ui_state["evaluation"])

    @app.post("/api/evaluation/run")
    def evaluation_run():
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        data = request.json or {}
        labels_path = Path((data.get("labels") or "").strip())
        if not labels_path.is_file():
            return _error("AIC JSONL ground-truth file is required.", "LABELS_REQUIRED", 400)
        if ui_state["evaluation"]["active"]:
            return _error("An evaluation run is already active.", "EVALUATION_ACTIVE", 409)
        ui_state["evaluation"] = {"active": True, "summary": None, "run_dir": None, "error": None}

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
                    event_text = [
                        item.get("event", "") if isinstance(item, dict) else str(item)
                        for item in events
                    ]
                    predictions, _ = engine.search_trake(event_text, max_results=100)
                    return [p.row() for p in predictions]

                summary, run_dir = evaluate_labels(labels, predictor, run_name="ui")
                ui_state["evaluation"] = {
                    "active": False, "summary": summary, "run_dir": str(run_dir), "error": None
                }
            except Exception as exc:
                ui_state["evaluation"] = {
                    "active": False, "summary": None, "run_dir": None, "error": str(exc)
                }

        Thread(target=worker, daemon=True).start()
        return jsonify(started=True), 202

    @app.post("/api/video/save")
    def video_save():
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        entry_dir = Path(state.cache_dir) / "entry"
        engine.entry.save(entry_dir)
        return jsonify(
            saved=["__aic2026__"], dir=str(entry_dir), generation=state.generation
        )

    @app.post("/api/submission/save")
    def submission_save():
        data = request.json or {}
        rows = data.get("rows") or []
        task = str(data.get("task") or "submission")
        if not rows:
            return _error("No rows to save.", "NO_ROWS", 400)
        if not task.replace("_", "").replace("-", "").isalnum():
            return _error(f"Invalid submission name {task!r}.", "INVALID_SUBMISSION_NAME", 400)
        # Local structural safety only; the full schema validator is Phase 11. A TRAKE
        # export must never write a row with fewer frames than the query had events.
        event_count = data.get("event_count")
        required = None
        if task.lower().startswith("trake") and event_count is not None:
            try:
                required = int(event_count) + 1
            except (TypeError, ValueError):
                return _error("event_count must be an integer.", "INVALID_EVENT_COUNT", 400)
        try:
            path = write_submission(
                rows, submission_dir / f"{task}.csv", require_row_length=required
            )
        except SubmissionStructureError as exc:
            return _error(str(exc), "MALFORMED_SUBMISSION_ROW", 422)
        return jsonify(path=str(path), rows=min(len(rows), MAX_PREDICTIONS))

    @app.get("/api/video/neighbors/<path:frame_id>")
    def video_neighbors(frame_id: str):
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        try:
            check_generation(state, request.args.get("generation"))
        except RuntimeStateError as exc:
            return _state_error_response(exc)
        if frame_id not in engine.entry.raws:
            return _error("Frame not found.", "FRAME_NOT_FOUND", 404)
        raws = sorted(
            (r for r in engine.entry.raws.values() if r.video_id == engine.entry.raws[frame_id].video_id),
            key=lambda r: r.timestamp,
        )
        index = next(i for i, r in enumerate(raws) if r.id == frame_id)
        chosen = raws[max(0, index - 4): index + 5]
        return jsonify(
            generation=state.generation,
            frames=[_frame_json(state, r.id, {"is_ref": r.id == frame_id}) for r in chosen],
        )

    @app.get("/api/video/file/<video_id>")
    def video_file(video_id: str):
        state = snapshot()
        try:
            check_generation(state, request.args.get("generation"))
            path = safe_video_path(state, video_id)
        except RuntimeStateError as exc:
            return _state_error_response(exc, status=404 if "not part of" in str(exc) else 409)
        if not path.is_file():
            return _error("Original MP4 is unavailable.", "VIDEO_FILE_MISSING", 404)
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/video/decoded_frame/<video_id>/<int:frame_idx>")
    def video_decoded_frame(video_id: str, frame_idx: int):
        """Serve an arbitrary decoded frame of an in-scope video.

        This is how a refined visual frame is displayed. It is NOT a submission frame:
        the official frame stays with the coarse candidate under the default policy, and
        the frontend labels the two differently. Access goes through the same scope and
        traversal checks as the MP4 route.
        """
        state = snapshot()
        try:
            check_generation(state, request.args.get("generation"))
            path = safe_video_path(state, video_id)
        except RuntimeStateError as exc:
            return _state_error_response(exc, status=404 if "not part of" in str(exc) else 409)
        if not path.is_file():
            return _error("Original MP4 is unavailable.", "VIDEO_FILE_MISSING", 404)
        if frame_idx < 0:
            return _error("frame_idx must be >= 0.", "FRAME_NOT_FOUND", 400)
        result = state.frame_provider.get_video_frame(
            video_id, frame_idx=frame_idx, source_video=path
        )
        if not result.available:
            return (
                jsonify(
                    error="That frame could not be decoded from the original MP4.",
                    error_code="FRAME_UNAVAILABLE",
                    generation=state.generation,
                    frame=result.to_dict(),
                ),
                422,
            )
        response = send_file(BytesIO(result.image_bytes), mimetype="image/jpeg")
        response.headers["X-Frame-Source"] = result.source
        response.headers["X-Frame-Id"] = str(frame_idx)
        response.headers["X-Frame-Role"] = "refined_visual_frame"
        response.headers["X-Runtime-Generation"] = str(state.generation)
        return response

    @app.get("/api/video/frame/<path:frame_id>")
    def video_frame(frame_id: str):
        state = snapshot()
        engine, err = _engine_or_error(state)
        if err:
            return err
        try:
            check_generation(state, request.args.get("generation"))
        except RuntimeStateError as exc:
            return _state_error_response(exc)
        raw = engine.entry.raws.get(frame_id)
        if raw is None:
            return _error("Frame not found.", "FRAME_NOT_FOUND", 404)
        # BTC keyframe JPEG first, then a decode of the exact mapped frame from the
        # original MP4 of the ACTIVE data root. Decoding happens only here.
        result = state.frame_provider.get_frame(raw)
        if result.available:
            response = send_file(BytesIO(result.image_bytes), mimetype="image/jpeg")
            response.headers["X-Frame-Source"] = result.source
            response.headers["X-Frame-Id"] = "" if result.frame_idx is None else str(result.frame_idx)
            response.headers["X-Runtime-Generation"] = str(state.generation)
            if result.warning:
                response.headers["X-Frame-Warning"] = result.warning
            return response
        # No visual source is a data state, not a server fault.
        return (
            jsonify(
                error="No visual source is available for this keyframe.",
                error_code="FRAME_UNAVAILABLE",
                generation=state.generation,
                frame=result.to_dict(),
            ),
            422,
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
