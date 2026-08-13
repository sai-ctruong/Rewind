"""Competition search service for AIC 2026 tasks."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ingestion.build_index import l2_normalize
from ingestion.embed_clip import deterministic_unit_vector
from ingestion.schemas import frame_jpeg_bytes
from retrieval.coarse_retriever import DEFAULT_FUSION_DEPTH, Candidate, CoarseRetriever
from retrieval.temporal_check import TemporalMatch, TemporalStep, temporal_consistency_filter
from retrieval.vqa_module import entry_records, retrieve_temporal_window  # noqa: F401 - re-exported for callers
from retrieval.video_engine import VideoIndexEntry, adaptive_bm25_weight

from .cache_manifest import (
    CACHE_MANIFEST_FILENAME,
    CacheManifest,
    CacheManifestError,
    LegacyCacheError,
    StaleCacheError,
    build_cache_manifest,
    cache_build_options_from_config,
    detect_code_version,
    inspect_cache,
    mismatch_message,
    write_cache_manifest_atomic,
)
from .config import AppConfig, config_hash, config_to_dict
from .dataset import AICDatasetLoader, AICDatasetStats, official_frame_id
from .dataset_validation import write_dataset_report
from .frame_provider import FrameProvider
from .frame_scorer import SCORER_STATE_UNAVAILABLE, FrameScorer, build_frame_scorer
from .fusion import CandidateEvidence, FusionConfig, RankedCandidate, fuse_candidates, object_match_score, token_overlap_score
from .local_refinement import (
    FRAME_OUTPUT_DECODED_FRAME,
    MODE_ALWAYS,
    MODE_DISABLED,
    LocalFrameRefiner,
    LocalRefinementRequest,
    LocalRefinementResult,
    RefinementCandidate,
)
from .budget import (
    ACTION_DENSE_TEMPORAL_ZOOM,
    ACTION_OFFICIAL_GRID_REFINE,
    BudgetAction,
    BudgetLedger,
    allocate,
    apply_channel_policy,
    channel_policy,
    kis_uncertainty,
    split_budget_by_uncertainty,
    trake_event_uncertainty,
)
from .cost import QueryCost
from .official_grid import OfficialGridRefiner
from .progressive_refinement import progressive_sample
from .query_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingKey,
    QueryExecutionContext,
    template_signature,
)
from .query_normalization import QueryRepresentation, normalize_query
from .retrieval_channels import (
    CHANNEL_BM25,
    CHANNEL_CLIP,
    CHANNEL_METADATA,
    CHANNEL_OBJECTS,
    ChannelUnion,
    PooledCandidate,
    RetrievalChannelRegistry,
    build_channel_registry,
    channel_depths,
)
from .qa import (
    ANSWER_STATUS_ABSTAINED,
    ANSWER_STATUS_ANSWERED,
    ANSWER_STATUS_BACKEND_FAILED,
    ANSWER_STATUS_BUDGET_EXHAUSTED,
    ANSWER_STATUS_VISUAL_UNAVAILABLE,
    ANSWER_TYPE_AUTO,
    UNKNOWN_ANSWER,
    QAAnswerResult,
    QAEvidenceBundle,
    QAEvidenceFrame,
    QAInput,
    QAVideoHypothesis,
    VisualQAAnswerer,
    answer_reliability_score,
    build_qa_answerer,
    build_retrieval_query,
    canonical_answer_type,
    group_hypotheses_by_video,
    is_unknown_answer,
    select_evidence_frames,
)
from .ranking import RankingConfig, cutoff_aware_top100, video_aware_top100
from .trake import (
    METHOD_BEAM_DP,
    AlignmentConfig,
    EventCandidate,
    TrakeAlignment,
    TrakePrediction,
    TrakeStructureError,
    align_trake,
    joint_trake_alignment,
)
from .trake_refinement import FrameBudget, TrakeSequenceRefiner, apply_refinement
from .text_encoder import (
    AutoCLIPTextEncoder,
    HashingTextEncoder,
    TextQueryEncoder,
    encode_many,
    encoder_status,
)

MAX_PREDICTIONS = 100


def _join_warnings(*parts: Optional[str]) -> Optional[str]:
    kept = [part for part in parts if part]
    return " ".join(kept) if kept else None


QUERY_TEMPLATES = (
    "{q}",
    "a photo of {q}.",
    "a video frame of {q}.",
    "mot khung hinh ve {q}.",
)


@dataclass
class AICPrediction:
    video_id: str
    frame_id: str
    keyframe_id: str
    score: float = 0.0
    answer: Optional[str] = None
    event_frame_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    # Local-refinement provenance for this candidate, or None when refinement did not
    # run. `frame_id` above stays the OFFICIAL submission frame under the default
    # preserve_coarse policy, whatever the refined visual frame turned out to be.
    refinement: Optional[dict] = None
    # Per-video-hypothesis Q&A provenance: which video the answer was produced from,
    # which evidence frames were used, and which backend answered.
    qa: Optional[dict] = None
    # TRAKE provenance: the complete event sequence, its status, and any recovered events.
    trake: Optional[dict] = None

    def row(self) -> list[str]:
        if self.event_frame_ids:
            return [self.video_id, *self.event_frame_ids]
        if self.answer is not None:
            return [self.video_id, self.frame_id, self.answer]
        return [self.video_id, self.frame_id]


@dataclass(frozen=True)
class KISSearchResult:
    """A KIS search plus everything Phase 5 can say about it without ground truth."""

    predictions: list[AICPrediction]
    refinement: Optional[LocalRefinementResult] = None
    coarse_search_ms: float = 0.0
    refinement_ms: float = 0.0
    total_search_ms: float = 0.0
    coarse: Optional["CoarseSearchResult"] = None
    cost: Optional[QueryCost] = None
    budget: Optional[dict] = None

    def diagnostics(self) -> dict:
        """Structural counters and timings. Never an accuracy measurement."""
        base = {
            "coarse_search_ms": round(self.coarse_search_ms, 3),
            "refinement_ms": round(self.refinement_ms, 3),
            "total_search_ms": round(self.total_search_ms, 3),
            "refinement_triggered": False,
            "candidates_refined": 0,
        }
        if self.refinement is not None:
            base.update(self.refinement.diagnostics)
        if self.coarse is not None:
            # Channel coverage counters; never named recall.
            base["channels"] = self.coarse.diagnostics
        base["coarse_search_ms"] = round(self.coarse_search_ms, 3)
        base["total_search_ms"] = round(self.total_search_ms, 3)
        if self.cost is not None:
            base["cost"] = self.cost.to_dict()
        # Present only when the experimental controller ran, so its absence is visible.
        base["adaptive_budget"] = self.budget or {"enabled": False}
        return base


@dataclass(frozen=True)
class CoarseSearchResult:
    """Coarse candidates plus which channels proposed them."""

    candidates: list[Candidate]
    union: Optional[ChannelUnion] = None
    query: Optional[QueryRepresentation] = None
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrakeSearchResult:
    """A TRAKE search: only complete sequences, plus what was discarded and why."""

    predictions: list[AICPrediction]
    matches: list[TemporalMatch]
    trake_predictions: tuple[TrakePrediction, ...] = ()
    discarded: tuple[TrakeAlignment, ...] = ()
    refinements: tuple = ()
    diagnostics: dict = field(default_factory=dict)

    def structural_summary(self) -> dict:
        """The invariants that must hold for every emitted row."""
        return {
            "returned_complete_predictions": len(self.predictions),
            "wrong_event_count_prediction_count": sum(
                1
                for prediction in self.predictions
                if len(prediction.event_frame_ids)
                != int(self.diagnostics.get("event_count", len(prediction.event_frame_ids)))
            ),
            "malformed_prediction_count": sum(
                1
                for prediction in self.trake_predictions
                if len(prediction.frame_ids) != prediction.event_count
            ),
            "cross_video_step_count": sum(
                1
                for prediction in self.trake_predictions
                for step in prediction.steps
                if step.video_id != prediction.video_id
            ),
            "discarded_incomplete_alignments": len(self.discarded),
        }


@dataclass
class AICLoadResult:
    entry: VideoIndexEntry
    stats: Optional[AICDatasetStats]
    build_seconds: float
    cache_hit: bool
    cache_valid: bool = True
    cache_legacy: bool = False
    cache_stale: bool = False
    cache_stale_reason: str | None = None
    cache_manifest_path: str | None = None
    cache_fingerprint: str | None = None
    cache_mismatches: list[dict] = field(default_factory=list)
    cache_warnings: list[dict] = field(default_factory=list)
    cache_created_at: str | None = None
    cache_code_version: str | None = None
    cache_manifest: CacheManifest | None = None

    def cache_status(self) -> dict:
        return {
            "exists": True,
            "hit": self.cache_hit,
            "valid": self.cache_valid,
            "legacy": self.cache_legacy,
            "stale": self.cache_stale,
            "stale_reason": self.cache_stale_reason,
            "manifest_path": self.cache_manifest_path,
            "fingerprint": self.cache_fingerprint,
            "created_at": self.cache_created_at,
            "code_version": self.cache_code_version,
            "mismatches": self.cache_mismatches,
            "warnings": self.cache_warnings,
        }


class AICCompetitionEngine:
    """One service for Textual KIS, Q&A, and TRAKE."""

    def __init__(
        self,
        entry: VideoIndexEntry,
        *,
        text_encoder: Optional[TextQueryEncoder] = None,
        query_templates: Sequence[str] = QUERY_TEMPLATES,
        fusion_depth: int = DEFAULT_FUSION_DEPTH,
        bm25_weight: float = 1.0,
        bm25_weight_high: float = 3.0,
        adaptive_bm25: bool = True,
        production_mode: Optional[bool] = None,
        allow_hashing_fallback: Optional[bool] = None,
        encoder_model_name: Optional[str] = None,
        device: Optional[str] = None,
        encoder_batch_size: Optional[int] = None,
        fusion_config: Optional[FusionConfig] = None,
        app_config: Optional[AppConfig] = None,
        frame_provider: Optional[FrameProvider] = None,
        frame_scorer: Optional[FrameScorer] = None,
        qa_answerer: Optional[VisualQAAnswerer] = None,
    ):
        self.entry = entry
        self.app_config = app_config or AppConfig()
        dim = int(getattr(getattr(entry.index, "_clip_index", None), "d", self.app_config.encoder.feature_dim) or self.app_config.encoder.feature_dim)
        effective_production = self.app_config.runtime.production_mode if production_mode is None else bool(production_mode)
        effective_allow_fallback = self.app_config.encoder.allow_hashing_fallback if allow_hashing_fallback is None else bool(allow_hashing_fallback)
        effective_model = encoder_model_name or self.app_config.encoder.model_name
        effective_device = device or (None if self.app_config.runtime.device == "auto" else self.app_config.runtime.device)
        effective_batch = int(encoder_batch_size or self.app_config.encoder.batch_size)
        if effective_production and isinstance(text_encoder, HashingTextEncoder):
            raise RuntimeError("HashingTextEncoder cannot be used when production_mode=true.")
        self.feature_dim = dim
        self.production_mode = bool(effective_production)
        self.text_encoder = text_encoder or AutoCLIPTextEncoder(
            feature_dim=dim,
            model_name=effective_model,
            device=effective_device,
            batch_size=effective_batch,
            production_mode=effective_production,
            allow_hashing_fallback=effective_allow_fallback,
        )
        self.query_templates = list(query_templates)
        self.fusion_depth = fusion_depth
        self.bm25_weight = bm25_weight
        self.bm25_weight_high = bm25_weight_high
        self.adaptive_bm25 = adaptive_bm25
        self.fusion_config = fusion_config or self.app_config.fusion
        self.ranking_config = self.app_config.ranking
        self.refinement_config = self.app_config.refinement
        self.trake_config = self.app_config.trake
        self.qa_config = self.app_config.qa
        # R1. Disabled by default; every gated path checks `.enabled` before running.
        self.budget_config = self.app_config.adaptive_budget
        self.config_hash = config_hash(self.app_config)
        # Visual access is on demand only: constructing this touches no video file.
        self.frame_provider = frame_provider or FrameProvider(
            self.app_config.dataset.root, cache_dir=self.app_config.dataset.frame_cache_dir
        )
        # The refiner and its scorer belong to THIS engine, and the engine belongs to one
        # runtime generation. That is what stops a refiner from ever pairing with the
        # frame provider or data root of a different generation.
        self.frame_scorer = frame_scorer if frame_scorer is not None else self._build_frame_scorer()
        self.local_refiner = LocalFrameRefiner(
            self.refinement_config,
            frame_provider=self.frame_provider,
            scorer=self.frame_scorer,
        )
        # Q&A backend selection is cheap and loads nothing; `auto` never downloads.
        self.qa_answerer = qa_answerer if qa_answerer is not None else build_qa_answerer(
            self.qa_config.backend_type,
            model_name=self.qa_config.backend_model_name,
            device=self.qa_config.backend_device,
            max_answer_tokens=self.qa_config.max_answer_tokens,
            temperature=self.qa_config.answer_temperature,
            max_images=self.qa_config.evidence_frame_count,
        )
        self._raws_by_video: Optional[dict[str, list]] = None
        # Channel indices are built once, on first use, and reused for every query.
        self._channels: Optional[RetrievalChannelRegistry] = None
        # Bounded, in-process, never persisted. The encoder is deterministic, so a hit
        # returns the identical vector; the key carries model, dimension and template
        # identity so it cannot survive a change to any of them.
        self._query_embeddings = QueryEmbeddingCache(
            self.app_config.runtime.query_embedding_cache_size
        )
        self._encoder_model_name = str(effective_model)
        self._template_signature = template_signature(self.query_templates)
        # Process-lifetime counters; per-query cost is their delta across one search.
        self._encode_calls = 0
        self._encode_vectors = 0
        self._encode_cache_hits = 0

    def _build_frame_scorer(self) -> Optional[FrameScorer]:
        """Construct the configured visual scorer without loading its weights.

        Construction is deliberately cheap; the checkpoint loads on the first refined
        query. A configuration error is fatal only when refinement is actually enabled,
        so a disabled-refinement deployment is never blocked by scorer settings.
        """
        config = self.refinement_config
        if config.effective_mode == MODE_DISABLED:
            return None
        try:
            return build_frame_scorer(
                config.scorer_type,
                model_name=config.scorer_model_name,
                device=None if config.scorer_device == "auto" else config.scorer_device,
                batch_size=config.batch_size,
                expected_dim=self.feature_dim,
            )
        except Exception:
            if config.scorer_required or self.production_mode:
                raise
            return None

    def refinement_status(self, *, initialize: bool = False) -> dict:
        """Refinement configuration and scorer state.

        `initialize=False` (the default, and what `/health` uses) never loads the CLIP
        checkpoint: the answer is simply `not_loaded` until a query needs it.
        """
        status = dict(self.refinement_config.summary())
        status["scorer"] = self.local_refiner.scorer_status(initialize=initialize)
        return status

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        rebuild: bool = False,
        limit_videos: Optional[int] = None,
        limit_frames_per_video: Optional[int] = None,
        load_objects: Optional[bool] = None,
        include_media_text: Optional[bool] = None,
        verify_keyframes: Optional[bool] = None,
        index_kind: Optional[str] = None,
        text_encoder: Optional[TextQueryEncoder] = None,
        frame_scorer: Optional[FrameScorer] = None,
        qa_answerer: Optional[VisualQAAnswerer] = None,
        production_mode: Optional[bool] = None,
        allow_hashing_fallback: Optional[bool] = None,
        allow_stale_cache: Optional[bool] = None,
        encoder_model_name: Optional[str] = None,
        device: Optional[str] = None,
        encoder_batch_size: Optional[int] = None,
        app_config: Optional[AppConfig] = None,
    ) -> tuple["AICCompetitionEngine", AICLoadResult]:
        app_config = app_config or AppConfig()
        data_root = data_root if data_root is not None else app_config.dataset.root
        cache_dir = Path(cache_dir if cache_dir is not None else app_config.dataset.cache_dir)
        load_objects = app_config.dataset.load_objects if load_objects is None else bool(load_objects)
        include_media_text = app_config.dataset.include_media_text if include_media_text is None else bool(include_media_text)
        verify_keyframes = app_config.dataset.verify_keyframes if verify_keyframes is None else bool(verify_keyframes)
        index_kind = index_kind or app_config.dataset.index_kind
        effective_production = app_config.runtime.production_mode if production_mode is None else bool(production_mode)
        allow_stale = app_config.cache.allow_stale_cache if allow_stale_cache is None else bool(allow_stale_cache)
        if effective_production:
            allow_stale = False

        start = time.perf_counter()
        entry_dir = cache_dir / "entry"
        entry_path = entry_dir / "entry.pkl"
        manifest_path = cache_dir / CACHE_MANIFEST_FILENAME
        stats_path = cache_dir / "dataset_stats.json"
        dataset_report_path = cache_dir / "dataset_report.json"
        expected = cache_build_options_from_config(
            app_config,
            data_root=data_root,
            load_objects=load_objects,
            include_media_text=include_media_text,
            verify_keyframes=verify_keyframes,
            index_kind=index_kind,
            limit_videos=limit_videos,
            limit_frames_per_video=limit_frames_per_video,
        )

        def read_stats() -> Optional[AICDatasetStats]:
            candidates = (stats_path, cache_dir / "stats.json")
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                try:
                    raw = json.loads(candidate.read_text(encoding="utf-8"))
                    return AICDatasetStats(
                        videos=int(raw.get("video_count", raw.get("videos", 0))),
                        frames=int(raw.get("frame_count", raw.get("frames", 0))),
                        missing_keyframes=int(raw.get("missing_keyframes", 0)),
                        missing_objects=int(raw.get("missing_objects", 0)),
                        missing_videos=int(raw.get("missing_videos", 0)),
                        feature_dim=int(raw.get("feature_dim", 0)),
                        feature_dtype=str(raw.get("feature_dtype", "unknown")),
                        dataset_validated=bool(raw.get("dataset_validated", False)),
                        dataset_report_path=raw.get("dataset_report_path"),
                        invalid_video_count=int(raw.get("invalid_video_count", 0)),
                        record_schema_version=int(raw.get("record_schema_version", 1)),
                        scope_mode=str(raw.get("scope_mode", "patterns")),
                        scope_include_patterns=tuple(raw.get("scope_include_patterns", ("*",))),
                        scope_exclude_patterns=tuple(raw.get("scope_exclude_patterns", ())),
                        discovered_videos=int(raw.get("discovered_video_count", 0)),
                        excluded_videos=int(raw.get("excluded_video_count", 0)),
                        selected_video_ids_hash=str(raw.get("selected_video_ids_hash", "")),
                        retrieval_valid_videos=int(raw.get("retrieval_valid_videos", 0)),
                        visual_accessible_videos=int(raw.get("visual_accessible_videos", 0)),
                        refinement_ready_videos=int(raw.get("refinement_ready_videos", 0)),
                        keyframe_jpeg_backed_videos=int(raw.get("keyframe_jpeg_backed_videos", 0)),
                        video_fallback_videos=int(raw.get("video_fallback_videos", 0)),
                    )
                except Exception:
                    continue
            return None

        def load_entry() -> VideoIndexEntry:
            try:
                return VideoIndexEntry.load(entry_dir)
            except Exception as exc:
                raise CacheManifestError(
                    f"Cannot load cache entry {entry_path}: {type(exc).__name__}: {exc}"
                ) from exc

        def make_engine(entry: VideoIndexEntry) -> "AICCompetitionEngine":
            return cls(
                entry,
                text_encoder=text_encoder,
                frame_scorer=frame_scorer,
                qa_answerer=qa_answerer,
                production_mode=production_mode,
                allow_hashing_fallback=allow_hashing_fallback,
                encoder_model_name=encoder_model_name,
                device=device,
                encoder_batch_size=encoder_batch_size,
                app_config=app_config,
            )

        if not rebuild:
            report = inspect_cache(
                cache_dir,
                expected,
                validate_data_signature=app_config.cache.validate_data_signature,
                code_version_policy=app_config.cache.code_version_policy,
                current_code_version=detect_code_version(),
            )
            if report["legacy"]:
                if not allow_stale:
                    mode = "production mode" if effective_production else "default policy"
                    raise LegacyCacheError(
                        f"Legacy cache rejected by {mode}: {entry_path}. "
                        "Rebuild it to create cache_manifest.json."
                    )
                import warnings

                warning = {
                    "field": "manifest",
                    "expected": CACHE_MANIFEST_FILENAME,
                    "actual": None,
                    "severity": "warning",
                    "message": "Loading explicitly allowed legacy cache without a manifest.",
                }
                warnings.warn(warning["message"], RuntimeWarning, stacklevel=2)
                entry = load_entry()
                result = AICLoadResult(
                    entry=entry,
                    stats=read_stats(),
                    build_seconds=time.perf_counter() - start,
                    cache_hit=True,
                    cache_valid=False,
                    cache_legacy=True,
                    cache_stale=True,
                    cache_stale_reason=warning["message"],
                    cache_manifest_path=str(manifest_path),
                    cache_mismatches=[],
                    cache_warnings=[warning],
                )
                return make_engine(entry), result
            if report["corrupt"]:
                raise CacheManifestError(
                    f"Corrupt cache manifest {manifest_path}: {mismatch_message(report)}"
                )
            if report["manifest_exists"]:
                if report["stale"] and not allow_stale:
                    raise StaleCacheError(
                        f"Stale cache rejected; mismatched field(s): {mismatch_message(report)}. "
                        "Use --rebuild to create a compatible cache."
                    )
                if report["stale"]:
                    import warnings

                    warnings.warn(
                        f"Loading explicitly allowed stale cache; mismatched field(s): "
                        f"{mismatch_message(report)}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                manifest = CacheManifest.from_dict(report["manifest"])
                entry = load_entry()
                entry_dim = int(getattr(getattr(entry.index, "_clip_index", None), "d", 0) or 0)
                if entry_dim != manifest.feature_dim:
                    raise CacheManifestError(
                        f"Cache entry feature_dim mismatch: manifest={manifest.feature_dim}, "
                        f"entry={entry_dim}."
                    )
                if int(entry.num_indexed) != manifest.frame_count:
                    raise CacheManifestError(
                        f"Cache entry frame_count mismatch: manifest={manifest.frame_count}, "
                        f"entry={entry.num_indexed}."
                    )
                entry_video_count = len({raw.video_id for raw in entry.raws.values()})
                if entry_video_count != manifest.video_count:
                    raise CacheManifestError(
                        f"Cache entry video_count mismatch: manifest={manifest.video_count}, "
                        f"entry={entry_video_count}."
                    )
                result = AICLoadResult(
                    entry=entry,
                    stats=read_stats(),
                    build_seconds=time.perf_counter() - start,
                    cache_hit=True,
                    cache_valid=bool(report["valid"]),
                    cache_stale=bool(report["stale"]),
                    cache_stale_reason=report.get("stale_reason"),
                    cache_manifest_path=str(manifest_path),
                    cache_fingerprint=manifest.cache_fingerprint,
                    cache_mismatches=list(report["hard_mismatches"]),
                    cache_warnings=list(report["warnings"]),
                    cache_created_at=manifest.created_at_utc,
                    cache_code_version=manifest.code_version,
                    cache_manifest=manifest,
                )
                return make_engine(entry), result

        loader = AICDatasetLoader(
            data_root,
            load_objects=load_objects,
            include_media_text=include_media_text,
            verify_keyframes=verify_keyframes,
            index_kind=index_kind,
            app_config=app_config,
        )
        entry, stats = loader.build_entry(
            limit_videos=limit_videos,
            limit_frames_per_video=limit_frames_per_video,
        )
        if int(stats.feature_dim) != int(app_config.encoder.feature_dim):
            raise CacheManifestError(
                f"Built feature_dim {stats.feature_dim} does not match configured "
                f"encoder.feature_dim {app_config.encoder.feature_dim}."
            )
        if loader.last_report is None or not loader.last_report.valid_for_index_build:
            raise CacheManifestError("Dataset build completed without a valid inspection report.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_report(loader.last_report, dataset_report_path)
        stats = replace(stats, dataset_report_path=str(dataset_report_path))
        if manifest_path.exists():
            manifest_path.unlink()
        entry.save(entry_dir)
        elapsed = time.perf_counter() - start
        resolved_config_path = cache_dir / "resolved_config.json"
        resolved_config_path.write_text(
            json.dumps(
                {"config_hash": config_hash(app_config), "config": config_to_dict(app_config)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stats_payload = {
            "video_count": stats.videos,
            "frame_count": stats.frames,
            "feature_dim": stats.feature_dim,
            "feature_dtype": stats.feature_dtype,
            "missing_keyframes": stats.missing_keyframes,
            "missing_objects": stats.missing_objects,
            "missing_media_info": len(loader.last_report.missing_sources.get("media_info", ())),
            "missing_videos": stats.missing_videos,
            "dataset_validated": stats.dataset_validated,
            "dataset_report_path": stats.dataset_report_path,
            "invalid_video_count": stats.invalid_video_count,
            "record_schema_version": stats.record_schema_version,
            "scope_mode": stats.scope_mode,
            "scope_include_patterns": list(stats.scope_include_patterns),
            "scope_exclude_patterns": list(stats.scope_exclude_patterns),
            "discovered_video_count": stats.discovered_videos,
            "selected_video_count": stats.videos,
            "excluded_video_count": stats.excluded_videos,
            "selected_video_ids_hash": stats.selected_video_ids_hash,
            "retrieval_valid_videos": stats.retrieval_valid_videos,
            "visual_accessible_videos": stats.visual_accessible_videos,
            "refinement_ready_videos": stats.refinement_ready_videos,
            "keyframe_jpeg_backed_videos": stats.keyframe_jpeg_backed_videos,
            "video_fallback_videos": stats.video_fallback_videos,
            "build_seconds": elapsed,
        }
        stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        inspected_video_ids = sorted({raw.video_id for raw in entry.raws.values()})
        files = {
            "entry": "entry/entry.pkl",
            "index": "entry/index/meta.pkl",
            "clip_index": "entry/index/clip.hnsw",
            "config_snapshot": "resolved_config.json",
            "dataset_stats": "dataset_stats.json",
            "dataset_report": "dataset_report.json",
            "manifest": CACHE_MANIFEST_FILENAME,
        }
        files = {
            key: relative
            for key, relative in files.items()
            if key == "manifest" or (cache_dir / relative).is_file()
        }
        manifest = build_cache_manifest(
            app_config=app_config,
            data_root=Path(data_root),
            video_ids=inspected_video_ids,
            video_count=len(inspected_video_ids),
            frame_count=stats.frames,
            feature_dim=stats.feature_dim,
            feature_dtype=stats.feature_dtype,
            index_kind=index_kind,
            index_params=expected.index_params,
            record_schema_version=expected.record_schema_version,
            files=files,
            load_objects=load_objects,
            include_media_text=include_media_text,
            scope=app_config.dataset.scope,
        )
        write_cache_manifest_atomic(manifest, manifest_path)
        result = AICLoadResult(
            entry=entry,
            stats=stats,
            build_seconds=time.perf_counter() - start,
            cache_hit=False,
            cache_valid=True,
            cache_manifest_path=str(manifest_path),
            cache_fingerprint=manifest.cache_fingerprint,
            cache_created_at=manifest.created_at_utc,
            cache_code_version=manifest.code_version,
            cache_manifest=manifest,
        )
        return make_engine(entry), result

    def dataset_identity(self) -> dict:
        """Immutable dataset identity, so a caller can verify what this engine indexes.

        Exposed so the runtime state can assert that the engine, the frame provider,
        and the routes all describe the same dataset instead of trusting they do.
        """
        return {
            "data_root": str(self.app_config.dataset.root),
            "cache_dir": str(self.app_config.dataset.cache_dir),
            "config_hash": self.config_hash,
            "video_ids": sorted({raw.video_id for raw in self.entry.raws.values()}),
            "frame_count": int(self.entry.num_indexed),
        }

    def encode_query(self, query: str) -> np.ndarray:
        """Prompt-ensembled, L2-normalized query embedding, cached per engine.

        The cache is an exact-recomputation saver, not an approximation: the same key
        can only produce the vector the encoder would have produced. A TRAKE query alone
        re-asks one event text at several candidate depths, so the saved work is real.
        """
        key = QueryEmbeddingKey(
            query=str(query),
            model_name=self._encoder_model_name,
            feature_dim=int(self.feature_dim),
            template_signature=self._template_signature,
        )
        if self._query_embeddings.peek(key) is not None:
            self._encode_cache_hits += 1
        else:
            self._encode_calls += 1
            self._encode_vectors += len(self.query_templates)
        return self._query_embeddings.get_or_compute(key, lambda: self._encode_query_uncached(query))

    def _cost_snapshot(self) -> tuple[int, int, int]:
        return (self._encode_calls, self._encode_vectors, self._encode_cache_hits)

    def _record_encoder_cost(self, cost: QueryCost, before: tuple[int, int, int]) -> None:
        """Attribute encoder work done since `before` to this query."""
        cost.text_encoder_calls += self._encode_calls - before[0]
        cost.text_vectors_computed += self._encode_vectors - before[1]
        cost.text_encoder_cache_hits += self._encode_cache_hits - before[2]

    def _encode_query_uncached(self, query: str) -> np.ndarray:
        variants = encode_many(
            self.text_encoder,
            [tpl.format(q=query) for tpl in self.query_templates],
            self.feature_dim,
        )
        mean = np.mean(variants, axis=0).astype(np.float32)
        return l2_normalize(mean.reshape(1, -1))[0]

    def prewarm(self, *, force: bool = False) -> dict:
        """Load the text encoder now instead of inside the first query.

        This moves a one-off cost and nothing else: the model, its weights, the produced
        vectors and every ranking are identical either way. Off unless configured, so a
        unit test never pays for it. Failure is reported, never raised — a system that
        cannot prewarm can still serve, it just pays on the first query as before.
        """
        state = {
            "prewarm_enabled": bool(self.app_config.runtime.prewarm_enabled),
            "requested": bool(force or self.app_config.runtime.prewarm_enabled),
            "performed": False,
            "prewarm_ms": 0.0,
            "model_state": "not_loaded",
            "error": None,
        }
        if not state["requested"]:
            return state
        started = time.perf_counter()
        try:
            status = self.encoder_status(initialize=True)
            state["model_state"] = str(status.get("state") or ("ready" if status.get("ready") else "unknown"))
            state["performed"] = True
        except Exception as exc:  # noqa: BLE001 - a failed prewarm must not block serving
            state["model_state"] = "failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
        state["prewarm_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return state

    def query_cache_status(self) -> dict:
        """Bounded-cache diagnostics. Structural counters only."""
        return {
            "query_embeddings": self._query_embeddings.to_dict(),
            "model_name": self._encoder_model_name,
            "template_signature": self._template_signature,
            "persisted": False,
        }

    def encoder_status(self, *, initialize: bool = False) -> dict:
        status_fn = getattr(self.text_encoder, "status", None)
        if initialize and callable(status_fn):
            try:
                status = status_fn(initialize=True)
            except TypeError:
                status = status_fn()
        else:
            status = encoder_status(self.text_encoder, self.feature_dim)
        return status.to_dict()

    @property
    def channels(self) -> RetrievalChannelRegistry:
        """Lazily built channel registry; the indices are built once per engine."""
        if self._channels is None:
            self._channels = build_channel_registry(
                self.entry, self.encode_query, self.app_config.retrieval_channels
            )
        return self._channels

    def channel_status(self) -> dict:
        """Which channels exist, which are enabled, and which have real data."""
        return self.channels.status()

    def search_candidates(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[dict] = None,
        depth_scale: float = 1.0,
        context: Optional[QueryExecutionContext] = None,
        cost: Optional[QueryCost] = None,
    ) -> list[Candidate]:
        coarse = self.search_candidates_detailed(
            query, top_k=top_k, filters=filters, depth_scale=depth_scale, context=context
        )
        if cost is not None:
            self._record_channel_cost(cost, coarse)
        return coarse.candidates

    def search_candidates_detailed(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[dict] = None,
        depth_scale: float = 1.0,
        context: Optional[QueryExecutionContext] = None,
    ) -> "CoarseSearchResult":
        """Coarse retrieval over the union of every enabled, available channel.

        Before Phase 9 the pool was CLIP union BM25, and objects/metadata could only
        rescore what was already inside it: a frame whose only strong signal was its
        detector labels had no way in. Now each channel proposes its own candidates and
        the pool is their union, with per-channel provenance kept all the way through.
        """
        started = time.perf_counter()
        top_k = max(1, min(self.fusion_depth, int(top_k)))
        # Normalization is deterministic, so within one request the same query text can
        # reuse its representation instead of re-folding accents and re-expanding terms.
        representation = (
            normalize_query(query)
            if context is None
            else context.representation(query, lambda: normalize_query(query))
        )
        depths = channel_depths(self.app_config.retrieval_channels, scale=depth_scale)
        policy: dict[str, str] = {}
        if self.budget_config.enabled and self.budget_config.channel_policy_enabled:
            # R1 experiment: ask a channel with nothing to match on less deeply. CLIP is
            # never reduced. Off by default; the baseline queries every channel fully.
            policy = channel_policy(
                representation,
                enabled_channels=[
                    name
                    for name, info in self.channels.status().items()
                    if info.get("usable")
                ],
            )
            depths = apply_channel_policy(depths, policy)
        union = self.channels.search(
            representation,
            depths=depths,
            default_top_k=self.fusion_depth,
        )
        channel_ms = (time.perf_counter() - started) * 1000.0

        # A filter restricts which rows may participate; it is applied to the pool rather
        # than to each channel, so a channel never has to know about it.
        allowed: Optional[set[str]] = None
        if filters:
            retriever = CoarseRetriever(self.entry.index, fusion_depth=self.fusion_depth)
            rows = retriever._apply_filters(filters)
            if rows is not None:
                allowed = {self.entry.index.ids[row] for row in rows}

        fusion_started = time.perf_counter()
        pooled = [
            item
            for item in union.candidates
            if allowed is None or item.keyframe_id in allowed
        ]
        ranked: list[RankedCandidate] = []
        provenance: dict[str, PooledCandidate] = {}
        for item in pooled:
            raw = self.entry.raws.get(item.keyframe_id)
            if raw is None:
                continue
            provenance[item.keyframe_id] = item
            object_evidence = item.by_channel.get(CHANNEL_OBJECTS)
            metadata_evidence = item.by_channel.get(CHANNEL_METADATA)
            ranked.append(RankedCandidate(
                video_id=raw.video_id,
                frame_id=official_frame_id(self.entry, item.keyframe_id),
                keyframe_id=item.keyframe_id,
                timestamp=raw.timestamp,
                keyframe_path=raw.image_path,
                # Channel scores are already normalized onto a comparable scale, so
                # fusion never adds a cosine to a BM25 sum to a detector confidence.
                dense_score=item.normalized_score(CHANNEL_CLIP),
                object_score=item.normalized_score(CHANNEL_OBJECTS),
                metadata_score=item.normalized_score(CHANNEL_METADATA),
                bm25_score=item.normalized_score(CHANNEL_BM25),
                evidence=CandidateEvidence(
                    matched_objects=() if object_evidence is None else object_evidence.evidence,
                    metadata_terms=() if metadata_evidence is None else metadata_evidence.evidence,
                    sparse_terms=tuple(item.channels),
                ),
            ))
        fused = fuse_candidates(query, ranked, self.fusion_config)
        row_by_id = {
            keyframe_id: self.entry.index._id_to_row[keyframe_id]
            for keyframe_id in provenance
            if keyframe_id in self.entry.index._id_to_row
        }
        out: list[Candidate] = []
        for item in fused.candidates[:top_k]:
            # Use the identity fusion carried through. Rebuilding it from
            # (video_id, frame_id) would alias the 192 official videos that repeat a
            # frame_idx and silently drop or mis-attribute their candidates.
            keyframe_id = item.keyframe_id
            row = None if keyframe_id is None else row_by_id.get(keyframe_id)
            if row is None:
                continue
            candidate = Candidate(
                keyframe_id=keyframe_id,
                row=row,
                score=item.fused_score,
                video_id=item.video_id,
                timestamp=item.timestamp,
                source_ranks={},
            )
            candidate.score_breakdown = {
                "dense": item.dense_score,
                "object": item.object_score,
                "metadata": item.metadata_score,
                "bm25": item.bm25_score,
                "fused": item.fused_score,
            }
            candidate.evidence = item.evidence
            pooled_item = provenance.get(keyframe_id)
            # Which channels found this candidate survives all the way to the response,
            # so an object-only or metadata-only candidate stays identifiable as such.
            candidate.channels = () if pooled_item is None else pooled_item.channels
            candidate.channel_evidence = (
                {} if pooled_item is None else pooled_item.to_dict()
            )
            out.append(candidate)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
        diagnostics = dict(union.diagnostics)
        diagnostics["channel_search_ms"] = round(channel_ms, 3)
        diagnostics["fusion_ms"] = round(fusion_ms, 3)
        diagnostics["total_coarse_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        diagnostics["filtered_out"] = len(union.candidates) - len(pooled)
        diagnostics["returned"] = len(out)
        diagnostics["query"] = representation.to_dict()
        if policy:
            diagnostics["channel_policy"] = {"policy": policy, "depths": depths}
        return CoarseSearchResult(
            candidates=out,
            union=union,
            query=representation,
            diagnostics=diagnostics,
        )
    def search_kis(
        self, query: str, *, top_k: Optional[int] = None, refine: Optional[bool] = None
    ) -> list[AICPrediction]:
        return self.search_kis_detailed(query, top_k=top_k, refine=refine).predictions

    def search_kis_detailed(
        self, query: str, *, top_k: Optional[int] = None, refine: Optional[bool] = None
    ) -> KISSearchResult:
        """Coarse retrieval, then bounded local refinement, then Top-100 allocation.

        Refinement runs BEFORE the final allocation so local evidence can actually
        rerank; refining after truncation would only relabel an already fixed list.
        `refine=False` forces refinement off for an A/B comparison; `None` follows the
        configured mode.
        """
        started = time.perf_counter()
        cost = QueryCost(task="kis", query=str(query))
        encoder_before = self._cost_snapshot()
        requested = max(1, min(MAX_PREDICTIONS, int(top_k if top_k is not None else self.ranking_config.final_top_k)))
        coarse = self.search_candidates_detailed(
            query, top_k=min(self.fusion_depth, max(requested * 3, 100))
        )
        pool = coarse.candidates
        coarse_ms = (time.perf_counter() - started) * 1000.0
        self._record_channel_cost(cost, coarse)

        refinement: Optional[LocalRefinementResult] = None
        refinement_ms = 0.0
        if refine is not False and self.refinement_config.effective_mode != MODE_DISABLED:
            refine_started = time.perf_counter()
            refinement = self._refine_candidates(query, pool)
            refinement_ms = (time.perf_counter() - refine_started) * 1000.0
            self._record_refinement_cost(cost, refinement, refinement_ms)
        by_keyframe = {} if refinement is None else refinement.by_keyframe()

        def nearby(anchor: Candidate, offsets: Sequence[int]) -> list[Candidate]:
            rows = [
                row for row, video in enumerate(self.entry.index.video_ids)
                if video == anchor.video_id
            ]
            try:
                position = rows.index(anchor.row)
            except ValueError:
                return []
            out: list[Candidate] = []
            for offset in offsets:
                target = position + offset
                if 0 <= target < len(rows):
                    row = rows[target]
                    out.append(Candidate(
                        keyframe_id=self.entry.index.ids[row],
                        row=row,
                        score=anchor.score * 0.98,
                        video_id=anchor.video_id,
                        timestamp=self.entry.index.timestamps[row],
                        source_ranks={"neighbor": abs(offset)},
                    ))
            return out

        # R1, off by default: bounded extra work chosen by uncertainty and by where the
        # official cutoffs make an improvement worth the most. With the budget disabled
        # this is skipped entirely and the baseline path below is unchanged.
        budget_info: Optional[dict] = None
        if self.budget_config.enabled:
            budget_info = self._run_kis_budget(query, coarse, pool, cost)

        allocation: dict[str, Any] = {}
        allocator_config = replace(self.ranking_config, final_top_k=requested, top_k=None)
        if self.budget_config.enabled and self.budget_config.cutoff_aware_enabled:
            ranked = cutoff_aware_top100(
                pool,
                video_id=lambda c: c.video_id,
                frame_id=lambda c: int(official_frame_id(self.entry, c.keyframe_id)),
                score=lambda c: c.score,
                neighbors=nearby,
                config=allocator_config,
                tail_min_videos=int(self.budget_config.cutoff_tail_min_videos),
                diagnostics=allocation,
            )
        else:
            ranked = video_aware_top100(
                pool,
                video_id=lambda c: c.video_id,
                frame_id=lambda c: int(official_frame_id(self.entry, c.keyframe_id)),
                score=lambda c: c.score,
                neighbors=nearby,
                config=allocator_config,
            )
        predictions = [
            self._from_candidate(c, refinement=by_keyframe.get(c.keyframe_id)) for c in ranked
        ]
        if budget_info is not None:
            budget_info["allocation"] = allocation
        self._record_encoder_cost(cost, encoder_before)
        cost.total_wall_ms = (time.perf_counter() - started) * 1000.0
        return KISSearchResult(
            predictions=predictions,
            refinement=refinement,
            coarse_search_ms=coarse_ms,
            refinement_ms=refinement_ms,
            total_search_ms=(time.perf_counter() - started) * 1000.0,
            coarse=coarse,
            cost=cost,
            budget=budget_info,
        )

    def _run_kis_budget(
        self, query: str, coarse: "CoarseSearchResult", pool: list[Candidate], cost: QueryCost
    ) -> dict:
        """R1 controller for one KIS query. Experimental; never runs when disabled.

        Order matters: the official-grid stage is bought first because it is ~100x
        cheaper than decoding the same neighbourhood, and because everything it can
        promote already carries an official `frame_idx`. Only what the grid cannot settle
        is worth paying an MP4 decode for.
        """
        config = self.budget_config
        ledger = BudgetLedger(max_cost_units=float(config.max_cost_units))
        channel_heads = []
        if coarse.union is not None:
            for name, info in (coarse.union.diagnostics.get("channels") or {}).items():
                head = info.get("head") or []
                if head:
                    channel_heads.append([str(item) for item in head])
        signals = kis_uncertainty(
            pool,
            channel_heads=channel_heads,
            use_margin=bool(config.uncertainty_margin),
            use_channel_disagreement=bool(config.uncertainty_channel_disagreement),
            use_temporal_ambiguity=bool(config.uncertainty_temporal_ambiguity),
        )
        info: dict[str, Any] = {
            "enabled": True,
            "uncertainty": signals.to_dict(),
            "official_grid": None,
            "actions": [],
        }

        actions: list[BudgetAction] = []
        if config.official_grid_enabled and pool:
            examined = min(int(config.official_grid_max_candidates), len(pool))
            actions.append(
                BudgetAction(
                    name=ACTION_OFFICIAL_GRID_REFINE,
                    target="top_candidates",
                    units=examined * max(1, int(config.official_grid_neighbors) * 2),
                    rank=1,
                    uncertainty=signals.uncertainty,
                    # Reading indexed vectors cannot make an answer worse and cannot
                    # produce an unsubmittable frame, so its proxy is 1.0 — that is a
                    # statement about safety, not about expected accuracy.
                    expected_gain_proxy=1.0,
                    detail={"candidates": examined},
                )
            )
        if config.progressive_video_enabled and pool:
            budget_frames = sum(int(value) for value in config.progressive_stage_frames)
            actions.append(
                BudgetAction(
                    name=ACTION_DENSE_TEMPORAL_ZOOM,
                    target=str(getattr(pool[0], "keyframe_id", "")),
                    units=budget_frames,
                    rank=1,
                    uncertainty=signals.uncertainty,
                    expected_gain_proxy=1.0,
                    detail={"frames": budget_frames},
                )
            )

        taken = allocate(actions, ledger)
        for action in taken:
            if action.name == ACTION_OFFICIAL_GRID_REFINE:
                grid = self._refine_official_grid(coarse, pool, int(action.detail["candidates"]))
                info["official_grid"] = grid
                cost.add_channel_search(
                    "official_grid", candidates=int(grid.get("vectors_read", 0)), ms=0.0
                )
            elif action.name == ACTION_DENSE_TEMPORAL_ZOOM:
                info["progressive_video"] = self._progressive_probe(
                    query, pool[0], int(action.detail["frames"]), cost
                )
        info["actions"] = ledger.to_dict()
        return info

    def _progressive_probe(
        self, query: str, candidate: Candidate, budget_frames: int, cost: QueryCost
    ) -> dict:
        """Staged MP4 sampling around one candidate, under a hard frame budget.

        The stage plan and the stop margin come from configuration; the decode and the
        scorer are the same ones the fixed refiner uses. When no scorer or no MP4 is
        available nothing is decoded and the refusal is reported — a missing capability
        is never silently treated as "found nothing".
        """
        config = self.budget_config
        scorer = self.frame_scorer
        if scorer is None:
            return {"skipped_reason": "no visual scorer configured", "frames_scored": 0}
        # `not_loaded` is lazy, not broken: the scorer loads on first use, and
        # `prepare_query` below is what actually decides. Only a scorer that has already
        # reported itself unusable is refused here.
        status = scorer.status()
        if status.state == SCORER_STATE_UNAVAILABLE:
            return {
                "skipped_reason": f"visual scorer unavailable: {status.fallback_reason or status.state}",
                "frames_scored": 0,
            }
        try:
            prepared = scorer.prepare_query(str(query))
        except Exception as exc:  # noqa: BLE001 - a scorer that cannot load is reported
            return {
                "skipped_reason": f"scorer could not prepare the query: {type(exc).__name__}",
                "frames_scored": 0,
            }
        raw = self.entry.raws.get(candidate.keyframe_id)
        if raw is None or raw.frame_idx is None:
            return {"skipped_reason": "candidate has no official frame", "frames_scored": 0}
        try:
            metadata = self.frame_provider.video_metadata(candidate.video_id)
        except Exception as exc:  # noqa: BLE001 - a missing video is reported, not raised
            return {"skipped_reason": f"video metadata unavailable: {type(exc).__name__}", "frames_scored": 0}
        if metadata is None or not getattr(metadata, "fps", 0):
            return {"skipped_reason": "video unavailable", "frames_scored": 0}

        fps = float(metadata.fps)
        frame_count = int(getattr(metadata, "frame_count", 0) or 0)
        window = max(1, int(round(float(self.refinement_config.window_before_s) * fps)))
        anchor = int(raw.frame_idx)
        low = max(0, anchor - window)
        high = anchor + window
        if frame_count > 0:
            high = min(high, frame_count - 1)

        decoded_total = 0

        def score_frames(indices):
            nonlocal decoded_total
            started = time.perf_counter()
            frames, _ = self.frame_provider.decode_frames(
                candidate.video_id, list(indices), source_video=raw.source_video
            )
            decode_ms = (time.perf_counter() - started) * 1000.0
            usable = [frame for frame in frames if getattr(frame, "image", None) is not None]
            decoded_total += len(usable)
            cost.add_decode(requested=len(list(indices)), decoded=len(usable), ms=decode_ms)
            if not usable:
                return {}
            embed_started = time.perf_counter()
            try:
                scores = scorer.score_frames(prepared, [frame.image for frame in usable])
            except Exception:  # noqa: BLE001 - degrades to coarse behaviour
                return {}
            cost.add_image_embeddings(
                len(usable), ms=(time.perf_counter() - embed_started) * 1000.0
            )
            # Keyed by the index the plan ASKED for: that is the identity the sampler
            # tracks, and the decoder's own landing point is recorded separately.
            return {
                int(frame.requested_frame_idx): float(value)
                for frame, value in zip(usable, scores)
            }

        result = progressive_sample(
            anchor=anchor,
            low=low,
            high=high,
            budget=int(budget_frames),
            stage_frames=tuple(int(v) for v in config.progressive_stage_frames),
            stop_margin=float(config.progressive_stop_margin),
            score_frames=score_frames,
            fps=fps,
        )
        payload = result.to_dict()
        payload["video_id"] = candidate.video_id
        payload["coarse_frame_idx"] = anchor
        payload["frames_decoded"] = decoded_total
        # Evidence only. A decoded frame index is NOT an official frame and never becomes
        # the submitted one; `frame_output_policy` stays `preserve_coarse`.
        payload["applied_to_submission"] = False
        return payload

    def _refine_official_grid(
        self, coarse: "CoarseSearchResult", pool: list[Candidate], candidates: int
    ) -> dict:
        """Score each strong candidate's official neighbours from indexed vectors."""
        refiner = OfficialGridRefiner(
            self.entry,
            neighbors=int(self.budget_config.official_grid_neighbors),
            max_candidates=int(self.budget_config.official_grid_max_candidates),
        )
        try:
            vector = self.encode_query(coarse.query.dense_query if coarse.query else "")
        except Exception as exc:  # noqa: BLE001 - a missing encoder skips the stage
            return {"skipped_reason": f"query vector unavailable: {type(exc).__name__}"}
        result = refiner.refine(vector, pool, budget_candidates=candidates)
        payload = result.to_dict()
        # Evidence only. The grid never rewrites a candidate's submitted frame here:
        # promoting a neighbour is a ranking decision, and R1 does not make it until
        # ground truth can say whether it helps.
        payload["applied_to_ranking"] = False
        return payload

    def _record_channel_cost(self, cost: QueryCost, coarse: "CoarseSearchResult") -> None:
        """Attribute one coarse retrieval's per-channel work to this query."""
        channels = (coarse.union.diagnostics.get("channels") or {}) if coarse.union else {}
        for name, info in channels.items():
            if not info.get("searched", True):
                continue
            cost.add_channel_search(
                name,
                candidates=int(info.get("candidates_returned") or 0),
                ms=float(info.get("search_ms") or 0.0),
            )

    def _record_refinement_cost(
        self, cost: QueryCost, refinement: Optional[LocalRefinementResult], ms: float
    ) -> None:
        """Decoded frames and image embeddings are the expensive part of refinement."""
        if refinement is None:
            return
        diagnostics = dict(getattr(refinement, "diagnostics", {}) or {})
        decoded = int(diagnostics.get("frames_decoded", 0) or 0)
        cost.add_decode(requested=decoded, decoded=decoded, ms=ms)
        # Every decoded frame that was scored required one image embedding.
        cost.add_image_embeddings(int(diagnostics.get("frames_scored", decoded) or 0))

    def _refine_candidates(
        self, query: str, pool: Sequence[Candidate]
    ) -> Optional[LocalRefinementResult]:
        """Run local refinement over the coarse pool and fold the result into scores.

        The refined score is `coarse_fusion_score + alpha * (best_visual - coarse_visual)`.
        It is the *improvement over the coarse frame's own visual score*, not the raw
        local similarity: the coarse CLIP vector is already inside the fused score, so
        adding the raw local score back would count the same evidence twice, and the two
        scales are unrelated anyway. Candidates that were not refined keep their coarse
        score untouched and are never dropped.
        """
        if not pool:
            return None
        candidates: list[RefinementCandidate] = []
        for candidate in pool:
            raw = self.entry.raws.get(candidate.keyframe_id)
            if raw is None:
                continue
            candidates.append(
                RefinementCandidate(
                    keyframe_id=candidate.keyframe_id,
                    video_id=candidate.video_id,
                    coarse_frame_idx=None if raw.frame_idx is None else int(raw.frame_idx),
                    timestamp=float(candidate.timestamp),
                    coarse_score=float(candidate.score),
                    source_video=raw.source_video,
                )
            )
        if not candidates:
            return None
        result = self.local_refiner.refine(
            LocalRefinementRequest(query=str(query), candidates=tuple(candidates))
        )
        refined = result.by_keyframe()
        for candidate in pool:
            item = refined.get(candidate.keyframe_id)
            if item is None or not item.applied:
                continue
            breakdown = dict(getattr(candidate, "score_breakdown", {}) or {})
            breakdown["coarse_fused"] = float(candidate.score)
            breakdown["visual_gain"] = float(item.score_gain)
            breakdown["refined"] = float(item.refined_score)
            candidate.score_breakdown = breakdown
            candidate.score = float(item.refined_score)
        return result

    # ------------------------------------------------------------------------ Q&A

    def qa_status(self) -> dict:
        """Q&A configuration plus backend capability. Never loads a model."""
        status = self.qa_answerer.status()
        return {
            "enabled": bool(self.qa_config.enabled),
            "top_video_hypotheses": int(self.qa_config.top_video_hypotheses),
            "frame_hypotheses_per_video": int(self.qa_config.frame_hypotheses_per_video),
            "evidence_frame_count": int(self.qa_config.evidence_frame_count),
            "default_answer_type": canonical_answer_type(self.qa_config.default_answer_type),
            "abstain_enabled": bool(self.qa_config.abstain_enabled),
            "abstain_threshold": float(self.qa_config.abstain_threshold),
            "use_local_refinement": bool(self.qa_config.use_local_refinement),
            "refinement_candidate_budget": int(self.qa_config.refinement_candidate_budget),
            "refinement_max_frames": int(self.qa_config.refinement_max_frames),
            "backend_required": bool(self.qa_config.backend_required),
            "backend": status.to_dict(),
            "backend_type": status.backend_type,
            "backend_state": status.state,
            "visual_capable": status.visual_capable,
            "supports_multi_image": status.supports_multi_image,
            "production_ready": status.production_ready,
            "model_name": status.model_name,
            "device": status.device,
            "warning": status.warning,
        }

    def _raws_for_video(self, video_id: str) -> list:
        """Every mapped keyframe of one video. The entry is immutable, so cache it."""
        if self._raws_by_video is None:
            index: dict[str, list] = {}
            for raw in self.entry.raws.values():
                index.setdefault(raw.video_id, []).append(raw)
            for items in index.values():
                items.sort(key=lambda raw: float(raw.timestamp))
            self._raws_by_video = index
        return self._raws_by_video.get(str(video_id), [])

    def _qa_refiner(self) -> LocalFrameRefiner:
        """A refiner with the Q&A budget, not the KIS budget.

        Q&A refines several video hypotheses per question, so it uses its own much
        smaller budget: reusing Phase 5's KIS budget would cost minutes per question.
        """
        config = replace(
            self.refinement_config,
            mode=MODE_ALWAYS,
            top_hypotheses=max(1, int(self.qa_config.refinement_candidate_budget)),
            candidate_budget=max(1, int(self.qa_config.refinement_candidate_budget)),
            max_frames=max(1, int(self.qa_config.refinement_max_frames)),
        )
        return LocalFrameRefiner(
            config, frame_provider=self.frame_provider, scorer=self.frame_scorer
        )

    def _qa_evidence_pool(
        self, hypothesis: QAVideoHypothesis, window_s: float, scores: dict[str, float]
    ) -> list[QAEvidenceFrame]:
        """Candidate evidence for ONE video: retrieved frames plus nearby context.

        Every frame is drawn from `hypothesis.video_id`, so nothing from another video
        can enter the bundle. Availability is probed cheaply; pixels are not read here.
        """
        anchor = hypothesis.submission_frame
        if anchor is None:
            return []
        center = float(anchor.timestamp)
        pool: list[QAEvidenceFrame] = []
        for raw in self._raws_for_video(hypothesis.video_id):
            distance = abs(float(raw.timestamp) - center)
            if distance > float(window_s) and raw.id not in scores:
                continue
            visual = self.frame_provider.describe(raw)
            # A retrieved frame always outranks mere temporal proximity.
            score = scores.get(raw.id)
            if score is None:
                score = 0.001 / (1.0 + distance)
            pool.append(
                QAEvidenceFrame(
                    video_id=hypothesis.video_id,
                    frame_idx=None if raw.frame_idx is None else int(raw.frame_idx),
                    timestamp=float(raw.timestamp),
                    source=visual["image_source"],
                    keyframe_id=raw.id,
                    retrieval_score=float(score),
                    text=self.entry.caption_by_id.get(raw.id, "") or "",
                    objects=tuple(raw.objects or ()),
                    image_available=bool(visual["image_available"]),
                )
            )
        return pool

    def _qa_refine_video(
        self, refiner: LocalFrameRefiner, hypothesis: QAVideoHypothesis, query: str
    ) -> tuple[Optional[dict], Optional[QAEvidenceFrame]]:
        """Bounded local refinement for one video. Failure degrades, never raises."""
        candidates = tuple(
            RefinementCandidate(
                keyframe_id=frame.keyframe_id,
                video_id=frame.video_id,
                coarse_frame_idx=frame.frame_idx,
                timestamp=float(frame.timestamp),
                coarse_score=float(frame.score),
                source_video=getattr(self.entry.raws.get(frame.keyframe_id), "source_video", None),
            )
            for frame in hypothesis.frames
        )
        if not candidates:
            return None, None
        result = refiner.refine(LocalRefinementRequest(query=str(query), candidates=candidates))
        item = next((entry for entry in result.refinements if entry.applied), None)
        if item is None or item.best_visual_frame_idx is None:
            return (
                result.refinements[0].to_dict() if result.refinements else None
            ), None
        frame = self.frame_provider.get_video_frame(
            hypothesis.video_id, frame_idx=int(item.best_visual_frame_idx)
        )
        evidence = QAEvidenceFrame(
            video_id=hypothesis.video_id,
            frame_idx=int(item.best_visual_frame_idx),
            timestamp=float(item.best_timestamp or 0.0),
            source="local_refinement",
            keyframe_id=None,
            # Refined evidence is by construction the strongest local view, so it is
            # ranked above the coarse candidates it was derived from.
            retrieval_score=float(item.refined_score) + 1e-6,
            visual_score=item.best_visual_score,
            image_available=frame.available,
        )
        return item.to_dict(), evidence

    def answer_qa(
        self,
        event_text: str,
        question: str,
        *,
        top_k: Optional[int] = None,
        window_s: float = 8.0,
        answerer: Optional[object] = None,
        expected_answer_type: Optional[str] = None,
        retrieval_query_mode: Optional[str] = None,
        evidence_frame_count: Optional[int] = None,
        answer_confidence_threshold: Optional[float] = None,
        abstain_enabled: Optional[bool] = None,
        use_local_refinement: Optional[bool] = None,
    ) -> tuple[list[AICPrediction], dict]:
        """Answer a question independently for each top VIDEO hypothesis.

        The pre-Phase-6 implementation answered once for the globally top-ranked
        candidate and attached that answer to every prediction row, so a row for video B
        carried an answer produced from video A's frames. Here the unit of answering is
        one video: evidence is collected from that video only, the backend is called for
        that video only, and the answer is written onto that video's rows only. An
        answer can no longer reach another video because it never leaves its hypothesis.
        """
        started = time.perf_counter()
        cost = QueryCost(task="qa", query=f"{event_text} | {question}")
        encoder_before = self._cost_snapshot()
        qa_config = self.qa_config
        retrieval_query_mode = retrieval_query_mode or qa_config.retrieval_query_mode
        answer_type = canonical_answer_type(
            expected_answer_type
            if expected_answer_type not in (None, "")
            else qa_config.default_answer_type
        )
        qa_input = QAInput(event_text, question, answer_type)
        ground_query = build_retrieval_query(qa_input, retrieval_query_mode)
        requested = max(
            1, min(MAX_PREDICTIONS, int(top_k if top_k is not None else qa_config.max_answers))
        )
        evidence_budget = max(
            1,
            int(
                evidence_frame_count
                if evidence_frame_count is not None
                else qa_config.evidence_frame_count
            ),
        )
        threshold = float(
            answer_confidence_threshold
            if answer_confidence_threshold is not None
            else qa_config.abstain_threshold
        )
        abstain = qa_config.abstain_enabled if abstain_enabled is None else bool(abstain_enabled)
        refine = (
            qa_config.use_local_refinement
            if use_local_refinement is None
            else bool(use_local_refinement)
        )
        backend = answerer if answerer is not None else self.qa_answerer
        backend_status = backend.status()
        if qa_config.backend_required and not backend_status.visual_capable:
            raise RuntimeError(
                "qa.backend.required is set but the active Q&A backend "
                f"({backend_status.backend_type}) cannot look at images."
            )

        retrieval_started = time.perf_counter()
        # A pool deep enough to contain several distinct videos, not just the top rows.
        pool_depth = min(
            self.fusion_depth,
            max(
                requested * 3,
                int(qa_config.top_video_hypotheses) * int(qa_config.frame_hypotheses_per_video) * 5,
                100,
            ),
        )
        candidates = self.search_candidates(ground_query, top_k=pool_depth)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        if not candidates:
            return [], self._empty_qa_info(
                ground_query, retrieval_query_mode, answer_type, backend_status, started
            )

        def official_idx(candidate) -> Optional[int]:
            raw = self.entry.raws.get(candidate.keyframe_id)
            return None if raw is None or raw.frame_idx is None else int(raw.frame_idx)

        hypotheses = group_hypotheses_by_video(
            candidates,
            top_video_hypotheses=int(qa_config.top_video_hypotheses),
            frame_hypotheses_per_video=int(qa_config.frame_hypotheses_per_video),
            diversity_s=float(qa_config.evidence_temporal_diversity_s),
            support_bonus=float(qa_config.video_support_bonus),
            frame_idx_of=official_idx,
        )
        scores_by_keyframe = {c.keyframe_id: float(c.score) for c in candidates}
        best_score = max(float(item.retrieval_score) for item in hypotheses)
        second_score = (
            float(hypotheses[1].retrieval_score) if len(hypotheses) > 1 else 0.0
        )
        margin = (best_score - second_score) / max(abs(best_score), 1e-9)

        refiner = self._qa_refiner() if refine else None
        evidence_ms = 0.0
        refinement_ms = 0.0
        vqa_ms = 0.0
        refinement_calls = 0
        decode_failures = 0
        answered: list[tuple[QAVideoHypothesis, QAEvidenceBundle, QAAnswerResult, float]] = []
        # A VLM call is the most expensive action in the system, so the number of calls
        # is capped per query rather than following the hypothesis count. The cap applies
        # to a backend that really looks at images; a non-visual mock costs nothing and
        # is recorded as zero VLM calls, which is what the cost trace must show.
        vlm_budget = max(1, int(qa_config.max_vlm_calls_per_query))
        vlm_calls_used = 0
        budget_skipped = 0

        for hypothesis in hypotheses:
            if backend_status.visual_capable and vlm_calls_used >= vlm_budget:
                # Out of budget. No call, no answer, and explicitly not submittable —
                # a spending limit is never a reason to guess.
                budget_skipped += 1
                bundle = QAEvidenceBundle(
                    video_id=hypothesis.video_id,
                    question=question or ground_query,
                    expected_answer_type=answer_type,
                    frames=(),
                )
                result = QAAnswerResult(
                    video_id=hypothesis.video_id,
                    answer="",
                    normalized_answer=UNKNOWN_ANSWER,
                    status=ANSWER_STATUS_BUDGET_EXHAUSTED,
                    backend_type=backend_status.backend_type,
                    visual=False,
                    warning=(
                        f"Per-query VLM budget of {vlm_budget} call(s) was already spent; "
                        f"video {hypothesis.video_id!r} was not answered."
                    ),
                )
                answered.append((hypothesis, bundle, result, 0.0))
                continue
            refinement_payload: Optional[dict] = None
            refined_evidence: Optional[QAEvidenceFrame] = None
            if refiner is not None:
                refine_started = time.perf_counter()
                try:
                    refinement_payload, refined_evidence = self._qa_refine_video(
                        refiner, hypothesis, question or ground_query
                    )
                    refinement_calls += 1
                except Exception:  # noqa: BLE001 - refinement is optional evidence
                    refinement_payload, refined_evidence = None, None
                refinement_ms += (time.perf_counter() - refine_started) * 1000.0
            hypothesis = replace(hypothesis, refinement=refinement_payload)

            evidence_started = time.perf_counter()
            pool = self._qa_evidence_pool(hypothesis, window_s, scores_by_keyframe)
            if refined_evidence is not None:
                pool.append(refined_evidence)
            # Two independent ceilings: how many frames evidence selection may choose,
            # and how many of them a single backend call is allowed to carry.
            selected = select_evidence_frames(
                pool,
                count=min(evidence_budget, int(qa_config.max_visual_frames_per_call)),
                diversity_s=float(qa_config.evidence_temporal_diversity_s),
            )
            # Pixels are read ONLY for the selected frames, and only when the backend can
            # actually use them; a non-visual backend never triggers a decode.
            loaded: list[QAEvidenceFrame] = []
            for frame in selected:
                if not (backend_status.visual_capable and frame.image_available):
                    loaded.append(frame)
                    continue
                payload = self._qa_image_bytes(frame)
                if payload is None:
                    decode_failures += 1
                    loaded.append(frame)
                else:
                    loaded.append(replace(frame, image_bytes=payload))
            bundle = QAEvidenceBundle(
                video_id=hypothesis.video_id,
                question=question or ground_query,
                expected_answer_type=answer_type,
                frames=tuple(loaded),
            )
            evidence_ms += (time.perf_counter() - evidence_started) * 1000.0

            vqa_started = time.perf_counter()
            result = self._answer_one_hypothesis(
                backend, question or ground_query, bundle, answer_type
            )
            call_ms = (time.perf_counter() - vqa_started) * 1000.0
            vqa_ms += call_ms
            # Only a backend that actually looked at images spends VLM budget.
            if backend_status.visual_capable and result.status != ANSWER_STATUS_VISUAL_UNAVAILABLE:
                vlm_calls_used += 1
                cost.add_vlm_call(images=len(bundle.visual_frames), ms=call_ms)

            reliability = answer_reliability_score(
                backend=backend_status,
                evidence_count=len(bundle.frames),
                visual_evidence_count=len(bundle.visual_frames),
                answer=result.normalized_answer,
                expected_answer_type=answer_type,
                retrieval_margin=margin if hypothesis.rank == 1 else 0.0,
            )
            if (
                abstain
                and result.status == ANSWER_STATUS_ANSWERED
                and (is_unknown_answer(result.normalized_answer) or reliability < threshold)
                and is_unknown_answer(result.normalized_answer)
            ):
                result = replace(
                    result,
                    status=ANSWER_STATUS_ABSTAINED,
                    answer=result.answer,
                    normalized_answer=UNKNOWN_ANSWER,
                    warning=_join_warnings(
                        result.warning, "Abstained: no usable answer for this video."
                    ),
                )
            answered.append((hypothesis, bundle, result, reliability))

        predictions = self._qa_predictions(answered, requested, qa_config)
        info = self._qa_info(
            predictions=predictions,
            answered=answered,
            ground_query=ground_query,
            retrieval_query_mode=retrieval_query_mode,
            answer_type=answer_type,
            backend_status=backend_status,
            timings={
                "retrieval_ms": retrieval_ms,
                "evidence_selection_ms": evidence_ms,
                "refinement_ms": refinement_ms,
                "vqa_ms": vqa_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            },
            refinement_calls=refinement_calls,
            decode_failures=decode_failures,
        )
        self._record_encoder_cost(cost, encoder_before)
        cost.total_wall_ms = (time.perf_counter() - started) * 1000.0
        info["diagnostics"]["vlm_budget"] = {
            "max_vlm_calls_per_query": vlm_budget,
            "vlm_calls_used": vlm_calls_used,
            "max_visual_frames_per_call": int(qa_config.max_visual_frames_per_call),
            "hypotheses_skipped_for_budget": budget_skipped,
            "backend_visual_capable": bool(backend_status.visual_capable),
        }
        info["diagnostics"]["cost"] = cost.to_dict()
        return predictions, info

    def _qa_image_bytes(self, frame: QAEvidenceFrame) -> Optional[bytes]:
        """Load one evidence frame's pixels: BTC JPEG, MP4 fallback, or refined frame."""
        try:
            if frame.keyframe_id is not None:
                raw = self.entry.raws.get(frame.keyframe_id)
                if raw is None:
                    return None
                result = self.frame_provider.get_frame(raw)
            else:
                result = self.frame_provider.get_video_frame(
                    frame.video_id, frame_idx=frame.frame_idx, timestamp=frame.timestamp
                )
        except Exception:  # noqa: BLE001 - a visual failure never fails the question
            return None
        return result.image_bytes if result.available else None

    def _answer_one_hypothesis(
        self,
        backend,
        question: str,
        bundle: QAEvidenceBundle,
        answer_type: str,
    ) -> QAAnswerResult:
        """Call the backend for ONE video. A failure is confined to this hypothesis."""
        status = backend.status()
        if status.visual_capable and not bundle.visual_frames:
            return QAAnswerResult(
                video_id=bundle.video_id,
                answer="",
                normalized_answer=UNKNOWN_ANSWER,
                status=ANSWER_STATUS_VISUAL_UNAVAILABLE,
                backend_type=status.backend_type,
                visual=False,
                warning=(
                    f"No visual evidence could be loaded for video {bundle.video_id!r}; "
                    "the backend was not called."
                ),
            )
        try:
            result = backend.answer(question, bundle, expected_answer_type=answer_type)
        except Exception as exc:  # noqa: BLE001 - never fabricate an answer on failure
            return QAAnswerResult(
                video_id=bundle.video_id,
                answer="",
                normalized_answer=UNKNOWN_ANSWER,
                status=ANSWER_STATUS_BACKEND_FAILED,
                backend_type=status.backend_type,
                visual=False,
                warning=f"Q&A backend failed for this video: {type(exc).__name__}: {exc}",
            )
        if result.video_id != bundle.video_id:
            # Structural guard: a backend must answer the video it was given.
            raise RuntimeError(
                f"Q&A backend returned an answer for {result.video_id!r} when asked about "
                f"{bundle.video_id!r}."
            )
        return result

    def _qa_predictions(
        self,
        answered: Sequence[tuple[QAVideoHypothesis, QAEvidenceBundle, QAAnswerResult, float]],
        requested: int,
        qa_config,
    ) -> list[AICPrediction]:
        """One row per (video, submission frame), each carrying its OWN video's answer."""
        weight = float(qa_config.answer_reliability_weight)
        rows: list[tuple[tuple, AICPrediction]] = []
        seen: set[tuple[str, str, str]] = set()
        for hypothesis, bundle, result, reliability in answered:
            for frame in hypothesis.frames:
                raw = self.entry.raws.get(frame.keyframe_id)
                if raw is None or raw.frame_idx is None:
                    continue
                # Phase 5 policy is unchanged: the submitted frame is the official mapped
                # frame_idx, never a decoded or refined frame index.
                submission_frame_id = official_frame_id(self.entry, frame.keyframe_id)
                key = (hypothesis.video_id, submission_frame_id, result.normalized_answer)
                if key in seen:
                    continue
                seen.add(key)
                # Reliability may nudge, never overturn: it scales the retrieval score
                # by at most +/- weight/2.
                score = float(frame.score) * (1.0 + weight * (float(reliability) - 0.5))
                prediction = AICPrediction(
                    video_id=hypothesis.video_id,
                    frame_id=submission_frame_id,
                    keyframe_id=frame.keyframe_id,
                    score=score,
                    answer=result.normalized_answer or result.answer,
                    timestamp=float(frame.timestamp),
                    score_breakdown={
                        "video_retrieval": round(float(hypothesis.retrieval_score), 6),
                        "frame_retrieval": round(float(frame.score), 6),
                        "answer_reliability": round(float(reliability), 6),
                        "qa_score": round(score, 6),
                    },
                    evidence={
                        "matched_objects": [],
                        "metadata_terms": [],
                        "sparse_terms": [],
                    },
                    qa={
                        "video_id": hypothesis.video_id,
                        # The video the ANSWER was produced from. Equal to `video_id` by
                        # construction; reported so the invariant is checkable, not assumed.
                        "answer_video_id": result.video_id,
                        "rank": hypothesis.rank,
                        "submission_frame_idx": int(raw.frame_idx),
                        "coarse_official_frame_idx": int(raw.frame_idx),
                        "best_visual_frame_idx": (
                            None
                            if not hypothesis.refinement
                            else hypothesis.refinement.get("best_visual_frame_idx")
                        ),
                        "raw_answer": result.answer,
                        "normalized_answer": result.normalized_answer,
                        "answer_status": result.status,
                        "expected_answer_type": bundle.expected_answer_type,
                        "backend_type": result.backend_type,
                        "backend_visual": result.visual,
                        "answer_reliability_score": round(float(reliability), 6),
                        "reasoning": result.reasoning,
                        "warning": result.warning,
                        "visual_available": bundle.visual_available,
                        "evidence": [item.to_dict() for item in bundle.frames],
                    },
                )
                rows.append(
                    ((-score, hypothesis.video_id, int(raw.frame_idx)), prediction)
                )
        rows.sort(key=lambda item: item[0])
        return [prediction for _, prediction in rows[:requested]]

    def _empty_qa_info(
        self, ground_query, retrieval_query_mode, answer_type, backend_status, started
    ) -> dict:
        return {
            "answer": UNKNOWN_ANSWER,
            "answer_normalized": UNKNOWN_ANSWER,
            "frame_ids": [],
            "center_time": None,
            "video_id": None,
            "answer_confidence": 0.0,
            "answer_reliability_score": 0.0,
            "ground_query": ground_query,
            "retrieval_query_mode": retrieval_query_mode,
            "expected_answer_type": answer_type,
            "hypotheses": [],
            "backend": backend_status.to_dict(),
            "warning": "No grounding candidate was retrieved.",
            "diagnostics": {
                "retrieved_video_hypotheses": 0,
                "answered_video_hypotheses": 0,
                "visual_hypotheses": 0,
                "nonvisual_hypotheses": 0,
                "abstentions": 0,
                "backend_failures": 0,
                "evidence_frames_used": 0,
                "local_refinement_calls": 0,
                "frame_decode_failures": 0,
                "cross_video_answer_copy_count": 0,
                "answer_without_matching_evidence_video_count": 0,
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        }

    def _qa_info(
        self,
        *,
        predictions,
        answered,
        ground_query,
        retrieval_query_mode,
        answer_type,
        backend_status,
        timings,
        refinement_calls,
        decode_failures,
    ) -> dict:
        """Response payload plus the structural checks that prove answer isolation."""
        # These two counters are the Phase 6 regression guards. They are computed from
        # the produced rows, not asserted from the algorithm, so a future refactor that
        # reintroduces cross-video copying makes them non-zero.
        cross_video = sum(
            1
            for prediction in predictions
            if prediction.qa and prediction.qa.get("answer_video_id") != prediction.video_id
        )
        mismatched_evidence = sum(
            1
            for prediction in predictions
            if prediction.qa
            and any(
                item.get("video_id") != prediction.video_id
                for item in prediction.qa.get("evidence", [])
            )
        )
        top = answered[0] if answered else None
        hypotheses_payload = []
        for hypothesis, bundle, result, reliability in answered:
            payload = hypothesis.to_dict()
            payload.update(
                {
                    "answer": result.answer,
                    "normalized_answer": result.normalized_answer,
                    "answer_status": result.status,
                    "answer_reliability_score": round(float(reliability), 6),
                    "backend_type": result.backend_type,
                    "backend_visual": result.visual,
                    "visual_available": bundle.visual_available,
                    "visual_evidence_loaded": len(bundle.visual_frames),
                    "evidence": [item.to_dict() for item in bundle.frames],
                    "warning": result.warning,
                }
            )
            hypotheses_payload.append(payload)

        info = {
            # Compatibility surface: the previous single-answer fields now describe the
            # TOP hypothesis only, and every hypothesis is available in `hypotheses`.
            "answer": top[2].answer if top else UNKNOWN_ANSWER,
            "value": top[2].value if top else None,
            "answer_normalized": top[2].normalized_answer if top else UNKNOWN_ANSWER,
            "answer_status": top[2].status if top else ANSWER_STATUS_BACKEND_FAILED,
            "reasoning": top[2].reasoning if top else "",
            "video_id": top[0].video_id if top else None,
            "center_time": (
                round(float(top[0].frames[0].timestamp), 3)
                if top and top[0].frames
                else None
            ),
            "frame_ids": (
                [item.evidence_id for item in top[1].frames if item.keyframe_id] if top else []
            ),
            "used_frame_ids": list(top[2].used_evidence_ids) if top else [],
            "evidence_roles": [item.role for item in top[1].frames] if top else [],
            "ground_query": ground_query,
            "retrieval_query_mode": retrieval_query_mode,
            "expected_answer_type": answer_type,
            "answer_reliability_score": round(float(top[3]), 6) if top else 0.0,
            # Retained so existing callers keep working; it is the same heuristic value.
            "answer_confidence": round(float(top[3]), 6) if top else 0.0,
            "grounding_score": round(float(top[0].retrieval_score), 6) if top else 0.0,
            "warning": top[2].warning if top else None,
            "backend": backend_status.to_dict(),
            "hypotheses": hypotheses_payload,
            "diagnostics": {
                "retrieved_video_hypotheses": len(answered),
                "answered_video_hypotheses": sum(
                    1 for _, _, result, _ in answered if result.status == ANSWER_STATUS_ANSWERED
                ),
                "visual_hypotheses": sum(1 for _, _, result, _ in answered if result.visual),
                "nonvisual_hypotheses": sum(
                    1 for _, _, result, _ in answered if not result.visual
                ),
                "abstentions": sum(
                    1 for _, _, result, _ in answered if result.status == ANSWER_STATUS_ABSTAINED
                ),
                "backend_failures": sum(
                    1
                    for _, _, result, _ in answered
                    if result.status == ANSWER_STATUS_BACKEND_FAILED
                ),
                "visual_unavailable": sum(
                    1
                    for _, _, result, _ in answered
                    if result.status == ANSWER_STATUS_VISUAL_UNAVAILABLE
                ),
                "evidence_frames_used": sum(len(bundle.frames) for _, bundle, _, _ in answered),
                "local_refinement_calls": refinement_calls,
                "frame_decode_failures": decode_failures,
                "predictions": len(predictions),
                "distinct_answer_videos": len({p.video_id for p in predictions}),
                "cross_video_answer_copy_count": cross_video,
                "answer_without_matching_evidence_video_count": mismatched_evidence,
                **{key: round(float(value), 3) for key, value in timings.items()},
            },
        }
        return info

    def search_trake(
        self,
        events: Sequence[str],
        *,
        per_event_k: Optional[int] = None,
        max_results: Optional[int] = None,
        refine_window_s: Optional[float] = None,
        refine: Optional[bool] = None,
    ) -> tuple[list[AICPrediction], list[TemporalMatch]]:
        """Align ordered events across one video.

        `refine_window_s` selects the local sampling window used by TRAKE's event-local
        refinement (Phase 8). It has no effect when refinement is off, which is the
        default; the response reports `refinement.status` either way.
        """
        outcome = self.search_trake_detailed(
            events,
            per_event_k=per_event_k,
            max_results=max_results,
            refine_window_s=refine_window_s,
            refine=refine,
        )
        return outcome.predictions, outcome.matches

    def search_trake_detailed(
        self,
        events: Sequence[str],
        *,
        per_event_k: Optional[int] = None,
        max_results: Optional[int] = None,
        refine_window_s: Optional[float] = None,
        refine: Optional[bool] = None,
    ) -> "TrakeSearchResult":
        """Align ordered events and return ONLY structurally complete sequences.

        An official TRAKE row needs exactly one frame per event. Before Phase 7 a skipped
        event was dropped from the row, which both shortened it and shifted every later
        event's label. Now the alignment keeps every event position, a deterministic
        recovery pass tries to fill the gaps from the same video's candidates for that
        event, and anything still incomplete is discarded instead of exported.
        """
        started = time.perf_counter()
        cost = QueryCost(task="trake", query=" ; ".join(str(item) for item in events))
        encoder_before = self._cost_snapshot()
        config = self.trake_config
        clean = [event.strip() for event in events if event and event.strip()]
        if len(clean) < 2:
            raise ValueError("TRAKE requires at least two ordered events.")
        per_event_k = int(per_event_k if per_event_k is not None else config.per_event_top_k)
        max_results = max(1, min(MAX_PREDICTIONS, int(max_results if max_results is not None else config.final_top_k)))
        refine = config.refinement_enabled if refine is None else bool(refine)

        retrieval_started = time.perf_counter()
        # One execution context per TRAKE request. Event texts are retrieved several
        # times as the depth expands, and the deterministic parts of that work — query
        # normalization, the text embedding, an identical (text, depth) retrieval — are
        # computed once. Request-local on purpose: a dataset change between requests must
        # never be able to serve a stale candidate list.
        context = QueryExecutionContext(label="trake")
        depth = {index: per_event_k for index in range(len(clean))}
        by_event: dict[int, list[EventCandidate]] = {
            index: self._trake_candidates(
                index, text, per_event_k, context=context, cost=cost
            )
            for index, text in enumerate(clean)
        }
        initial_counts = {index: len(rows) for index, rows in by_event.items()}
        report = align_trake(clean, by_event, config, max_results=max_results)
        complete_before_expansion = int(report.diagnostics["returned_complete_predictions"])

        # Adaptive expansion: only the events that are actually holding videos back are
        # re-retrieved deeper, and only until enough videos can cover every event. A
        # blanket deep retrieval for every event would cost far more for no reason.
        expansion = self._expand_trake_candidates(
            clean, by_event, depth, report, config, max_results, context=context, cost=cost
        )
        report, expansion_info = expansion
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        refinement_result = self._refine_trake_sequences(
            report, config, refine=refine, refine_window_s=refine_window_s
        )
        refined_predictions, refinements, refinement_stats = refinement_result

        matches: list[TemporalMatch] = []
        predictions: list[AICPrediction] = []
        for result in refined_predictions[:max_results]:
            # Every step is present, so step i is event i: the UI can no longer shift.
            steps = [
                TemporalStep(
                    event=step.event_text,
                    keyframe_id=str(step.keyframe_id),
                    timestamp=float(step.timestamp or 0.0),
                )
                for step in result.steps
            ]
            match = TemporalMatch(result.video_id, steps, result.final_sequence_score)
            matches.append(match)
            predictions.append(self._from_trake(result))

        diagnostics = dict(report.diagnostics)
        diagnostics["per_event_top_k"] = per_event_k
        diagnostics["initial_candidate_counts"] = initial_counts
        diagnostics["candidate_retrieval_ms"] = round(retrieval_ms, 3)
        diagnostics["complete_alignments_before_expansion"] = complete_before_expansion
        diagnostics["complete_alignments_after_expansion"] = int(
            report.diagnostics["returned_complete_predictions"]
        )
        diagnostics.update(expansion_info)
        diagnostics.update(refinement_stats)
        diagnostics["query_execution"] = context.to_dict()
        diagnostics["query_embedding_cache"] = self._query_embeddings.to_dict()
        diagnostics["adaptive_budget"] = (
            self._trake_event_budget(by_event, expansion_info)
            if self.budget_config.enabled
            else {"enabled": False}
        )
        self._record_encoder_cost(cost, encoder_before)
        cost.add_decode(
            requested=int(refinement_stats.get("frames_decoded", 0) or 0),
            decoded=int(refinement_stats.get("frames_decoded", 0) or 0),
            ms=float(refinement_stats.get("refinement_ms", 0.0) or 0.0),
        )
        cost.add_image_embeddings(int(refinement_stats.get("frames_scored", 0) or 0))
        cost.total_wall_ms = (time.perf_counter() - started) * 1000.0
        diagnostics["cost"] = cost.to_dict()
        # `refine_window_s` now genuinely selects the local sampling window.
        diagnostics["refine_window_s_requested"] = (
            None if refine_window_s is None else float(refine_window_s)
        )
        diagnostics["alignment_ms"] = round(
            max(0.0, (time.perf_counter() - started) * 1000.0 - retrieval_ms
                - float(refinement_stats.get("refinement_ms", 0.0))),
            3,
        )
        diagnostics["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return TrakeSearchResult(
            predictions=predictions,
            matches=matches,
            trake_predictions=tuple(refined_predictions[:max_results]),
            discarded=report.discarded,
            refinements=tuple(refinements),
            diagnostics=diagnostics,
        )

    def _trake_candidates(
        self,
        event_index: int,
        text: str,
        depth: int,
        context: Optional[QueryExecutionContext] = None,
        cost: Optional[QueryCost] = None,
    ) -> list[EventCandidate]:
        """Retrieve one event's candidates through the EXISTING retrieval path.

        Phase 8 deliberately does not add object/OCR/ASR generators; deeper coverage comes
        from asking the same CLIP+BM25 fusion for more rows.

        R0 reuses an identical `(text, depth)` result within one request — two events
        that share wording, or a stage that asks for a depth already retrieved. It does
        NOT slice a deeper result to answer a shallower request: channel scores are
        rank-normalized over the pool each channel returned, so a top-40 slice of a
        depth-300 retrieval is not the same list as a depth-40 retrieval. Reuse is only
        allowed where the result is provably identical.
        """
        key = (str(text), int(depth))
        if context is not None:
            cached = context.channel_results.get(key)
            if cached is not None:
                context.reused_channel_results += 1
                return [replace(row, event_index=event_index) for row in cached]
        rows: list[EventCandidate] = []
        # Expansion must deepen the CHANNELS, not just widen the fused slice; otherwise a
        # deeper request would silently fall back to the original channel depths.
        scale = max(1.0, float(depth) / max(1.0, float(self.trake_config.per_event_top_k)))
        for candidate in self.search_candidates(
            text, top_k=int(depth), depth_scale=scale, context=context, cost=cost
        ):
            raw = self.entry.raws.get(candidate.keyframe_id)
            if raw is None or raw.frame_idx is None:
                # An unmapped keyframe has no official frame to submit, so it cannot
                # stand for an event. It is skipped rather than given a placeholder.
                continue
            rows.append(
                EventCandidate(
                    event_index=event_index,
                    video_id=candidate.video_id,
                    keyframe_id=candidate.keyframe_id,
                    frame_id=official_frame_id(self.entry, candidate.keyframe_id),
                    timestamp=candidate.timestamp,
                    score=float(candidate.score),
                )
            )
        if context is not None:
            context.channel_results[key] = tuple(rows)
        return rows

    def _expand_trake_candidates(
        self,
        events: Sequence[str],
        by_event: dict[int, list[EventCandidate]],
        depth: dict[int, int],
        report,
        config,
        max_results: int,
        context: Optional[QueryExecutionContext] = None,
        cost: Optional[QueryCost] = None,
    ):
        """Re-retrieve only the events that are blocking completeness, and only deeper.

        The Phase 7 smoke showed 59 missing positions where the event simply had no
        candidate for that video at depth 40. Expansion targets exactly those events;
        it never fabricates a candidate and never exceeds `candidate_depth_max`.
        """
        info: dict[str, Any] = {
            "candidate_expansion_triggered": False,
            "candidate_expansion_stages": 0,
            "events_expanded": [],
            "depth_before": dict(depth),
            "depth_after": dict(depth),
            "new_candidates_added": 0,
            "expanded_candidate_counts": {i: len(rows) for i, rows in by_event.items()},
            "new_complete_video_hypotheses": 0,
        }
        target = max(1, int(config.target_complete_video_hypotheses))
        maximum = max(int(config.per_event_top_k), int(config.candidate_depth_max))
        stages = [int(value) for value in config.candidate_depth_expansion if int(value) > 0]
        if not stages:
            return report, info
        coverage_before = int(report.diagnostics.get("videos_with_full_event_coverage", 0))
        for stage in stages:
            if int(report.diagnostics.get("videos_with_full_event_coverage", 0)) >= target:
                break
            # An event is "weak" when it reaches fewer videos than the others do; those
            # are the ones actually preventing complete sequences.
            reach = {index: len({row.video_id for row in rows}) for index, rows in by_event.items()}
            best_reach = max(reach.values(), default=0)
            weak = sorted(index for index, value in reach.items() if value < best_reach)
            if not weak:
                weak = sorted(by_event)
            wanted = min(int(stage), maximum)
            changed = False
            for index in weak:
                if wanted <= depth[index]:
                    continue
                before = len(by_event[index])
                by_event[index] = self._trake_candidates(
                    index, events[index], wanted, context=context, cost=cost
                )
                depth[index] = wanted
                info["new_candidates_added"] += max(0, len(by_event[index]) - before)
                if index not in info["events_expanded"]:
                    info["events_expanded"].append(index)
                changed = True
            if not changed:
                continue
            info["candidate_expansion_triggered"] = True
            info["candidate_expansion_stages"] += 1
            report = align_trake(events, by_event, config, max_results=max_results)
        info["depth_after"] = dict(depth)
        info["expanded_candidate_counts"] = {i: len(rows) for i, rows in by_event.items()}
        info["new_complete_video_hypotheses"] = max(
            0,
            int(report.diagnostics.get("videos_with_full_event_coverage", 0)) - coverage_before,
        )
        return report, info

    def _trake_event_budget(self, by_event: dict, expansion_info: dict) -> dict:
        """R1: hand the optional per-event budget to the structurally weakest event.

        The organizer gives zero for the wrong video, so finding a complete video
        hypothesis comes first and is untouched here. What this changes is what happens
        *after* one exists: an equal split spends as much on a settled event as on the
        one that is holding the sequence together.

        Every invariant of Phase 7/8 survives: all N events remain present, ordering is
        unchanged, candidates stay within one video, and the per-event allocations sum to
        exactly the global cap with no event allowed to take more than
        `trake_event_frame_cap`.
        """
        config = self.budget_config
        if not config.trake_weakest_event_enabled:
            return {"enabled": True, "trake_weakest_event": False}
        expanded = list(expansion_info.get("events_expanded") or ())
        signals = trake_event_uncertainty(by_event, expanded=expanded)
        weights = {item.event_index: item.uncertainty for item in signals}
        requested = max(1, int(self.trake_config.refinement_max_frames_per_query))
        allocation = split_budget_by_uncertainty(
            weights, requested, minimum=0, maximum=int(config.trake_event_frame_cap)
        )
        # The per-event cap can make the requested total unreachable. Both numbers are
        # reported: what was asked for, and what the caps actually allowed.
        allocated = sum(allocation.values())
        weakest = max(weights, key=lambda key: (weights[key], key)) if weights else None
        return {
            "enabled": True,
            "trake_weakest_event": True,
            "event_uncertainty": [item.to_dict() for item in signals],
            "frame_budget_requested": requested,
            "frame_budget_total": allocated,
            "frame_budget_by_event": allocation,
            "event_frame_cap": int(config.trake_event_frame_cap),
            "weakest_event_index": weakest,
            "note": (
                "Allocation only. Which event is structurally weakest is measurable; "
                "whether spending more on it improves the answer is not, without "
                "ground truth."
            ),
        }

    def _refine_trake_sequences(self, report, config, *, refine: bool, refine_window_s):
        """Locally refine only the top few COMPLETE sequences, within a frame budget."""
        predictions = list(report.predictions)
        stats: dict[str, Any] = {
            "alignments_refined": 0,
            "events_refined": 0,
            "frames_decoded": 0,
            "frames_scored": 0,
            "refinement_failures": 0,
            "order_violations_detected": 0,
            "order_violations_resolved": 0,
            "refinement_budget_exhausted": False,
            "refinement_ms": 0.0,
            "refinement_applied": False,
            "refinement_status": "disabled" if not refine else "no_sequences",
        }
        if not refine or not predictions:
            return predictions, [], stats

        started = time.perf_counter()
        refiner = TrakeSequenceRefiner(
            config,
            frame_provider=self.frame_provider,
            scorer=self.frame_scorer,
            source_video_for=lambda video_id, keyframe_id: getattr(
                self.entry.raws.get(keyframe_id or ""), "source_video", None
            ),
            window_s=None if refine_window_s is None else float(refine_window_s),
        )
        budget = FrameBudget(limit=max(1, int(config.refinement_max_frames_per_query)))
        limit = max(1, int(config.refinement_top_alignment_budget))
        refinements: list = []
        out: list[TrakePrediction] = []
        for position, prediction in enumerate(predictions):
            if position >= limit:
                out.append(prediction)
                continue
            outcome = refiner.refine(prediction, budget)
            refinements.append(outcome)
            if outcome.applied:
                stats["alignments_refined"] += 1
                stats["events_refined"] += outcome.events_refined
                stats["frames_decoded"] += outcome.frames_decoded
                stats["frames_scored"] += outcome.frames_scored
                stats["order_violations_detected"] += int(outcome.order_violation_detected)
                stats["order_violations_resolved"] += int(outcome.order_violation_resolved)
                out.append(apply_refinement(prediction, outcome))
            else:
                stats["refinement_failures"] += 1
                out.append(prediction)
        stats["refinement_budget_exhausted"] = budget.exhausted
        stats["refinement_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        stats["refinement_applied"] = stats["alignments_refined"] > 0
        stats["refinement_status"] = (
            "refined" if stats["refinement_applied"] else "unavailable"
        )
        stats["refinement_window_s"] = refiner.window_s
        # Refinement may reorder the top sequences; the row contents never change.
        out.sort(key=lambda p: (-float(p.final_sequence_score), p.video_id, p.frame_ids))
        return [replace(p, rank=i) for i, p in enumerate(out, start=1)], refinements, stats
    def _from_candidate(
        self,
        c: Candidate,
        answer: Optional[str] = None,
        refinement=None,
    ) -> AICPrediction:
        # The submission frame is the OFFICIAL mapped frame_idx. A refined visual frame
        # replaces it only under the explicit `decoded_frame` policy, which is not the
        # default because AIC has not confirmed those frame-ID semantics.
        frame_id = official_frame_id(self.entry, c.keyframe_id)
        if (
            refinement is not None
            and refinement.applied
            and refinement.submission_frame_idx is not None
            and str(self.refinement_config.frame_output_policy) == FRAME_OUTPUT_DECODED_FRAME
        ):
            frame_id = str(int(refinement.submission_frame_idx))
        return AICPrediction(
            video_id=c.video_id,
            frame_id=frame_id,
            keyframe_id=c.keyframe_id,
            score=float(c.score),
            answer=answer,
            timestamp=float(c.timestamp),
            score_breakdown=dict(getattr(c, "score_breakdown", {})),
            evidence={
                "matched_objects": list(getattr(getattr(c, "evidence", None), "matched_objects", ())),
                "metadata_terms": list(getattr(getattr(c, "evidence", None), "metadata_terms", ())),
                "sparse_terms": list(getattr(getattr(c, "evidence", None), "sparse_terms", ())),
                # Which independent channels proposed this candidate. A candidate that
                # only objects or only metadata found stays identifiable as such.
                "channels": list(getattr(c, "channels", ()) or ()),
            },
            refinement=None if refinement is None else refinement.to_dict(),
        )

    def _from_trake(self, result: TrakePrediction) -> AICPrediction:
        """Build a submission row from a COMPLETE alignment, re-checking the invariant."""
        frame_ids = [str(step.submission_frame_idx) for step in result.steps]
        if len(frame_ids) != result.event_count:
            raise TrakeStructureError(
                f"TRAKE row for {result.video_id!r} would carry {len(frame_ids)} frames "
                f"for {result.event_count} events."
            )
        return AICPrediction(
            video_id=result.video_id,
            frame_id=frame_ids[0],
            keyframe_id=str(result.steps[0].keyframe_id),
            score=float(result.score),
            timestamp=float(result.steps[0].timestamp or 0.0),
            event_frame_ids=frame_ids,
            trake=result.to_dict(),
        )

    def _from_match(self, match: TemporalMatch) -> AICPrediction:
        """Legacy conversion, retained for callers that hold a `TemporalMatch`.

        It no longer filters anything: `match.steps` is event-preserving by construction
        since Phase 7, so the row length equals the event count.
        """
        frame_ids = [official_frame_id(self.entry, s.keyframe_id) for s in match.steps]
        key = match.steps[0].keyframe_id if match.steps else ""
        return AICPrediction(
            video_id=match.video_id,
            frame_id=frame_ids[0] if frame_ids else "",
            keyframe_id=key,
            score=float(match.total_score),
            event_frame_ids=frame_ids,
        )