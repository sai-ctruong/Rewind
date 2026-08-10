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
from retrieval.vqa_module import VqaAnswerer, VqaModule, default_answerer, entry_records, retrieve_temporal_window
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
from .fusion import CandidateEvidence, FusionConfig, RankedCandidate, fuse_candidates, object_match_score, token_overlap_score
from .qa import QAInput, build_retrieval_query, confidence_from_evidence, normalize_answer, select_diverse_evidence
from .ranking import RankingConfig, video_aware_top100
from .trake import AlignmentConfig, EventCandidate, joint_trake_alignment
from .text_encoder import (
    AutoCLIPTextEncoder,
    HashingTextEncoder,
    TextQueryEncoder,
    encode_many,
    encoder_status,
)

MAX_PREDICTIONS = 100
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

    def row(self) -> list[str]:
        if self.event_frame_ids:
            return [self.video_id, *self.event_frame_ids]
        if self.answer is not None:
            return [self.video_id, self.frame_id, self.answer]
        return [self.video_id, self.frame_id]


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
        self.config_hash = config_hash(self.app_config)
        # Visual access is on demand only: constructing this touches no video file.
        self.frame_provider = frame_provider or FrameProvider(
            self.app_config.dataset.root, cache_dir=self.app_config.dataset.frame_cache_dir
        )

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
        variants = encode_many(
            self.text_encoder,
            [tpl.format(q=query) for tpl in self.query_templates],
            self.feature_dim,
        )
        mean = np.mean(variants, axis=0).astype(np.float32)
        return l2_normalize(mean.reshape(1, -1))[0]

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

    def search(
        self,
        entry: VideoIndexEntry,
        query: str,
        top_k: int = 20,
        rerank: bool = False,
    ) -> list[Candidate]:
        return self.search_candidates(query, top_k=top_k)

    def search_candidates(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[Candidate]:
        top_k = max(1, min(self.fusion_depth, int(top_k)))
        qvec = self.encode_query(query)
        retriever = CoarseRetriever(self.entry.index, fusion_depth=self.fusion_depth)
        rows = retriever._apply_filters(filters)
        depth = min(self.fusion_depth, len(self.entry.index.ids))
        if rows is None:
            dense_pairs = self.entry.index.dense_search(qvec, "clip", depth)
        else:
            dense_pairs = self.entry.index.exact_dense_search(qvec, "clip", rows, depth)
        sparse_pairs = self.entry.index.sparse_search(query, depth, rows)
        dense = dict(dense_pairs)
        sparse = dict(sparse_pairs)
        union = set(dense) | set(sparse)
        ranked: list[RankedCandidate] = []
        for row in union:
            keyframe_id = self.entry.index.ids[row]
            raw = self.entry.raws[keyframe_id]
            object_score, matched_objects = object_match_score(
                query, getattr(raw, "object_detections", None) or [
                    {"label": label, "confidence": 0.5} for label in raw.objects
                ]
            )
            metadata = self.entry.caption_by_id.get(keyframe_id, "")
            metadata_score, metadata_terms = token_overlap_score(query, metadata)
            ranked.append(RankedCandidate(
                video_id=raw.video_id,
                frame_id=official_frame_id(self.entry, keyframe_id),
                keyframe_id=keyframe_id,
                timestamp=raw.timestamp,
                keyframe_path=raw.image_path,
                dense_score=float(dense.get(row, 0.0)),
                object_score=object_score,
                metadata_score=metadata_score,
                bm25_score=float(sparse.get(row, 0.0)),
                evidence=CandidateEvidence(
                    matched_objects=matched_objects,
                    metadata_terms=metadata_terms,
                    sparse_terms=metadata_terms,
                ),
            ))
        fused = fuse_candidates(query, ranked, self.fusion_config)
        row_by_id = {self.entry.index.ids[row]: row for row in union}
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
            out.append(candidate)
        return out
    def search_kis(self, query: str, *, top_k: Optional[int] = None) -> list[AICPrediction]:
        requested = max(1, min(MAX_PREDICTIONS, int(top_k if top_k is not None else self.ranking_config.final_top_k)))
        pool = self.search_candidates(query, top_k=min(self.fusion_depth, max(requested * 3, 100)))

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

        ranked = video_aware_top100(
            pool,
            video_id=lambda c: c.video_id,
            frame_id=lambda c: int(official_frame_id(self.entry, c.keyframe_id)),
            score=lambda c: c.score,
            neighbors=nearby,
            config=replace(self.ranking_config, final_top_k=requested, top_k=None),
        )
        return [self._from_candidate(c) for c in ranked]

    def answer_qa(
        self,
        event_text: str,
        question: str,
        *,
        top_k: Optional[int] = None,
        window_s: float = 8.0,
        answerer: Optional[VqaAnswerer] = None,
        expected_answer_type: Optional[str] = None,
        retrieval_query_mode: Optional[str] = None,
        evidence_frame_count: Optional[int] = None,
        answer_confidence_threshold: Optional[float] = None,
        abstain_enabled: Optional[bool] = None,
    ) -> tuple[list[AICPrediction], dict]:
        retrieval_query_mode = retrieval_query_mode or self.qa_config.retrieval_query_mode
        qa_input = QAInput(event_text, question, expected_answer_type)
        ground_query = build_retrieval_query(qa_input, retrieval_query_mode)
        requested = max(1, min(MAX_PREDICTIONS, int(top_k if top_k is not None else self.qa_config.max_answers)))
        evidence_frame_count = int(evidence_frame_count if evidence_frame_count is not None else self.qa_config.evidence_frame_count)
        answer_confidence_threshold = float(answer_confidence_threshold if answer_confidence_threshold is not None else self.qa_config.answer_confidence_threshold)
        abstain_enabled = self.qa_config.abstain_enabled if abstain_enabled is None else bool(abstain_enabled)
        candidates = self.search_candidates(ground_query, top_k=requested)
        if not candidates:
            return [], {
                "answer": "unknown", "answer_normalized": "unknown", "frame_ids": [],
                "center_time": None, "video_id": None, "answer_confidence": 0.0,
                "warning": "No grounding candidate was retrieved.",
            }
        center = candidates[0]
        records = entry_records(self.entry, video_id=center.video_id)
        window = retrieve_temporal_window(records, center.timestamp, window_s, center.video_id)
        evidence = select_diverse_evidence(
            window, center.timestamp, max(1, evidence_frame_count), query=question
        )
        selected_records = [item.record for item in evidence]
        # Decode lazily and only for the handful of selected evidence frames; the
        # Top-100 search path never touches a video file.
        images: dict[str, bytes] = {}
        for frame in selected_records:
            raw = self.entry.raws.get(frame.id)
            if raw is None:
                continue
            result = self.frame_provider.get_frame(raw)
            if result.available:
                images[frame.id] = result.image_bytes
        ans = VqaModule(answerer or default_answerer()).answer(
            question or ground_query,
            selected_records,
            video_id=center.video_id,
            center_time=None,
            window_s=window_s,
            images=images,
        )
        answer_text = str(ans.value) if ans.value is not None else ans.answer
        normalized = normalize_answer(answer_text)
        # RRF scores are small, so use rank-relative grounding evidence rather than pretending
        # they are calibrated probabilities.
        margin = max(0.0, float(center.score - candidates[1].score)) if len(candidates) > 1 else float(center.score)
        grounding_score = min(1.0, 0.5 + margin * 10.0)
        confidence = confidence_from_evidence(grounding_score, len(selected_records), answer_text)
        warning = None
        if confidence < answer_confidence_threshold:
            warning = "Weak grounding or answer evidence; prediction is uncertain."
            if abstain_enabled and not answer_text.strip():
                answer_text = "unknown"
                normalized = "unknown"
        preds = [self._from_candidate(candidate, answer=answer_text) for candidate in candidates]
        info = {
            "answer": ans.answer,
            "value": ans.value,
            "answer_normalized": normalized,
            "reasoning": ans.reasoning,
            "used_frame_ids": ans.used_frame_ids,
            "frame_ids": [item.record.id for item in evidence],
            "evidence_roles": [item.role for item in evidence],
            "center_time": round(center.timestamp, 3),
            "video_id": center.video_id,
            "ground_query": ground_query,
            "retrieval_query_mode": retrieval_query_mode,
            "grounding_score": grounding_score,
            "answer_confidence": confidence,
            "warning": warning,
        }
        return preds, info
    def search_trake(
        self,
        events: Sequence[str],
        *,
        per_event_k: Optional[int] = None,
        max_results: Optional[int] = None,
        refine_window_s: Optional[float] = None,
    ) -> tuple[list[AICPrediction], list[TemporalMatch]]:
        clean = [event.strip() for event in events if event and event.strip()]
        if len(clean) < 2:
            raise ValueError("TRAKE requires at least two ordered events.")
        per_event_k = int(per_event_k if per_event_k is not None else self.trake_config.per_event_top_k)
        max_results = max(1, min(MAX_PREDICTIONS, int(max_results if max_results is not None else self.trake_config.final_top_k)))
        by_event: dict[int, list[EventCandidate]] = {}
        for event_index, event in enumerate(clean):
            by_event[event_index] = [
                EventCandidate(
                    event_index=event_index,
                    video_id=c.video_id,
                    keyframe_id=c.keyframe_id,
                    frame_id=official_frame_id(self.entry, c.keyframe_id),
                    timestamp=c.timestamp,
                    score=float(c.score),
                )
                for c in self.search_candidates(event, top_k=per_event_k)
            ]
        aligned = joint_trake_alignment(
            clean,
            by_event,
            self.trake_config,
            max_results=max_results,
        )
        matches: list[TemporalMatch] = []
        predictions: list[AICPrediction] = []
        for result in aligned:
            selected = [alignment for alignment in result.alignments if alignment.candidate is not None]
            steps = [
                TemporalStep(
                    event=alignment.event.text,
                    keyframe_id=alignment.candidate.keyframe_id,
                    timestamp=alignment.candidate.timestamp,
                )
                for alignment in selected
            ]
            if not steps:
                continue
            match = TemporalMatch(result.video_id, steps, result.score)
            matches.append(match)
            predictions.append(self._from_match(match))
        return predictions[:max_results], matches[:max_results]
    def _from_candidate(self, c: Candidate, answer: Optional[str] = None) -> AICPrediction:
        return AICPrediction(
            video_id=c.video_id,
            frame_id=official_frame_id(self.entry, c.keyframe_id),
            keyframe_id=c.keyframe_id,
            score=float(c.score),
            answer=answer,
            timestamp=float(c.timestamp),
            score_breakdown=dict(getattr(c, "score_breakdown", {})),
            evidence={
                "matched_objects": list(getattr(getattr(c, "evidence", None), "matched_objects", ())),
                "metadata_terms": list(getattr(getattr(c, "evidence", None), "metadata_terms", ())),
                "sparse_terms": list(getattr(getattr(c, "evidence", None), "sparse_terms", ())),
            },
        )

    def _from_match(self, match: TemporalMatch) -> AICPrediction:
        frame_ids = [official_frame_id(self.entry, s.keyframe_id) for s in match.steps]
        key = match.steps[0].keyframe_id if match.steps else ""
        return AICPrediction(
            video_id=match.video_id,
            frame_id=frame_ids[0] if frame_ids else "",
            keyframe_id=key,
            score=float(match.total_score),
            event_frame_ids=frame_ids,
        )