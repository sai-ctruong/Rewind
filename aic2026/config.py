"""Validated runtime configuration for the AIC 2026 pipeline."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fusion import FusionConfig
from .local_refinement import (
    FRAME_OUTPUT_POLICIES,
    REFINEMENT_MODES,
    MODE_UNCERTAINTY,
    MODE_ALWAYS,
    RefinementConfig,
)
from .ranking import RankingConfig
from .trake import AlignmentConfig as TrakeConfig


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    production_mode: bool = False
    seed: int = 42
    device: str = "auto"
    time_budget_seconds: float = 20.0


DATASET_SCOPE_MODES = ("patterns", "existing_videos")


@dataclass(frozen=True)
class DatasetScopeConfig:
    """Which canonical video IDs belong to the active dataset.

    A development subset (for example only collection ``L21``) is expressed here so
    that inspection, index build, and cache identity all agree on one selection. The
    default selects the whole collection, so the full L21-L30 dataset needs no code
    change: only configuration.

    ``mode`` adds a source constraint underneath the patterns:

    ``patterns``
        select by video-ID glob alone.
    ``existing_videos``
        additionally require that the original MP4 exists on disk *and* that the video
        has map + CLIP support. Resolved against ``DATA_ROOT`` at run time, so the
        resolved ID set — not the mode string — is what identifies the cache.
    """

    include_patterns: tuple[str, ...] = ("*",)
    exclude_patterns: tuple[str, ...] = ()
    mode: str = "patterns"


@dataclass(frozen=True)
class DatasetValidationConfig:
    strict_alignment: bool = True
    # Phase 3.2: BTC keyframe JPEGs are SUPPORTING data, so their absence no longer
    # blocks the official-CLIP global index. `require_visual_source` asks the weaker,
    # architecturally correct question: does each selected video have *some* visual
    # source, a keyframe JPEG or the original MP4? Off by default, because a
    # retrieval-only development subset is legitimate.
    require_visual_source: bool = False
    strict_objects: bool = False
    verify_feature_finite: bool = True
    verify_feature_norms: bool = True
    verify_image_decode: bool = False
    max_image_decode_checks_per_video: int | None = None
    supported_feature_dtypes: tuple[str, ...] = ("float16", "float32")
    expected_feature_dim: int | None = None


@dataclass(frozen=True)
class DatasetConfig:
    root: str = "data"
    cache_dir: str = "artifacts/aic2026_index"
    # Derived frames decoded from the original MP4s. A disposable visual artifact
    # cache: never part of the retrieval cache manifest, and never inside DATA_ROOT.
    frame_cache_dir: str = "artifacts/video_frame_cache"
    load_objects: bool = False
    include_media_text: bool = False
    verify_keyframes: bool = False
    index_kind: str = "flat"
    scope: DatasetScopeConfig = field(default_factory=DatasetScopeConfig)
    validation: DatasetValidationConfig = field(default_factory=DatasetValidationConfig)


@dataclass(frozen=True)
class CacheConfig:
    allow_stale_cache: bool = False
    validate_data_signature: bool = True
    code_version_policy: str = "warn"


@dataclass(frozen=True)
class EncoderConfig:
    type: str = "auto_clip"
    model_name: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 512
    normalize: bool = True
    batch_size: int = 32
    allow_hashing_fallback: bool = True


@dataclass(frozen=True)
class RetrievalChannelConfig:
    clip_enabled: bool = True
    clip_top_k: int = 1200
    bm25_enabled: bool = True
    bm25_top_k: int = 800
    objects_enabled: bool = True
    objects_top_k: int = 800
    object_confidence_threshold: float = 0.25
    metadata_enabled: bool = True
    metadata_top_k: int = 300


@dataclass(frozen=True)
class QAConfig:
    top_video_hypotheses: int = 8
    frame_hypotheses_per_video: int = 3
    evidence_frame_count: int = 8
    answerer_batch_size: int = 1
    max_answers: int = 100
    retrieval_query_mode: str = "event_only"
    answer_confidence_threshold: float = 0.35
    abstain_enabled: bool = True
    default_answer_type: str = "auto"


@dataclass(frozen=True)
class SubmissionConfig:
    max_predictions: int = 100
    max_answers: int = 100


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple[int, ...] = (1, 5, 20, 50, 100)
    output_dir: str = "evaluation/benchmarks/aic2026"
    save_predictions: bool = True
    save_errors: bool = True


@dataclass(frozen=True)
class UIConfig:
    enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retrieval_channels: RetrievalChannelConfig = field(default_factory=RetrievalChannelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    trake: TrakeConfig = field(default_factory=TrakeConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _construct(cls, raw: Mapping[str, Any] | None):
    raw = dict(raw or {})
    names = {field.name for field in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def _pattern_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Accept a single pattern or a list of patterns; keep YAML order, reject junk."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a string or a list of strings")
    patterns: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{name} entries must be non-empty strings, got {item!r}")
        patterns.append(item.strip())
    return tuple(patterns)


def _refinement_raw(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept the nested Phase 5 shape and the older flat keys for refinement.

    `scorer:` and `trigger:` read naturally in YAML, but `RefinementConfig` is a flat
    frozen dataclass, so they are flattened here rather than mirrored as a second
    settings structure. `uncertainty_only: true` from a pre-Phase-5 file means exactly
    `mode: uncertainty`, so it is translated instead of being silently dropped.
    """
    source = dict(raw or {})
    scorer = source.pop("scorer", None)
    if isinstance(scorer, Mapping):
        for key, value in scorer.items():
            name = "scorer_type" if key == "type" else f"scorer_{key}"
            source.setdefault(name, value)
    trigger = source.pop("trigger", None)
    if isinstance(trigger, Mapping):
        for key, value in trigger.items():
            source.setdefault(str(key), value)
    legacy = source.pop("uncertainty_only", None)
    if legacy is not None and "mode" not in source:
        source["mode"] = MODE_UNCERTAINTY if bool(legacy) else MODE_ALWAYS
    return source


def app_config_from_dict(data: Mapping[str, Any]) -> AppConfig:
    source = copy.deepcopy(dict(data))
    raw = source.get("aic2026", source)
    dataset_raw = dict(raw.get("dataset") or {})
    validation_raw = dict(dataset_raw.pop("validation", None) or {})
    if "require_keyframe_images" in validation_raw:
        # Removed in Phase 3.2 rather than silently ignored: its meaning changed, and a
        # dropped key would quietly re-enable an index build this config meant to block.
        raise ConfigError(
            "dataset.validation.require_keyframe_images was removed in Phase 3.2 because "
            "BTC keyframe JPEGs are supporting data, not a retrieval requirement. Use "
            "dataset.validation.require_visual_source to demand a keyframe JPEG or an "
            "original MP4 for every selected video."
        )
    dataset_raw["validation"] = _construct(DatasetValidationConfig, validation_raw)
    scope_raw = dict(dataset_raw.pop("scope", None) or {})
    for key in ("include_patterns", "exclude_patterns"):
        if key in scope_raw:
            scope_raw[key] = _pattern_tuple(scope_raw[key], f"dataset.scope.{key}")
    dataset_raw["scope"] = _construct(DatasetScopeConfig, scope_raw)
    cfg = AppConfig(
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
        dataset=_construct(DatasetConfig, dataset_raw),
        cache=_construct(CacheConfig, raw.get("cache")),
        encoder=_construct(EncoderConfig, raw.get("encoder")),
        retrieval_channels=_construct(RetrievalChannelConfig, raw.get("retrieval_channels")),
        fusion=_construct(FusionConfig, raw.get("fusion")),
        ranking=_construct(RankingConfig, raw.get("ranking")),
        refinement=_construct(RefinementConfig, _refinement_raw(raw.get("refinement"))),
        trake=_construct(TrakeConfig, raw.get("trake")),
        qa=_construct(QAConfig, raw.get("qa")),
        submission=_construct(SubmissionConfig, raw.get("submission")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
        ui=_construct(UIConfig, raw.get("ui")),
    )
    validate_app_config(cfg)
    return cfg


def load_app_config(
    path: str | Path = "configs/settings.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError("config root must be a mapping")
    merged = copy.deepcopy(dict(data))
    if overrides:
        overlay = {"aic2026": overrides} if "aic2026" not in overrides else overrides
        merged = _deep_merge(merged, overlay)
    return app_config_from_dict(merged)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    def convert(value):
        if is_dataclass(value):
            out = {}
            for item in fields(value):
                child = getattr(value, item.name)
                if item.name == "top_k" and child is None:
                    continue
                out[item.name] = convert(child)
            return out
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items() if not (str(k) == "top_k" and v is None)}
        return value

    return convert(config)


def config_hash(config: AppConfig) -> str:
    blob = json.dumps(config_to_dict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def validate_app_config(config: AppConfig) -> None:
    _validate_runtime(config.runtime)
    _validate_dataset(config.dataset)
    if config.dataset.validation.expected_feature_dim is not None:
        _require(
            int(config.dataset.validation.expected_feature_dim) == int(config.encoder.feature_dim),
            "dataset.validation.expected_feature_dim must equal encoder.feature_dim",
        )
    _validate_cache(config.cache)
    _validate_encoder(config.runtime, config.encoder)
    _validate_fusion(config.fusion)
    _validate_ranking(config.ranking)
    _validate_refinement(config.refinement)
    _validate_trake(config.trake)
    _validate_qa(config.qa)
    _validate_submission(config.submission)
    _validate_evaluation(config.evaluation)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate_runtime(cfg: RuntimeConfig) -> None:
    _require(int(cfg.seed) >= 0, "runtime.seed must be >= 0")
    _require(
        cfg.device in {"auto", "cpu", "cuda"} or bool(re.fullmatch(r"cuda:\d+", str(cfg.device))),
        'runtime.device must be "auto", "cpu", "cuda", or cuda:N',
    )


def _validate_scope(cfg: DatasetScopeConfig) -> None:
    _require(
        cfg.mode in DATASET_SCOPE_MODES,
        f"dataset.scope.mode must be one of {', '.join(DATASET_SCOPE_MODES)}",
    )
    _require(
        bool(cfg.include_patterns),
        "dataset.scope.include_patterns must be non-empty; use [\"*\"] for the full dataset",
    )
    for name, patterns in (
        ("include_patterns", cfg.include_patterns),
        ("exclude_patterns", cfg.exclude_patterns),
    ):
        for pattern in patterns:
            _require(
                isinstance(pattern, str) and bool(pattern.strip()),
                f"dataset.scope.{name} entries must be non-empty strings",
            )
            _require(
                "/" not in pattern and "\\" not in pattern,
                f"dataset.scope.{name} entries match a video ID, not a path: {pattern!r}",
            )


def _validate_dataset(cfg: DatasetConfig) -> None:
    validation = cfg.validation
    _validate_scope(cfg.scope)
    _require(
        bool(str(cfg.frame_cache_dir).strip()),
        "dataset.frame_cache_dir must be non-empty",
    )
    _require(
        Path(cfg.frame_cache_dir).resolve(strict=False)
        != Path(cfg.root).resolve(strict=False)
        and Path(cfg.root).resolve(strict=False)
        not in Path(cfg.frame_cache_dir).resolve(strict=False).parents,
        "dataset.frame_cache_dir must not live inside DATA_ROOT; derived frames must "
        "never be mistaken for BTC-provided data",
    )
    _require(cfg.index_kind in {"flat", "hnsw"}, "dataset.index_kind must be flat or hnsw")
    _require(
        bool(validation.strict_alignment),
        "dataset.validation.strict_alignment must remain true; non-strict mode is inspection-only",
    )
    _require(
        bool(validation.supported_feature_dtypes),
        "dataset.validation.supported_feature_dtypes must be non-empty",
    )
    supported = {str(item) for item in validation.supported_feature_dtypes}
    _require(
        supported <= {"float16", "float32"},
        "dataset.validation.supported_feature_dtypes contains an unsupported dtype",
    )
    if validation.expected_feature_dim is not None:
        _require(
            int(validation.expected_feature_dim) > 0,
            "dataset.validation.expected_feature_dim must be > 0",
        )
    if validation.max_image_decode_checks_per_video is not None:
        _require(
            int(validation.max_image_decode_checks_per_video) > 0,
            "dataset.validation.max_image_decode_checks_per_video must be > 0",
        )


def _validate_cache(cfg: CacheConfig) -> None:
    _require(
        cfg.code_version_policy in {"ignore", "warn", "reject"},
        "cache.code_version_policy must be one of ignore, warn, reject",
    )


def _validate_encoder(runtime: RuntimeConfig, cfg: EncoderConfig) -> None:
    _require(int(cfg.feature_dim) > 0, "encoder.feature_dim must be > 0")
    _require(int(cfg.batch_size) > 0, "encoder.batch_size must be > 0")
    _require(bool(str(cfg.model_name).strip()), "encoder.model_name must be non-empty")
    if runtime.production_mode:
        _require(cfg.type not in {"hashing", "hashing_fallback", "hashing-only"}, "encoder.type cannot be hashing-only in production mode")


def _validate_fusion(cfg: FusionConfig) -> None:
    weights = [cfg.clip_weight, cfg.object_weight, cfg.metadata_weight, cfg.sparse_weight]
    for name, value in zip(("clip_weight", "object_weight", "metadata_weight", "sparse_weight"), weights):
        _require(float(value) >= 0, f"fusion.{name} must be >= 0")
    _require(any(float(v) > 0 for v in weights), "fusion weights must not all be zero")
    _require(str(cfg.method).lower() in {"adaptive", "weighted", "rrf", "rrf_full"}, "fusion.method must be one of adaptive, weighted, rrf, rrf_full")
    _require(int(cfg.rrf_k) > 0, "fusion.rrf_k must be > 0")


def _validate_ranking(cfg: RankingConfig) -> None:
    final_top_k = int(cfg.final_top_k)
    _require(1 <= final_top_k <= 100, "ranking.final_top_k must be in [1, 100]")
    _require(int(cfg.precision_head_size) >= 0, "ranking.precision_head_size must be >= 0")
    _require(int(cfg.recall_tail_size) >= 0, "ranking.recall_tail_size must be >= 0")
    _require(int(cfg.min_frame_gap) >= 0, "ranking.min_frame_gap must be >= 0")
    _require(int(cfg.max_frames_per_video) > 0, "ranking.max_frames_per_video must be > 0")
    _require(0 <= float(cfg.diversity_lambda) <= 1, "ranking.diversity_lambda must be in [0, 1]")


def _validate_refinement(cfg: RefinementConfig) -> None:
    _require(
        str(cfg.mode) in REFINEMENT_MODES,
        f"refinement.mode must be one of {', '.join(REFINEMENT_MODES)}",
    )
    _require(float(cfg.window_before_s) >= 0, "refinement.window_before_s must be >= 0")
    _require(float(cfg.window_after_s) >= 0, "refinement.window_after_s must be >= 0")
    _require(float(cfg.fine_fps) > 0, "refinement.fine_fps must be > 0")
    _require(int(cfg.max_frames) > 0, "refinement.max_frames must be > 0")
    _require(int(cfg.batch_size) > 0, "refinement.batch_size must be > 0")
    _require(int(cfg.cache_size_mb) >= 0, "refinement.cache_size_mb must be >= 0")
    _require(
        math.isfinite(float(cfg.margin_threshold)) and float(cfg.margin_threshold) >= 0,
        "refinement.margin_threshold must be finite and >= 0",
    )
    _require(int(cfg.candidate_budget) > 0, "refinement.candidate_budget must be > 0")
    _require(int(cfg.top_hypotheses) > 0, "refinement.top_hypotheses must be > 0")
    _require(
        int(cfg.top_hypotheses) >= int(cfg.candidate_budget),
        "refinement.top_hypotheses must be >= refinement.candidate_budget; the budget "
        "is chosen from the considered candidates",
    )
    _require(float(cfg.region_merge_s) >= 0, "refinement.region_merge_s must be >= 0")
    _require(
        math.isfinite(float(cfg.rerank_alpha)) and float(cfg.rerank_alpha) >= 0,
        "refinement.rerank_alpha must be finite and >= 0",
    )
    _require(
        int(cfg.scorer_input_max_side) >= 32,
        "refinement.scorer_input_max_side must be >= 32",
    )
    _require(
        str(cfg.frame_output_policy) in FRAME_OUTPUT_POLICIES,
        f"refinement.frame_output_policy must be one of {', '.join(FRAME_OUTPUT_POLICIES)}",
    )
    _require(
        str(cfg.scorer_type).strip().lower() == "clip",
        "refinement.scorer_type must be 'clip'; there is no production fake scorer",
    )
    _require(
        bool(str(cfg.scorer_model_name).strip()),
        "refinement.scorer_model_name must be non-empty",
    )
    _require(
        cfg.scorer_device in {"auto", "cpu", "cuda"}
        or bool(re.fullmatch(r"cuda:\d+", str(cfg.scorer_device))),
        'refinement.scorer_device must be "auto", "cpu", "cuda", or cuda:N',
    )


def _validate_trake(cfg: TrakeConfig) -> None:
    _require(str(cfg.alignment_method) == "beam_dp", "trake.alignment_method must be beam_dp for the current implementation")
    _require(int(cfg.per_event_top_k) > 0, "trake.per_event_top_k must be > 0")
    _require(int(cfg.top_video_hypotheses) > 0, "trake.top_video_hypotheses must be > 0")
    _require(int(cfg.beam_width) > 0, "trake.beam_width must be > 0")
    _require(float(cfg.min_gap_s) >= 0, "trake.min_gap_s must be >= 0")
    if cfg.max_gap_s is not None:
        _require(float(cfg.max_gap_s) >= float(cfg.min_gap_s), "trake.max_gap_s must be >= trake.min_gap_s")
    _require(float(cfg.missing_event_penalty) >= 0, "trake.missing_event_penalty must be >= 0")
    _require(float(cfg.transition_penalty) >= 0, "trake.transition_penalty must be >= 0")
    _require(1 <= int(cfg.final_top_k) <= 100, "trake.final_top_k must be in [1, 100]")


def _validate_qa(cfg: QAConfig) -> None:
    _require(int(cfg.top_video_hypotheses) > 0, "qa.top_video_hypotheses must be > 0")
    _require(int(cfg.frame_hypotheses_per_video) > 0, "qa.frame_hypotheses_per_video must be > 0")
    _require(int(cfg.evidence_frame_count) > 0, "qa.evidence_frame_count must be > 0")
    _require(1 <= int(cfg.max_answers) <= 100, "qa.max_answers must be in [1, 100]")
    _require(str(cfg.default_answer_type) in {"auto", "number", "yes/no", "boolean", "color", "text"}, "qa.default_answer_type must be a supported answer type")


def _validate_submission(cfg: SubmissionConfig) -> None:
    _require(1 <= int(cfg.max_predictions) <= 100, "submission.max_predictions must be in [1, 100]")
    _require(1 <= int(cfg.max_answers) <= 100, "submission.max_answers must be in [1, 100]")


def _validate_evaluation(cfg: EvaluationConfig) -> None:
    ks = list(cfg.ks)
    _require(bool(ks), "evaluation.ks must be non-empty")
    _require(ks == sorted(ks), "evaluation.ks must be sorted ascending")
    for k in ks:
        _require(1 <= int(k) <= 100, "evaluation.ks values must be in [1, 100]")


__all__ = [
    "AppConfig",
    "CacheConfig",
    "ConfigError",
    "DatasetConfig",
    "DatasetScopeConfig",
    "DatasetValidationConfig",
    "EncoderConfig",
    "EvaluationConfig",
    "FusionConfig",
    "QAConfig",
    "RankingConfig",
    "RefinementConfig",
    "RetrievalChannelConfig",
    "RuntimeConfig",
    "SubmissionConfig",
    "TrakeConfig",
    "UIConfig",
    "app_config_from_dict",
    "config_hash",
    "config_to_dict",
    "load_app_config",
    "validate_app_config",
]
