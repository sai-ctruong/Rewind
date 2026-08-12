"""Reproducibility identity and competition readiness.

Two questions this module answers, both of which have to be answerable *before* a
submission is trusted:

* **What exactly ran?** `SystemProfile` records the code, config, cache, dataset and
  capability identity of one runtime, so a result can be tied back to the thing that
  produced it.
* **Is this system fit to compete?** `evaluate_readiness` runs the same checks the CLI
  preflight runs and classifies the outcome as `READY`, `READY_WITH_WARNINGS` or
  `NOT_READY`.

The readiness verdict is deliberately allowed to be `READY_WITH_WARNINGS`. On this
machine there is no production visual Q&A backend, and reporting that as a warning is the
truthful answer; forcing `READY` for appearance would make the classification worthless.

Nothing here evaluates quality. No AIC ground truth exists in this repository, so every
number below is structural.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .cache_manifest import (
    cache_build_options_from_config,
    cache_fingerprint,
    canonical_data_root,
    detect_code_version,
    inspect_cache,
)
from .config import AppConfig, config_hash
from .dataset_scope import hash_selected_video_ids, resolve_dataset_scope, scope_payload
from .retrieval_channels import CHANNEL_SCHEMA_VERSION
from .submission_validation import MAX_SUBMISSION_ROWS, SUBMISSION_TASKS
from .version import PROJECT_VERSION, git_commit, git_is_dirty
from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION

# The validator's contract version, bumped when a submission rule changes meaning.
SUBMISSION_VALIDATION_VERSION = 1

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

READY = "READY"
READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
NOT_READY = "NOT_READY"

# Channels the competition tasks genuinely depend on. The rest are optional: their
# absence is reported, never fatal.
REQUIRED_CHANNELS = ("clip",)
OPTIONAL_CHANNELS = ("bm25", "objects", "metadata", "ocr", "asr", "caption")


@dataclass(frozen=True)
class CheckResult:
    """One preflight check. `FAIL` blocks a release; `WARN` does not."""

    name: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SystemProfile:
    """Everything needed to identify and reproduce one runtime."""

    project_version: str
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    python_version: str
    platform: str
    config_path: str
    config_hash: str
    cache_dir: str
    cache_fingerprint: Optional[str]
    cache_valid: Optional[bool]
    data_root_identity: str
    scope_mode: str
    selected_video_count: int
    selected_video_ids_hash: str
    runtime_generation: Optional[int]
    record_schema_version: int
    cache_schema_version: Optional[int]
    channel_schema_version: int
    submission_validation_version: int
    retrieval_channels: dict[str, Any] = field(default_factory=dict)
    kis: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)
    trake: dict[str, Any] = field(default_factory=dict)
    submission: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> dict[str, Any]:
        """The subset that must match for two runs to be considered the same setup."""
        return {
            "project_version": self.project_version,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "cache_fingerprint": self.cache_fingerprint,
            "selected_video_ids_hash": self.selected_video_ids_hash,
            "record_schema_version": self.record_schema_version,
            "channel_schema_version": self.channel_schema_version,
            "submission_validation_version": self.submission_validation_version,
        }


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    checks: tuple[CheckResult, ...] = ()
    profile: Optional[SystemProfile] = None

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(item for item in self.checks if item.status == STATUS_FAIL)

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(item for item in self.checks if item.status == STATUS_WARN)

    @property
    def ready(self) -> bool:
        return self.status in {READY, READY_WITH_WARNINGS}

    def exit_code(self) -> int:
        """0 ready, 1 ready with warnings, 2 not ready."""
        return {READY: 0, READY_WITH_WARNINGS: 1, NOT_READY: 2}[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "checks": [item.to_dict() for item in self.checks],
            "failures": [item.name for item in self.failures],
            "warnings": [item.name for item in self.warnings],
            "profile": None if self.profile is None else self.profile.to_dict(),
            "note": (
                "Readiness is STRUCTURAL: it reports whether the system can run and "
                "produce well-formed submissions. It says nothing about answer quality, "
                "and no AIC ground truth exists in this repository."
            ),
        }


def _cache_report(config: AppConfig, cache_dir: str | Path) -> dict[str, Any]:
    expected = cache_build_options_from_config(config)
    return inspect_cache(
        cache_dir,
        expected,
        validate_data_signature=config.cache.validate_data_signature,
        code_version_policy=config.cache.code_version_policy,
        current_code_version=detect_code_version(),
    )


def build_system_profile(
    config: AppConfig,
    *,
    config_path: str = "<in-memory>",
    engine: Any = None,
    runtime_generation: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
) -> SystemProfile:
    """Snapshot the runtime. Never loads a model and never touches the network."""
    resolved_cache = str(cache_dir or config.dataset.cache_dir)
    report: dict[str, Any] = {}
    try:
        report = _cache_report(config, resolved_cache)
    except Exception:  # noqa: BLE001 - a missing/broken cache must not break the profile
        report = {}
    manifest = report.get("manifest") or {}

    try:
        resolved_scope = resolve_dataset_scope(config.dataset.scope, config.dataset.root)
        scope = scope_payload(resolved_scope)
    except Exception:  # noqa: BLE001 - an unresolvable scope is reported, not raised
        scope = scope_payload(config.dataset.scope)

    selected: Sequence[str] = ()
    if engine is not None:
        selected = sorted({raw.video_id for raw in engine.entry.raws.values()})
    elif manifest.get("selected_video_ids"):
        selected = list(manifest["selected_video_ids"])

    channels: dict[str, Any] = {}
    if engine is not None:
        channels = engine.channel_status()
    else:
        settings = config.retrieval_channels
        for name in ("clip", "bm25", "objects", "metadata", "ocr", "asr", "caption"):
            channels[name] = {
                "name": name,
                "enabled": bool(getattr(settings, f"{name}_enabled")),
                "available": None,
                "records": None,
                "reason": "engine not loaded",
            }

    refinement = config.refinement
    qa_status: dict[str, Any] = {}
    if engine is not None:
        qa_status = engine.qa_status()
    backend = qa_status.get("backend", {}) if qa_status else {}

    return SystemProfile(
        project_version=PROJECT_VERSION,
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        config_path=str(config_path),
        config_hash=config_hash(config),
        cache_dir=resolved_cache,
        cache_fingerprint=manifest.get("cache_fingerprint") or cache_fingerprint(config),
        cache_valid=report.get("valid"),
        data_root_identity=canonical_data_root(config.dataset.root),
        scope_mode=str(scope.get("mode", "patterns")),
        selected_video_count=len(selected),
        selected_video_ids_hash=hash_selected_video_ids(selected),
        runtime_generation=runtime_generation,
        record_schema_version=AIC_RECORD_SCHEMA_VERSION,
        cache_schema_version=manifest.get("schema_version"),
        channel_schema_version=CHANNEL_SCHEMA_VERSION,
        submission_validation_version=SUBMISSION_VALIDATION_VERSION,
        retrieval_channels=channels,
        kis={
            "refinement_mode": refinement.effective_mode,
            "candidate_budget": int(refinement.candidate_budget),
            "frame_output_policy": str(refinement.frame_output_policy),
            "scorer_backend": str(refinement.scorer_type),
            "scorer_model_name": str(refinement.scorer_model_name),
            "scorer_device": str(refinement.scorer_device),
        },
        qa={
            "backend": backend.get("backend_type", config.qa.backend_type),
            "backend_state": backend.get("state"),
            "visual_capable": backend.get("visual_capable"),
            "production_ready": backend.get("production_ready", False),
            "top_video_hypotheses": int(config.qa.top_video_hypotheses),
        },
        trake={
            "alignment_method": str(config.trake.alignment_method),
            "k_best_per_video": int(config.trake.k_best_per_video),
            "max_alignments_per_video": int(config.trake.max_alignments_per_video),
            "refinement_enabled": bool(config.trake.refinement_enabled),
            "candidate_depth_max": int(config.trake.candidate_depth_max),
        },
        submission={
            "validation_version": SUBMISSION_VALIDATION_VERSION,
            "tasks": list(SUBMISSION_TASKS),
            "max_rows": MAX_SUBMISSION_ROWS,
            "frame_policy": "submission_frame_idx",
        },
    )


def evaluate_readiness(
    config: AppConfig,
    *,
    config_path: str = "<in-memory>",
    engine: Any = None,
    runtime_generation: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    profile: Optional[SystemProfile] = None,
) -> ReadinessReport:
    """Classify the system as READY / READY_WITH_WARNINGS / NOT_READY.

    A missing optional capability is a warning; a broken or misidentified dataset,
    cache, or validator is a failure. The distinction matters: this environment has no
    visual Q&A backend and is still perfectly able to produce valid KIS and TRAKE
    submissions.
    """
    checks: list[CheckResult] = []
    resolved_cache = str(cache_dir or config.dataset.cache_dir)
    snapshot = profile or build_system_profile(
        config,
        config_path=config_path,
        engine=engine,
        runtime_generation=runtime_generation,
        cache_dir=resolved_cache,
    )

    # --- configuration ------------------------------------------------------------
    checks.append(
        CheckResult(
            "config",
            STATUS_PASS,
            f"Configuration is valid (hash {snapshot.config_hash}).",
            {"config_path": snapshot.config_path, "config_hash": snapshot.config_hash},
        )
    )
    if config.cache.allow_stale_cache:
        checks.append(
            CheckResult(
                "cache_policy",
                STATUS_FAIL,
                "cache.allow_stale_cache is enabled; a competition run must never load a "
                "cache that does not match its data.",
            )
        )
    else:
        checks.append(CheckResult("cache_policy", STATUS_PASS, "Stale caches are rejected."))

    # --- data root and scope ------------------------------------------------------
    root = Path(config.dataset.root)
    if not root.is_dir():
        checks.append(
            CheckResult("data_root", STATUS_FAIL, f"DATA_ROOT does not exist: {root}")
        )
    else:
        checks.append(
            CheckResult(
                "data_root", STATUS_PASS, f"DATA_ROOT resolves to {snapshot.data_root_identity}."
            )
        )

    if snapshot.selected_video_count > 0:
        checks.append(
            CheckResult(
                "dataset_scope",
                STATUS_PASS,
                f"Scope {snapshot.scope_mode!r} selects {snapshot.selected_video_count} video(s).",
                {"selected_video_ids_hash": snapshot.selected_video_ids_hash},
            )
        )
    else:
        checks.append(
            CheckResult(
                "dataset_scope",
                STATUS_FAIL,
                f"Scope {snapshot.scope_mode!r} selects no videos; there is nothing to search.",
            )
        )

    # --- cache --------------------------------------------------------------------
    try:
        report = _cache_report(config, resolved_cache)
    except Exception as exc:  # noqa: BLE001
        report = {}
        checks.append(
            CheckResult("cache", STATUS_FAIL, f"Cache could not be inspected: {exc}")
        )
    if report:
        if not report.get("exists"):
            checks.append(
                CheckResult("cache", STATUS_FAIL, f"No cache at {resolved_cache}; build one first.")
            )
        elif report.get("corrupt"):
            checks.append(CheckResult("cache", STATUS_FAIL, "Cache manifest is corrupt."))
        elif report.get("legacy"):
            checks.append(
                CheckResult("cache", STATUS_FAIL, "Cache is legacy (no manifest); rebuild it.")
            )
        elif report.get("stale"):
            checks.append(
                CheckResult(
                    "cache",
                    STATUS_FAIL,
                    f"Cache is stale: {report.get('stale_reason') or 'fingerprint mismatch'}.",
                    {"mismatches": report.get("hard_mismatches", [])},
                )
            )
        else:
            checks.append(
                CheckResult(
                    "cache",
                    STATUS_PASS,
                    f"Cache is valid (fingerprint {snapshot.cache_fingerprint}).",
                    {"cache_dir": resolved_cache},
                )
            )

    # --- engine-dependent checks --------------------------------------------------
    if engine is None:
        checks.append(
            CheckResult(
                "engine",
                STATUS_WARN,
                "No engine was loaded, so channel availability and Q&A capability could "
                "not be measured from real data.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "engine",
                STATUS_PASS,
                f"Engine holds {int(engine.entry.num_indexed)} frames across "
                f"{snapshot.selected_video_count} video(s).",
            )
        )
        for name in REQUIRED_CHANNELS:
            info = snapshot.retrieval_channels.get(name, {})
            if info.get("usable"):
                checks.append(
                    CheckResult(
                        f"channel_{name}",
                        STATUS_PASS,
                        f"Channel {name} is available ({info.get('records')} records).",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        f"channel_{name}",
                        STATUS_FAIL,
                        f"Required channel {name} is unusable: {info.get('reason')}.",
                    )
                )
        for name in OPTIONAL_CHANNELS:
            info = snapshot.retrieval_channels.get(name, {})
            if info.get("usable"):
                checks.append(
                    CheckResult(
                        f"channel_{name}",
                        STATUS_PASS,
                        f"Channel {name} is available ({info.get('records')} records).",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        f"channel_{name}",
                        STATUS_WARN,
                        f"Channel {name} is not contributing: {info.get('reason')}.",
                    )
                )
        # Visual access is what makes local refinement and visual evidence possible.
        if getattr(engine, "frame_provider", None) is None:
            checks.append(
                CheckResult("frame_provider", STATUS_FAIL, "No frame provider is attached.")
            )
        else:
            checks.append(
                CheckResult("frame_provider", STATUS_PASS, "Frame provider is ready.")
            )

    # --- Q&A capability -----------------------------------------------------------
    if snapshot.qa.get("visual_capable"):
        checks.append(
            CheckResult(
                "qa_backend",
                STATUS_PASS,
                f"Q&A backend {snapshot.qa.get('backend')!r} can look at images.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "qa_backend",
                STATUS_WARN,
                f"No production visual Q&A backend is available (backend "
                f"{snapshot.qa.get('backend')!r}). Q&A retrieval and evidence work; "
                "engine-produced answers are not exportable.",
            )
        )

    # --- refinement cost ----------------------------------------------------------
    device = str(snapshot.kis.get("scorer_device", "auto"))
    if device in {"auto", "cpu"}:
        checks.append(
            CheckResult(
                "refinement_device",
                STATUS_WARN,
                "Local visual refinement runs on CPU here and costs seconds per query; "
                "it is off by default for that reason.",
                {"scorer_device": device},
            )
        )

    # --- submission validator -------------------------------------------------------
    try:
        from .submission_validation import validate_submission

        probe = validate_submission("kis", [["L21_V001", "1"]])
        if probe.valid:
            checks.append(
                CheckResult(
                    "submission_validator",
                    STATUS_PASS,
                    f"Submission validator v{SUBMISSION_VALIDATION_VERSION} is working for "
                    f"{', '.join(SUBMISSION_TASKS)}.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "submission_validator",
                    STATUS_FAIL,
                    "Submission validator rejected a known-good row.",
                )
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            CheckResult("submission_validator", STATUS_FAIL, f"Validator unavailable: {exc}")
        )

    # --- runtime identity -----------------------------------------------------------
    if engine is not None:
        engine_root = canonical_data_root(engine.app_config.dataset.root)
        if engine_root != snapshot.data_root_identity:
            checks.append(
                CheckResult(
                    "runtime_identity",
                    STATUS_FAIL,
                    f"Engine was built for {engine_root!r} but the config declares "
                    f"{snapshot.data_root_identity!r}.",
                )
            )
        else:
            checks.append(
                CheckResult("runtime_identity", STATUS_PASS, "Engine and config agree on the dataset.")
            )

    failures = [item for item in checks if item.status == STATUS_FAIL]
    warnings = [item for item in checks if item.status == STATUS_WARN]
    status = NOT_READY if failures else (READY_WITH_WARNINGS if warnings else READY)
    return ReadinessReport(status=status, checks=tuple(checks), profile=snapshot)


__all__ = [
    "NOT_READY",
    "OPTIONAL_CHANNELS",
    "READY",
    "READY_WITH_WARNINGS",
    "REQUIRED_CHANNELS",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_WARN",
    "SUBMISSION_VALIDATION_VERSION",
    "CheckResult",
    "ReadinessReport",
    "SystemProfile",
    "build_system_profile",
    "evaluate_readiness",
]
