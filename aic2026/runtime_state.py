"""One authoritative runtime dataset state for the running application.

Before Phase 4 the UI kept its dataset facts in a mutable dict plus module-level
`DATA_ROOT` / `AIC_CACHE_DIR` constants, and the "load another folder" endpoint swapped
the engine without updating anything else. A single request could then mix an engine
built from root B with a frame provider, video route, health report, and cache status
still pointing at root A.

This module makes that impossible by construction:

* `RuntimeDatasetState` is frozen. Every dataset-dependent fact — config, data root,
  cache directory, scope, resolved video IDs, engine, frame provider — lives in one
  immutable object, so there is no way to update half of it.
* `RuntimeStateManager` publishes states atomically behind a lock. A request takes one
  snapshot and uses only that snapshot, so it can never straddle two generations.
* Every published state carries a `generation`, which lets a client detect that the
  result it is holding predates a dataset switch.

Building a new state never mutates the active one: if any step fails, the caller keeps
serving the previous state.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .cache_manifest import (
    cache_build_options_from_config,
    canonical_data_root,
    detect_code_version,
    inspect_cache,
)
from .config import AppConfig, config_hash
from .dataset_scope import (
    ResolvedDatasetScope,
    hash_selected_video_ids,
    resolve_dataset_scope,
    scope_payload,
    select_video_ids,
)
from .engine import AICCompetitionEngine, AICLoadResult
from .frame_provider import FrameProvider
from .video_inventory import discover_videos

# Error codes shared by every dataset-dependent surface.
RUNTIME_STATE_UNINITIALIZED = "RUNTIME_STATE_UNINITIALIZED"
DATA_ROOT_INVALID = "DATA_ROOT_INVALID"
CACHE_INVALID = "CACHE_INVALID"
STATE_BUILD_FAILED = "STATE_BUILD_FAILED"
STALE_RESULT_GENERATION = "STALE_RESULT_GENERATION"
VIDEO_NOT_IN_ACTIVE_SCOPE = "VIDEO_NOT_IN_ACTIVE_SCOPE"
FRAME_NOT_AVAILABLE = "FRAME_NOT_AVAILABLE"
ACTIVE_STATE_NOT_CHANGED = "ACTIVE_STATE_NOT_CHANGED"


class RuntimeStateError(RuntimeError):
    """Raised when a runtime state cannot be built or is internally inconsistent."""

    def __init__(self, message: str, *, error_code: str = STATE_BUILD_FAILED) -> None:
        super().__init__(message)
        self.error_code = error_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def derived_cache_dir(app_config: AppConfig, data_root: str | Path, scope: Any) -> str:
    """Where a cache for a *different* data root should live.

    Switching DATA_ROOT must never silently reuse the cache built for another root.
    The manifest would reject it anyway, but routing a distinct directory means the
    user gets a clean build target instead of a confusing stale-cache error.
    """
    digest = hashlib.sha256(
        f"{canonical_data_root(data_root)}|{scope_payload(scope)}".encode("utf-8")
    ).hexdigest()[:12]
    base = Path(app_config.dataset.cache_dir)
    parent = base.parent if str(base.parent) not in {"", "."} else Path("artifacts")
    return str(parent / f"aic2026_index_{digest}")


def resolve_cache_dir(
    app_config: AppConfig,
    data_root: str | Path,
    *,
    explicit: str | Path | None = None,
    scope: Any = None,
) -> str:
    """Pick the cache directory for a root: explicit, configured, or derived."""
    if explicit:
        return str(explicit)
    if canonical_data_root(data_root) == canonical_data_root(app_config.dataset.root):
        # The configured cache belongs to the configured root.
        return str(app_config.dataset.cache_dir)
    return derived_cache_dir(app_config, data_root, scope or app_config.dataset.scope)


@dataclass(frozen=True)
class RuntimeDatasetState:
    """An immutable snapshot of everything the application knows about its dataset."""

    generation: int
    created_at_utc: str
    app_config: AppConfig
    config_path: str
    config_hash: str
    data_root: str
    cache_dir: str
    frame_cache_dir: str
    dataset_scope: dict[str, Any]
    resolved_video_ids: tuple[str, ...]
    selected_video_ids_hash: str
    frame_provider: FrameProvider
    engine: Optional[AICCompetitionEngine] = None
    load: Optional[AICLoadResult] = None
    cache_status: Optional[dict[str, Any]] = None
    dataset_status: Optional[dict[str, Any]] = None
    video_inventory_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def engine_loaded(self) -> bool:
        return self.engine is not None

    @property
    def selected_video_count(self) -> int:
        return len(self.resolved_video_ids)

    def knows_video(self, video_id: str) -> bool:
        """Is this video ID part of the ACTIVE selection? Used to gate file serving."""
        return str(video_id) in set(self.resolved_video_ids)

    def identity(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "data_root": self.data_root,
            "cache_dir": self.cache_dir,
            "config_hash": self.config_hash,
            "selected_video_ids_hash": self.selected_video_ids_hash,
        }

    def verify_engine_identity(self) -> None:
        """Fail loudly if the engine was built against a different dataset.

        This is the invariant the Phase 0 bug violated. It is cheap, so it runs on
        every state publication rather than being trusted implicitly.
        """
        if self.engine is None:
            return
        engine_root = canonical_data_root(self.engine.app_config.dataset.root)
        if engine_root != canonical_data_root(self.data_root):
            raise RuntimeStateError(
                "Runtime state is inconsistent: engine was built for data root "
                f"{engine_root!r} but the state declares {canonical_data_root(self.data_root)!r}."
            )
        provider_root = self.frame_provider.data_root
        if provider_root is not None and canonical_data_root(provider_root) != engine_root:
            raise RuntimeStateError(
                "Runtime state is inconsistent: frame provider serves data root "
                f"{canonical_data_root(provider_root)!r} but the engine was built for "
                f"{engine_root!r}."
            )
        # Phase 5: the local refiner decodes original MP4s, so it must never pair with a
        # frame provider from another generation.
        refiner = getattr(self.engine, "local_refiner", None)
        if refiner is not None and refiner.frame_provider is not self.frame_provider:
            raise RuntimeStateError(
                "Runtime state is inconsistent: the local refiner does not use this "
                "state's frame provider."
            )

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "created_at_utc": self.created_at_utc,
            "data_root": self.data_root,
            "cache_dir": self.cache_dir,
            "frame_cache_dir": self.frame_cache_dir,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "scope": self.dataset_scope,
            "selected_video_count": self.selected_video_count,
            "selected_video_ids_hash": self.selected_video_ids_hash,
            "engine_loaded": self.engine_loaded,
        }


def _resolve_selection(
    app_config: AppConfig, data_root: str | Path
) -> tuple[ResolvedDatasetScope, tuple[str, ...]]:
    """Resolve the configured scope against a root without building anything."""
    resolved = resolve_dataset_scope(app_config.dataset.scope, data_root)
    feature_dir = Path(data_root) / "clip-features-32"
    discovered = (
        sorted(path.stem for path in feature_dir.glob("*.npy")) if feature_dir.is_dir() else []
    )
    return resolved, select_video_ids(discovered, resolved, allow_empty=True)


def _dataset_status(load: AICLoadResult | None) -> dict[str, Any] | None:
    """Capability counts, taken from the build/load stats rather than re-inspecting."""
    if load is None or load.stats is None:
        return None
    stats = load.stats
    return {
        "selected_video_count": stats.videos,
        "discovered_video_count": stats.discovered_videos,
        "excluded_video_count": stats.excluded_videos,
        "frame_count": stats.frames,
        "retrieval_valid_count": stats.retrieval_valid_videos,
        "visual_accessible_count": stats.visual_accessible_videos,
        "refinement_ready_count": stats.refinement_ready_videos,
        "keyframe_jpeg_backed_count": stats.keyframe_jpeg_backed_videos,
        "video_fallback_count": stats.video_fallback_videos,
        "invalid_video_count": stats.invalid_video_count,
        "record_schema_version": stats.record_schema_version,
        "dataset_validated": stats.dataset_validated,
    }


def _cache_status_for(app_config: AppConfig, data_root: str | Path, cache_dir: str | Path) -> dict[str, Any]:
    """Inspect a cache directory without loading a pickle or an encoder."""
    expected = cache_build_options_from_config(app_config, data_root=data_root)
    report = inspect_cache(
        cache_dir,
        expected,
        validate_data_signature=app_config.cache.validate_data_signature,
        code_version_policy=app_config.cache.code_version_policy,
        current_code_version=detect_code_version(),
    )
    manifest = report.get("manifest") or {}
    return {
        "exists": report["exists"],
        "hit": False,
        "valid": report["valid"],
        "legacy": report["legacy"],
        "stale": report["stale"],
        "stale_reason": report.get("stale_reason"),
        "manifest_path": report["manifest_path"],
        "fingerprint": manifest.get("cache_fingerprint"),
        "created_at": manifest.get("created_at_utc"),
        "code_version": manifest.get("code_version"),
        "mismatches": report["hard_mismatches"],
        "warnings": report["warnings"],
    }


def build_runtime_state(
    *,
    app_config: AppConfig,
    config_path: str,
    generation: int,
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    engine: AICCompetitionEngine | None = None,
    load: AICLoadResult | None = None,
) -> RuntimeDatasetState:
    """Assemble a complete state. Never touches or mutates any active state.

    `app_config` is normalized so that its dataset root and cache directory match the
    state being built; that keeps engine identity, cache identity, and frame-provider
    identity from drifting apart.
    """
    root = str(data_root if data_root is not None else app_config.dataset.root)
    resolved_scope, _ = _resolve_selection(app_config, root)
    effective_cache_dir = resolve_cache_dir(
        app_config, root, explicit=cache_dir, scope=resolved_scope
    )
    effective_config = replace(
        app_config,
        dataset=replace(
            app_config.dataset, root=root, cache_dir=str(effective_cache_dir)
        ),
    )
    _, selected = _resolve_selection(effective_config, root)
    # An engine's own selection is authoritative once one exists: it is what actually
    # got indexed.
    if engine is not None:
        indexed = sorted({raw.video_id for raw in engine.entry.raws.values()})
        if indexed:
            selected = tuple(indexed)

    inventory = discover_videos(root)
    # One frame provider per generation. When an engine exists its provider is adopted
    # rather than duplicated, so the routes, the engine, and the local refiner all
    # decode from the same object — and switching DATA_ROOT replaces all three at once.
    frame_provider = (
        engine.frame_provider
        if engine is not None
        else FrameProvider(root, cache_dir=effective_config.dataset.frame_cache_dir)
    )
    state = RuntimeDatasetState(
        generation=int(generation),
        created_at_utc=_now(),
        app_config=effective_config,
        config_path=str(config_path),
        config_hash=config_hash(effective_config),
        data_root=canonical_data_root(root),
        cache_dir=str(effective_cache_dir),
        frame_cache_dir=str(effective_config.dataset.frame_cache_dir),
        dataset_scope=scope_payload(resolved_scope),
        resolved_video_ids=tuple(selected),
        selected_video_ids_hash=hash_selected_video_ids(selected),
        frame_provider=frame_provider,
        engine=engine,
        load=load,
        cache_status=(
            load.cache_status()
            if load is not None
            else _cache_status_for(effective_config, root, effective_cache_dir)
        ),
        dataset_status=_dataset_status(load),
        video_inventory_summary={
            "available_count": len(inventory.video_ids),
            "collections": inventory.collections,
            "total_bytes": inventory.total_bytes,
        },
    )
    state.verify_engine_identity()
    return state


class RuntimeStateManager:
    """Publishes runtime states atomically and hands out stable snapshots.

    Deliberately small: this is a single-process application, so a lock plus immutable
    snapshots is the whole correctness story. A request that started under generation N
    finishes under generation N, which is fine; what must never happen is one request
    mixing components from two generations.
    """

    def __init__(self, state: RuntimeDatasetState):
        self._lock = threading.RLock()
        self._state = state

    def get_state(self) -> RuntimeDatasetState:
        """Take one snapshot. Callers must reuse it for the whole request."""
        with self._lock:
            return self._state

    def replace_state(self, new_state: RuntimeDatasetState) -> RuntimeDatasetState:
        """Publish a fully built state. The old one stays active until this returns."""
        new_state.verify_engine_identity()
        with self._lock:
            self._state = new_state
            return self._state

    def next_generation(self) -> int:
        with self._lock:
            return self._state.generation + 1

    def build_and_replace(self, **kwargs: Any) -> RuntimeDatasetState:
        """Build a new state, then publish it only if construction fully succeeded."""
        with self._lock:
            current = self._state
            generation = kwargs.pop("generation", current.generation + 1)
            config_path = kwargs.pop("config_path", current.config_path)
        candidate = build_runtime_state(
            generation=generation, config_path=config_path, **kwargs
        )
        return self.replace_state(candidate)

    def status(self) -> dict[str, Any]:
        return self.get_state().runtime_summary()


def check_generation(state: RuntimeDatasetState, requested: Any) -> None:
    """Reject a request that was issued against a superseded dataset.

    Without this, a result URL captured before a DATA_ROOT switch could quietly resolve
    to an identically named frame from the *new* dataset.
    """
    if requested in (None, "", "None"):
        return
    try:
        wanted = int(requested)
    except (TypeError, ValueError):
        raise RuntimeStateError(
            f"Invalid generation {requested!r}; expected an integer.",
            error_code=STALE_RESULT_GENERATION,
        ) from None
    if wanted != state.generation:
        raise RuntimeStateError(
            f"This result was produced for dataset generation {wanted}, but generation "
            f"{state.generation} is active. Re-run the query against the active dataset.",
            error_code=STALE_RESULT_GENERATION,
        )


def safe_video_path(state: RuntimeDatasetState, video_id: str) -> Path:
    """Resolve a video ID to a file inside the ACTIVE data root, or refuse.

    Rejects traversal and separators outright, requires the ID to be part of the active
    selection, and finally confirms the resolved path really is inside the video
    directory. No filesystem path from the request ever reaches the filesystem.
    """
    identifier = str(video_id)
    if not identifier or identifier in {".", ".."}:
        raise RuntimeStateError(
            f"Invalid video id {video_id!r}.", error_code=VIDEO_NOT_IN_ACTIVE_SCOPE
        )
    if any(character in identifier for character in ("/", "\\", "\0")) or ".." in identifier:
        raise RuntimeStateError(
            f"Invalid video id {video_id!r}.", error_code=VIDEO_NOT_IN_ACTIVE_SCOPE
        )
    if not state.knows_video(identifier):
        raise RuntimeStateError(
            f"Video {identifier!r} is not part of the active dataset selection "
            f"(generation {state.generation}).",
            error_code=VIDEO_NOT_IN_ACTIVE_SCOPE,
        )
    video_dir = (Path(state.app_config.dataset.root) / "video").resolve(strict=False)
    candidate = (video_dir / f"{identifier}.mp4").resolve(strict=False)
    try:
        candidate.relative_to(video_dir)
    except ValueError:
        raise RuntimeStateError(
            f"Refusing to serve a path outside the active video directory: {video_id!r}.",
            error_code=VIDEO_NOT_IN_ACTIVE_SCOPE,
        ) from None
    return candidate


__all__ = [
    "ACTIVE_STATE_NOT_CHANGED",
    "CACHE_INVALID",
    "DATA_ROOT_INVALID",
    "FRAME_NOT_AVAILABLE",
    "RUNTIME_STATE_UNINITIALIZED",
    "STALE_RESULT_GENERATION",
    "STATE_BUILD_FAILED",
    "VIDEO_NOT_IN_ACTIVE_SCOPE",
    "RuntimeDatasetState",
    "RuntimeStateError",
    "RuntimeStateManager",
    "build_runtime_state",
    "check_generation",
    "derived_cache_dir",
    "resolve_cache_dir",
    "safe_video_path",
]
