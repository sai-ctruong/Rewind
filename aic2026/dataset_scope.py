"""Deterministic dataset scope selection for the AIC collection.

Two concepts are deliberately kept apart everywhere in the pipeline:

DISCOVERED SOURCES
    every canonical video ID that exists somewhere under ``DATA_ROOT``.

SELECTED DATASET
    the discovered IDs that the configured scope includes. This is the *only*
    validation domain: required sources are demanded for selected IDs and for no
    others, so developing on ``L21_*`` while ``L22_*``-``L30_*`` keyframe packages are
    still undownloaded is a legitimate, reproducible state rather than a corrupt one.

Patterns are ``fnmatch`` globs over the canonical video ID (``L21_V006``), never over
paths, and never hard-coded here: the batch/collection names live in configuration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import DATASET_SCOPE_MODES, DatasetScopeConfig
from .video_inventory import (
    existing_video_ids_with_retrieval_support,
    retrieval_ready_video_ids,
)

FULL_DATASET_SCOPE = DatasetScopeConfig(include_patterns=("*",), exclude_patterns=())


class DatasetScopeError(ValueError):
    """Raised when a dataset scope is unusable against the discovered sources."""


@dataclass(frozen=True)
class ResolvedDatasetScope:
    """A scope with every data-dependent part already resolved against DATA_ROOT.

    `source_video_ids` is the extra constraint imposed by a non-`patterns` mode:
    `None` means unconstrained, a tuple means "only these IDs may be selected". The
    patterns still apply on top, so `--scope-existing-videos --video-exclude X` works.
    """

    mode: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    source_video_ids: tuple[str, ...] | None = None


def _clean_patterns(patterns: Iterable[str] | None, name: str) -> tuple[str, ...]:
    cleaned: list[str] = []
    for pattern in patterns or ():
        if not isinstance(pattern, str) or not pattern.strip():
            raise DatasetScopeError(f"{name} entries must be non-empty strings, got {pattern!r}")
        text = pattern.strip()
        if "/" in text or "\\" in text:
            raise DatasetScopeError(
                f"{name} entries match a canonical video ID, not a path: {pattern!r}"
            )
        cleaned.append(text)
    return tuple(cleaned)


def normalize_scope(scope: DatasetScopeConfig | None) -> DatasetScopeConfig:
    """Validate a scope and return it with cleaned patterns."""
    if scope is None:
        return FULL_DATASET_SCOPE
    include = _clean_patterns(scope.include_patterns, "include_patterns")
    exclude = _clean_patterns(scope.exclude_patterns, "exclude_patterns")
    mode = str(getattr(scope, "mode", "patterns"))
    if mode not in DATASET_SCOPE_MODES:
        raise DatasetScopeError(
            f"Unknown dataset scope mode {mode!r}; expected one of {list(DATASET_SCOPE_MODES)}"
        )
    if not include:
        raise DatasetScopeError(
            'include_patterns must be non-empty; use ["*"] to select the full dataset'
        )
    return DatasetScopeConfig(include_patterns=include, exclude_patterns=exclude, mode=mode)


def resolve_dataset_scope(
    scope: DatasetScopeConfig | ResolvedDatasetScope | None,
    data_root: str | Path,
) -> ResolvedDatasetScope:
    """Turn a configured scope into one whose data-dependent parts are already read.

    Two data-dependent modes, deliberately separate:

    `retrieval_ready`
        canonical ID INTERSECT valid map INTERSECT valid CLIP feature. This is global
        retrieval coverage: what the coarse index can search at all.

    `existing_videos`
        the above INTERSECT an MP4 on disk. This is *visual* coverage: preview, local
        refinement and visual Q&A. Using it as the retrieval scope discards searchable
        videos for a reason retrieval does not care about.

    Resolution happens once per run so that inspection, index build, and cache identity
    all see the same IDs.
    """
    if isinstance(scope, ResolvedDatasetScope):
        return scope
    normalized = normalize_scope(scope)
    source_video_ids: tuple[str, ...] | None = None
    if normalized.mode == "retrieval_ready":
        source_video_ids = tuple(retrieval_ready_video_ids(data_root))
    elif normalized.mode == "existing_videos":
        source_video_ids = tuple(existing_video_ids_with_retrieval_support(data_root))
    return ResolvedDatasetScope(
        mode=normalized.mode,
        include_patterns=normalized.include_patterns,
        exclude_patterns=normalized.exclude_patterns,
        source_video_ids=source_video_ids,
    )


def matches_any(video_id: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(video_id, pattern) for pattern in patterns)


def select_video_ids(
    available_video_ids: Iterable[str],
    scope: DatasetScopeConfig | ResolvedDatasetScope | None = None,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return the sorted, de-duplicated video IDs a scope selects.

    The mode's source constraint applies first, then include rules, then exclude
    rules. Duplicate patterns are harmless and the result never depends on filesystem
    iteration order. An empty selection from a non-empty collection is an explicit
    error unless ``allow_empty`` is set, which callers use when they merely need cache
    facts for a data root that may not exist yet.
    """
    if isinstance(scope, ResolvedDatasetScope):
        resolved = scope
    else:
        normalized = normalize_scope(scope)
        if normalized.mode != "patterns":
            # Resolving needs DATA_ROOT, which this function does not take. Failing
            # loudly beats silently ignoring the mode and selecting too much.
            raise DatasetScopeError(
                f"Dataset scope mode {normalized.mode!r} must be resolved against "
                "DATA_ROOT first; call resolve_dataset_scope(scope, data_root)."
            )
        resolved = ResolvedDatasetScope(
            mode=normalized.mode,
            include_patterns=normalized.include_patterns,
            exclude_patterns=normalized.exclude_patterns,
        )
    available = sorted({str(item) for item in available_video_ids})
    allowed = (
        None if resolved.source_video_ids is None else {str(item) for item in resolved.source_video_ids}
    )
    selected = tuple(
        video_id
        for video_id in available
        if (allowed is None or video_id in allowed)
        and matches_any(video_id, resolved.include_patterns)
        and not matches_any(video_id, resolved.exclude_patterns)
    )
    if available and not selected and not allow_empty:
        constraint = (
            ""
            if allowed is None
            else f" scope mode {resolved.mode!r} allows {len(allowed)} source video ID(s)."
        )
        raise DatasetScopeError(
            f"Dataset scope selected 0 of {len(available)} discovered video ID(s). "
            f"include_patterns={list(resolved.include_patterns)}, "
            f"exclude_patterns={list(resolved.exclude_patterns)}.{constraint} "
            f"Example discovered IDs: {available[:5]}."
        )
    return selected


def excluded_video_ids(
    available_video_ids: Iterable[str], selected_video_ids: Iterable[str]
) -> tuple[str, ...]:
    selected = {str(item) for item in selected_video_ids}
    return tuple(sorted({str(item) for item in available_video_ids} - selected))


def scope_payload(scope: DatasetScopeConfig | ResolvedDatasetScope | None) -> dict[str, Any]:
    """JSON/manifest-safe view of a scope; the same shape everywhere it is recorded.

    The resolved ID set is intentionally NOT folded in here — it is recorded separately
    as `selected_video_ids` / `selected_video_ids_hash`, which is what the cache
    fingerprint depends on. The mode string alone never identifies a cache.
    """
    if isinstance(scope, ResolvedDatasetScope):
        return {
            "mode": scope.mode,
            "include_patterns": list(scope.include_patterns),
            "exclude_patterns": list(scope.exclude_patterns),
        }
    scope = normalize_scope(scope)
    return {
        "mode": scope.mode,
        "include_patterns": list(scope.include_patterns),
        "exclude_patterns": list(scope.exclude_patterns),
    }


def scope_from_payload(payload: Any) -> DatasetScopeConfig:
    if not isinstance(payload, dict):
        raise DatasetScopeError(f"dataset scope payload must be an object, got {type(payload).__name__}")
    return normalize_scope(
        DatasetScopeConfig(
            include_patterns=_clean_patterns(payload.get("include_patterns") or (), "include_patterns"),
            exclude_patterns=_clean_patterns(payload.get("exclude_patterns") or (), "exclude_patterns"),
            mode=str(payload.get("mode", "patterns")),
        )
    )


def hash_selected_video_ids(video_ids: Iterable[str]) -> str:
    """Stable hash of a selection; input order and duplicates never change it."""
    payload = json.dumps(
        sorted({str(item) for item in video_ids}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "FULL_DATASET_SCOPE",
    "DatasetScopeError",
    "ResolvedDatasetScope",
    "excluded_video_ids",
    "hash_selected_video_ids",
    "matches_any",
    "normalize_scope",
    "resolve_dataset_scope",
    "scope_from_payload",
    "scope_payload",
    "select_video_ids",
]
