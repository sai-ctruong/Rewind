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
from fnmatch import fnmatchcase
from typing import Any, Iterable, Sequence

from .config import DatasetScopeConfig

FULL_DATASET_SCOPE = DatasetScopeConfig(include_patterns=("*",), exclude_patterns=())


class DatasetScopeError(ValueError):
    """Raised when a dataset scope is unusable against the discovered sources."""


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
    if not include:
        raise DatasetScopeError(
            'include_patterns must be non-empty; use ["*"] to select the full dataset'
        )
    return DatasetScopeConfig(include_patterns=include, exclude_patterns=exclude)


def matches_any(video_id: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(video_id, pattern) for pattern in patterns)


def select_video_ids(
    available_video_ids: Iterable[str],
    scope: DatasetScopeConfig | None = None,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return the sorted, de-duplicated video IDs a scope selects.

    Include rules run first, exclude rules afterwards. Duplicate patterns are
    harmless and the result never depends on filesystem iteration order. An empty
    selection from a non-empty collection is an explicit error unless ``allow_empty``
    is set, which callers use when they merely need cache facts for a data root that
    may not exist yet.
    """
    scope = normalize_scope(scope)
    available = sorted({str(item) for item in available_video_ids})
    selected = tuple(
        video_id
        for video_id in available
        if matches_any(video_id, scope.include_patterns)
        and not matches_any(video_id, scope.exclude_patterns)
    )
    if available and not selected and not allow_empty:
        raise DatasetScopeError(
            f"Dataset scope selected 0 of {len(available)} discovered video ID(s). "
            f"include_patterns={list(scope.include_patterns)}, "
            f"exclude_patterns={list(scope.exclude_patterns)}. "
            f"Example discovered IDs: {available[:5]}."
        )
    return selected


def excluded_video_ids(
    available_video_ids: Iterable[str], selected_video_ids: Iterable[str]
) -> tuple[str, ...]:
    selected = {str(item) for item in selected_video_ids}
    return tuple(sorted({str(item) for item in available_video_ids} - selected))


def scope_payload(scope: DatasetScopeConfig | None) -> dict[str, list[str]]:
    """JSON/manifest-safe view of a scope; the same shape everywhere it is recorded."""
    scope = normalize_scope(scope)
    return {
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
    "excluded_video_ids",
    "hash_selected_video_ids",
    "matches_any",
    "normalize_scope",
    "scope_from_payload",
    "scope_payload",
    "select_video_ids",
]
