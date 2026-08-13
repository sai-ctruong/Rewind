"""Named baseline configurations and the identity every experiment must record.

An experiment result is worthless if you cannot say what produced it. This module fixes
three named configurations and records, for each run, the exact code, configuration,
index, dataset, query set, label set, device and compute budgets behind it.

The configurations are DEFINED here and deliberately not tuned: `B0_RELEASE` is the
frozen release behaviour, `B0_CLEAN` is R0 (same rankings, less wasted work), and
`R1_ADAPTIVE` turns on the experimental controller. Comparing them requires ground
truth, so nothing here runs a semantic comparison — it only makes one reproducible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from aic2026.config import AppConfig, config_hash
from aic2026.dataset_scope import hash_selected_video_ids
from aic2026.version import PROJECT_VERSION, git_commit, git_is_dirty

# The frozen release. Recorded so a later run can prove it was compared against the
# right baseline, and so a drifted checkout is visible rather than assumed.
B0_RELEASE_COMMIT = "7dfe06e84197fe1805ce7acf25aa96eaa31eb7dc"
B0_RELEASE_TAG = "aic2026-competition-ready"

EXPERIMENT_B0_RELEASE = "B0_RELEASE"
EXPERIMENT_B0_CLEAN = "B0_CLEAN"
EXPERIMENT_R1_ADAPTIVE = "R1_ADAPTIVE"
EXPERIMENTS = (EXPERIMENT_B0_RELEASE, EXPERIMENT_B0_CLEAN, EXPERIMENT_R1_ADAPTIVE)

# Config overrides that DEFINE each named variant. Empty means "the config as written".
EXPERIMENT_OVERRIDES: dict[str, dict[str, Any]] = {
    EXPERIMENT_B0_RELEASE: {},
    EXPERIMENT_B0_CLEAN: {},
    EXPERIMENT_R1_ADAPTIVE: {"adaptive_budget": {"enabled": True}},
}

EXPERIMENT_NOTES: dict[str, str] = {
    EXPERIMENT_B0_RELEASE: (
        f"Exact behaviour of the frozen release {B0_RELEASE_COMMIT[:7]} "
        f"({B0_RELEASE_TAG}). Run it from a worktree at that tag; the overrides here are "
        "empty because the difference is the CODE, not the configuration."
    ),
    EXPERIMENT_B0_CLEAN: (
        "R0: dead UI and config removed, query caching and prewarm allowed. Verified to "
        "produce identical rankings to B0_RELEASE on the fixed fixture."
    ),
    EXPERIMENT_R1_ADAPTIVE: (
        "R1 controller enabled. EXPERIMENTAL. Its stages are evidence-producing today, "
        "so rankings match B0_CLEAN; that is a fact about the current stages, not a "
        "quality result."
    ),
}


def hash_payload(value: Any) -> str:
    """Stable SHA-256 of any JSON-able value; key order never changes the answer."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def hash_query_set(queries: Sequence[Any]) -> str:
    """Identity of the query set a run was scored on."""
    return hash_payload(list(queries))


@dataclass(frozen=True)
class ComputeBudget:
    """The hard ceilings in force for a run. Recorded, never inferred afterwards."""

    max_cost_units: Optional[float] = None
    kis_max_frames: Optional[int] = None
    trake_max_frames_per_query: Optional[int] = None
    trake_event_frame_cap: Optional[int] = None
    qa_max_vlm_calls_per_query: Optional[int] = None
    qa_max_visual_frames_per_call: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cost_units": self.max_cost_units,
            "kis_max_frames": self.kis_max_frames,
            "trake_max_frames_per_query": self.trake_max_frames_per_query,
            "trake_event_frame_cap": self.trake_event_frame_cap,
            "qa_max_vlm_calls_per_query": self.qa_max_vlm_calls_per_query,
            "qa_max_visual_frames_per_call": self.qa_max_visual_frames_per_call,
        }


def budget_of(config: AppConfig) -> ComputeBudget:
    return ComputeBudget(
        max_cost_units=float(config.adaptive_budget.max_cost_units),
        kis_max_frames=int(config.refinement.max_frames),
        trake_max_frames_per_query=int(config.trake.refinement_max_frames_per_query),
        trake_event_frame_cap=int(config.adaptive_budget.trake_event_frame_cap),
        qa_max_vlm_calls_per_query=int(config.qa.max_vlm_calls_per_query),
        qa_max_visual_frames_per_call=int(config.qa.max_visual_frames_per_call),
    )


def build_manifest(
    config: AppConfig,
    *,
    name: str,
    gt: Any = None,
    queries: Optional[Sequence[Any]] = None,
    load: Any = None,
    selected_video_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Everything needed to reproduce and to trust one experiment run."""
    manifest: dict[str, Any] = {
        "experiment": name,
        "note": EXPERIMENT_NOTES.get(name, ""),
        "project_version": PROJECT_VERSION,
        "git_commit": git_commit(short=False),
        "git_dirty": git_is_dirty(),
        "b0_release_commit": B0_RELEASE_COMMIT,
        "config_hash": config_hash(config),
        "scope_mode": str(config.dataset.scope.mode),
        "cache_dir": str(config.dataset.cache_dir),
        "cache_fingerprint": getattr(load, "cache_fingerprint", None),
        "selected_video_ids_hash": (
            hash_selected_video_ids(selected_video_ids) if selected_video_ids is not None else None
        ),
        "model": {
            "encoder": str(config.encoder.model_name),
            "device": str(config.runtime.device),
            "scorer": str(config.refinement.scorer_model_name),
            "qa_backend": str(config.qa.backend_type),
        },
        "compute_budget": budget_of(config).to_dict(),
        "adaptive_budget_enabled": bool(config.adaptive_budget.enabled),
        "query_set_hash": None if queries is None else hash_query_set(queries),
        "ground_truth": None,
        "ground_truth_hash": None,
    }
    if gt is not None:
        manifest["ground_truth"] = gt.provenance()
        manifest["ground_truth_hash"] = gt.content_hash()
        manifest["official_ground_truth"] = bool(gt.is_official)
        # The query set of a labelled run IS the labelled queries, so it is derived
        # rather than left None and later mistaken for "not recorded".
        if queries is None:
            manifest["query_set_hash"] = hash_query_set(
                [entry.query_id for entry in gt.real_entries]
            )
    return manifest


def comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Two runs are comparable only if they scored the same labels on the same index.

    A MISSING discriminator is treated as "cannot confirm", not as "identical". Two runs
    that never recorded their cache fingerprint are not thereby proven to have used the
    same cache, and silently calling them comparable is how a 29-video result gets
    compared against an 873-video one.
    """
    problems: list[str] = []
    for field_name, label in (
        ("config_hash", "different configuration"),
        ("scope_mode", "different dataset scope"),
        ("cache_fingerprint", "different index"),
        ("selected_video_ids_hash", "different dataset selection"),
        ("ground_truth_hash", "different labels"),
        ("query_set_hash", "different query set"),
    ):
        a, b = left.get(field_name), right.get(field_name)
        if a is None or b is None:
            problems.append(f"unverifiable: {field_name} not recorded on both runs")
        elif a != b:
            problems.append(f"{label} ({field_name})")
    return (not problems), problems


__all__ = [
    "B0_RELEASE_COMMIT",
    "B0_RELEASE_TAG",
    "EXPERIMENTS",
    "EXPERIMENT_B0_CLEAN",
    "EXPERIMENT_B0_RELEASE",
    "EXPERIMENT_NOTES",
    "EXPERIMENT_OVERRIDES",
    "EXPERIMENT_R1_ADAPTIVE",
    "ComputeBudget",
    "budget_of",
    "build_manifest",
    "comparable",
    "hash_payload",
    "hash_query_set",
]
