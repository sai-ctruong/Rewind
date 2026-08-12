"""Phase 11: the reproducibility identity of a runtime.

A submission is only trustworthy if you can say what produced it. These tests pin the
contents of `SystemProfile`, its determinism, and what it does when parts of the system
are missing. Nothing here evaluates retrieval quality.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aic2026.retrieval_channels import CHANNEL_SCHEMA_VERSION
from aic2026.system_profile import (
    SUBMISSION_VALIDATION_VERSION,
    build_system_profile,
)
from aic2026.version import PROJECT_VERSION, RELEASE_TAG, VERSION, git_commit, git_is_dirty
from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION
from tests.release_support import (
    FakeEngine,
    build_engine,
    channel_status,
    make_config,
    make_data,
    qa_status,
)


@pytest.fixture()
def config(tmp_path):
    return make_config(make_data(tmp_path / "data"), tmp_path / "cache")


# --------------------------------------------------------------------------- version


def test_version_strings_are_consistent() -> None:
    assert VERSION == "0.11.0"
    assert PROJECT_VERSION.startswith(VERSION)
    assert PROJECT_VERSION.endswith("-aic2026")
    assert RELEASE_TAG


def test_pyproject_version_matches_version_module() -> None:
    """The packaged version must not drift from the module the profile reports."""
    from pathlib import Path

    text = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert f'version = "{VERSION}"' in text


def test_git_helpers_degrade_instead_of_raising(monkeypatch) -> None:
    """Outside a git checkout the profile still builds; the commit is simply unknown."""
    import subprocess

    def boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert git_commit() is None
    assert git_is_dirty() is None


# --------------------------------------------------------------------------- contents


def test_profile_records_full_identity(config) -> None:
    profile = build_system_profile(config, config_path="configs/test.yaml")
    assert profile.project_version == PROJECT_VERSION
    assert profile.config_path == "configs/test.yaml"
    assert profile.config_hash
    assert profile.record_schema_version == AIC_RECORD_SCHEMA_VERSION
    assert profile.channel_schema_version == CHANNEL_SCHEMA_VERSION
    assert profile.submission_validation_version == SUBMISSION_VALIDATION_VERSION
    assert profile.python_version and profile.platform
    assert profile.data_root_identity


def test_profile_reports_every_task_block(config) -> None:
    profile = build_system_profile(config)
    assert set(profile.kis) >= {"refinement_mode", "candidate_budget", "frame_output_policy"}
    assert set(profile.qa) >= {"backend", "visual_capable", "top_video_hypotheses"}
    assert set(profile.trake) >= {"alignment_method", "k_best_per_video", "refinement_enabled"}
    assert profile.submission["validation_version"] == SUBMISSION_VALIDATION_VERSION
    assert profile.submission["frame_policy"] == "submission_frame_idx"
    assert set(profile.retrieval_channels) >= {"clip", "bm25", "objects", "metadata"}


def test_profile_is_json_serializable_and_deterministic(config) -> None:
    first = build_system_profile(config).to_dict()
    second = build_system_profile(config).to_dict()
    assert first == second
    assert json.loads(json.dumps(first, ensure_ascii=False)) == first


def test_identity_subset_is_the_reproducibility_key(config) -> None:
    identity = build_system_profile(config).identity()
    assert set(identity) == {
        "project_version",
        "git_commit",
        "config_hash",
        "cache_fingerprint",
        "selected_video_ids_hash",
        "record_schema_version",
        "channel_schema_version",
        "submission_validation_version",
    }


def test_identity_changes_when_the_config_changes(config) -> None:
    changed = replace(config, dataset=replace(config.dataset, load_objects=True))
    before = build_system_profile(config).identity()
    after = build_system_profile(changed).identity()
    assert before["config_hash"] != after["config_hash"]
    assert before["cache_fingerprint"] != after["cache_fingerprint"]


def test_query_time_change_moves_config_hash_but_not_cache_fingerprint(config) -> None:
    """Build-time and query-time identity are deliberately separate."""
    changed = replace(config, ranking=replace(config.ranking, final_top_k=7))
    before = build_system_profile(config)
    after = build_system_profile(changed)
    assert before.config_hash != after.config_hash
    assert before.cache_fingerprint == after.cache_fingerprint


# ------------------------------------------------------------------- missing pieces


def test_profile_survives_a_missing_cache(tmp_path) -> None:
    config = make_config(make_data(tmp_path / "data"), tmp_path / "nope")
    profile = build_system_profile(config)
    assert profile.cache_valid in (None, False)
    assert profile.cache_schema_version is None
    # A fingerprint is still derivable from the config alone: it says what the cache
    # *should* be, which is exactly what a rebuild needs.
    assert profile.cache_fingerprint


def test_profile_survives_a_missing_data_root(tmp_path) -> None:
    config = make_config(tmp_path / "absent", tmp_path / "cache")
    profile = build_system_profile(config)
    assert profile.selected_video_count == 0
    assert profile.data_root_identity


def test_channels_are_marked_unmeasured_without_an_engine(config) -> None:
    profile = build_system_profile(config)
    clip = profile.retrieval_channels["clip"]
    assert clip["available"] is None
    assert clip["reason"] == "engine not loaded"
    assert profile.selected_video_count == 0


# ------------------------------------------------------------------------- engine


def test_engine_supplies_measured_channels_and_video_ids(config) -> None:
    engine = FakeEngine(config, video_ids=("L01_V001", "L02_V003"), frames=3)
    profile = build_system_profile(config, engine=engine)
    assert profile.selected_video_count == 2
    assert profile.selected_video_ids_hash
    assert profile.retrieval_channels["clip"]["usable"] is True
    assert profile.retrieval_channels["ocr"]["usable"] is False


def test_selected_video_ids_hash_is_order_independent(config) -> None:
    forward = build_system_profile(
        config, engine=FakeEngine(config, video_ids=("L01_V001", "L02_V003"))
    )
    backward = build_system_profile(
        config, engine=FakeEngine(config, video_ids=("L02_V003", "L01_V001"))
    )
    assert forward.selected_video_ids_hash == backward.selected_video_ids_hash


def test_qa_block_reports_the_real_backend_capability(config) -> None:
    mock = build_system_profile(config, engine=FakeEngine(config, qa=qa_status(visual=False)))
    real = build_system_profile(
        config, engine=FakeEngine(config, qa=qa_status(visual=True, backend_type="vlm"))
    )
    assert mock.qa["visual_capable"] is False and mock.qa["production_ready"] is False
    assert real.qa["visual_capable"] is True and real.qa["backend"] == "vlm"


def test_runtime_generation_is_carried_through(config) -> None:
    assert build_system_profile(config, runtime_generation=4).runtime_generation == 4


def test_profile_never_contains_secret_looking_values(config) -> None:
    engine = FakeEngine(config, channels=channel_status(optional=True))
    text = json.dumps(build_system_profile(config, engine=engine).to_dict()).lower()
    for needle in ("api_key", "api-key", "secret", "token", "password", "authorization"):
        assert needle not in text


# ---------------------------------------------------------------------- real engine


def test_real_engine_profile_matches_the_index(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    profile = build_system_profile(config, engine=engine)
    assert profile.selected_video_count == 1
    assert profile.cache_valid is True
    assert profile.cache_schema_version is not None
    assert profile.retrieval_channels["clip"]["usable"] is True
