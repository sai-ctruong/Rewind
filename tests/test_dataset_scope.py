"""Phase 3.1: dataset scope selection, scoped validation, and scoped cache identity.

All fixtures are tiny temporary directories; nothing here touches the real dataset,
a GPU, a model download, or the network.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from aic2026.benchmark import BenchmarkLogger, QueryLog
from aic2026.cache_manifest import (
    CACHE_MANIFEST_FILENAME,
    cache_build_options_from_config,
    cache_fingerprint,
    read_cache_manifest,
)
from aic2026.cli import main as cli_main
from aic2026.config import ConfigError, DatasetScopeConfig, app_config_from_dict
from aic2026.dataset_scope import (
    DatasetScopeError,
    excluded_video_ids,
    hash_selected_video_ids,
    scope_payload,
    select_video_ids,
)
from aic2026.dataset_validation import inspect_aic_dataset
from aic2026.engine import AICCompetitionEngine
from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION

AVAILABLE = ("L21_V001", "L21_V002", "L22_V001", "L30_V999")


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def scope(include=("*",), exclude=()) -> DatasetScopeConfig:
    return DatasetScopeConfig(include_patterns=tuple(include), exclude_patterns=tuple(exclude))


def make_video(root: Path, video_id: str, *, write_images: bool = True) -> Path:
    for relative in ("map-keyframes", "clip-features-32", f"keyframes/{video_id}"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "map-keyframes" / f"{video_id}.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,1.0,30.0,30\n", encoding="utf-8"
    )
    np.save(
        root / "clip-features-32" / f"{video_id}.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    if write_images:
        for ordinal in (1, 2):
            Image.new("RGB", (8, 8), (ordinal * 60, 0, 0)).save(
                root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
            )
    return root


def make_collection(root: Path, video_ids=AVAILABLE, *, without_images=()) -> Path:
    for video_id in video_ids:
        make_video(root, video_id, write_images=video_id not in without_images)
    return root


def make_config(root: Path, *, cache_dir: Path | None = None, include=("*",), exclude=()):
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir or root.parent / "cache"),
                    "scope": {
                        "include_patterns": list(include),
                        "exclude_patterns": list(exclude),
                    },
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
            }
        }
    )


def write_cli_config(path: Path, root: Path, *, include=("*",), exclude=(), cache_dir: Path | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "aic2026": {
                    "dataset": {
                        "root": str(root),
                        "cache_dir": str(cache_dir or root.parent / "cache"),
                        "scope": {
                            "include_patterns": list(include),
                            "exclude_patterns": list(exclude),
                        },
                        "validation": {"expected_feature_dim": 2},
                    },
                    "encoder": {"feature_dim": 2},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------- selection


def test_include_one_collection_selects_only_that_collection() -> None:
    assert select_video_ids(AVAILABLE, scope(["L21_*"])) == ("L21_V001", "L21_V002")


def test_include_star_selects_everything() -> None:
    assert select_video_ids(AVAILABLE, scope(["*"])) == tuple(sorted(AVAILABLE))


def test_default_scope_is_the_full_dataset() -> None:
    assert select_video_ids(AVAILABLE, None) == tuple(sorted(AVAILABLE))


def test_two_include_patterns_select_their_union() -> None:
    selected = select_video_ids(AVAILABLE, scope(["L21_*", "L22_*"]))
    assert selected == ("L21_V001", "L21_V002", "L22_V001")


def test_exclude_is_applied_after_include() -> None:
    selected = select_video_ids(AVAILABLE, scope(["*"], ["L30_*"]))
    assert selected == ("L21_V001", "L21_V002", "L22_V001")
    assert select_video_ids(AVAILABLE, scope(["L21_*"], ["L21_V002"])) == ("L21_V001",)


def test_scope_selecting_nothing_raises_a_clear_error() -> None:
    with pytest.raises(DatasetScopeError, match="selected 0 of 4"):
        select_video_ids(AVAILABLE, scope(["L99_*"]))


def test_empty_include_patterns_are_rejected_by_config() -> None:
    with pytest.raises(ConfigError, match="include_patterns"):
        app_config_from_dict({"aic2026": {"dataset": {"scope": {"include_patterns": []}}}})


def test_path_like_pattern_is_rejected() -> None:
    with pytest.raises(ConfigError, match="video ID, not a path"):
        app_config_from_dict(
            {"aic2026": {"dataset": {"scope": {"include_patterns": ["keyframes/L21_*"]}}}}
        )


def test_selection_is_deterministic_regardless_of_input_order_and_duplicates() -> None:
    shuffled = ["L30_V999", "L21_V002", "L21_V001", "L21_V001", "L22_V001"]
    repeated_patterns = scope(["L21_*", "L21_*", "L22_*"], ["L30_*", "L30_*"])
    first = select_video_ids(shuffled, repeated_patterns)
    second = select_video_ids(sorted(shuffled, reverse=True), repeated_patterns)
    assert first == second == ("L21_V001", "L21_V002", "L22_V001")


def test_excluded_ids_are_the_complement_of_the_selection() -> None:
    selected = select_video_ids(AVAILABLE, scope(["L21_*"]))
    assert excluded_video_ids(AVAILABLE, selected) == ("L22_V001", "L30_V999")


# -------------------------------------------------------------- scoped validation


def test_missing_keyframes_outside_scope_do_not_invalidate_the_selection(tmp_path) -> None:
    root = make_collection(tmp_path / "data", without_images=("L22_V001", "L30_V999"))
    report = inspect_aic_dataset(root, app_config=make_config(root, include=["L21_*"]))
    assert report.valid_for_index_build
    assert report.selected_video_count == 2
    assert report.invalid_video_count == 0
    assert "KEYFRAME_IMAGE_MISSING" not in report.issue_counts
    assert "REQUIRED_SOURCE_MISSING" not in report.issue_counts


def test_missing_keyframe_inside_scope_still_invalidates_the_selection(tmp_path) -> None:
    root = make_collection(tmp_path / "data", without_images=("L21_V002",))
    report = inspect_aic_dataset(root, app_config=make_config(root, include=["L21_*"]))
    assert not report.valid_for_index_build
    assert report.invalid_selected_video_ids == ["L21_V002"]
    assert report.videos[1].missing_keyframe_ordinals == [1, 2]


def test_report_counts_discovered_selected_and_excluded_separately(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    report = inspect_aic_dataset(root, app_config=make_config(root, include=["L21_*"]))
    assert report.discovered_video_count == 4
    assert report.selected_video_count == report.video_count == 2
    assert report.excluded_video_count == 2
    assert report.excluded_video_ids_sample == ["L22_V001", "L30_V999"]
    assert report.scope == {"include_patterns": ["L21_*"], "exclude_patterns": []}
    assert report.selected_video_ids_hash == hash_selected_video_ids(["L21_V001", "L21_V002"])


def test_scope_selecting_nothing_is_reported_as_an_error_not_an_empty_pass(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    with pytest.raises(DatasetScopeError):
        inspect_aic_dataset(root, app_config=make_config(root, include=["L99_*"]))


# ------------------------------------------------------------------------- CLI


def test_cli_scope_override_beats_yaml(tmp_path, capsys) -> None:
    root = make_collection(tmp_path / "data")
    config_path = write_cli_config(tmp_path / "settings.yaml", root, include=["*"])
    original = config_path.read_text(encoding="utf-8")
    assert cli_main(["--config", str(config_path), "--video-include", "L21_*", "inspect-data"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"]["include_patterns"] == ["L21_*"]
    assert payload["selected_video_count"] == 2
    assert payload["discovered_video_count"] == 4
    # The override is runtime-only.
    assert config_path.read_text(encoding="utf-8") == original


def test_cli_without_override_keeps_the_yaml_scope(tmp_path, capsys) -> None:
    root = make_collection(tmp_path / "data")
    config_path = write_cli_config(tmp_path / "settings.yaml", root, include=["L22_*"])
    assert cli_main(["--config", str(config_path), "inspect-data"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"]["include_patterns"] == ["L22_*"]
    assert payload["selected_video_count"] == 1


def test_cli_repeated_include_and_exclude_options(tmp_path, capsys) -> None:
    root = make_collection(tmp_path / "data")
    config_path = write_cli_config(tmp_path / "settings.yaml", root, include=["*"])
    code = cli_main(
        [
            "--config", str(config_path),
            "--video-include", "L21_*",
            "--video-include", "L22_*",
            "--video-exclude", "L21_V002",
            "inspect-data",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"] == {
        "include_patterns": ["L21_*", "L22_*"],
        "exclude_patterns": ["L21_V002"],
    }
    assert payload["selected_video_count"] == 2


def test_cli_empty_scope_exits_with_a_dataset_error(tmp_path, capsys) -> None:
    root = make_collection(tmp_path / "data")
    config_path = write_cli_config(tmp_path / "settings.yaml", root)
    assert cli_main(["--config", str(config_path), "--video-include", "L99_*", "inspect-data"]) == 5
    assert "EMPTY_DATASET_SCOPE" in capsys.readouterr().err


# ------------------------------------------------------------------ cache identity


def test_scoped_cache_fingerprint_differs_from_full_collection(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    full = cache_fingerprint(make_config(root, include=["*"]))
    scoped = cache_fingerprint(make_config(root, include=["L21_*"]))
    assert full != scoped


def test_two_different_collections_have_different_fingerprints(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    l21 = cache_fingerprint(make_config(root, include=["L21_*"]))
    l22 = cache_fingerprint(make_config(root, include=["L22_*"]))
    both = cache_fingerprint(make_config(root, include=["L21_*", "L22_*"]))
    assert len({l21, l22, both}) == 3


def test_exclude_pattern_changes_the_fingerprint(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    without_exclude = cache_fingerprint(make_config(root, include=["*"]))
    with_exclude = cache_fingerprint(make_config(root, include=["*"], exclude=["L30_*"]))
    assert without_exclude != with_exclude


def test_selected_video_ids_hash_ignores_order_and_duplicates() -> None:
    assert hash_selected_video_ids(["L21_V002", "L21_V001", "L21_V001"]) == hash_selected_video_ids(
        ["L21_V001", "L21_V002"]
    )
    assert hash_selected_video_ids(["L21_V001"]) != hash_selected_video_ids(["L21_V002"])


def test_query_time_ranking_change_still_does_not_change_the_fingerprint(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    config = make_config(root, include=["L21_*"])
    changed = replace(config, ranking=replace(config.ranking, final_top_k=7))
    assert cache_fingerprint(config) == cache_fingerprint(changed)


def test_record_schema_v3_makes_a_v2_cache_stale(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    options = cache_build_options_from_config(make_config(root, include=["L21_*"]))
    assert options.record_schema_version == AIC_RECORD_SCHEMA_VERSION == 3
    older = replace(options, record_schema_version=2)
    assert cache_fingerprint(older) != cache_fingerprint(options)


def test_built_manifest_records_scope_and_selected_ids_hash(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    cache = tmp_path / "cache_l21"
    config = make_config(root, cache_dir=cache, include=["L21_*"])
    _, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    manifest = read_cache_manifest(cache / CACHE_MANIFEST_FILENAME)
    assert manifest.dataset_scope == {"include_patterns": ["L21_*"], "exclude_patterns": []}
    assert manifest.selected_video_count == 2
    assert manifest.selected_video_ids_hash == hash_selected_video_ids(["L21_V001", "L21_V002"])
    assert manifest.record_schema_version == AIC_RECORD_SCHEMA_VERSION
    assert load.stats.discovered_videos == 4
    assert load.stats.excluded_videos == 2
    assert load.stats.scope_include_patterns == ("L21_*",)
    # The build really is scoped: only L21 keyframes are indexed.
    assert {raw.video_id for raw in load.entry.raws.values()} == {"L21_V001", "L21_V002"}


def test_a_scoped_cache_is_rejected_for_a_different_scope(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    cache = tmp_path / "cache_l21"
    AICCompetitionEngine.from_data_root(
        app_config=make_config(root, cache_dir=cache, include=["L21_*"]),
        text_encoder=TinyTextEncoder(),
        rebuild=True,
    )
    from aic2026.cache_manifest import StaleCacheError

    with pytest.raises(StaleCacheError, match="dataset_scope"):
        AICCompetitionEngine.from_data_root(
            app_config=make_config(root, cache_dir=cache, include=["L22_*"]),
            text_encoder=TinyTextEncoder(),
        )


def test_benchmark_run_records_the_dataset_scope(tmp_path) -> None:
    root = make_collection(tmp_path / "data")
    cache = tmp_path / "cache_l21"
    config = make_config(root, cache_dir=cache, include=["L21_*"])
    _, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    run = BenchmarkLogger(tmp_path / "bench").write_run(
        "scope",
        {},
        [QueryLog("kis", "1", "q", 1.0, [["L21_V001", "0"]])],
        cache_manifest=load.cache_manifest,
        dataset_report=load.stats.dataset_report_path,
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    assert summary["dataset_scope"] == {"include_patterns": ["L21_*"], "exclude_patterns": []}
    assert summary["selected_video_count"] == 2
    assert summary["selected_video_ids_hash"] == load.cache_manifest.selected_video_ids_hash
    assert environment["dataset_validation"]["dataset_scope"] == scope_payload(config.dataset.scope)
