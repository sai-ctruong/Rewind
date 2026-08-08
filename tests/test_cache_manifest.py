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
    CACHE_MANIFEST_SCHEMA_VERSION,
    CacheManifestError,
    LegacyCacheError,
    StaleCacheError,
    build_data_signature,
    cache_build_options_from_config,
    cache_fingerprint,
    hash_video_ids,
    inspect_cache,
    mismatch_message,
    read_cache_manifest,
    validate_cache_manifest,
    write_cache_manifest_atomic,
)
from aic2026.cli import main as cli_main
from aic2026.config import (
    CacheConfig,
    app_config_from_dict,
    config_to_dict,
)
from aic2026.engine import AICCompetitionEngine
import ui.app as appmod


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def make_data(root: Path) -> Path:
    for relative in (
        "clip-features-32",
        "map-keyframes",
        "keyframes/L01_V001",
        "objects/L01_V001",
        "media-info",
        "video",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "map-keyframes" / "L01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30,100\n2,1.0,30,130\n",
        encoding="utf-8",
    )
    np.save(
        root / "clip-features-32" / "L01_V001.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    for ordinal in (1, 2):
        Image.new("RGB", (8, 8), (ordinal * 64, 0, 0)).save(
            root / "keyframes" / "L01_V001" / f"{ordinal:03d}.jpg"
        )
        (root / "objects" / "L01_V001" / f"{ordinal:03d}.json").write_text(
            "{}", encoding="utf-8"
        )
    (root / "media-info" / "L01_V001.json").write_text("{}", encoding="utf-8")
    return root


def make_config(root: Path, cache: Path, **aic_overrides):
    data = {
        "aic2026": {
            "runtime": {"production_mode": False, "seed": 42, "device": "auto"},
            "dataset": {
                "root": str(root),
                "cache_dir": str(cache),
                "load_objects": False,
                "include_media_text": False,
                "verify_keyframes": False,
                "index_kind": "flat",
            },
            "cache": {
                "allow_stale_cache": False,
                "validate_data_signature": True,
                "code_version_policy": "warn",
            },
            "encoder": {
                "type": "auto_clip",
                "model_name": "openai/clip-vit-base-patch32",
                "feature_dim": 2,
                "normalize": True,
                "batch_size": 2,
            },
            "ranking": {"final_top_k": 25},
            "evaluation": {"ks": [1, 5, 20]},
        }
    }
    for section, values in aic_overrides.items():
        data["aic2026"].setdefault(section, {}).update(values)
    return app_config_from_dict(data)


def build_cache(tmp_path: Path):
    root = make_data(tmp_path / "data")
    cache = tmp_path / "cache"
    cfg = make_config(root, cache)
    _, load = AICCompetitionEngine.from_data_root(
        app_config=cfg, text_encoder=TinyTextEncoder(), rebuild=True
    )
    return root, cache, cfg, load


def write_config(path: Path, cfg) -> Path:
    path.write_text(
        yaml.safe_dump({"aic2026": config_to_dict(cfg)}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_build_manifest_is_valid(tmp_path) -> None:
    _, cache, _, load = build_cache(tmp_path)
    assert load.cache_valid and not load.cache_hit
    assert (cache / CACHE_MANIFEST_FILENAME).is_file()
    assert load.cache_manifest.schema_version == CACHE_MANIFEST_SCHEMA_VERSION


def test_validate_matching_manifest_passes(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    manifest = read_cache_manifest(cache / CACHE_MANIFEST_FILENAME)
    expected = cache_build_options_from_config(cfg)
    result = validate_cache_manifest(
        manifest,
        expected,
        current_data_signature=expected.data_signature,
        current_video_ids_hash=expected.video_ids_hash,
        cache_dir=cache,
    )
    assert result.valid and not result.hard_mismatches


def test_manifest_signature_and_video_hash_are_deterministic(tmp_path) -> None:
    root = make_data(tmp_path / "data")
    kwargs = dict(
        video_ids=["L01_V001"], include_objects=True, include_media_text=True
    )
    assert build_data_signature(root, **kwargs) == build_data_signature(root, **kwargs)
    assert hash_video_ids(["b", "a", "a"]) == hash_video_ids(["a", "b"])


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("data_root", "X:/different"),
        ("video_count", 2),
        ("frame_count", 3),
        ("feature_dim", 3),
        ("load_objects", True),
        ("include_media_text", True),
        ("index_kind", "hnsw"),
        ("encoder_feature_space", "different-space"),
    ],
)
def test_required_manifest_field_mismatch_is_hard(tmp_path, field, changed) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    manifest = replace(read_cache_manifest(cache / CACHE_MANIFEST_FILENAME), **{field: changed})
    expected = cache_build_options_from_config(cfg)
    result = validate_cache_manifest(
        manifest,
        expected,
        current_data_signature=expected.data_signature,
        current_video_ids_hash=expected.video_ids_hash,
    )
    assert not result.valid
    assert field in {item.field for item in result.hard_mismatches}


def test_query_time_ranking_change_does_not_change_fingerprint(tmp_path) -> None:
    root = make_data(tmp_path / "data")
    cfg = make_config(root, tmp_path / "cache")
    changed = replace(cfg, ranking=replace(cfg.ranking, final_top_k=10))
    assert cache_fingerprint(cfg) == cache_fingerprint(changed)


def test_query_time_evaluation_change_does_not_change_fingerprint(tmp_path) -> None:
    root = make_data(tmp_path / "data")
    cfg = make_config(root, tmp_path / "cache")
    changed = replace(cfg, evaluation=replace(cfg.evaluation, ks=(1, 10)))
    assert cache_fingerprint(cfg) == cache_fingerprint(changed)


@pytest.mark.parametrize("change", ["objects", "dimension", "root"])
def test_build_affecting_change_changes_fingerprint(tmp_path, change) -> None:
    root = make_data(tmp_path / "data")
    cfg = make_config(root, tmp_path / "cache")
    if change == "objects":
        changed = replace(cfg, dataset=replace(cfg.dataset, load_objects=True))
    elif change == "dimension":
        changed = replace(cfg, encoder=replace(cfg.encoder, feature_dim=3))
    else:
        changed = replace(cfg, dataset=replace(cfg.dataset, root=str(tmp_path / "other")))
    assert cache_fingerprint(cfg) != cache_fingerprint(changed)


def test_legacy_cache_rejected_by_default(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    (cache / CACHE_MANIFEST_FILENAME).unlink()
    with pytest.raises(LegacyCacheError, match="Legacy cache"):
        AICCompetitionEngine.from_data_root(
            app_config=cfg, text_encoder=TinyTextEncoder()
        )


def test_legacy_cache_explicitly_allowed_with_warning(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    (cache / CACHE_MANIFEST_FILENAME).unlink()
    cfg = replace(cfg, cache=replace(cfg.cache, allow_stale_cache=True))
    with pytest.warns(RuntimeWarning, match="legacy cache"):
        _, load = AICCompetitionEngine.from_data_root(
            app_config=cfg, text_encoder=TinyTextEncoder()
        )
    assert load.cache_hit and load.cache_legacy and load.cache_stale
    assert not load.cache_valid and load.cache_warnings


def test_corrupt_manifest_rejected(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    (cache / CACHE_MANIFEST_FILENAME).write_text("{bad", encoding="utf-8")
    with pytest.raises(CacheManifestError, match="Corrupt"):
        AICCompetitionEngine.from_data_root(
            app_config=cfg, text_encoder=TinyTextEncoder()
        )


def test_missing_referenced_cache_file_is_corrupt(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    (cache / "resolved_config.json").unlink()
    report = inspect_cache(cache, cache_build_options_from_config(cfg))
    assert report["corrupt"]
    assert "files.config_snapshot" in {
        item["field"] for item in report["hard_mismatches"]
    }


def test_wrong_manifest_schema_rejected(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    path = cache / CACHE_MANIFEST_FILENAME
    manifest = replace(read_cache_manifest(path), schema_version=999)
    write_cache_manifest_atomic(manifest, path)
    with pytest.raises(CacheManifestError, match="schema_version"):
        AICCompetitionEngine.from_data_root(
            app_config=cfg, text_encoder=TinyTextEncoder()
        )


def test_atomic_manifest_write_leaves_no_temp_file(tmp_path) -> None:
    _, cache, _, load = build_cache(tmp_path)
    path = cache / CACHE_MANIFEST_FILENAME
    write_cache_manifest_atomic(load.cache_manifest, path)
    assert path.is_file()
    assert not list(cache.glob(f".{CACHE_MANIFEST_FILENAME}.*.tmp"))


def test_engine_does_not_load_stale_cache_by_default(tmp_path) -> None:
    _, _, cfg, _ = build_cache(tmp_path)
    changed = replace(cfg, dataset=replace(cfg.dataset, load_objects=True))
    with pytest.raises(StaleCacheError, match="load_objects"):
        AICCompetitionEngine.from_data_root(
            app_config=changed, text_encoder=TinyTextEncoder()
        )


def test_engine_loads_stale_cache_only_when_explicit(tmp_path) -> None:
    _, _, cfg, _ = build_cache(tmp_path)
    changed = replace(
        cfg,
        dataset=replace(cfg.dataset, load_objects=True),
        cache=replace(cfg.cache, allow_stale_cache=True),
    )
    with pytest.warns(RuntimeWarning, match="stale cache"):
        _, load = AICCompetitionEngine.from_data_root(
            app_config=changed, text_encoder=TinyTextEncoder()
        )
    assert load.cache_hit and load.cache_stale and not load.cache_valid
    assert "load_objects" in {item["field"] for item in load.cache_mismatches}


def test_rebuild_bypasses_stale_and_writes_new_manifest(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    changed = replace(cfg, dataset=replace(cfg.dataset, load_objects=True))
    _, load = AICCompetitionEngine.from_data_root(
        app_config=changed, text_encoder=TinyTextEncoder(), rebuild=True
    )
    assert not load.cache_hit and load.cache_valid
    assert read_cache_manifest(cache / CACHE_MANIFEST_FILENAME).load_objects


def test_cli_inspect_cache_returns_valid_state(tmp_path, capsys) -> None:
    _, _, cfg, _ = build_cache(tmp_path)
    config_path = write_config(tmp_path / "settings.yaml", cfg)
    assert cli_main(["--config", str(config_path), "inspect-cache"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["manifest_exists"] and body["valid"] and body["status"] == "valid"


def test_cli_stale_cache_exits_nonzero(tmp_path, capsys) -> None:
    _, _, cfg, _ = build_cache(tmp_path)
    changed = replace(cfg, dataset=replace(cfg.dataset, load_objects=True))
    config_path = write_config(tmp_path / "settings.yaml", changed)
    code = cli_main(["--config", str(config_path), "build-index"])
    assert code == 3
    assert "STALE_CACHE" in capsys.readouterr().err


def test_ui_health_returns_cache_status(tmp_path) -> None:
    _, _, cfg, _ = build_cache(tmp_path)
    app = appmod.create_app(app_config=cfg)
    app.testing = True
    body = app.test_client().get("/api/health").get_json()
    assert body["cache"]["valid"] is True
    assert body["cache"]["manifest_path"].endswith(CACHE_MANIFEST_FILENAME)


def test_benchmark_saves_manifest_snapshot_and_cache_metadata(tmp_path) -> None:
    _, _, _, load = build_cache(tmp_path)
    status = {
        "cache_hit": load.cache_hit,
        "cache_valid": load.cache_valid,
        "cache_stale": load.cache_stale,
    }
    run = BenchmarkLogger(tmp_path / "bench").write_run(
        "cache",
        {},
        [QueryLog("kis", "1", "q", 1.0, [["V", "1"]])],
        cache_manifest=load.cache_manifest,
        cache_status=status,
        dataset_report=load.stats.dataset_report_path,
    )
    snapshot = json.loads((run / "cache_manifest_snapshot.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    dataset_snapshot = json.loads((run / "dataset_report_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot == load.cache_manifest.to_dict()
    assert dataset_snapshot["valid_for_index_build"] is True
    assert summary["dataset_validated"] is True
    assert summary["cache_valid"] is True
    assert summary["cache_fingerprint"] == load.cache_fingerprint


def test_config_serialization_keeps_cache_config(tmp_path) -> None:
    cfg = make_config(make_data(tmp_path / "data"), tmp_path / "cache")
    data = config_to_dict(cfg)
    assert data["cache"] == {
        "allow_stale_cache": False,
        "validate_data_signature": True,
        "code_version_policy": "warn",
    }
    assert isinstance(app_config_from_dict({"aic2026": data}).cache, CacheConfig)


def test_production_mode_rejects_legacy_even_with_fallback_enabled(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    (cache / CACHE_MANIFEST_FILENAME).unlink()
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, production_mode=True),
        cache=replace(cfg.cache, allow_stale_cache=True),
    )
    with pytest.raises(LegacyCacheError, match="production mode"):
        AICCompetitionEngine.from_data_root(
            app_config=cfg, text_encoder=TinyTextEncoder()
        )


def test_error_message_names_mismatched_fields(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    manifest = replace(read_cache_manifest(cache / CACHE_MANIFEST_FILENAME), frame_count=99)
    expected = cache_build_options_from_config(cfg)
    result = validate_cache_manifest(manifest, expected)
    assert "frame_count" in mismatch_message(result)
    assert "frame_count" in result.hard_mismatches[0].message or len(result.hard_mismatches) > 1


def test_cache_manifest_contains_no_secret(tmp_path) -> None:
    _, _, _, load = build_cache(tmp_path)
    text = json.dumps(load.cache_manifest.to_dict()).casefold()
    for forbidden in ("api_key", "secret", "password", "anthropic_api_key"):
        assert forbidden not in text


def test_code_version_mismatch_is_warning_by_default(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    manifest = read_cache_manifest(cache / CACHE_MANIFEST_FILENAME)
    expected = cache_build_options_from_config(cfg)
    result = validate_cache_manifest(
        manifest, expected, current_code_version="different-commit"
    )
    assert result.valid
    assert "code_version" in {item.field for item in result.warnings}


def test_full_config_hash_change_is_warning_not_stale(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    changed = replace(cfg, ranking=replace(cfg.ranking, final_top_k=10))
    expected = cache_build_options_from_config(changed)
    manifest = read_cache_manifest(cache / CACHE_MANIFEST_FILENAME)
    result = validate_cache_manifest(
        manifest,
        expected,
        current_data_signature=expected.data_signature,
        current_video_ids_hash=expected.video_ids_hash,
    )
    assert result.valid
    assert "config_hash" in {item.field for item in result.warnings}


def test_mp4_metadata_does_not_change_data_signature(tmp_path) -> None:
    root = make_data(tmp_path / "data")
    kwargs = dict(
        video_ids=["L01_V001"], include_objects=False, include_media_text=False
    )
    before = build_data_signature(root, **kwargs)
    (root / "video" / "L01_V001.mp4").write_bytes(b"not-used-by-cache-build")
    assert build_data_signature(root, **kwargs) == before


def test_corrupt_pickle_is_not_silently_rebuilt(tmp_path) -> None:
    _, cache, cfg, _ = build_cache(tmp_path)
    (cache / "entry" / "entry.pkl").write_bytes(b"not a pickle")
    with pytest.raises(CacheManifestError, match="Cannot load cache entry"):
        AICCompetitionEngine.from_data_root(
            app_config=cfg, text_encoder=TinyTextEncoder()
        )
