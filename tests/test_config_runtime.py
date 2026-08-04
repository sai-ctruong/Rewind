from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic2026.benchmark import BenchmarkLogger, QueryLog
from aic2026.cli import main as cli_main
from aic2026.config import (
    ConfigError,
    app_config_from_dict,
    config_hash,
    config_to_dict,
    load_app_config,
)
from aic2026.dataset import AICDatasetLoader
from aic2026.engine import AICCompetitionEngine
from aic2026.text_encoder import HashingTextEncoder
import ui.app as appmod


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32) if "red" in text.lower() else np.array([0.0, 1.0], dtype=np.float32)


def write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def minimal_config(**overrides):
    base = {
        "aic2026": {
            "runtime": {"production_mode": False, "seed": 42, "device": "auto"},
            "dataset": {"root": "data", "cache_dir": "artifacts/aic2026_index"},
            "encoder": {"type": "auto_clip", "model_name": "openai/clip-vit-base-patch32", "feature_dim": 2, "batch_size": 4},
            "fusion": {"method": "adaptive", "clip_weight": 1.0, "object_weight": 0.2, "metadata_weight": 0.1, "sparse_weight": 0.25},
            "ranking": {"final_top_k": 25, "max_frames_per_video": 12, "min_frame_gap": 0},
            "trake": {"alignment_method": "beam_dp", "per_event_top_k": 3, "top_video_hypotheses": 2, "beam_width": 4, "final_top_k": 25},
            "qa": {"top_video_hypotheses": 2, "frame_hypotheses_per_video": 1, "evidence_frame_count": 2, "max_answers": 25},
            "submission": {"max_predictions": 25, "max_answers": 25},
            "evaluation": {"ks": [1, 5, 20], "output_dir": "evaluation/benchmarks/aic2026"},
        }
    }

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a

    return merge(base, overrides)


def make_aic_root(root: Path) -> Path:
    for rel in ("clip-features-32", "map-keyframes", "keyframes/L01_V001", "objects/L01_V001", "media-info", "video"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "map-keyframes" / "L01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,100\n2,1.0,30.0,130\n",
        encoding="utf-8",
    )
    np.save(root / "clip-features-32" / "L01_V001.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    for i in (1, 2):
        Image.new("RGB", (8, 8), (255 if i == 1 else 0, 0, 255 if i == 2 else 0)).save(
            root / "keyframes" / "L01_V001" / f"{i:03d}.jpg"
        )
        (root / "objects" / "L01_V001" / f"{i:03d}.json").write_text("{}", encoding="utf-8")
    (root / "media-info" / "L01_V001.json").write_text("{}", encoding="utf-8")
    return root


def test_load_yaml_valid_and_missing_optional_defaults(tmp_path) -> None:
    cfg_path = write_yaml(tmp_path / "settings.yaml", "aic2026:\n  ranking:\n    final_top_k: 25\n")
    cfg = load_app_config(cfg_path)
    assert cfg.ranking.final_top_k == 25
    assert cfg.encoder.model_name == "openai/clip-vit-base-patch32"
    assert cfg.evaluation.ks == (1, 5, 20, 50, 100) or list(cfg.evaluation.ks) == [1, 5, 20, 50, 100]


@pytest.mark.parametrize("patch,error", [
    ({"aic2026": {"ranking": {"final_top_k": 101}}}, "ranking.final_top_k"),
    ({"aic2026": {"fusion": {"clip_weight": -1}}}, "fusion.clip_weight"),
    ({"aic2026": {"fusion": {"clip_weight": 0, "object_weight": 0, "metadata_weight": 0, "sparse_weight": 0}}}, "fusion weights"),
    ({"aic2026": {"trake": {"min_gap_s": 5, "max_gap_s": 1}}}, "trake.max_gap_s"),
    ({"aic2026": {"evaluation": {"ks": [5, 1]}}}, "evaluation.ks"),
])
def test_invalid_config_rejected(patch, error) -> None:
    with pytest.raises(ConfigError, match=error):
        app_config_from_dict(minimal_config(**patch))


def test_config_hash_deterministic_and_changes() -> None:
    a = app_config_from_dict(minimal_config())
    b = app_config_from_dict(minimal_config())
    c = app_config_from_dict(minimal_config(aic2026={"ranking": {"final_top_k": 20}}))
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_cli_override_precedence_and_no_implicit_override(tmp_path, capsys) -> None:
    cfg_path = write_yaml(tmp_path / "settings.yaml", "aic2026:\n  dataset:\n    root: yaml-root\n  ranking:\n    final_top_k: 25\n")
    original = cfg_path.read_text(encoding="utf-8")
    assert cli_main(["--config", str(cfg_path), "--data-root", "cli-root", "show-config"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["dataset"]["root"] == "cli-root"
    assert cfg_path.read_text(encoding="utf-8") == original
    assert cli_main(["--config", str(cfg_path), "show-config"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["dataset"]["root"] == "yaml-root"


def test_engine_receives_runtime_config_and_yaml_changes_behavior(tmp_path) -> None:
    root = make_aic_root(tmp_path / "data")
    cfg = app_config_from_dict(minimal_config(aic2026={"dataset": {"root": str(root), "cache_dir": str(tmp_path / "cache")}, "ranking": {"final_top_k": 1}, "fusion": {"method": "weighted"}}))
    entry, _ = AICDatasetLoader(root).build_entry()
    engine = AICCompetitionEngine(entry, text_encoder=TinyTextEncoder(), query_templates=("{q}",), app_config=cfg)
    assert engine.fusion_config.method == "weighted"
    assert engine.ranking_config.final_top_k == 1
    assert engine.trake_config.alignment_method == "beam_dp"
    assert engine.feature_dim == 2
    assert len(engine.search_kis("red")) == 1


def test_from_data_root_uses_config_dataset_and_cache(tmp_path) -> None:
    root = make_aic_root(tmp_path / "data")
    cfg = app_config_from_dict(minimal_config(aic2026={"dataset": {"root": str(root), "cache_dir": str(tmp_path / "cache")}, "ranking": {"final_top_k": 1}}))
    engine, load = AICCompetitionEngine.from_data_root(app_config=cfg, text_encoder=TinyTextEncoder(), rebuild=True)
    assert load.cache_hit is False
    assert engine.ranking_config.final_top_k == 1
    assert (tmp_path / "cache" / "entry" / "entry.pkl").exists()


def test_ui_health_returns_config_hash(tmp_path) -> None:
    cfg = app_config_from_dict(minimal_config(aic2026={"dataset": {"root": str(tmp_path / "data"), "cache_dir": str(tmp_path / "cache")}}))
    app = appmod.create_app(app_config=cfg)
    app.testing = True
    body = app.test_client().get("/api/health").get_json()
    assert body["config"]["hash"] == config_hash(cfg)
    assert body["config"]["final_top_k"] == cfg.ranking.final_top_k


def test_benchmark_snapshot_records_resolved_config_and_hash(tmp_path) -> None:
    cfg = app_config_from_dict(minimal_config(aic2026={"ranking": {"final_top_k": 25}}))
    h = config_hash(cfg)
    run = BenchmarkLogger(tmp_path).write_run(
        "cfg", config_to_dict(cfg), [QueryLog("kis", "1", "q", 1.0, [["V", "1"]])], config_hash=h
    )
    assert (run / "config_hash.txt").read_text(encoding="utf-8") == h
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["ranking"]["final_top_k"] == 25


def test_production_mode_rejects_hashing_only_encoder() -> None:
    with pytest.raises(ConfigError, match="encoder.type"):
        app_config_from_dict(minimal_config(aic2026={"runtime": {"production_mode": True}, "encoder": {"type": "hashing"}}))
    with pytest.raises(RuntimeError):
        HashingTextEncoder(2, production_mode=True)


def test_config_serialization_round_trip() -> None:
    cfg = app_config_from_dict(minimal_config())
    data = config_to_dict(cfg)
    assert config_to_dict(app_config_from_dict({"aic2026": data})) == data