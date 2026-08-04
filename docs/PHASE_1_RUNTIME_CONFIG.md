# Phase 1 Runtime Config

Phase 1 makes `configs/settings.yaml` a real runtime source for the AIC 2026 competition pipeline without changing retrieval quality logic or starting cache/dataset/refinement phases.

## Architecture

Runtime config is centralized in `aic2026/config.py`.

Core API:

```python
load_app_config(path="configs/settings.yaml", overrides=None) -> AppConfig
app_config_from_dict(data) -> AppConfig
validate_app_config(config) -> None
config_to_dict(config) -> dict
config_hash(config) -> str
```

`AppConfig` sections:

- `runtime`
- `dataset`
- `encoder`
- `retrieval_channels`
- `fusion`
- `ranking`
- `refinement`
- `trake`
- `qa`
- `submission`
- `evaluation`
- `ui`

`configs/settings.yaml` keeps legacy top-level sections for existing modules/tests, and adds `aic2026:` as the competition runtime section used by CLI/UI/engine.

## Precedence

Runtime values resolve in this order:

1. Explicit CLI override.
2. YAML `aic2026:` section.
3. Dataclass default.

CLI overrides are temporary and do not mutate the YAML file.

## Validation

`validate_app_config` raises `ConfigError` with field-specific messages, for example:

```text
ranking.final_top_k must be in [1, 100]
trake.max_gap_s must be >= trake.min_gap_s
fusion.clip_weight must be >= 0
```

Validation covers runtime/device, encoder, fusion weights/method, ranking bounds, refinement bounds, TRAKE method/bounds, Q&A bounds, submission caps and evaluation cutoffs.

Current TRAKE method is intentionally named `beam_dp`, because the implementation is beam-pruned DP, not exact exhaustive DP.

## CLI

Show resolved config and hash:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml show-config
```

Examples of explicit overrides:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data --cache-dir artifacts\aic2026_index show-config
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --production-mode --device cpu search --task kis --query "red shirt"
```

## UI Health

`/api/health` now includes a compact config status:

```json
{
  "config": {
    "path": "...",
    "hash": "...",
    "production_mode": false,
    "device": "auto",
    "encoder_type": "auto_clip",
    "feature_dim": 512,
    "fusion_method": "adaptive",
    "final_top_k": 100,
    "refinement_enabled": true,
    "trake_alignment_method": "beam_dp",
    "qa_top_video_hypotheses": 8
  }
}
```

## Benchmark Logging

Benchmark runs now receive the resolved `config_to_dict(AppConfig)` snapshot and can write `config_hash.txt`. The snapshot reflects CLI overrides when they are used.

## Tests Run

```text
python -m compileall -q aic2026 ingestion retrieval evaluation ui tests: passed
python -m pytest -q: passed; one legacy lazy-import test skipped because torch is installed
python -m aic2026.cli --config configs\settings.yaml show-config: passed
```

## Remaining Limits

Phase 1 does not implement cache manifest, strict dataset validation, dynamic DATA_ROOT propagation, local refinement integration, Q&A per-video answers, TRAKE k-best, retrieval channels, submission validation or UI manual-edit scoping. Those remain pending later phases.