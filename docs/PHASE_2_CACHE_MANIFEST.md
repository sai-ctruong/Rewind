# Phase 2 Cache Manifest And Cache Safety

Phase 2 makes persisted AIC 2026 indexes self-describing and rejects incompatible,
legacy, or corrupt caches before normal cache loading. It does not change retrieval,
dataset alignment, ranking, Q&A, TRAKE, submission, or frontend interaction logic.

## Artifacts

A successful build writes these files below the configured cache directory:

```text
cache_manifest.json
resolved_config.json
dataset_stats.json
dataset_report.json  # record schema v2 and later
entry/entry.pkl
entry/index/meta.pkl
entry/index/clip.hnsw
```

Only files actually produced by the build are listed in `manifest.files`. The final
`cache_manifest.json` is written atomically after the entry, index, resolved config,
dataset statistics, and (from Phase 3) the validated dataset report have been
written. A rebuild validates data before removing the old usability marker and
replacing cache contents.

## Manifest Schema

`CACHE_MANIFEST_SCHEMA_VERSION = 1`.

The manifest contains:

- identity: `schema_version`, `created_at_utc`, `code_version`;
- config: `config_hash`, `cache_fingerprint`;
- data: normalized `data_root`, `data_signature`, `video_ids_hash`, `video_count`,
  `frame_count`;
- feature space: `feature_dim`, `feature_dtype`, `encoder_feature_space`;
- source options: `load_objects`, `include_media_text`, `include_ocr`, `include_asr`,
  `include_captions`;
- index/record contract: `index_kind`, `index_params`, `record_schema_version`;
- artifact paths: `files` using forward-slash relative paths.

Unsupported schema versions are not migrated or loaded automatically.

Phase 3 keeps manifest schema v1 but advances the single record contract constant to
`AIC_RECORD_SCHEMA_VERSION = 2`. New record-v2 manifests reference
`dataset_report.json`; record-v1 caches become stale through the fingerprint/version
comparison rather than being treated as corrupt solely for lacking that newer
artifact.

## Cache Fingerprint

`cache_fingerprint` hashes only build-affecting options:

- normalized data root;
- object/media/OCR/ASR/caption inclusion;
- index kind and build parameters;
- video/frame build limits, keyframe verification, and expected feature dimension
  through `index_params`;
- configured feature dimension and encoder feature space;
- record schema version.

It intentionally excludes query-time settings such as `ranking.final_top_k`,
`evaluation.ks`, Q&A evidence count, TRAKE beam width, and UI settings. The full
`config_hash` is retained for reproducibility; a full hash change alone is a warning,
not a hard cache mismatch.

## Data Signature

`data_signature` is deterministic SHA-256 over:

- canonical resolved data root;
- sorted unique video IDs;
- map-keyframe and CLIP feature relative paths, sizes, and `mtime_ns` values;
- object JSON metadata when objects are included;
- media-info JSON metadata when media text is included;
- keyframe file metadata only when keyframe verification affects the build;
- record schema version.

Large image, video, feature, and index file contents are not hashed. MP4 files are not
part of the current index-build signature because their contents do not build the
persisted retrieval index.

## Load Policy

| State | Default behavior | Explicit development override |
|---|---|---|
| Valid | Load cache | Not needed |
| Legacy: `entry.pkl` without manifest | Reject and require rebuild | Allowed only with `allow_stale_cache=true` outside production, with warning and invalid/stale status |
| Stale data/build options | Reject and report mismatched fields | Allowed outside production with warning and invalid/stale status |
| Corrupt JSON/schema/referenced file/pickle/index | Reject | Never bypassed by stale override |

Production mode always rejects legacy and stale caches. Code-version differences warn
by default; `cache.code_version_policy` supports `ignore`, `warn`, or `reject`.

## Configuration

```yaml
aic2026:
  cache:
    allow_stale_cache: false
    validate_data_signature: true
    code_version_policy: warn
```

CLI overrides are runtime-only and do not modify YAML:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --allow-stale-cache build-index
```

## CLI

Inspect without loading pickle or initializing the encoder:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --cache-dir artifacts\aic2026_index inspect-cache
```

The JSON result includes `exists`, `manifest_exists`, `entry_exists`, `status`,
`valid`, `legacy`, `stale`, `corrupt`, `hard_mismatches`, `warnings`, and the parsed
manifest when available.

Exit codes are:

- `0`: valid, missing cache inspection, or successful build;
- `2`: invalid config/arguments;
- `3`: stale or legacy cache rejected;
- `4`: corrupt manifest/cache rejected.

Use `--rebuild` to bypass an old cache and produce a new manifest:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --rebuild build-index
```

## UI API

`GET /api/health` and successful index responses include:

```json
{
  "cache": {
    "exists": true,
    "hit": true,
    "valid": true,
    "legacy": false,
    "stale": false,
    "manifest_path": ".../cache_manifest.json",
    "fingerprint": "...",
    "created_at": "...",
    "code_version": "...",
    "mismatches": [],
    "warnings": []
  }
}
```

Index endpoints return HTTP 409 for `STALE_CACHE` or `LEGACY_CACHE`, and HTTP 422
for `CORRUPT_CACHE_MANIFEST`.

## Benchmark Artifacts

`BenchmarkLogger.write_run` can write `cache_manifest_snapshot.json` and, when
available, `dataset_report_snapshot.json`. Cache metadata
is copied into `environment.json` and `summary.json`:

- `cache_hit`, `cache_valid`, `cache_stale`;
- `cache_fingerprint`, `cache_manifest_schema`;
- `data_signature`, `code_version`.

## Verification

Phase 2 adds 37 offline tests in `tests/test_cache_manifest.py`. The full suite result
at implementation time is 297 passed and 1 skipped. The skip is the existing
lazy-import guard test that is inapplicable because torch is installed.

The repository cache at `artifacts/aic2026_index` is currently detected as `legacy`:
`entry.pkl` exists, `cache_manifest.json` does not, and the cache is rejected by
default. It has not been rebuilt or accepted with the stale override.

## Remaining Limits

Strict map/feature alignment and complete dataset inspection were added in Phase 3.
Dynamic DATA_ROOT propagation, local refinement integration, per-video Q&A hypotheses, TRAKE schema and
k-best output, independent retrieval channels, submission validation, and scoped UI
manual editing remain pending in later phases. Phase 3 also makes record schema v2 and
dataset reports part of every newly built cache contract.
