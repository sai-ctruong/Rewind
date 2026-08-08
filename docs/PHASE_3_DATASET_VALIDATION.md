# Phase 3 Strict Dataset Validation

Phase 3 removes silent map/feature truncation and makes dataset inspection a hard
prerequisite for every new AIC index. It does not implement dynamic UI `DATA_ROOT`,
retrieval changes, local refinement, Q&A hypotheses, TRAKE output, or submission
editing.

## Alignment Policy

The old loader used `min(len(rows), features.shape[0])`, which could silently discard
map rows or feature vectors. The production loader now requires exact equality and
raises `DatasetAlignmentError` with the video ID, expected/actual counts, both source
paths, and a remediation hint.

`inspect_aic_dataset(strict=False)` is diagnostic only. It continues across videos,
records `MAP_FEATURE_COUNT_MISMATCH`, marks the video invalid, and never builds an
index. `strict=True` still completes the report, then raises
`DatasetValidationError(report)` when invalid. Index builds always use strict validation;
no configuration or stale-cache override permits a partial mismatch index.

## Source Policy

Required for the current keyframe index:

- `map-keyframes/{video_id}.csv`
- `clip-features-32/{video_id}.npy`
- `keyframes/{video_id}` with one mapped image for each keyframe ordinal

Optional:

- `objects/{video_id}`; validated when `load_objects=true`, with missing/corrupt files
  controlled by `strict_objects`
- `media-info/{video_id}.json`; absence or malformed metadata is diagnostic only
- `video/{video_id}.mp4`; used only as a display/decode fallback and never required to
  build the current index

Optional orphan sources are warnings. They do not create false map/feature missing
errors and do not invalidate an otherwise complete required inventory.

## Validation Coverage

Map CSV checks include required columns (`n`, `pts_time`, `fps`, `frame_idx`), integer
and non-negative frame IDs, uniqueness, strict monotonicity, timestamps, and
FPS/timestamp consistency diagnostics.

Feature checks use `np.load(..., mmap_mode="r", allow_pickle=False)`. Arrays must be
2-D, aligned with map rows, dimensionally consistent, use an allowed dtype, contain
only finite values, and contain no zero vectors. Finite/norm scans run in chunks and
do not retain full arrays in the report.

Fast image inspection checks inventory and mapped paths. `--deep` additionally uses
Pillow verification, optionally capped per video. Object JSON checks the official
label/entity, score, and `[ymin, xmin, ymax, xmax]` arrays without storing raw JSON.
Media JSON is parsed into separate title, description, tags, channel, duration, and
FPS fields.

Principal issue codes are:

```text
REQUIRED_SOURCE_MISSING
MAP_MISSING MAP_READ_ERROR MAP_REQUIRED_COLUMN_MISSING MAP_FEATURE_COUNT_MISMATCH
FRAME_IDX_INVALID FRAME_IDX_NEGATIVE FRAME_IDX_DUPLICATE FRAME_IDX_NON_MONOTONIC
TIMESTAMP_INVALID TIMESTAMP_NON_MONOTONIC MAP_FPS_INVALID
TIMESTAMP_FRAME_FPS_INCONSISTENT
FEATURE_MISSING FEATURE_LOAD_ERROR FEATURE_INVALID_NDIM FEATURE_COUNT_MISMATCH
FEATURE_DIM_MISMATCH FEATURE_DTYPE_UNSUPPORTED FEATURE_NON_FINITE FEATURE_ZERO_VECTOR
KEYFRAME_FOLDER_MISSING KEYFRAME_IMAGE_MISSING KEYFRAME_IMAGE_ORPHAN
KEYFRAME_IMAGE_CORRUPT KEYFRAME_COUNT_MISMATCH
OBJECT_FILE_MISSING OBJECT_JSON_CORRUPT OBJECT_SCHEMA_INVALID
OBJECT_SCORE_INVALID OBJECT_BOX_INVALID
MEDIA_INFO_MISSING MEDIA_JSON_CORRUPT MEDIA_SCHEMA_INVALID
MEDIA_DURATION_INVALID MEDIA_FPS_INVALID
ORPHAN_MAP ORPHAN_FEATURE ORPHAN_KEYFRAME_FOLDER
ORPHAN_OBJECT_FOLDER ORPHAN_MEDIA_INFO ORPHAN_VIDEO
```

## Report And CLI

`DatasetInspectionReport` contains collection totals, per-video facts, source
inventory differences, structured issues, and `valid_for_index_build`. Reports are
UTF-8 JSON written atomically through `write_dataset_report`; feature vectors and raw
object payloads are never embedded.

Fast inspection:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data inspect-data --output artifacts\dataset_report.json
```

Deep inspection on a small deterministic prefix:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data inspect-data --deep --limit-videos 3 --output artifacts\dataset_report_deep.json
```

`inspect-data` prints a compact summary and does not load a cache, initialize an
encoder, or build an index. Exit codes are `0` valid, `2` config/arguments, `5`
dataset invalid, and `6` inspection/runtime error. `--summary-only` controls report
file detail but still writes the requested output file.

## Record And Cache Contract

`AIC_RECORD_SCHEMA_VERSION = 2` is defined once in `ingestion/schemas.py` and used by
records, data signatures, cache fingerprints, and manifests. Video metadata now uses
`media_title`, `media_description`, `media_tags`, `media_channel`, and
`media_duration`, and `media_fps`. `frame_caption` is separate; legacy
`llm_caption` remains an alias
only for a real frame caption. Search text prefixes every source to preserve
provenance and avoid double counting.

A successful build writes `dataset_report.json`, records its path and validation
state in `dataset_stats.json`, and references it from the manifest. The manifest is
written last. Invalid data produces no new entry or usable manifest. Schema-v1 record
caches are stale as expected after this change.

## Real Dataset Inspection

Fast inspection ran on 2026-08-08 without `--deep`:

| Fact | Result |
|---|---:|
| Inspection time | 9.157 s |
| Videos / map files / feature files | 873 / 873 / 873 |
| Map rows / feature vectors | 177,321 / 177,321 |
| Feature shape contract | dimension 512, dtype float16 |
| Keyframe folders / images | 25 / 6,624 |
| Object folders / media-info files | 873 / 873 |
| MP4 files / optional MP4 missing | 29 / 844 |
| Valid / invalid videos | 15 / 858 |
| `valid_for_index_build` | false |

Issue counts:

```text
KEYFRAME_FOLDER_MISSING: 848
REQUIRED_SOURCE_MISSING: 848
FRAME_IDX_DUPLICATE: 192
FRAME_IDX_NON_MONOTONIC: 192
KEYFRAME_IMAGE_MISSING: 1
KEYFRAME_COUNT_MISMATCH: 1
```

`L21_V027` has 292 mapped rows but only 180 images (112 missing). A representative
frame-ID problem is `L21_V006`, whose first two rows both map to official
`frame_idx=0`. The full report is generated at `artifacts/dataset_report.json` and is
ignored by Git.

Because the required data is invalid, the real cache was not rebuilt. `inspect-cache`
still reports the pre-existing cache as `legacy=true`, `stale=true`, `valid=false`,
and `corrupt=false` because it has an entry but no manifest. The 844 missing MP4
files did not contribute any error and were not used as a reason to block build.

## Verification And Limits

`compileall` passed. The full offline suite collected 357 tests: 356 passed and one
existing lazy-import test was skipped because torch is installed. The new Phase 3
module contributes 59 passing cases, including fast/deep inspection and build-gate
coverage.

A real deep inspection of the first three deterministic videos also passed: 3/3
valid, 855 map rows/features/images, no issues, in 14.536 seconds. Deep inspection
of all 873 videos was not run because fast inspection already proved that required
keyframes and frame IDs are invalid. UI inspection remains pending with dynamic
`DATA_ROOT` in Phase 4.
