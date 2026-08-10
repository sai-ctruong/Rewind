# Phase 3.1 Dataset Scope And Keyframe Mapping Reconciliation

Phase 3.1 makes a development subset of the AIC collection a first-class, configurable
concept, and reconciles the internal keyframe identity with what the official
`map-keyframes` CSVs actually contain. It does not implement Phase 4 dynamic
`DATA_ROOT` propagation, local refinement integration, per-video Q&A hypotheses, TRAKE
k-best output, independent retrieval channels, submission validation, or scoped UI
manual editing.

## 1. Why Development Subsets Exist

The official preliminary collection is 873 videos across `L21`-`L30`. `map-keyframes`,
`clip-features-32`, `objects`, and `media-info` are present for all 873, but the
`Keyframes` image packages are large and are downloaded in parts. Before Phase 3.1,
inspection validated all 873 discovered IDs, so 848 not-yet-downloaded keyframe folders
were reported as `REQUIRED_SOURCE_MISSING` / `KEYFRAME_FOLDER_MISSING` and the whole
dataset was invalid.

Absent packages outside the dataset you are actually working on are not corruption.
The fix is not to make keyframe images globally optional — that would hide real missing
data — but to state explicitly which videos form the active dataset.

## 2. Current Development Scope

The current working scope is collection `L21` only. This is **configuration, not code**:
nothing under `aic2026/` mentions `L21`, `L22`, or any AIC batch name. The same code
serves the full `L21`-`L30` dataset by changing one setting.

## 3. Scope Selection Semantics

`aic2026/dataset_scope.py` owns selection.

```yaml
aic2026:
  dataset:
    scope:
      include_patterns:
        - "L21_*"
      exclude_patterns: []
```

`select_video_ids(available_video_ids, scope)`:

- patterns are `fnmatch` globs over the canonical video ID (`L21_V006`), never paths;
- include rules run first, exclude rules afterwards;
- the result is de-duplicated and sorted, so it never depends on filesystem iteration
  order or on the order patterns are written;
- duplicate patterns are harmless;
- patterns are validated (non-empty strings, no `/` or `\`);
- selecting nothing from a non-empty collection is an explicit `DatasetScopeError`, not
  a silent empty run.

The default is the full dataset:

```yaml
include_patterns: ["*"]
exclude_patterns: []
```

Three concepts stay separate everywhere:

| Concept | Meaning |
|---|---|
| Discovered sources | every video ID present under `DATA_ROOT` |
| Selected dataset | discovered IDs the scope includes |
| Validation domain | the selected dataset, and nothing else |

## 4. Scope Is Applied Before Validation

`inspect_aic_dataset` discovers sources, applies the scope, and only then validates.
Videos outside the scope produce no `REQUIRED_SOURCE_MISSING`,
`KEYFRAME_FOLDER_MISSING`, or `invalid_video_count` entries. They are still visible as
counts rather than vanishing:

```json
{
  "scope": {"include_patterns": ["L21_*"], "exclude_patterns": []},
  "discovered_video_count": 873,
  "selected_video_count": 29,
  "excluded_video_count": 844,
  "selected_video_ids_hash": "9c60623...",
  "excluded_video_ids_sample": ["L22_V001", "..."]
}
```

Only a bounded sample of excluded IDs is reported, so an 844-video exclusion never
floods the logs. `video_count` remains the number of validated videos, which is now
exactly `selected_video_count`; `DATASET_INSPECTION_SCHEMA_VERSION` is `2`.

## 5. Required Versus Optional Sources Within A Scope

Required for every **selected** video:

- `map-keyframes/{video_id}.csv`
- `clip-features-32/{video_id}.npy`
- `keyframes/{video_id}` with one image per keyframe ordinal

Optional, exactly as in Phase 3: `objects/` (policy from `load_objects` /
`strict_objects`), `media-info/`, and `video/{video_id}.mp4`. Missing MP4s stay a
statistic and never invalidate a dataset — the user has intentionally not downloaded
most of them, and Phase 5 local refinement will handle availability and fallback.

## 6. Cache Isolation

A scoped cache must never be mistaken for a full-collection cache. `cache_fingerprint`
now hashes `dataset_scope`, `selected_video_count`, and `selected_video_ids_hash`
alongside the existing build-affecting options, and the manifest records all three.

So `L21` != `L22` != `L21+L22` != full, and a mismatched scope is a hard
`StaleCacheError` naming `dataset_scope`. Query-time settings such as
`ranking.final_top_k` and `evaluation.ks` still do not affect the fingerprint.

Both `cache_fingerprint` overloads now resolve build options first, so the `AppConfig`
and `CacheBuildOptions` forms cannot drift apart.

Recommended (not enforced) cache layout:

```text
artifacts/aic2026_index_L21/
artifacts/aic2026_index_full/
```

Cache safety comes from the manifest and fingerprint, never from the directory name.

## 7. Real Map CSV Schema

`tools/inspect_map_schema.py` reads every map CSV under a data root and writes
`artifacts/map_schema_report.json`. Result over all 873 official files (177,321 rows):

| Fact | Result |
|---|---|
| Distinct schemas | 1 — `n,pts_time,fps,frame_idx` |
| Keyframe ordinal column | `n`, strictly `1..N` in every file |
| Official frame index column | `frame_idx` |
| Timestamp column | `pts_time` |
| `frame_idx == int(pts_time * fps)` | 177,321 / 177,321 rows |
| `frame_idx == round(pts_time * fps)` | 154,399 / 177,321 rows |
| Videos with duplicate `frame_idx` | 192 |
| Videos with equal-consecutive `frame_idx` | 192 |
| Videos with strictly decreasing `frame_idx` | **0** |

Representative head of `data/map-keyframes/L21_V006.csv`:

```text
n,pts_time,fps,frame_idx
1,0.0,30.0,0
2,0.0333333,30.0,0
3,4.03333,30.0,120
```

Rows 1 and 2 are one source frame apart, but `0.0333333 * 30 = 0.999999` truncates to
`0`, so both carry official `frame_idx=0`.

## 8. keyframe_ordinal, frame_idx, keyframe_id

| Name | Definition | Unique? |
|---|---|---|
| `keyframe_ordinal` | 1-based position in the map CSV (`n`); the same position as the CLIP feature row and the keyframe image file | yes, within a video |
| `frame_idx` | official source-video frame index; **the AIC submission value** | no |
| `keyframe_id` | internal `{video_id}/kf_{keyframe_ordinal:06d}` | yes, globally |

Example, `L21_V006`:

| ordinal | frame_idx | keyframe_id | submitted as |
|---:|---:|---|---|
| 1 | 0 | `L21_V006/kf_000001` | `L21_V006, 0` |
| 2 | 0 | `L21_V006/kf_000002` | `L21_V006, 0` |
| 3 | 120 | `L21_V006/kf_000003` | `L21_V006, 120` |

The submission value is always read from `RawKeyframe.frame_idx`. `official_frame_id`
no longer has a parse-the-ID fallback and raises instead, so an ordinal can never be
emitted as an official frame. `aic2026/engine.py` used to rebuild the internal ID as
`f"{video_id}/{frame_id}"` after fusion; `RankedCandidate` now carries `keyframe_id`
through fusion instead.

## 9. Duplicate frame_idx Conclusion

**CASE B, from evidence.** The parser was already reading the correct column. Repeated
`frame_idx` is genuine official data produced by truncation, so:

- `FRAME_IDX_DUPLICATE` is **info**, not an error;
- equal-consecutive `frame_idx` (`FRAME_IDX_NON_MONOTONIC`) is **info**; non-decreasing
  order is the real invariant;
- strictly decreasing `frame_idx` is a new hard error, `FRAME_IDX_DECREASING`, and does
  not occur anywhere in the official collection;
- `frame_idx` that is negative or non-integer stays a hard error;
- a repeated keyframe ordinal is a new hard error, `KEYFRAME_ORDINAL_DUPLICATE`, since
  that is the only thing that can collide internal IDs or keyframe image lookup;
- map/feature count mismatch stays a hard error.

The old behavior was not merely noisy: with `{video_id}/{frame_idx}` IDs, the second
keyframe of each of those 192 videos overwrote the first in `entry.raws` while both
stayed in the index list.

## 10. Record Schema Version

`AIC_RECORD_SCHEMA_VERSION = 3` (`ingestion/schemas.py`). The keyframe ID semantics
changed, so v2 caches are stale by design through the fingerprint and manifest
comparison. Pickle caches are never silently migrated. `RawKeyframe` gains
`keyframe_ordinal`, which is persisted by `VideoIndexEntry.save`.

## 11. Ordering Invariant

```text
CLIP feature array row ordering follows keyframe ordering,
while AIC submission uses the original source frame_idx.
```

feature vector `i` ↔ map row `i` ↔ keyframe ordinal `i` ↔ keyframe image `i` ↔ official
`frame_idx` read from row `i`. Official packages name images after the ordinal
(`001.jpg` … `257.jpg` for the 257 map rows of `L21_V006`), so image lookup uses the
ordinal and never `frame_idx`. `tests/test_keyframe_mapping_semantics.py` pins each
link of that chain.

## 12. Commands

Inspect the current development scope:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data `
  --video-include "L21_*" inspect-data --output artifacts\dataset_report_L21.json
```

Build a scoped cache once the required data is complete:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data `
  --cache-dir artifacts\aic2026_index_L21 --video-include "L21_*" build-index --rebuild
```

`--video-include` and `--video-exclude` are global options placed before the
subcommand, and both may be repeated. Precedence is explicit CLI override > YAML >
default; `configs/settings.yaml` is never rewritten.

No `configs/settings_l21.yaml` was added. The config loader has no inheritance, so a
second file would silently drop every tuned value in the main file (`expected_feature_dim`,
fusion weights, ranking) and become a trap. The CLI override above is the supported
path.

## 13. Real L21 Inspection Result

Fast inspection, `--video-include "L21_*"`, run on 2026-08-10:

| Fact | Result |
|---|---:|
| Discovered videos | 873 |
| Selected videos | 29 |
| Excluded videos | 844 |
| Valid / invalid selected videos | 24 / 5 |
| Map rows / feature vectors | 7,800 / 7,800 |
| Keyframe images | 6,624 |
| Feature contract | dimension 512, dtype float16 |
| Duplicate official `frame_idx` | 10 (info) |
| Duplicate internal keyframe IDs | 0 |
| Missing required map / features / keyframes | 0 / 0 / 4 |
| `valid_for_index_build` | **false** |
| Inspection time | 5.157 s |

```text
FRAME_IDX_DUPLICATE: 10        (info)
FRAME_IDX_NON_MONOTONIC: 10    (info)
KEYFRAME_FOLDER_MISSING: 4     (error)
REQUIRED_SOURCE_MISSING: 4     (error)
KEYFRAME_IMAGE_MISSING: 1      (error)
KEYFRAME_COUNT_MISMATCH: 1     (error)
```

The 844 excluded videos contributed zero errors, down from 848 before Phase 3.1. Every
remaining failure is real, missing L21 data:

| Video | Map rows | Keyframe images | Missing |
|---|---:|---:|---|
| `data/keyframes/L21_V027` | 292 | 180 | 112 images, first ordinals 3, 4, 11, 18, 19, 21, 22, 25, 26, 29 |
| `data/keyframes/L21_V028` | 235 | 0 | whole folder |
| `data/keyframes/L21_V029` | 290 | 0 | whole folder |
| `data/keyframes/L21_V030` | 254 | 0 | whole folder |
| `data/keyframes/L21_V031` | 285 | 0 | whole folder |

The full list of the 112 missing `L21_V027` ordinals is in
`artifacts/dataset_report_L21.json` under that video's `missing_keyframe_ordinals`.

As read-only evidence that the scope mechanism itself is sound, inspecting the 24
completely downloaded L21 videos passes cleanly:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data `
  --video-include "L21_*" --video-exclude "L21_V027" --video-exclude "L21_V028" `
  --video-exclude "L21_V029" --video-exclude "L21_V030" --video-exclude "L21_V031" `
  inspect-data --output artifacts\dataset_report_L21_complete.json
```

`valid_for_index_build: true`, 24/24 valid, 6,444 map rows = 6,444 feature vectors =
6,444 keyframe images, 9 informational duplicate `frame_idx`, 0 internal ID collisions,
2.198 s.

## 14. Cache Rebuild Status

**No cache was rebuilt.** The `L21_*` scope is not `valid_for_index_build`, so per
policy no new index was produced. The pre-existing `artifacts/aic2026_index` was not
touched and still reports `legacy=true`, `stale=true`, `valid=false`, `corrupt=false`
because it has an entry and no manifest. No full-collection cache was rebuilt either.

To unblock the `L21_*` cache, download the keyframe packages for `L21_V028`-`L21_V031`
and the 112 missing images of `L21_V027`, then re-run the inspect command in section 12
and the build command once it reports `valid_for_index_build: true`.

## 15. Switching To The Full Dataset Later

No code change is required. Either leave `configs/settings.yaml` at its default
(`include_patterns: ["*"]`) and omit `--video-include`, or pass `--video-include "*"`.
Point `--cache-dir` at a different directory (for example
`artifacts/aic2026_index_full`) and rebuild; the scope difference alone already
guarantees the two caches cannot be confused.

## 16. Verification

`compileall` passed over `aic2026`, `ingestion`, `retrieval`, `evaluation`, `ui`,
`tests`, and `tools`. The offline suite is 404 passed, 1 skipped; the skip is the
pre-existing lazy-import guard that does not apply because torch is installed. Phase 3.1
adds 43 tests across `tests/test_dataset_scope.py` and
`tests/test_keyframe_mapping_semantics.py`. `git diff --check` is clean. No accuracy
benchmark was run, because no AIC ground truth exists.

## 17. Not Started

Phase 4 dynamic `DATA_ROOT` propagation has not been started. Local refinement
integration, per-video Q&A hypotheses, TRAKE k-best, independent retrieval channels,
submission validation, and scoped UI manual editing all remain pending in later phases.
