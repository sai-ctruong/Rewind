# Phase 3.2 Video-Backed Development And Visual Fallback

Phase 3.2 corrects the data-role model, uses the original MP4s already on disk as a
visual source, and makes the currently available video-backed subset a reproducible
development scope. It does not start Phase 4, and it does not implement advanced local
refinement, Q&A per-video hypotheses, TRAKE k-best, new retrieval channels, a
submission validator, or paper benchmarking.

## 1. Architectural Rationale

Official AIC documentation makes the **video** the competition data. Keyframes,
objects, CLIP features, and metadata are **supporting** data.

Phase 3 encoded the opposite: `map + CLIP + keyframe JPEG` were all universally
required, so 848 not-yet-downloaded keyframe folders made the collection invalid, and
even inside the `L21` scope four missing folders plus one partial folder blocked every
build. That treated a supporting artifact as the source of truth.

Phase 3.2 restores the official hierarchy. The global index is built from the official
CLIP feature arrays; keyframe JPEGs are a *convenience* visual source; the original
MP4 is the authoritative one and can always regenerate a frame.

## 2. Capability Model

`valid` / `invalid` was too coarse to express "searchable but not viewable". Each
video now carries four independent capabilities.

| Capability | Requires | Notes |
|---|---|---|
| `retrieval_valid` | `map-keyframes` + `clip-features-32`, aligned and passing feature validation | Gates the global index. JPEG and MP4 are irrelevant here. |
| `visual_accessible` | a keyframe JPEG **or** the original MP4 | At least one real pixel source. |
| `refinement_ready` | the original MP4 | What Phase 5 local refinement will need. |
| `qa_visual_ready` | `visual_accessible` | Text-only evidence is explicitly *not* visual readiness. |

`visual_source` is one of `keyframe_jpeg`, `video_decode`, or `none`. A partially
downloaded JPEG folder reports `video_decode`, because the MP4 is what makes *every*
mapped keyframe of that video viewable.

`retrieval_valid=true` with `visual_accessible=false` is a legitimate, supported state.

## 3. What Stayed Strict

Unchanged hard errors: map/feature row mismatch, invalid map schema, missing required
map or CLIP source, invalid or negative official `frame_idx`, strictly decreasing
`frame_idx` (the verified Phase 3.1 policy), duplicate keyframe ordinal, wrong feature
dimension, unsupported dtype, NaN/Inf, zero vectors, and corrupt required sources.

What changed: a missing keyframe JPEG is no longer a hard error. It emits
`KEYFRAME_JPEG_UNAVAILABLE` at `info` when the MP4 can cover the gap and `warning`
when nothing can, alongside the existing `KEYFRAME_FOLDER_MISSING` /
`KEYFRAME_IMAGE_MISSING` / `KEYFRAME_COUNT_MISMATCH` diagnostics at the same severity.
Keyframes are also no longer listed under `REQUIRED_SOURCE_MISSING`.

New codes:

```text
KEYFRAME_JPEG_UNAVAILABLE                    info | warning
VISUAL_SOURCE_UNAVAILABLE                    error, only when require_visual_source=true
VIDEO_PRESENT_BUT_RETRIEVAL_SUPPORT_MISSING  warning
```

### Configuration change

`dataset.validation.require_keyframe_images` was **removed**. Its meaning changed, and
silently ignoring the old key could have re-enabled a build a config meant to block, so
loading a config that still contains it raises a `ConfigError` naming the replacement:

```yaml
aic2026:
  dataset:
    validation:
      # Off by default: a retrieval-only development subset is legitimate.
      # true => every selected video must have a keyframe JPEG or an original MP4.
      require_visual_source: false
```

`AICDataPaths.validate()` no longer requires a `keyframes/` directory to exist at all.

## 4. Report Fields

`DatasetInspectionReport` adds, over the **selected** videos:

```text
retrieval_valid_video_count      visual_accessible_video_count
refinement_ready_video_count     qa_visual_ready_video_count
keyframe_jpeg_backed_video_count video_fallback_video_count
no_visual_source_video_count     selected_video_ids
```

and per video: `retrieval_valid`, `visual_accessible`, `refinement_ready`,
`qa_visual_ready`, `visual_source`, `keyframe_jpeg_available`, `video_available`,
`map_available`, `features_available`.

## 5. FrameProvider

`aic2026/frame_provider.py` resolves one mapped keyframe to JPEG bytes.

```python
FrameProvider(data_root, cache_dir=...).get_frame(record, prefer_keyframe_jpeg=True)
```

`FrameResult` carries `image_bytes`, `source`, `video_id`, `frame_idx`, `timestamp`,
`requested_frame_idx`, `decoded_frame_idx`, `seek_method`, `cache_hit`, `cache_path`,
and `warning`; `available` is false rather than raising.

Priority:

1. the existing BTC keyframe JPEG;
2. otherwise decode the mapped frame from the original MP4;
3. otherwise an explicit unavailable result.

Three invariants:

- **The official `frame_idx` never changes.** `frame_idx` echoes what was requested;
  where the decoder actually landed is reported separately as `decoded_frame_idx`.
  A discrepancy becomes a warning, never a rewritten submission value.
- **Nothing is invented.** No placeholder image, no substituted neighbouring keyframe.
- **A visualization failure never fails retrieval.** Search runs on CLIP features.

Decoding reuses the project's existing OpenCV approach (the same primitive as
`ingestion/schemas._decode_from_source`); `ingestion/video_ingest.py` bulk sampling and
`aic2026/local_refinement.py` window decoding were audited and left alone. Frame-index
seek is preferred and was verified exact against the real L21 MP4s (requested == landed
at the first, second, middle, and last mapped rows); timestamp seek remains a fallback,
and `seek_method` records which was used.

`describe(record)` returns availability facts with no file reads and no decoding, which
is what search results use.

## 6. Derived Frame Cache

Decoded frames are written **only** under `dataset.frame_cache_dir`, default
`artifacts/video_frame_cache/`:

```text
artifacts/video_frame_cache/L21_V027/frame_00000300.jpg
artifacts/video_frame_cache/L21_V027/frame_00000300.source
```

- deterministic key: video ID + requested official `frame_idx`;
- atomic write via a temporary file plus `os.replace`;
- no duplicate decode once cached;
- staleness from the source MP4's size and mtime, recorded in the `.source` sidecar, so
  re-downloading one video invalidates only its frames;
- bounded disk usage, no unbounded RAM store;
- config validation rejects a `frame_cache_dir` inside `DATA_ROOT`.

Nothing is ever written into `data/keyframes/`. These are derived, disposable artifacts
and are deliberately **not** part of the retrieval cache manifest.

## 7. Existing-Video Development Scope

`dataset.scope.mode` accepts `patterns` (default) and `existing_videos`.
`existing_videos` adds a source constraint underneath the patterns:

```text
MP4 present under DATA_ROOT/video   INTERSECT   map-keyframes present   INTERSECT   CLIP features present
```

It is resolved from disk on every run by `resolve_dataset_scope`, so no video ID is
ever hard-coded. Patterns still apply on top, so
`--scope-existing-videos --video-exclude "L21_V027"` works. An unresolved
`existing_videos` scope passed to `select_video_ids` raises rather than silently
selecting too much.

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data `
  --scope-existing-videos inspect-data --output artifacts\dataset_report_existing_videos.json
```

Note the argparse ordering: `--scope-existing-videos`, `--video-include`,
`--video-exclude`, `--cache-dir`, and `--rebuild` are **global** options placed before
the subcommand.

## 8. Cache Identity

The manifest records `dataset_scope.mode`, `selected_video_ids`,
`selected_video_ids_hash`, and `selected_video_count`. The fingerprint depends on the
**resolved** IDs, never on the string `"existing_videos"`, so downloading one more MP4
changes the hash and the old cache becomes stale by design. Query-time settings such as
`ranking.final_top_k` still do not affect the fingerprint.

## 9. Retrieval Pipeline Is Unchanged

```text
query -> official BTC CLIP index -> candidate record -> keyframe JPEG if present
                                                     -> otherwise MP4 decode for display/evidence
```

No re-embedding, no generated embeddings replacing the official features. Decoding is
lazy: it happens when the UI requests a frame or when Q&A selects a handful of evidence
frames, never for all Top-100 candidates. Search results carry `image_available`,
`image_source`, and `video_available`, all computed without touching a video file.

An MP4 with no map or CLIP is never given an invented mapping. It is reported as
`VIDEO_PRESENT_BUT_RETRIEVAL_SUPPORT_MISSING`, kept out of the global index, and
excluded by the `existing_videos` scope. A separate video-ingestion pipeline is out of
scope for this phase.

## 10. UI Frame Serving

`GET /api/video/frame/<keyframe_id>` now serves the JPEG when present, otherwise a
decoded MP4 frame, and returns a machine-readable error instead of a 500 when neither
exists:

| State | Response |
|---|---|
| JPEG or MP4 available | `200` image/jpeg, `X-Frame-Source: keyframe_jpeg \| video_decode`, `X-Frame-Id: <official frame_idx>` |
| Unknown keyframe ID | `404` `{"error_code": "FRAME_NOT_FOUND"}` |
| No visual source | `422` `{"error_code": "FRAME_UNAVAILABLE", "frame": {...}}` |

The frontend was not redesigned.

## 11. Real Local MP4 Inventory

`tools/inspect_video_inventory.py --data-root data --probe-readable`, run 2026-08-10,
writing `artifacts/video_inventory.json` and `artifacts/video_support_coverage.json`:

| Fact | Result |
|---|---:|
| Video root | `data/video` |
| MP4 files | 29 |
| Collections | `L21`: 29 |
| Duplicate video IDs | 0 |
| Unreadable containers | 0 |
| Total size | 3,378,944,552 bytes (3.147 GiB) |

Coverage of those 29 videos:

| Fact | Result |
|---|---:|
| video + map + CLIP | 29 |
| video + map + CLIP + JPEG folder | 25 |
| needing MP4 fallback for some or all frames | 5 |
| video without map | 0 |
| video without CLIP | 0 |

The video IDs are `L21_V001`-`L21_V003`, `L21_V005`-`L21_V019`, and
`L21_V021`-`L21_V031` (`L21_V004` and `L21_V020` exist in neither videos nor support
data).

## 12. Real Inspection Result

`--scope-existing-videos`, written to `artifacts/dataset_report_existing_videos.json`:

| Fact | Result |
|---|---:|
| Discovered support-data videos | 873 |
| Available MP4 videos | 29 |
| Selected video-backed videos | 29 |
| Excluded | 844 |
| `retrieval_valid` | 29 |
| `visual_accessible` | 29 |
| `refinement_ready` | 29 |
| JPEG-backed | 24 |
| MP4-fallback | 5 |
| No visual source | 0 |
| Invalid | 0 |
| Missing map / CLIP | 0 / 0 |
| Corrupt or unreadable MP4 | 0 |
| Map rows / feature vectors / keyframe images | 7,800 / 7,800 / 6,624 |
| Feature contract | dimension 512, dtype float16 |
| Duplicate official `frame_idx` | 10 (info) |
| Duplicate internal keyframe IDs | 0 |
| `valid_for_index_build` | **true** |
| Inspection time | 3.252 s |

```text
FRAME_IDX_DUPLICATE: 10        (info)
FRAME_IDX_NON_MONOTONIC: 10    (info)
KEYFRAME_JPEG_UNAVAILABLE: 5   (info — MP4 covers every gap)
KEYFRAME_FOLDER_MISSING: 4     (info)
KEYFRAME_IMAGE_MISSING: 1      (info)
KEYFRAME_COUNT_MISMATCH: 1     (info)
```

The five videos that were blocking Phase 3.1 — `L21_V027` (112 of 292 JPEGs missing)
and `L21_V028`-`L21_V031` (no JPEG folder) — all have their original MP4, so they are
now fully usable. No additional data was downloaded.

## 13. Real MP4 Fallback Smoke Test

`tools/smoke_video_frame.py --data-root data`, writing
`artifacts/video_frame_smoke.json`. It picks a mapped keyframe whose BTC JPEG is
genuinely absent and never deletes anything.

| Case | Result |
|---|---|
| `L21_V027/kf_000003`, JPEG genuinely missing | decoded, `source=video_decode`, `seek_method=frame_index`, requested 300 → landed 300, 1280×720, 171,428 valid JPEG bytes |
| `L21_V001/kf_000002`, JPEG present, `prefer_keyframe_jpeg=False` | decoded, requested 90 → landed 90, 141,423 valid JPEG bytes; the JPEG on disk untouched |
| same request repeated | `cache_hit=true`, `seek_method=cache`, byte-identical |

Official `frame_idx` was preserved in every case. `data/keyframes/L21_V027` still holds
exactly its original 180 files, and no `frame_*.jpg` exists anywhere under `data/`.

## 14. Development Cache

Built, because the video-backed scope is `valid_for_index_build`:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\settings.yaml --data-root data `
  --cache-dir artifacts\aic2026_index_existing_videos --scope-existing-videos --rebuild build-index
```

`--allow-stale-cache` was not used, and neither `artifacts/aic2026_index` nor an L21
wildcard cache was reused.

| Fact | Result |
|---|---|
| Cache directory | `artifacts/aic2026_index_existing_videos` |
| `inspect-cache` | `status=valid`, `valid=true`, `legacy=false`, `stale=false`, `corrupt=false` |
| Scope mode | `existing_videos` |
| Selected video count | 29 |
| Selected video IDs hash | `9c6062381f9139a87c8049516e6b4634e37cb289a95c51cc41dd6ac2370691fe` |
| Cache fingerprint | `199cd0fdf099c5605e1939e719996a3df97f78b63775a90ab652acaaf51a7be9` |
| Record schema version | 3 |
| Frames indexed | 7,800 |
| Build time | 2.278 s |

The selected-IDs hash matches the `L21_*` pattern scope's hash because the two resolve
to the same 29 IDs today; the fingerprints still differ, because the scope payload is
part of the fingerprint. The pre-existing `artifacts/aic2026_index` was left untouched
and remains legacy.

## 15. Limitations

- Retrieval still uses the official CLIP features only; nothing is re-embedded, so the
  4 videos without any JPEG contribute exactly the frames their official features
  describe.
- `frame_idx` is derived by the organisers as `int(pts_time * fps)`, so a decoded frame
  can be up to one source frame away from the visually intended instant. This is a
  property of the official mapping, not of the decoder, and it does not affect the
  submitted value.
- Frame-index seek was verified exact on the local L21 MP4s only; other containers may
  fall back to timestamp seek, which `seek_method` reports.
- The derived frame cache is unbounded in total size; it is disposable and can be
  deleted at any time.
- Videos with an MP4 but no map/CLIP are reported and excluded, not ingested.
- No accuracy benchmark was run: no AIC ground truth exists.

## 16. Not Started

Phase 4 dynamic `DATA_ROOT` propagation has not been started. Advanced local refinement
was **not** implemented — `FrameProvider` gives Phase 5 exact, on-demand visual access
around known mapped frames, and nothing more: no uncertainty trigger, no dense temporal
sampling, no candidate rescoring, no semantic local search, no TRAKE refinement. Q&A
per-video hypotheses, TRAKE k-best, independent retrieval channels, submission
validation, and scoped UI manual editing all remain pending.
