# Phase 11 — Final Integration, Diagnostics, Reproducibility, Release

Date: 2026-08-12 · Branch: `feat/aic2026-competition-research` · Version: `0.11.0-aic2026`

This is the final implementation phase. It adds no retrieval capability. Its job is to
make the system startable one way, identifiable after the fact, honest about what it can
and cannot do, and safe to hand to someone else.

Everything measured here is **structural**. This repository holds no AIC ground truth, so
no number below is accuracy, recall, or a quality claim of any kind.

---

## 1. What Phase 11 added

| Component | File | Purpose |
|---|---|---|
| Version identity | `aic2026/version.py` | `VERSION`, `PROJECT_VERSION`, `RELEASE_TAG`, lazy git commit/dirty that degrade to `None` outside a checkout. |
| Runtime identity | `aic2026/system_profile.py` | `SystemProfile`: code, config, cache, dataset, channel and capability identity of one runtime. |
| Readiness preflight | `aic2026/system_profile.py` | `evaluate_readiness` → `READY` / `READY_WITH_WARNINGS` / `NOT_READY`, exit codes 0/1/2. |
| Release config | `configs/competition.yaml` | One fixed artifact a competition run is identified by; parsed by the same validated loader as `settings.yaml`. |
| CLI surface | `aic2026/cli.py` | `--version`, `competition-check`, `system-profile`, `serve`. |
| API surface | `ui/app.py` | `GET /api/readiness` (200 / 503), system profile + submission contract inside `/api/health`. |
| Ground-truth guard | `aic2026/metrics.py`, `evaluation/official_eval.py` | `GroundTruthRequired` — semantic metrics refuse to run without real labels instead of returning zeros. |
| Release smoke | `tools/run_competition_smoke.py` | Fixed-fixture end-to-end run with a real submission round trip. |
| Ablation scaffold | `tools/run_ablation.py` | Structural variant comparison that explicitly refuses to name a best variant. |
| Release manifest | `tools/build_release_manifest.py` | Generated (gitignored) inventory of version, identity, caches and artifacts. |
| Tests | 5 new modules, 108 tests | Readiness, profile, smoke contract, GT guard, release config. |

## 2. One way to start the system

```powershell
.venv\Scripts\python.exe -m aic2026.cli competition-check --config configs/competition.yaml
.venv\Scripts\python.exe -m aic2026.cli serve --config configs/competition.yaml
```

`serve` runs the readiness preflight first and **refuses to start on `NOT_READY`** (exit
2). `--activate` (default) loads the index at startup, so a broken dataset fails there
rather than on a user's first search.

## 3. Readiness verdict on this machine

`competition-check --load-engine` against the real 29-video development set:

```text
PASS config                 hash 8661704d8946018d
PASS cache_policy           stale caches rejected
PASS data_root              .../KISC_module/data
PASS dataset_scope          'existing_videos' selects 29 video(s)
PASS cache                  valid, fingerprint 6ec64f85638889a2…
PASS engine                 7800 frames across 29 videos
PASS channel_clip           7800 records
PASS channel_bm25           7800 records
PASS channel_objects        7663 records
PASS channel_metadata       29 records
WARN channel_ocr            no_populated_source_data
WARN channel_asr            no_populated_source_data
WARN channel_caption        no_populated_source_data
PASS frame_provider         ready
WARN qa_backend             no production visual Q&A backend (backend 'mock')
WARN refinement_device      CPU refinement costs seconds per query
PASS submission_validator   v1 for kis, qa, trake
PASS runtime_identity       engine and config agree on the dataset

READY_WITH_WARNINGS   (exit code 1)
```

`READY_WITH_WARNINGS` is the truthful verdict, not a softened failure. Every warning
names a real, specific gap. `READY` is reachable — a test constructs a fully capable
system and asserts it — so the classification is not warning-padded into meaninglessness.

## 4. Release smoke (12 queries, real data)

`tools/run_competition_smoke.py --config configs/competition.yaml`

| Task | Queries | Result |
|---|---|---|
| KIS | 6 (2 Vietnamese) | 100 rows each; submission valid, 100 rows, byte-identical round trip |
| Q&A | 3 | 8/8 video hypotheses answered each; export **correctly refused** |
| TRAKE | 3 × 3 events | 34 / 52 / 30 complete sequences; submission valid, 30 rows, byte-identical round trip |

Timing: startup 650 ms · first (cold) query 6510 ms · 5 warm queries min 89 / p50 136 /
max 299 ms (with only 5 samples the nearest-rank p95 *is* the max) · estimated one-off
model load 6373 ms — labelled an estimate because it is the difference between the cold
first query and the warm median, not a measured load.
Memory: 38 MB before engine → 245 MB after engine → 758 MB after 12 queries.

Structural invariants, all zero: malformed predictions, wrong-event-count predictions,
cross-video TRAKE steps, unordered submission sequences, cross-video Q&A answer copies,
answers without matching evidence.

The refused Q&A export is the **correct** outcome: the backend is a non-visual mock, and
its output is rejected by `QA_ANSWER_TOO_LONG` and the non-submittable `mock_backend`
status. A submission would only be produced after a deliberate human answer.

## 5. Ground-truth guard

`R@k` and `Final Score` are semantic claims. Without official labels there is nothing to
compare against, so `require_ground_truth` raises `GroundTruthRequired`
(`error_code="GROUND_TRUTH_REQUIRED"`) rather than degrading to a fabricated baseline.
The guard rejects `None` and label objects that carry no frame ranges, and
`evaluate_labels([])` now refuses instead of returning a full report of zeros — a report
of zeros reads like a measured result rather than like nothing having been measured.

Structural diagnostics stay available with no labels at all.

## 6. Live UI/API walkthrough

A real server (`serve --port 5057`) answered 22 scripted steps; abbreviated:

| Step | Result |
|---|---|
| `GET /` | 200, 37,598-byte UI |
| `GET /api/readiness` | 200 `READY_WITH_WARNINGS`, 5 named warnings |
| `GET /api/health` | 7800 frames, 29 videos, version `0.11.0-aic2026` |
| channel status | clip 7800 · bm25 7800 · objects 7663 · metadata 29 · ocr/asr/caption unavailable |
| `POST /api/video/search` (Vietnamese) | 100 rows, candidate union 2172 |
| top row identity | `id=L21_V017/kf_000110`, `frame_id=10862`, `submission_frame_id=10862` |
| `POST /api/submission/preflight` | valid, 100 rows |
| `POST /api/results/<id>/edit` | 1 row edited; the other 99 unchanged (a frame value occurring twice stayed intact) |
| `POST /api/results/<id>/reset` | original frame restored |
| `POST /api/submission/save` (KIS) | 200, 100 rows written + sidecar report |
| `POST /api/video/vqa` | 8/8 answered, backend `mock`, visual `false`, 0 cross-video copies |
| `POST /api/submission/save` (mock Q&A) | **422 `QA_ANSWER_TOO_LONG`** |
| `POST /api/video/temporal` | 20 sequences, every row exactly 3 frames, 0 malformed, 8 incomplete alignments discarded |
| `POST /api/submission/save` (TRAKE) | 200, 20 rows |
| unknown batch | 404 `UNKNOWN_RESULT_BATCH` |
| frame / decoded-frame / video list | 200 (140 KB JPEG, 115 KB decoded, 29 videos) |
| re-activate, then export a pre-activation batch | **409 `STALE_RESULT_GENERATION`**, no file written |
| export a fresh batch afterwards | 200 |

The server was stopped afterwards; no listener remained on port 5057.

## 7. Structural ablation scaffold

`tools/run_ablation.py` runs the same queries through config variants and reports what
each did structurally. It never ranks them: choosing a "best" variant is a quality
judgement that requires ground truth. It also refuses to treat variants as comparable if
their cache fingerprints differ.

Retrieval group, 6 KIS queries (mean candidate union):

| Variant | Mean union | Cold 1st query | Mean warm |
|---|---|---|---|
| `clip_only` | 1200.0 | 7033 ms | 68 ms |
| `clip_sparse` | 1688.3 | 104 ms | 194 ms |
| `all_channels` | 1897.8 | 127 ms | 137 ms |

TRAKE group, 3 sequences:

| Variant | Complete sequences | Full-coverage videos |
|---|---|---|
| `beam_dp_k1` | 42 | 12 / 18 / 12 |
| `beam_dp_k4` (shipped) | 116 | 12 / 18 / 12 |
| `no_recovery` | 116 | 12 / 18 / 12 |
| `no_expansion` | 35 | 6 / 5 / 5 |

These reproduce what Phases 8–9 reported: k-best multiplies sequence yield at unchanged
coverage; adaptive expansion is what raises coverage; recovery fires zero times on these
queries. Candidate counts are **candidate coverage, not recall**.

## 8. Data, cache and channel integrity (final pass)

Data (`inspect-data`, real `data/`): 873 videos discovered, 29 selected by
`existing_videos`, 844 excluded, 0 invalid; 7,800 map rows == 7,800 feature vectors;
6,624 keyframe JPEGs; 24 videos JPEG-backed, 5 relying on MP4 fallback, 0 with no visual
source; `valid_for_index_build: true`. 10 duplicate official `frame_idx` values and 10
non-monotonic rows exist in the BTC data and are informational; **0 duplicate internal
keyframe ids**, which is the invariant that matters.

Caches on disk (nothing was deleted):

| Directory | Role | Size |
|---|---|---|
| `artifacts/aic2026_index_channels` | **active** (objects + media text, 29 videos) | 47.7 MB |
| `artifacts/aic2026_index` | legacy CLIP-only | 371.6 MB |
| `artifacts/aic2026_index_existing_videos` | pre-Phase-9 scoped cache | 16.5 MB |
| `artifacts/aic2026_multisignal_smoke`, `artifacts/aic2026_smoke_index` | test caches | 0.9 MB |

Query normalization final pass: accent-folded forms of a query produce identical folded
tokens (`một người đang đi bộ` == `mot nguoi dang di bo`), the dense query keeps the
user's own words for CLIP, negation is detected (`người không đội mũ` marks `không` and
excludes hat terms from positive object matching), and empty or punctuation-only input
degrades to empty tokens without raising.

## 9. Two real bugs found and fixed inside Phase 11

1. **`model_load_ms` was always 0.0.** It was computed as `elapsed - total_search_ms`
   while the model load happens *inside* `total_search_ms`. Replaced with
   `model_load_estimate_ms = first_query_ms - warm_median_ms`, explicitly labelled an
   estimate.
2. **`git_commit(short=True)` passed `HEAD` twice** (`git rev-parse --short HEAD HEAD`),
   which git rejects as "needed a single revision". Fixed, and a non-zero return code now
   yields `None` instead of a possibly partial string.

Both are integration/release defects, so they were fixed here rather than deferred.

## 10. What Phase 11 did not do

- No retrieval, ranking, fusion, refinement or TRAKE parameter was changed. Nothing was
  tuned to make smoke output look better.
- No ground truth was created, and no accuracy was reported.
- No cache or artifact was deleted; the inventory classifies, the human decides.
- No model was downloaded and no network call was made.
- `--allow-stale-cache` was never used.

See `docs/KNOWN_LIMITATIONS.md` for the honest list of what remains unproven, and
`docs/COMPETITION_RELEASE_CHECKLIST.md` for the pre-submission procedure.
