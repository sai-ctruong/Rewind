# KISC_module / Rewind - AIC 2026 Competition Context

## Goal
This repository is now oriented to the AIC 2026 preliminary round, with only three official tasks:

1. Textual KIS: submit `<video_id>, <frame_id>`.
2. Q&A: submit `<video_id>, <frame_id>, <answer>`.
3. TRAKE: submit `<video_id>, <frame_id_1>, ..., <frame_id_N>`.

Each query can output at most 100 ranked predictions. Official-style scoring is implemented as R@1, R@5, R@20, R@50, R@100, where R@k is the best R-score within the first k predictions. Final Score is the mean of those five values.

## Data Root
Default AIC data root:

```text
C:\Users\ad\Downloads\codepython\project\KISC_module\data
```

Expected layout:

```text
data/
  clip-features-32/{video_id}.npy
  map-keyframes/{video_id}.csv
  keyframes/{video_id}/{n:03d}.jpg
  objects/{video_id}/{n:03d}.json
  media-info/{video_id}.json
  video/{video_id}.mp4
```

Observed local data:

- `clip-features-32`: 873 `.npy` files.
- `map-keyframes`: 873 `.csv` files.
- `keyframes`: image folders matching video ids.
- `objects`: object JSON folders matching keyframes.
- `media-info`: 873 JSON files.
- `video`: only 29 mp4 files are present, so the system must not require original mp4s.
- Full AIC cache build loaded 873 video ids and 177,321 keyframes with feature dimension 512.

The official submitted `frame_id` is `frame_idx` from `map-keyframes/{video_id}.csv`, not the internal keyframe ordinal. Internal ids are stored as `{video_id}/{frame_idx}` to avoid collisions.

## Important Files

```text
aic2026/
  dataset.py      # AIC data loader: .npy + map csv + keyframe paths + optional objects/media
  engine.py       # Competition engine for KIS, Q&A, TRAKE
  metrics.py      # Official-style R-score, R@k, Final Score, CSV export
  benchmark.py    # JSONL benchmark/ablation logging
  cli.py          # Command-line build/search helpers

ui/
  app.py          # Flask backend using AICCompetitionEngine
  index.html      # Simple competition UI: KIS, Q&A, TRAKE

retrieval/
  coarse_retriever.py  # RRF fusion over dense/BM25 signals
  temporal_check.py    # Hard temporal ordering filter for TRAKE
  vqa_module.py        # Ground temporal window first, then answer
  video_engine.py      # Legacy video indexing engine kept for tests/backward utilities, trimmed

evaluation/
  metrics.py      # General IR/QA metrics
  run_eval.py     # Mock/offline report runner

tests/
  test_aic2026.py       # AIC loader/engine/official metrics
  test_ui_phase10.py    # Competition UI/backend contract
```

Removed legacy runtime modules:

- `dialogue/`
- `retrieval/search_agent.py`
- `retrieval/agent_tools.py`
- `retrieval/session_memory.py`
- `retrieval/image_filter.py`
- dialogue/image-filter/agent demo and tests

## Build And Run

Install light dependencies and run tests:

```powershell
cd C:\Users\ad\Downloads\codepython\project\KISC_module
.venv\Scripts\python.exe -m pytest -q
```

Build the full local AIC cache from precomputed features:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --data-root data --cache-dir artifacts\aic2026_index --rebuild build-index
```

Current full build result on this machine:

```json
{
  "cache_hit": false,
  "build_seconds": 36.759,
  "stats": {
    "videos": 873,
    "frames": 177321,
    "missing_keyframes": 0,
    "missing_objects": 0,
    "missing_videos": 844,
    "feature_dim": 512
  }
}
```

Start the web UI:

```powershell
.venv\Scripts\python.exe -m ui.app
```

Open:

```text
http://127.0.0.1:5000
```

Stop the server with `Ctrl+C`. If it was started in the background, find and stop the process listening on port 5000:

```powershell
Get-NetTCPConnection -LocalPort 5000 -State Listen
Stop-Process -Id <OwningProcess> -Force
```

## CLI Usage

KIS:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --data-root data --cache-dir artifacts\aic2026_index search --task kis --query "red shirt" --top-k 100 --out artifacts\submissions\kis.csv
```

Q&A:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --data-root data --cache-dir artifacts\aic2026_index search --task qa --query "event description" --question "question text" --top-k 100 --out artifacts\submissions\qa.csv
```

TRAKE:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --data-root data --cache-dir artifacts\aic2026_index search --task trake --events "event 1" "event 2" "event 3" --top-k 100 --out artifacts\submissions\trake.csv
```

## Accuracy Notes

The active .venv includes torch 2.13.0+cpu and transformers 5.14.1. CLIP ViT-B/32 is cached locally and production retrieval passes with 512-dimensional normalized text embeddings. Hashing remains available only for explicit test/smoke fallback.

For real AIC retrieval against `clip-features-32`, install full dependencies:

```powershell
.venv\Scripts\pip.exe install -r requirements-full.txt
```

Then the engine can use `openai/clip-vit-base-patch32` text embeddings, matching the provided 512-dimensional CLIP features.

## Scoring Logic

Implemented in `aic2026/metrics.py`:

- KIS R-score: 1 if video id matches and submitted frame id is inside any GT range, else 0.
- Q&A R-score: KIS condition plus answer match. Exact normalized answer and very-high token F1 are supported; a semantic matcher hook can be injected later.
- TRAKE R-score: 0 if video id is wrong; otherwise the fraction of event frame ids that fall in their corresponding GT frame ranges.
- R@k: max R-score among the top-k predictions.
- Final Score: average of R@1, R@5, R@20, R@50, R@100.

## TRAKE Design

`AICCompetitionEngine.search_trake()` searches each event independently, applies a hard same-video and strictly increasing timestamp filter via `retrieval.temporal_check`, then refines each event inside a local time window. This keeps alignment frame-level and avoids treating temporal order as a soft fusion score.

## Q&A Design

`AICCompetitionEngine.answer_qa()` grounds first: it searches with event text plus question, chooses a center frame, collects a temporal window around that frame, and only then calls `VqaModule`. This follows the prompt priority: ground before answering.

## Benchmarking

`aic2026/benchmark.py` writes:

- `config.json`
- `queries.jsonl`
- optional `summary.json`

under `evaluation/benchmarks/aic2026/{timestamp}-{name}/`. Use this for ablations such as objects on/off, metadata on/off, HNSW vs flat, per-event K, and TRAKE refine window.
## 2026-08-04 Accuracy Upgrade

Read AUDIT.md and PHASE_REPORT.md before continuing. The active code now includes explicit production CLIP safeguards, multi-signal fusion, diversified Top-100 ranking, local refinement fallback, TRAKE monotonic DP, grounded multi-frame Q&A, complete benchmark artifacts, annotation templates and an evaluation UI.

Current hard blockers for accuracy claims:

- torch 2.13.0+cpu and transformers 5.14.1 are installed; CLIP ViT-B/32 is cached and production smoke passed on CPU.
- No AIC-format KIS/Q&A/TRAKE ground-truth file exists.
- The existing full cache was built without object/media signals; rebuild with --load-objects --include-media-text for ablations.
- Do not restore Agent, KISC, dialogue, image filter, image query, explore, similar search or session memory.