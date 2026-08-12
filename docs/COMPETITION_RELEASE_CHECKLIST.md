# Competition Release Checklist

Version `0.11.0-aic2026`. Run top to bottom before a competition session, then keep the
"during the session" part open beside you.

Every check below is **structural**. Passing all of them means the system runs and emits
well-formed submissions. It does not mean any answer is correct — see
`docs/KNOWN_LIMITATIONS.md`.

---

## A. Before the session (once, on the competition machine)

### 1. Environment

```powershell
cd <repo>
.venv\Scripts\pip.exe install -r requirements-full.txt
.venv\Scripts\python.exe -m aic2026.cli --version      # rewind-aic2026 0.11.0-aic2026
```

- [ ] `--version` prints `0.11.0-aic2026`.
- [ ] The CLIP checkpoint `openai/clip-vit-base-patch32` is present in the local HF cache.
      Verify offline: `$env:HF_HUB_OFFLINE="1"` and run step 4 — it must still work.

### 2. Data

- [ ] `data/` holds the competition media (`video/`, `map-keyframes/`, `clip-features-32/`,
      `keyframes/`, `objects/`, `media-info/`).
- [ ] Inspect it:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs/competition.yaml inspect-data
```

- [ ] `valid_for_index_build: true`.
- [ ] `total_map_rows == total_feature_vectors`.
- [ ] `selected_video_count` matches what you expect to search. Note the
      `selected_video_ids_hash`; it identifies the dataset in every later artifact.
- [ ] `no_visual_source_video_count: 0`.

### 3. Index

The competition config points at `artifacts/aic2026_index_channels`, which **must** be
built with objects and media text or the object and metadata channels will be empty:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs/competition.yaml `
    --rebuild build-index --load-objects --include-media-text
```

- [ ] Build finished without `--allow-stale-cache`. Never use that flag to get past an
      incompatibility; rebuild instead.
- [ ] `inspect-cache` reports `valid: true`, `stale: false`, `legacy: false`.

### 4. Readiness

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs/competition.yaml `
    competition-check --load-engine --output artifacts\readiness.json
```

- [ ] Status is `READY` or `READY_WITH_WARNINGS` (exit 0 or 1). `NOT_READY` (exit 2)
      blocks the session — fix the named FAIL.
- [ ] `channel_clip` PASS. Everything else being WARN is acceptable and expected.
- [ ] Read every WARN out loud and confirm you accept it. On this machine the expected
      warnings are `channel_ocr`, `channel_asr`, `channel_caption` (no source data),
      `qa_backend` (no visual backend), `refinement_device` (CPU).

### 5. Tests and smoke

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe tools\run_competition_smoke.py --config configs\competition.yaml
```

- [ ] Full suite green.
- [ ] Smoke: KIS and TRAKE submissions valid with `byte_identical_roundtrip: true`.
- [ ] Smoke: all six `structural_invariants` counters are 0.
- [ ] Smoke: the Q&A export is refused unless you have configured a real visual backend.
      **A refused mock Q&A export is the correct result.**
- [ ] Note the timings in `artifacts/final_release_smoke/performance.json` so you know
      your per-query budget.

### 6. Reproducibility record

```powershell
git status                                   # commit anything you want identified
.venv\Scripts\python.exe tools\build_release_manifest.py --config configs\competition.yaml
```

- [ ] `git_dirty: false` — run from a clean tree so the commit identifies the code.
- [ ] Save `artifacts/release_manifest.json` somewhere outside `artifacts/`; it records
      version, commit, config hash, cache fingerprint and dataset hash.
- [ ] Optionally tag: `git tag aic2026-competition-ready`.

---

## B. Starting the system

One command, no alternatives:

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\competition.yaml serve
```

- [ ] Startup printed the readiness verdict and activated the dataset.
- [ ] `http://127.0.0.1:5000` loads.
- [ ] `GET /api/readiness` returns 200 (503 means `NOT_READY` — stop and fix).
- [ ] `GET /api/health` shows the expected `dataset_size`, `videos`, `project_version` and
      `cache_fingerprint`.

If you change `DATA_ROOT` from the UI, the runtime generation advances and every result
batch produced before the change becomes non-exportable by design.

---

## C. During the session, per query

### KIS

- [ ] Search. Confirm `count` is the top-k you expect (100 for submission).
- [ ] Check `diagnostics.channels` — if only `clip` contributed, your index was probably
      built without objects/media text.
- [ ] Fix wrong frames with the row editor. Each edit touches **one row**; use reset to
      undo. Never edit by value.
- [ ] Preflight (`/api/submission/preflight`) before saving. `valid: true`, expected row
      count, and `active_generation` matching the current run.
- [ ] Save. Keep the `.validation.json` sidecar with the CSV.

### Q&A

- [ ] Confirm the backend badge. If it says non-visual, the engine's answers are
      **not** submittable — type the answers yourself; a manual answer is exportable.
- [ ] Confirm `cross_video_answer_copy_count: 0` in diagnostics.
- [ ] Answers must be short. Anything over 512 characters is refused, and anything that
      long is not an answer.

### TRAKE

- [ ] Every returned row must have exactly one frame per event. `structural` reports
      `malformed_prediction_count: 0` and `wrong_event_count_prediction_count: 0`.
- [ ] `discarded_incomplete_alignments` > 0 is normal: incomplete alignments are dropped,
      never shortened.
- [ ] Save with the correct event count; the validator enforces it.

---

## D. Before submitting a file to the organisers

- [ ] Re-validate the exact file you are about to upload:

```powershell
.venv\Scripts\python.exe -m aic2026.cli validate-submission --task kis --input <file>.csv
.venv\Scripts\python.exe -m aic2026.cli validate-submission --task trake --input <file>.csv --event-count 3
```

- [ ] Row count ≤ 100 and in the intended rank order.
- [ ] `video_id` values look like `L21_V001` and frames are non-negative integers.
- [ ] For TRAKE, every row has `1 + event_count` columns.
- [ ] For Q&A, every answer is one you are willing to defend — a mock-produced answer must
      never reach a submission.
- [ ] The sidecar report's `selected_video_ids_hash` and `config_hash` match the manifest
      from step 6.

---

## E. After the session

- [ ] Copy `artifacts/submissions/` and `artifacts/release_manifest.json` somewhere
      durable; `artifacts/` is gitignored and regenerable.
- [ ] Stop the server (Ctrl+C) and confirm nothing is still listening on the port.
- [ ] Record which config, commit and cache fingerprint produced each submitted file.

---

## Fast failure lookup

| Symptom | Meaning | Action |
|---|---|---|
| `competition-check` exit 2, `cache` FAIL | no cache, corrupt, legacy or stale | rebuild the index; never `--allow-stale-cache` |
| `dataset_scope` FAIL | scope selects no videos | check `data/video` and the scope patterns |
| `channel_clip` FAIL | index has no usable vectors | rebuild |
| Only `clip` contributes candidates | index built without objects/media text | rebuild with `--load-objects --include-media-text` |
| Export → `409 STALE_RESULT_GENERATION` | results predate the current dataset activation | re-run the query, then export |
| Export → `422 QA_NON_SUBMITTABLE_STATUS` | mock backend answer | type the answer manually |
| Export → `422 QA_ANSWER_TOO_LONG` | over 512 characters | write a real short answer |
| Export → `422 TRAKE_EVENT_COUNT_MISMATCH` | wrong `--event-count` or a short row | pass the right event count; short rows are refused by design |
| `GroundTruthRequired` | a semantic metric was requested | expected: there is no ground truth here |
