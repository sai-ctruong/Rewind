# AIC 2026 Phase Report

Chronological history of the project, last updated 2026-08-12 (version `0.11.0-aic2026`).

**Two phase numberings exist and must not be confused.** The first section below is the
original 2026-08-04 series (Phases 0–8), which built the baseline pipeline. The second
section is the AIC2026 competition-hardening series (Phases 1–11), each with a design
document under `docs/`. Where they collide — both have a "Phase 5", for example — the
document name disambiguates.

## Index of the AIC2026 series

| Phase | Subject | Document |
|---|---|---|
| 1 | Validated runtime configuration | `docs/PHASE_1_RUNTIME_CONFIG.md` |
| 2 | Cache manifest, fingerprint, stale/legacy policy | `docs/PHASE_2_CACHE_MANIFEST.md` |
| 3 | Strict dataset validation | `docs/PHASE_3_DATASET_VALIDATION.md` |
| 3.1 | Dataset scope and keyframe identity | `docs/PHASE_3_1_DATASET_SCOPE_AND_MAPPING.md` |
| 3.2 | Video-backed development, frame provider | `docs/PHASE_3_2_VIDEO_BACKED_DEVELOPMENT.md` |
| 4 | Dynamic DATA_ROOT runtime state | `docs/PHASE_4_DYNAMIC_DATA_ROOT.md` |
| 5 | Query-conditioned local refinement (KIS) | `docs/PHASE_5_LOCAL_REFINEMENT.md` |
| 6 | Grounded per-hypothesis Q&A | `docs/PHASE_6_GROUNDED_QA.md` |
| 7 | TRAKE structural correctness | `docs/PHASE_7_TRAKE_STRUCTURAL_CORRECTNESS.md` |
| 8 | TRAKE k-best, expansion, refinement | `docs/PHASE_8_TRAKE_KBEST_AND_REFINEMENT.md` |
| 9 | Multi-channel retrieval, Vietnamese queries | `docs/PHASE_9_MULTI_CHANNEL_RETRIEVAL.md` |
| 10 | Submission validation and UI result safety | `docs/PHASE_10_SUBMISSION_AND_UI_SAFETY.md` |
| 11 | Final integration, reproducibility, release | `docs/PHASE_11_FINAL_INTEGRATION.md` |

---

# Original series (2026-08-04)

## Phase 0 - Audit

- Added: `AUDIT.md`.
- Verified: branch/history, 873 feature/map files, 29 MP4 files, 512-dimensional features, 225 baseline tests.
- Result: current context was partly correct, but production fallback, multi-signal scoring, Top-100 diversification and TRAKE DP were missing.

## Phase 1 - Production Text Encoder

- Added: `aic2026/text_encoder.py`, `tests/test_text_encoder.py`.
- Design: batch-capable CLIP ViT-B/32 encoder, CPU/CUDA selection, 512-dimension/dtype/finite/normalization checks, explicit status, strict production guard.
- Test: real-model integration test passes from the local cache; offline fallback and strict production-policy tests also pass.
- Verified environment: torch 2.13.0+cpu, transformers 5.14.1, cached CLIP ViT-B/32, 512-dimensional output and production_ready=true.

## Phase 2 - Multi-Signal Fusion

- Added: `aic2026/fusion.py`, `tests/test_signal_fusion.py`.
- Modified: AIC loader now preserves object labels, confidence and bounding boxes; engine fuses dense, object, metadata and BM25 scores.
- Ablations: clip-only, objects, metadata, sparse, RRF full and adaptive full.
- Smoke benchmark: 2 videos / 200 frames built in 0.13 s; 196 frames had object evidence and 200 had metadata.
- Limitation: OCR/ASR/caption signals are used only when source files actually populate them.

## Phase 3 - Video-Aware Top-100

- Added: `aic2026/ranking.py`, `tests/test_top100_ranking.py`.
- Design: precision head, neighbor expansion, temporal suppression, per-video cap and recall tail.
- Verified: no duplicates, output cap, rank-one preservation, neighbor inclusion and video diversity.

## Phase 4 - Local Refinement

- Added: `aic2026/local_refinement.py`, `tests/test_local_refinement.py`.
- Design: bounded OpenCV decoding, interval cache/memory cap, injected visual scorer and explicit map-keyframe fallback.
- Verified: synthetic MP4 selects the stronger local frame and never exceeds `max_frames`; missing MP4 returns `keyframe_only` warning.
- Superseded by the AIC2026 Phase 5 work (`docs/PHASE_5_LOCAL_REFINEMENT.md`): a production
  image scorer now exists (`aic2026/frame_scorer.CLIPFrameScorer`, sharing one CLIP
  checkpoint with the text tower), and the refiner is called end-to-end by `search_kis`.

## Phase 5 - TRAKE Joint Alignment

- Added: `aic2026/trake.py`, `tests/test_trake_dp.py`.
- Modified: `AICCompetitionEngine.search_trake` now uses event-coverage video hypotheses and monotonic beam-pruned DP (`beam_dp`, not exact DP) with gap/transition/missing penalties.
- Verified: order, min/max gap, incomplete-video penalty, deterministic output, duplicate suppression and partial R-score.
- Limitation: TRAKE still does not use local refinement. The AIC2026 Phase 5 work enabled
  refinement for Textual KIS only; `search_trake(refine_window_s=...)` remains unused.

## Phase 6 - Grounded Q&A

- Added: `aic2026/qa.py`, Q&A grounding/normalization tests.
- Modified: engine separates event and question, defaults retrieval to event-only, chooses ordered diverse evidence, reports normalized answer/confidence/warning.
- Limitation: default answerer is offline mock. A local/API VLM must be configured for real answer accuracy.

## Phase 7 - Evaluation

- Added: official task runners, ablation registry, error categories, latency helper, report helper, annotation tool and `evaluation/labels/template.jsonl`.
- Modified: benchmark runs now save config, environment, queries, predictions, summary and errors.
- Verified: synthetic official-evaluation test and complete artifact-contract test pass.
- Blocker: no AIC-format development/official labels exist in the repository. Legacy `evaluation/labels*.json` are not AIC task labels, so no AIC Final Score was reported.

## Phase 8 - Competition UI

- Modified: `ui/app.py`, `ui/index.html`.
- Added: encoder/device/MP4 health, score/evidence display, MP4 link, frame/answer correction, separate Q&A event/question, confidence, TRAKE order status and background evaluation.
- HTTP smoke: health returned 873 feature videos and 29 MP4 files; server was stopped after the check.
- Limitation: no browser backend was available in this session, so screenshot/mobile visual QA could not be completed.

## Verification

```text
python -m compileall -q aic2026 retrieval evaluation ui tests: passed
python -m pytest -q: passed; one legacy lazy-import test skipped because torch is installed
CLI production smoke: 5 rows returned with CLIP/CPU/512, production_ready=true, latency 5255.6 ms
Port check: no Flask listener remained on port 5000 after verification
```

## User Action Required

1. Completed: real CLIP dependencies and model are installed/cached on CPU.
2. Provide AIC JSONL ground truth, or annotate a development set from `evaluation/labels/template.jsonl` (suggested minimum: 50 KIS, 30 Q&A, 30 TRAKE).
3. Rebuild a full multi-signal cache with `--load-objects --include-media-text` before running ablations.

---

# AIC2026 competition series

Each entry records what changed, what was verified on real data, and what remained open.
Full detail lives in the matching `docs/PHASE_*.md`.

## AIC2026 Phase 1 - Validated Runtime Configuration

- Added `aic2026/config.py`: a validated `AppConfig` with a deterministic `config_hash`,
  wired into engine, CLI, UI and benchmark snapshots. `configs/settings.yaml` gained the
  `aic2026:` section as the single runtime source.

## AIC2026 Phase 2 - Cache Manifest And Validation

- Added `aic2026/cache_manifest.py`: every cache carries an atomic manifest with a build
  fingerprint and data signature; legacy, stale and corrupt caches have explicit policies
  and are rejected before unpickling.
- Build-time and query-time configuration are separated: changing `ranking.final_top_k`
  does not invalidate a cache; changing `dataset.load_objects` does.

## AIC2026 Phase 3 / 3.1 / 3.2 - Dataset Truth

- Strict map/feature alignment (`DatasetAlignmentError`), structured inspection reports,
  and `valid_for_index_build` as the build gate.
- **3.1**: `DatasetScopeConfig` selects active videos by pattern; the internal keyframe id
  became `{video_id}/kf_{ordinal:06d}` after real data showed 192 videos repeating a
  `frame_idx`, which the old id silently collapsed. Record schema advanced to v3.
- **3.2**: official data roles encoded correctly — the MP4 is the competition data and
  keyframes/features/objects/metadata are supporting data. `frame_provider.py` serves a
  JPEG, else decodes the mapped frame from the MP4, else reports unavailable. Scope mode
  `existing_videos` resolves the 29-video video-backed subset from disk.

## AIC2026 Phase 4 - Dynamic DATA_ROOT

- One frozen `RuntimeDatasetState` per activation, replaced atomically; a failed switch
  leaves the previous state serving. Every state carries a `generation`, and a superseded
  request gets `409 STALE_RESULT_GENERATION`.

## AIC2026 Phase 5 - Query-Conditioned Local Refinement (KIS)

- `LocalFrameRefiner` is genuinely called by `search_kis`: bounded MP4 decoding
  (candidate budget → window → `max_frames`), a real `CLIPFrameScorer` sharing one
  checkpoint with the text tower through `aic2026/clip_backend.py`, and an
  `always`/`uncertainty`/`disabled` trigger policy.
- Three frame identities kept distinct — `coarse_official_frame_idx`,
  `best_visual_frame_idx`, `submission_frame_idx` — with `preserve_coarse` as the default
  output policy.
- **No accuracy claim**: verified with synthetic MP4s and structural diagnostics only.

## AIC2026 Phase 6 - Grounded Q&A

- `answer_qa` groups the retrieval pool by video and answers each hypothesis from its own
  evidence. `cross_video_answer_copy_count` and
  `answer_without_matching_evidence_video_count` are computed from the produced rows and
  are 0; a `QAEvidenceBundle` refuses frames from a second video.
- Backends report truthful `QAAnswererStatus`; on this machine `auto` resolves to the
  **non-visual mock**, labelled as such everywhere.
- Real smoke caught `normalize_answer("không có mô tả", expected_type="yes/no")` returning
  a confident `"no"`; refusal forms now short-circuit ahead of type handling.

## AIC2026 Phase 7 - TRAKE Structural Correctness

- N events always produce exactly N submitted frames. A `TrakeAlignment` holds one step
  per event, a deterministic recovery pass fills gaps from the same video's candidates for
  that event, and anything still incomplete is discarded. `TrakePrediction` raises unless
  it carries exactly `event_count` frames.
- `trake_r_score` checks length before scoring; `zip` had been truncating silently.
- The method is named `beam_dp` everywhere; `exact_dp` is rejected by config validation.
- Cost reported honestly: 65 of 77 alignments discarded rather than emitted short.

## AIC2026 Phase 8 - TRAKE K-Best, Expansion, Refinement

- k-best path enumeration plus diversity filtering: complete sequences went 12 → 187 on
  the same four queries.
- Adaptive candidate expansion re-retrieves only under-covered events through the existing
  retriever, bounded by `candidate_depth_max`: full-coverage videos went 12 → 66.
- `align_video_exact_dp` added as a bounded **test oracle** only.
- `trake_refinement.py` refines a few complete sequences event by event against each
  event's own text; off by default (~10 s per query on CPU).

## AIC2026 Phase 9 - Multi-Channel Retrieval And Vietnamese Queries

- A real-data audit found the shipped cache had been built with `load_objects: false` and
  `include_media_text: false`: the BM25 corpus was 7,800 empty sentinels and the pipeline
  was effectively CLIP-only. A new cache was built.
- CLIP, BM25, objects and metadata became independent candidate generators; OCR, ASR and
  captions report `available: false` rather than being hidden.
- `query_normalization.py` keeps the original query for CLIP and gives folded/expanded
  views to the lexical channels; negation is preserved and excluded from positive matching.
- **Honest regression reported, not tuned away**: TRAKE full coverage fell 50 → 42 videos
  because a more diverse pool spreads a fixed per-event `top_k` across more videos.
- Candidate counts are **candidate coverage, not recall**.

## AIC2026 Phase 10 - Submission Validation And UI Result Safety

- Added: `aic2026/submission_validation.py`, `aic2026/result_batch.py`,
  `docs/PHASE_10_SUBMISSION_AND_UI_SAFETY.md`, and four test modules.
- Design: one validator for all three official tasks, used by the CLI export, the CLI
  `validate-submission` command, the UI preflight and the UI export. Manual edits address
  `result_id + row_id` (plus `event_index` for TRAKE) instead of matching values.
- Verified on real L21: KIS 100 rows, TRAKE 34 sequences of 3 events, Q&A exported only
  after deliberate manual answers, a stale generation refused with 409 and no file
  written, and an edit to one row of a natural duplicate frame leaving 99 others intact.
- Limitation: validation is structural. It says a submission has the right FORMAT and
  says nothing about whether any answer or frame is correct; no AIC ground truth exists.

## AIC2026 Phase 11 - Final Integration, Reproducibility, Release

- Added `aic2026/version.py`, `aic2026/system_profile.py`, `configs/competition.yaml`,
  `tools/run_competition_smoke.py`, `tools/run_ablation.py`,
  `tools/build_release_manifest.py`, and five test modules (108 tests).
- **One way to start the system**: `aic2026.cli serve --config configs/competition.yaml`,
  which runs the readiness preflight first and refuses to start on `NOT_READY`.
- **Readiness** is `READY` / `READY_WITH_WARNINGS` / `NOT_READY` with exit codes 0/1/2.
  On this machine the verdict is `READY_WITH_WARNINGS`: 13 PASS, and 5 warnings that each
  name a real gap (no OCR/ASR/caption source data, no visual Q&A backend, CPU refinement).
  `READY` was left reachable and is asserted by a test, so the verdict means something.
- **Reproducibility**: `SystemProfile` records version, commit, config hash, cache
  fingerprint, dataset hash, schema versions and capability state; `identity()` is the
  subset two runs must share to be considered the same setup.
- **Ground-truth guard**: semantic metrics now raise `GroundTruthRequired` instead of
  scoring against absent labels, and an empty label set refuses rather than reporting a
  full table of zeros.
- **Release smoke** on real data: KIS and TRAKE submissions valid with byte-identical
  round trips, all six structural invariants 0, and the mock Q&A export correctly refused.
- **Live API walkthrough** (22 steps) verified single-row edits, reset, preflight, export,
  frame serving, and `409 STALE_RESULT_GENERATION` after a dataset re-activation.
- **Two real bugs found and fixed here**: `model_load_ms` was structurally always 0.0, and
  `git_commit(short=True)` passed `HEAD` twice.
- **No accuracy claim.** No retrieval, ranking or refinement parameter was changed; no
  ground truth was created; no cache or artifact was deleted.
- See `docs/PHASE_11_FINAL_INTEGRATION.md`, `docs/KNOWN_LIMITATIONS.md` and
  `docs/COMPETITION_RELEASE_CHECKLIST.md`.
