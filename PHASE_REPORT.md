# AIC 2026 Phase Report

Date: 2026-08-04

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
- Limitation: production MP4 refinement needs a compatible image scorer; it does not pretend the text tower can score decoded images.

## Phase 5 - TRAKE Joint Alignment

- Added: `aic2026/trake.py`, `tests/test_trake_dp.py`.
- Modified: `AICCompetitionEngine.search_trake` now uses event-coverage video hypotheses and monotonic DP with gap/transition/missing penalties.
- Verified: order, min/max gap, incomplete-video penalty, deterministic output, duplicate suppression and partial R-score.
- Limitation: local refinement is implemented independently but not enabled without a production frame scorer.

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
