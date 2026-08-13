# Known Limitations

Version `0.11.0-aic2026` · last reviewed 2026-08-12

This is the honest list. Everything here is a real constraint of the shipped system, not
a hypothetical. If something in this repository sounds like a quality claim and is not
listed here as verified, treat it as unverified.

---

## 1. The one that governs everything else: no ground truth

**This repository contains no AIC ground truth.** Not a partial set, not a development
split — none.

Consequences, all enforced in code:

- No accuracy, recall, precision, R@k or Final Score is reported anywhere for this system.
- `aic2026.metrics.require_ground_truth` raises `GroundTruthRequired` rather than scoring
  against absent, synthetic or self-authored labels.
- Every count this project publishes is **structural**: how many candidates a channel
  contributed, how many complete sequences were produced, how many rows validated, how
  long something took. Candidate counts are *candidate coverage*, never recall.
- No parameter has been tuned against quality. Where a phase reported a structural
  regression (Phase 9: TRAKE full-coverage videos 50 → 42), it was reported rather than
  tuned away.

`evaluation/labels.json` is **not** AIC ground truth: it annotates the bundled demo
clips. `evaluation/labels/template.jsonl` is an empty template for annotating a real
development set. A test asserts no other label file is shipped.

## 2. No production visual Q&A backend

On this machine `qa.backend.type: auto` resolves to the **non-visual mock**, which reads
retrieved text and reasons about it. It does not look at images.

- The mock reports `visual_capable: false` and `production_ready: false` everywhere:
  health, search payload, system profile, readiness (`WARN qa_backend`), and the UI.
- Its answers are **not exportable**. The submission validator rejects them with the
  non-submittable `mock_backend` status, and rejects over-long text with
  `QA_ANSWER_TOO_LONG` (512 characters) — a rule added after a real smoke caught the mock
  echoing an entire YouTube description as an "answer".
- Q&A **retrieval**, per-video hypothesis grounding and evidence selection are fully
  exercised and correct; only the answering step is unavailable.
- A human-typed answer is marked `manual` and *is* exportable. That is the intended path
  until a real image-capable backend is configured.

What is required to lift this: an API key for a hosted vision model, or a local VLM
checkpoint plus its runtime. Neither is downloaded automatically.

## 3. Local visual refinement is off by default

Both KIS refinement (`refinement.enabled`) and TRAKE refinement
(`trake.refinement.enabled`) ship **disabled** in `configs/competition.yaml`.

The reason is latency, not quality: on this CPU-only machine KIS refinement costs roughly
14 s per query and TRAKE refinement roughly 10 s per query. Readiness reports this as
`WARN refinement_device`. Enable them deliberately when a CUDA device or a larger time
budget is available. **Whether refinement improves results is unknown** — that is a
quality question and there is no ground truth.

When enabled, refinement is bounded (candidate budget → window → `max_frames`, with a
hard per-query frame ceiling) and never decodes a whole video.

## 4. The submitted frame stays the coarse mapped `frame_idx`

`refinement.frame_output_policy: preserve_coarse`. A refined frame is **evidence only**:
`coarse_official_frame_idx`, `best_visual_frame_idx` and `submission_frame_idx` are three
distinct values, and only the last reaches a CSV.

This is deliberate and unresolved: the organisers have not confirmed how an arbitrary
decoded frame index is interpreted against their keyframe mapping. Until they do,
submitting anything other than the official mapped `frame_idx` risks a well-formed row
that means something different to the grader than to us.

## 5. TRAKE alignment is beam-pruned, not exact

`trake.alignment_method: beam_dp`. States are truncated by `beam_width`, so the returned
alignment is not guaranteed optimal. This is named honestly in config, code, API, UI and
docs; `exact_dp` is **rejected** by config validation, and a test scans the sources for
false claims of exactness.

`align_video_exact_dp` exists as a bounded **test oracle** only. It is not the production
path.

Completeness is enforced ahead of yield: a sequence carries exactly one frame per event
or it is discarded. On the walkthrough query, 8 incomplete alignments were discarded to
return 20 complete ones. That trade costs candidates and is intentional — a short TRAKE
row is not a partial answer, it is a malformed one.

## 6. OCR, ASR and frame captions do not exist in this data

Three channels are constructed and report `available: false` with
`no_populated_source_data`. The sources are genuinely empty in the BTC data available
here. Nothing is substituted for them: object labels are not OCR, and media metadata is
not a frame caption.

**R0 update:** they are now **disabled** in `configs/competition.yaml`. Disabling does not
hide the absence — each channel still measures its own source and still reports
`available: false, no_populated_source_data`, and readiness lists them as `INFO` rather
than `WARN`. Re-enable them the moment a source is populated.

## 7. Dataset scope is 29 videos, not the full collection

`scope.mode: existing_videos` selects videos that have an MP4 **and** map + CLIP support:
29 of 873 discovered videos on this machine. The other 844 are excluded because their
MP4s were not downloaded, not because they are broken.

**R0 update — this is a configuration choice, not a capability limit.** All 873 videos are
retrieval-ready (map + CLIP + objects + media-info all present); only 29 have MP4s. Scope
mode `retrieval_ready` searches all 873 and `configs/competition_full_retrieval.yaml`
uses it. That config has **no index built** in this checkout: building it is a large,
deliberate step, and until it is built the shipped competition config still searches 29
videos. What an absent MP4 actually costs is preview, local refinement and visual Q&A —
never retrieval.

Everything measured in this repository was measured on those 29 videos (7,800 frames).
Latency, memory and candidate counts will change on the full collection; the structural
invariants will not.

Known BTC data quirks in that scope, all handled: 10 duplicate official `frame_idx`
values and 10 non-monotonic rows (informational), 4 missing keyframe folders and 5 videos
needing MP4 fallback for some frames (`retrieval_valid=true, visual_accessible=false` is
a supported state).

## 8. Structural validation is not correctness

`aic2026/submission_validation.py` proves a submission has the right **format**: correct
column count, valid video ids, non-negative integer frames, no duplicate rows, at most
100 rows, an answer present and submittable for Q&A, exactly one frame per event for
TRAKE, and a current runtime generation.

It says **nothing** about whether any frame or answer is right. A perfectly valid
submission can be entirely wrong.

## 9. Performance numbers are hardware-specific and small-scale

Measured on this machine only: CPU-only, 29 videos, 7,800 frames.

- Startup ~650 ms; cold first query ~6.5 s (dominated by the one-off CLIP text-encoder
  load); warm KIS queries p50 ~136 ms over a 89–299 ms spread; TRAKE 0.7–1.3 s per
  3-event query. Five warm samples is too few for a meaningful p95 — the reported value is
  simply the slowest of them.
- Memory 38 MB → 245 MB after engine load → 758 MB after 12 queries.

Nothing here predicts behaviour on competition hardware or on the full collection.

## 10. Smaller open items

- **Video dropdown in the UI** does not filter search; search runs over the whole indexed
  collection. The control is misleading and should either filter or be removed.
- **`evaluation/run_eval.py`** is a legacy end-to-end harness over *synthetic mock data*
  with self-generated labels. Its numbers describe that synthetic set and nothing else;
  do not quote them as system quality.
- **`serve` uses the Flask development server.** Fine for a single operator on localhost,
  not a production deployment.
- **No browser-level UI testing** has been done in this environment (no browser backend
  available). API behaviour is verified; visual/mobile rendering is not.
- **`git_dirty: true`** appears in profiles generated from a working tree with
  uncommitted changes. That is information, not an error — but a submission run should be
  made from a clean tree so the commit identifies the code.

## 11. What would change these answers

Only two things:

1. **Official AIC ground truth.** Then `evaluate_labels` supplies the quality half,
   `tools/run_ablation.py` variants become rankable, and refinement can be judged rather
   than assumed.
2. **A real visual Q&A backend.** Then Q&A answers become exportable without a human in
   the loop, and `readiness` reaches `READY` on the Q&A axis.

Neither is a code change in this repository.
