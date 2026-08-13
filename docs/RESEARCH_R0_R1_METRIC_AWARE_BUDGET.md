# Metric-Aware Budgeted Retrieval (R0 / R1)

**Status: CURRENT.** Branch `research/aic2026-metric-budget`, baselined on the frozen
release `0.11.0-aic2026` (commit `7dfe06e`, tag `aic2026-competition-ready`).

> **No AIC ground truth exists in this repository.** Nothing in this document reports
> accuracy, recall or Final Score for this system, and no parameter was tuned by looking
> at retrieval output. Structural counts are not recall; candidate coverage is not
> accuracy; more TRAKE sequences is not better.

---

## 1. Problem

The organizer evaluates each query at five cutoffs — R@1, R@5, R@20, R@50, R@100 — and
the Final Score is their mean. Two things follow that the system did not previously act
on.

**The metric is not flat.** Moving a correct row from rank 21 to rank 5 is worth as much
Final Score as moving it from rank 6 to rank 1. Ranks 51–100 are worth a fifth of rank 1
and are still worth something, so a near-duplicate row occupying rank 60 has a real
opportunity cost.

**Compute is not free and is not uniform.** On this CPU-only machine a coarse retrieval
costs ~130 ms, a full local refinement pass ~14 s, and a VLM call would cost more than
either. The release spends the *same* compute on a query whose top candidate is an
obvious runaway winner as on one where the top twenty candidates are separated by noise.

## 2. Official metric geometry

For KIS, where a row is either right or wrong, the rank of the first correct row
determines which cutoffs it can still satisfy:

| First correct rank | Cutoffs satisfied | Maximum Final-Score contribution |
|---|---|---|
| 1 | 1, 5, 20, 50, 100 | 1.0 |
| 2–5 | 5, 20, 50, 100 | 0.8 |
| 6–20 | 20, 50, 100 | 0.6 |
| 21–50 | 50, 100 | 0.4 |
| 51–100 | 100 | 0.2 |
| >100 | none | 0.0 |

This is arithmetic about the metric, not a prediction about the system. R1 implements it
as `rank_cutoff_utility` and treats it as an *allocation prior*: it says where an
improvement would be worth the most if it happened, never that one will.

For Q&A and TRAKE the same geometry applies to the rank axis, combined with each task's
own R-score semantics — a Q&A row needs the right frame *and* the right answer, a TRAKE
row earns `hits / N` across its events. Without ground truth those remain priors too.

## 3. The fixed-compute bottleneck in the release

| Stage | Release behaviour | Consequence |
|---|---|---|
| Query encoding | Re-encoded per retrieval call | A 3-event TRAKE query encoded 20–24 prompt variants for 3 distinct texts |
| Channel depth | Fixed per channel, scaled only by TRAKE expansion | An easy query pays the same as an ambiguous one |
| KIS refinement | All-or-nothing, fixed 5×32 frame plan | Off by default because the worst case is unaffordable, so it is never used at all |
| TRAKE refinement | Budget split across events by position | The event holding a sequence together gets no more attention than a settled one |
| Q&A | One backend call per video hypothesis | Unbounded by design; 100 hypotheses would be 100 calls |
| Top-100 allocation | One diversity rule for ranks 1–100 | Rank 2 and rank 90 are treated the same although their metric value differs 4× |

## 4. Related work

Adaptive frame sampling, temporal zoom, uncertainty-driven routing and budgeted top-k
are **existing ideas**. This project claims none of them.

| Area | Representative work |
|---|---|
| Text–video retrieval, uncertainty-aware | UATVR (arXiv:2301.06309) |
| CLIP reranking | CLIPRerank (arXiv:2401.08449) |
| Hierarchical/adaptive video structure | VideoTree (arXiv:2405.19209), MDP3 (arXiv:2501.02885) |
| Adaptive keyframe selection | Adaptive Keyframe Sampling (arXiv:2502.21271), Q-Frame (arXiv:2506.22139), EcoFrame (arXiv:2608.03918) |
| Long-video temporal search | BOLT (arXiv:2503.21483), T\* (arXiv:2504.02259), Adaptive Bidirectional Temporal Search (arXiv:2504.09298) |
| Budgeted / cascaded ranking | AcuRank (arXiv:2505.18512), Top-k on a Budget (arXiv:2601.20989), U-CESE (arXiv:2605.23274) |
| Competition systems | Fusionista2.0 (arXiv:2511.12255) |
| Small VLMs for evidence | SmolVLM (arXiv:2504.05299), Qwen2.5-VL (arXiv:2502.13923) |

## 5. Novelty hypothesis — stated as a hypothesis

> **Multi-cutoff metric-aware heterogeneous compute allocation for video retrieval.**

A training-free controller allocates *heterogeneous* retrieval and visual computation —
channel depth, official-grid rescoring, MP4 decoding, VLM calls — according to (1)
ranking uncertainty, (2) task-specific structural uncertainty, (3) the utility of the
official multi-cutoff rank region, and (4) estimated cost.

What is plausibly new is the combination: existing budgeted-retrieval work optimises a
single cutoff or a single modality of compute; existing adaptive-sampling work is
uncertainty-driven but metric-agnostic. Whether that combination *helps* is unknown and
cannot be known here. This is a hypothesis to test when labels exist, not a claim.

## 6. R0 — engineering changes (complete)

R0 changes no ranking. Verified below.

### 6.1 Removed things that were not real

| Removed | Evidence it was dead |
|---|---|
| KIS `Rerank` UI control | Sent `rerank` to an endpoint that never read it; `AICCompetitionEngine.search()` ignored the flag and delegated to the plain candidate search |
| `AICCompetitionEngine.search()` | No caller in the repository; the legacy `retrieval.video_engine.VideoSearchEngine.search()` is a different method that does implement reranking |
| `ranking.diversity_lambda`, `ranking.recall_tail_size` | Defined, validated, present in both YAMLs, read by nothing — diversity is `min_frame_gap` + `max_frames_per_video`, and the recall tail has no size of its own |
| `trake.alignments_per_video`, `trake.sequence_overlap_threshold` | Superseded by `k_best_per_video` / `max_alignments_per_video` and by `sequence_diversity.*` in Phase 8; never read after that |
| `evaluation.save_predictions`, `evaluation.save_errors` | `BenchmarkLogger.write_run` always writes both files |

Each removed key is now **rejected with an explanation** rather than silently ignored,
because `_construct` drops unknown keys and a dropped knob in a config that looks tuned
is worse than an error.

### 6.2 Told the truth about capability

- **Source-empty channels.** OCR, ASR and frame captions are disabled in
  `configs/competition.yaml`. They are still constructed and still report
  `available: false, no_populated_source_data`; readiness classifies a deliberately
  disabled empty source as a new `INFO` status that never affects the verdict. A warning
  nobody can act on teaches readers to ignore warnings.
- **Display count vs competition pool.** The KIS UI defaulted to Top-K 20 and that number
  sized the *engine* call, so a 20-row view exported a 20-row submission and discarded 80
  legal ranks. Pool and display are now separate: the pool is `ranking.final_top_k`, the
  batch keeps every row, and `display_limit` only controls rendering. Same for TRAKE.
- **Retrieval coverage vs visual coverage.** Scope mode `retrieval_ready` (valid map +
  valid CLIP feature) is what global coarse retrieval needs. `existing_videos`
  additionally requires an MP4 and is a *visual* development scope. On the current data
  root the difference is stark:

| Capability | Videos | Requirement |
|---|---|---|
| Retrieval-ready | **873** | map + CLIP |
| Visual-accessible | 29 | keyframe JPEG or MP4 |
| Refinement-ready | 29 | MP4 |
| Q&A-visual-ready | 29 | keyframe JPEG or MP4 |

  844 videos have complete supporting data (map, CLIP, objects, media-info) and no local
  MP4. The release scope discarded all of them from *retrieval*, for a reason retrieval
  does not care about. `configs/competition_full_retrieval.yaml` uses the global scope
  and a separate cache; building that index is a deliberate step and nothing does it
  automatically.

### 6.3 Zero-risk speed and cost work

- **Bounded query-embedding cache** (`aic2026/query_cache.py`). Keyed on query, model
  name, feature dimension and template-set signature; LRU with a hard capacity; never
  persisted; vectors handed out read-only so a caller cannot corrupt a later hit.
- **Per-request execution context.** One TRAKE request reuses its query representation
  and any *identical* `(text, depth)` retrieval. It deliberately does **not** slice a
  deeper result to answer a shallower request: channel scores are rank-normalized over
  the pool each channel returned, so a top-40 slice of a depth-300 retrieval is not a
  depth-40 retrieval. Reuse is allowed only where the result is provably identical.
- **Optional prewarm.** `runtime.prewarm_enabled` / `serve --prewarm` moves the one-off
  text-encoder load out of the first query. It cannot change a result, is off by default
  so tests never pay for it, and reports failure instead of raising.

### 6.4 Measurement before algorithm

- `aic2026/cost.py` — `QueryCost` / `StageCost`: encoder calls and vectors, per-channel
  searches and candidates, frames requested/decoded, image embeddings, VLM calls and
  images, stage timings, wall time, RSS. GPU metrics report *unavailable* rather than 0.
  `cost_proxy()` exists for within-machine action ranking and is documented as not a
  score.
- `evaluation/ground_truth.py` — a private-development GT schema. Every file must declare
  `label_source: official | private_dev`; a file annotated by `system`/`model`/`auto` is
  refused as circular; reports carry `PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE`.
  With no file at all, semantic evaluation still raises `GROUND_TRUTH_REQUIRED`.
- `evaluation/pareto.py` — three-axis reporting. Quality columns exist only when GT was
  supplied; otherwise the report carries efficiency and cost and says why. Dominance is
  strict Pareto dominance and no single "best" configuration is declared.

### 6.5 R0 results — measured, on real data

**Semantic equivalence.** The 12-query fixture was run in a git worktree at the release
tag and in the R0 tree, against the same data root and the same cache:

| Task | Queries | Identical rows and scores |
|---|---|---|
| KIS | 6 | 6 / 6 |
| Q&A | 3 | 3 / 3 |
| TRAKE | 3 | 3 / 3 |

Cache fingerprint identical. **R0 changed no ranking.**

**Work removed** (same fixture, same outputs):

| Measurement | B0_RELEASE | B0_CLEAN |
|---|---|---|
| Encoder invocations, TRAKE query 1 (3 events) | 20 | **12** |
| Encoder invocations, TRAKE query 2 | 24 | **12** |
| Encoder invocations, TRAKE query 3 | 20 | **12** |
| Encoder invocations, repeated KIS query | 4 | **0** |

12 is the floor: 3 distinct event texts × 4 prompt templates, encoded once each. The
release re-encoded the same text at every expansion depth.

Latency moved in the right direction on a single run (warm KIS mean 163.9 → 158.5 ms,
TRAKE mean 1021 → 973 ms), but one run of six queries is not a latency result and is not
claimed as one. The encoder-call counts are exact and are the honest measurement.

## 7. R1 — method (experimental, disabled by default)

R1 is gated behind `adaptive_budget.enabled: false`. With it off, the system reproduces
B0_CLEAN exactly.

*(Populated by the R1 commit: uncertainty signals, budget actions, official-grid
refinement, progressive video refinement, cutoff-aware allocation, weakest-event
allocation, channel budget.)*

## 8. Evaluation protocol

Four named configurations:

| Name | Meaning |
|---|---|
| `B0_RELEASE` | Exact `0.11.0-aic2026` behaviour |
| `B0_CLEAN` | R0: dead UI/config removed, caching and prewarm allowed, ranking equivalent |
| `FIXED_REFINEMENT` | Existing fixed frame-budget refinement |
| `ADAPTIVE_BUDGET` | R1 controller |

Rules that make a comparison mean anything:

1. **Matched budgets.** Fixed vs adaptive refinement is compared under the same hard
   maximum decoded frames, image embeddings and VLM calls; *actual* consumption is also
   reported. An adaptive method that simply spends more is not an efficiency result.
2. **Same index.** Variants whose cache fingerprints differ are reported as not
   comparable rather than compared.
3. **Separate axes.** Final Score, latency, decoded frames, VLM calls and memory are
   reported as raw quantities. The controller may use a cost proxy internally; the
   evaluation never collapses the axes.
4. **Statistics.** With enough labelled queries: paired per-query comparison, bootstrap
   confidence intervals, retained error cases. No significance claim without a test.

## 9. No-ground-truth status

Today the only reportable quantities are: candidate structure, latency, compute counts,
memory, determinism, failure counts, and human-inspection artifacts.

Not reportable, and not reported: accuracy, recall, precision, Final Score, "better
retrieval", "improved correctness", SOTA.

A ranking that changes between B0_CLEAN and ADAPTIVE_BUDGET is neither success nor
failure. Without labels it is only a difference.

## 10. Threats to validity

- **Small scope.** Everything is measured on 29 video-backed videos / 7,800 frames. The
  873-video retrieval-ready index has not been built.
- **One machine.** CPU-only, one hardware profile. Latency ratios will not transfer.
- **No visual Q&A backend.** The Q&A compute path is exercised structurally; its
  expensive stage has never actually run.
- **Cost proxy weights are guesses.** They are order-of-magnitude placeholders for
  within-machine action ranking, never calibrated against outcomes.
- **Uncertainty signals are unvalidated.** They are transparent functions of existing
  scores. Whether they correlate with correctness is exactly what cannot be checked here.
- **Fixture bias.** The smoke fixture is fixed and small; it proves the pipeline runs, not
  that behaviour generalises.

## 11. Open questions for the organizers

1. What are the frame-ID semantics of an arbitrary decoded frame that is not in
   `map-keyframes`? Until answered, `frame_output_policy` stays `preserve_coarse` and a
   decoded frame remains evidence only.
2. Is a submission scored per query independently, with the Final Score averaged over
   queries?
3. For TRAKE, is a row with the right video but one wrong event scored `hits/N`, as
   assumed here?
4. Are duplicate `(video_id, frame_id)` rows rejected, ignored, or counted against the
   100-row budget?
