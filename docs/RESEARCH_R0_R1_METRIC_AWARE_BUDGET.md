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
  and a separate cache.

  **That index has since been built** — see §12. R0 made the scope expressible; the
  build made it real.

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

R1 is gated behind `adaptive_budget.enabled: false`. With it off, every gated path is
skipped — not given a neutral parameter — and the system reproduces B0_CLEAN exactly.
Two tests assert that the diagnostics carry `{"enabled": false}` and nothing else.

### 7.1 Rank-cutoff utility (`aic2026/rank_utility.py`)

`rank_cutoff_utility(r)` returns the fraction of the Final Score a query can still earn
if its first correct row lands at rank `r`. It reads the cutoffs from the scorer's own
`TOP_KS`, so it cannot drift from the metric, and a test grounds it against
`final_score_from_r_scores` for real rank positions.

Its most useful property is what it is *not*: a smooth decay. `marginal_utility(20 → 6)`
is exactly 0 — promoting a row inside a bucket crosses no cutoff and changes no score —
while `marginal_utility(21 → 20)` is 0.2. An allocator using a smooth rank prior would
spend compute on moves worth nothing.

It is an allocation prior. It never claims a row is correct.

### 7.2 Uncertainty signals (`aic2026/budget.py`)

Four training-free views, each in [0, 1], each logged:

| Signal | 0 means | 1 means |
|---|---|---|
| `score_margin` | the top candidate is a runaway | a photo finish |
| `channel_disagreement` | active channels return the same head | heads are disjoint |
| `support_concentration` | the head is one video | every row a different video |
| `temporal_ambiguity` | one tight temporal cluster | every candidate its own region |

The reported `uncertainty` is the mean of the **enabled** components; a disabled one is
excluded rather than counted as zero, which would quietly make every query look settled.
None of them is a probability, none is calibrated, and the payload says so.

For TRAKE, `EventUncertainty` combines coverage scarcity, score margin, feasible
candidate count and whether the event needed expansion. One deliberate asymmetry: an
event with a **single** candidate contributes the maximum margin term, not zero. For a
KIS head one candidate means nothing is competing; for a TRAKE event it means there is no
alternative at all — the thinnest possible support, and exactly where a budget belongs.
(The first implementation got this backwards and a test caught it.)

### 7.3 Budget actions and the ledger

Five named actions — `DEEPEN_CHANNEL`, `OFFICIAL_GRID_REFINE`, `SPARSE_VIDEO_SAMPLE`,
`DENSE_TEMPORAL_ZOOM`, `QA_VLM_CALL` — each with an order-of-magnitude unit cost in the
same units as `QueryCost.cost_proxy` (reading an indexed vector 0.05, decoding a frame
4.0, a VLM call 200.0). Priority is transparent:

```text
priority = rank_cutoff_utility(rank) × uncertainty × expected_gain_proxy / max(cost, ε)
```

`expected_gain_proxy` is named a proxy in the code, in the payload and here. It is not
expected accuracy and will not be until it is calibrated against held-out labels.

`BudgetLedger.try_spend` is the only way to buy anything. It refuses rather than
overshooting, and a refusal is recorded with its reason, so "the controller wanted more
budget" stays visible instead of looking like a decision not to act. Allocation is greedy
in priority order — a knapsack solver would buy precision the cost weights do not have.

### 7.4 Stage 1: official-grid refinement (`aic2026/official_grid.py`)

Before any MP4 is opened, the controller rescores a candidate's **official mapped
neighbours** using vectors already inside the index: `index.neighbor_rows` →
`index.vectors_for_rows` → one dot product each. No decode, no JPEG read, no image
encoder. It produces a local score curve with a best neighbour, a peak margin, a temporal
stability and a slope.

Two reasons this is the right first stage:

* it costs ~0.05 cost-units per vector against ~4.0 for a decoded frame;
* every point it can surface carries an official `frame_idx`, so anything it suggests is
  submission-safe by construction. An arbitrary decoded frame is not.

A boundary bug is worth recording: `neighbor_rows` originally returned bare rows, and the
refiner zipped them against the requested offsets. At the first or last frame of a video
some offsets do not exist, so every surviving neighbour was mislabelled. The API now
returns `(offset, row)` pairs and a test pins the boundary case.

### 7.5 Stage 2/3: progressive video refinement (`aic2026/progressive_refinement.py`)

The same hard frame budget as the fixed sampler, spent in stages: a sparse sweep, then a
zoom on the strongest peak, then whatever remains near the unresolved peak. It stops as
soon as the peak is separated by `stop_margin`.

Invariants, all tested: the coarse frame is always in stage A; the budget is never
exceeded; no frame is scored twice; indices stay inside the real video using that video's
own fps; a decode or scorer failure falls back to coarse behaviour instead of raising.

**Verified on real MP4s** (competition data, refinement enabled, budget 32 frames,
stages 8/8/16): 3 stages entered, 26 frames scored, 6 saved, best index 1–2 frames from
the coarse frame, `applied_to_submission: false` throughout.

### 7.6 KIS cutoff-aware allocation (`ranking.cutoff_aware_top100`)

Ranks 1 and 2–5 stay dominated by relevance evidence: promoting a weaker-but-different
row into rank 1 trades the most valuable slot in the metric for variety. Ranks 6–20 keep
the baseline rule including neighbour expansion, because official ground truth is an
interval. Ranks 21–50 and 51–100 require each bucket to reach `tail_min_videos` distinct
videos before one video may take a second slot there — a near-duplicate at rank 60
consumes a legal slot worth 0.2 for no additional coverage.

Bucket survival is logged (`1`, `2-5`, `6-20`, `21-50`, `51-100`).

### 7.7 TRAKE weakest-event allocation

Finding a complete video hypothesis is untouched — the organizer awards zero for the
wrong video. What R1 changes is the *optional* budget afterwards:
`split_budget_by_uncertainty` divides the per-query frame cap in proportion to event
uncertainty, subject to a per-event ceiling, and the parts sum exactly to what the caps
allow. Both numbers are reported: `frame_budget_requested` and the achievable
`frame_budget_total`.

The default ceiling is half the per-query budget (48 of 96). A first attempt used 24,
which with three events binds on all of them and silently produces a uniform split — the
exact behaviour the stage exists to replace. The real smoke exposed that, not a test.

With the corrected ceiling, the three fixture queries allocate `37/22/37`, `32/32/32` and
`37/37/22` frames: the structurally weaker events get more, and a query whose events are
equally uncertain still splits evenly, which is the right answer for that query.

All Phase 7/8 invariants survive: N events → N frames, ordering intact, one video per
sequence, and enabling the budget does not change the returned sequences.

### 7.8 Q&A compute escalation

`qa.max_vlm_calls_per_query` and `qa.max_visual_frames_per_call` are hard caps. A
hypothesis reached after the budget is spent gets status `budget_exhausted`, which is
non-submittable — a spending limit is never a reason to guess. Only a backend that really
looks at images spends budget; the non-visual mock costs nothing and is recorded as zero
VLM calls, which is what the cost trace shows today.

No VLM is downloaded. The existing `LocalVlmQAAnswerer` contract is unchanged and would
still require an explicit local path with `local_files_only=true`.

### 7.9 Channel budget (off even within R1)

`channel_policy` maps each usable channel to `full` / `shallow` / `skip` from signals the
query representation already carries. CLIP is never reduced. It is disabled by default
inside R1 because its whole purpose is to ask whether equivalent quality survives less
work — a question that needs labels. Every decision is logged.

### 7.10 What R1 measurably does today

Real 12-query smoke, `B0_CLEAN` vs `ADAPTIVE_BUDGET`, same data and cache:

* **rankings identical** on all 6 KIS, 3 TRAKE and 3 Q&A queries — the shipped stages are
  evidence-producing, not re-ranking;
* official-grid stage read 180 indexed vectors per query and decoded **0** frames;
* uncertainty ranged 0.60–0.72 across the fixture, so the signal is not constant;
* the ledger spent 136 of 400 cost units and reported the remainder;
* TRAKE event allocation differentiated (`37/22/37`, `32/32/32`, `37/37/22`) with every
  structural counter still 0 and the returned sequences unchanged;
* with refinement enabled, the progressive sampler entered 3 stages and used 26 of 32
  frames on real MP4s;
* Q&A VLM calls: 0, because no visual backend exists here.

Warm latency on that single run: KIS 179.5 → 142.2 ms, TRAKE 893.6 → 930.2 ms. Six and
three queries respectively is not a latency measurement, and the direction is not claimed
in either case — the official-grid stage does real extra work, so a TRAKE query getting
slower is expected, not a regression to explain away.

**Mechanism verified. Quality unmeasured, and not claimed.**

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

## 11. Full retrieval-ready collection

R0 made the global scope *expressible*. This section records making it *real*.

> **Full retrieval coverage does NOT mean full visual refinement coverage.** They are
> separate capabilities and the system keeps them separate. Indexing a video whose MP4 is
> absent adds it to search; it adds nothing to preview, local refinement or visual Q&A,
> and every one of those reports itself unavailable rather than pretending.

### 11.1 What the collection actually contains

Read-only inventory of the current data root — measured, not assumed:

| Quantity | Value |
|---|---|
| Discovered video ids | 873 |
| Retrieval-ready (valid map + valid CLIP) | **873** |
| Map rows | **177,321** |
| CLIP vectors | **177,321** |
| Map/vector mismatches | **0** |
| Feature dimension / dtype | 512 / float16 |
| Object coverage | 873 / 873 |
| Metadata coverage | 873 / 873 |
| Invalid videos | **0** |
| Visual-accessible (JPEG or MP4) | 29 |
| Refinement-ready (MP4) | 29 |
| Q&A-visual-ready | 29 |
| CLIP features on disk | 173.3 MB |
| Free disk at build time | 142.8 GB of 456.2 GB |

844 videos have complete supporting data and no local MP4. Nothing about that prevents
coarse retrieval, which scores the CLIP vectors the organizers supplied.

### 11.2 A note on the inspection pass

A deep `inspect-data` run over the full scope was started first and stopped after 40
minutes: it had used 116 s of CPU and was I/O-bound reading 177k object JSONs, duplicating
validation the index build performs anyway. The numbers above come from a fast read-only
pass; the **build's own validation gate is the authoritative full check**. Nothing was
weakened to make the build pass — the build refuses on invalid data, and a validation
error would have stopped this work rather than been worked around.

### 11.3 The build

`build-index --load-objects --include-media-text`, no `--allow-stale-cache`, against
`configs/competition_full_retrieval.yaml`.

| | |
|---|---|
| Cache path | `artifacts/aic2026_index_retrieval_ready` |
| Scope mode | `retrieval_ready` |
| Selected videos | **873** |
| Indexed records | **177,321** |
| Feature dim / dtype | 512 / float16 |
| Cache fingerprint | `d09cbd66426dc9e22fc3596f729dada9df29b605320b56c9b7c95391ac24d15e` |
| Selected-video-IDs hash | `12011cea2c134f09990fbda71a3edb874c58705aa4ba258b2431315e0af334de` |
| Cache schema / record schema | 1 / 3 |
| Size | 1,040.5 MB |
| Build duration | 113.6 minutes |
| `valid` / `stale` / `legacy` / `corrupt` | **true / false / false / false**, 0 hard and 0 soft mismatches |

The three pre-existing caches were not touched.

Channel records on the full index, beside the 29-video one:

| Channel | 873-video | 29-video |
|---|---|---|
| clip | 177,321 | 7,800 |
| bm25 | 177,321 | 7,800 |
| objects | 169,909 | 7,663 |
| metadata | 873 | 29 |
| ocr / asr / caption | 0, disabled | 0, disabled |

OCR, ASR and captions stay disabled with empty sources: still reported, still not a
readiness warning, because the operator switched them off deliberately.

### 11.4 Structural smoke: 29 videos vs 873

Same fixed fixture, same code, two indexes. **No semantic comparison is possible or
attempted** — the searchable collection changed, so results changed; that is a fact, not
an improvement.

| Measure | 29-video | 873-video |
|---|---|---|
| Indexed videos / records | 29 / 7,800 | 873 / 177,321 |
| Mean candidate union per KIS query | 1,897.8 | 2,675.8 |
| Mean distinct videos per KIS query | 25.0 | 51.5 |
| KIS result rows from videos with no MP4 | 0 | **245** |
| Warm KIS mean latency | 166.9 ms | 1,606.5 ms |
| Engine startup | 914 ms | 23,109 ms |
| RSS after the fixture | 753.7 MB | 5,353.7 MB |
| Query-embedding cache | 18 entries, 7 hits / 18 misses | 19 entries, 12 hits / 19 misses |
| Distinct cache fingerprints | — | **yes**: separate identities |

The cost of 22.7× more records is roughly 10× warm latency and 7× resident memory on this
CPU-only machine. That is a resource fact to plan around, not a quality statement.

### 11.5 A video with no pixels behaves correctly

Concrete examples returned by the fixture on the full index: **`L30_V053`** (submitted
frame 7087), `L30_V091` (2149), `L28_V008` (19918) — all retrieved, none refined.

Direct probe of `L22_V001` (`L22_V001/kf_000001`, official frame 0):

| Capability | Result |
|---|---|
| Coarse KIS retrieval | reached, official mapped `frame_idx` preserved |
| Frame / preview request | `available: false`, no bytes, **no exception** |
| Local refinement | ran, 1 candidate, **0 applied, 0 frames decoded**; candidate not dropped |
| Submission eligibility | structurally valid on the official mapped frame |

TRAKE on the full index returned 30 / 13 / 30 complete sequences, of which 8 / 4 / 6 of
the represented videos have no MP4. Every structural counter stayed **0**:
`malformed_prediction_count`, `wrong_event_count_prediction_count`,
`cross_video_step_count`, and no unordered submission sequence.

Q&A, probed with a wider hypothesis count so pixel-less videos were included: 30
hypotheses, **19 without pixels**, every one of them with `evidence_with_image: 0`,
`backend_visual: false`, and the warning *"Answered by the non-visual mock backend from
caption/OCR/ASR text; this is not visual Q&A."* **0 VLM calls.** The mock echoed media
text, and the submission validator refused the whole batch with `QA_ANSWER_TOO_LONG`.
Nothing was fabricated and nothing was exportable.

Artifacts: `artifacts/full_retrieval_smoke/{summary.json, comparison_29_vs_full.json}`
(gitignored). Neither contains an accuracy, recall, precision or Final Score figure,
because no ground truth exists.

## 12. Open questions for the organizers

1. What are the frame-ID semantics of an arbitrary decoded frame that is not in
   `map-keyframes`? Until answered, `frame_output_policy` stays `preserve_coarse` and a
   decoded frame remains evidence only.
2. Is a submission scored per query independently, with the Final Score averaged over
   queries?
3. For TRAKE, is a row with the right video but one wrong event scored `hits/N`, as
   assumed here?
4. Are duplicate `(video_id, frame_id)` rows rejected, ignored, or counted against the
   100-row budget?
