# Phase 5 Query-Conditioned Local Refinement

Phase 5 is the first phase whose purpose is retrieval **quality** rather than
infrastructure correctness. It makes `LocalFrameRefiner` a real, end-to-end part of
Textual KIS: bounded local sampling of the original MP4s, a real query-conditioned
visual scorer, and a defined reranking rule.

> **No accuracy claim.** This repository contains no AIC ground truth. Nothing in this
> phase was tuned against retrieval accuracy, no threshold was chosen by measuring
> quality, and none of the diagnostics below is a precision, recall, or accuracy
> figure. They record what the system *did*, not whether it was right.

Phase 5 does **not** implement Q&A per-video hypotheses, TRAKE k-best, TRAKE semantic
refinement, independent object/OCR/ASR retrieval channels, Vietnamese query
normalization, a submission validator, a manual-edit redesign, or paper experiments.

## 1. Why A Local Layer At All

BTC keyframes are sparse in time. The official CLIP features index only those mapped
keyframes, so the frame that actually shows the queried moment is frequently *between*
two indexed keyframes and can never be scored by the global index. Local refinement
looks at that gap — for a handful of candidates only.

```text
query
  -> BTC precomputed CLIP index      (global recall; unchanged)
  -> fusion + coarse ranking
  -> candidate regions (deduplicated in time, budgeted)
  -> trigger decision (always | uncertainty | disabled)
  -> bounded dense sampling of the ORIGINAL MP4
  -> batched visual scoring against the query
  -> rerank
  -> video-aware Top-100
```

The global layer is untouched: no video is re-embedded, no index is replaced, and a
query with refinement disabled follows exactly the pre-Phase-5 path.

## 2. What Was There Before

| | Before Phase 5 |
|---|---|
| Called by the engine | **No** — only by its own unit test |
| Query conditioning | none; the scorer signature took frames only |
| Scorer | none; a callable had to be injected |
| Trigger | `uncertainty_only` existed in config and was never read |
| Budget / dedup | `top_hypotheses`, `batch_size` never read |
| Video access | its own `cv2.VideoCapture`, unrelated to `FrameProvider` |
| Frame identity | `selected_frame_id` conflated a decoded index with a frame ID |
| Diagnostics | none |
| `search_trake(refine_window_s=...)` | accepted, unused |

## 3. Components

`aic2026/local_refinement.py`

| Concept | Role |
|---|---|
| `RefinementConfig` | all settings, frozen, validated in `aic2026/config.py` |
| `RefinementCandidate` | one coarse candidate offered for refinement |
| `LocalRefinementRequest` | a query plus its candidates |
| `CandidateRegion` | deduplicated region: an anchor plus merged members |
| `RefinementDecision` | mode, triggered, reason, margin, threshold, counts |
| `LocalRefinementFrame` | one sampled frame: index, timestamp, score, is-coarse |
| `CandidateRefinement` | full provenance for one candidate (§7) |
| `LocalRefinementResult` | decision + refinements + diagnostics + warnings |
| `LocalFrameRefiner` | the algorithm; knows nothing about CLIP |

`LocalFrameRefiner` takes a `FrameProvider` and a `FrameScorer` and nothing else, which
is what lets the whole algorithm be driven offline by a fake scorer and 31-frame
synthetic MP4s.

## 4. FrameScorer

```python
class FrameScorer(Protocol):
    def prepare_query(self, query: str) -> Any: ...
    def score_frames(self, prepared_query, frames: Sequence[np.ndarray]) -> Sequence[float]: ...
    def status(self, *, initialize: bool = False) -> ScorerStatus: ...
```

`frames` are BGR uint8 arrays as OpenCV decodes them. Scores are plain floats, higher
is better, and `validate_scores` rejects a non-finite value at the boundary rather than
letting it into ranking.

**`CLIPFrameScorer`** (`aic2026/frame_scorer.py`) is the only production backend:
query text embedding, batched image embedding, L2 normalization on both sides, cosine
similarity as a normalized dot product, `model.eval()`, `torch.inference_mode()`, no
gradients, configurable device with CPU and CUDA support, and lazy loading throughout.
`build_frame_scorer` accepts `"clip"` and nothing else — there is no production fake.

## 5. Model Ownership: One Shared Checkpoint

`transformers.CLIPModel` contains **both** towers. Before Phase 5 only the text tower
was used; the image tower is needed now, and it must be the *same* checkpoint or the
two embedding spaces are not comparable.

`aic2026/clip_backend.py` owns the checkpoint. `get_clip_backend(model_name, device)`
returns one `CLIPBackend` per `(model_name, device)`, and both `CLIPTextEncoder` and
`CLIPFrameScorer` obtain their model from it. So:

- one `openai/clip-vit-base-patch32` in memory, not two (~600 MB saved);
- the text tower and the image tower are provably in one space;
- `CLIPTextEncoder`'s public behaviour is unchanged: same model, same offline-first
  load policy, same projection-dimension check, same `.model` / `.processor` /
  `.device` attributes;
- the refiner still does not depend on the text encoder, and the text encoder does not
  depend on the refiner. They share a *backend*, not each other.

Verified on this machine: the cosine similarities computed from the two towers match
CLIP's own `logits_per_image / logit_scale` to 9e-8.

`transformers` 5 returns `get_*_features` wrapped in `BaseModelOutputWithPooling`;
`_feature_tensor` accepts both that and the plain tensor of 4.x.

## 6. Configuration

```yaml
refinement:
  enabled: true
  mode: uncertainty          # always | uncertainty | disabled
  top_hypotheses: 10         # coarse candidates examined when forming regions
  candidate_budget: 5        # regions actually decoded and scored
  region_merge_s: 1.0
  window_before_s: 4.0
  window_after_s: 4.0
  fine_fps: 4.0              # sample rate inside the window
  max_frames: 32             # hard cap per candidate
  batch_size: 16
  cache_size_mb: 256
  scorer_input_max_side: 336
  rerank_alpha: 0.10
  frame_output_policy: preserve_coarse
  trigger:
    margin_threshold: 0.03
  scorer:
    type: clip
    model_name: openai/clip-vit-base-patch32
    device: auto
    required: false
```

`scorer:` and `trigger:` are flattened into the frozen dataclass at load time rather
than mirrored as a second settings structure. A pre-Phase-5 `uncertainty_only: true` is
translated to `mode: uncertainty` instead of being silently dropped.

Validation rejects: an unknown mode or frame-output policy, a non-`clip` scorer type,
a non-finite or negative `margin_threshold` or `rerank_alpha`, a zero or negative
budget / window / sample rate / frame cap / batch size, a `top_hypotheses` smaller than
`candidate_budget`, and a device that is not `auto`/`cpu`/`cuda`/`cuda:N`.

Pre-Phase-5 defaults (`window_before_s`, `window_after_s`, `fine_fps`, `max_frames`,
`batch_size`, `margin_threshold`, `cache_size_mb`) are preserved exactly.

## 7. Frame-ID Safety

Three identities are kept separate and all three are reported:

| Field | Meaning |
|---|---|
| `coarse_official_frame_idx` | the BTC mapped `frame_idx` of the coarse candidate |
| `best_visual_frame_idx` | the decoded frame the scorer preferred |
| `submission_frame_idx` | what goes into the AIC submission row |

Under the default `frame_output_policy = "preserve_coarse"`,
`submission_frame_idx == coarse_official_frame_idx` **always**, even when the refined
frame scored higher. The refined frame is evidence, visualization, a local score, and a
reranking contribution — never a change to the submitted row.

AIC has not confirmed the frame-ID semantics of an arbitrary decoded frame. The
alternative `frame_output_policy = "decoded_frame"` is implemented and tested so it can
be switched on later, but it is not the default and nothing selects it implicitly.

## 8. Bounded Sampling

`build_sample_plan(coarse_frame_idx, fps, frame_count, config)`:

- `step = round(fps / fine_fps)` — computed from **this** video's fps, because AIC
  videos do not share one frame rate;
- window `[coarse - window_before_s*fps, coarse + window_after_s*fps]`, clamped to
  `[0, frame_count-1]`;
- the coarse frame is always included, and sampling expands symmetrically outward, so
  the budget is spent nearest the coarse hit first;
- no duplicates, `max_frames` is a hard cap, output is sorted ascending;
- deterministic: same inputs, same plan.

Nothing outside the window is ever read, and no video is decoded in full.

## 9. One Shared Video-Access Implementation

`FrameProvider` gained three methods and keeps its keyframe-record API exactly as it
was:

| Method | Purpose |
|---|---|
| `video_metadata(video_id)` | fps / frame count / duration, cached per file identity |
| `decode_frames(video_id, indices)` | many frames from **one** capture |
| `get_video_frame(video_id, frame_idx=, timestamp=)` | one arbitrary frame as JPEG |

`decode_frames` opens the container once and reads forward between nearby targets,
re-seeking only when the gap exceeds `SEQUENTIAL_READ_LIMIT`. That is what makes a
32-frame window affordable on a 100 MB MP4; one open-and-seek per frame would not be.
It never raises for a missing or unreadable video — it returns what it got plus a
warning, because a refinement failure must not fail retrieval.

The refiner no longer has its own OpenCV code. Derived images still go only to the
configured artifact cache; `data/` is never written.

## 10. Batching

- the query is embedded **once** per refinement request;
- every sampled frame of every selected candidate is gathered and passed to
  `score_frames` in a **single** call, which then batches internally by `batch_size`;
- frames are downscaled to `scorer_input_max_side` (336) immediately after decode, since
  CLIP works at 224 px — roughly a 30x memory saving with no change to the embedding;
- `decode_ms`, `inference_ms`, and `total_ms` are reported separately.

## 11. Budget And Region Deduplication

`top_hypotheses` coarse candidates are examined, merged into regions, and at most
`candidate_budget` regions are decoded. Two candidates of one video within
`region_merge_s` are one region: the highest-scoring one anchors it and the others are
recorded in `merged_keyframe_ids` rather than discarded, so no candidate disappears
from the response. Selection is deterministic — sorted by
`(-coarse_score, video_id, timestamp, keyframe_id)`.

## 12. Uncertainty Trigger

Deliberately simple, deterministic, documented, and logged. It compares the two best
candidate **regions**:

```text
margin           = score(region_1) - score(region_2)
relative_margin  = margin / max(|score(region_1)|, 1e-9)
triggered        = relative_margin <= margin_threshold
```

Fused scores are min-max normalized weighted sums, not calibrated probabilities and not
bounded to `[0, 1]`, which is exactly why the raw gap is normalized before it meets the
threshold. With fewer than two regions there is no evidence of separation at all, so
the top hit is treated as unconfirmed (`single_candidate_region`) rather than as
confidently correct.

The decision is always reported:

```json
{"mode": "uncertainty", "triggered": true,
 "reason": "top_score_margin_below_threshold",
 "margin": 0.0105, "relative_margin": 0.0105, "threshold": 0.03,
 "candidates_considered": 10, "regions_found": 8, "regions_selected": 5}
```

The 0.03 threshold is the pre-existing configured default. It was **not** tuned.

## 13. Reranking Formula

```text
refined_score = coarse_fusion_score + rerank_alpha * clamp(best_visual - coarse_visual, -1, 1)
```

The contribution is the *improvement over the coarse frame's own visual score*, not the
raw local similarity. Two reasons:

1. **No double counting.** The coarse CLIP vector is already inside the fused score;
   adding the raw local score back would count the same evidence twice.
2. **No scale mismatch.** Fused scores and cosine similarities live on unrelated
   scales; a difference of cosines does not, and it is 0 exactly when the coarse frame
   is already the best.

Every component survives: `score_breakdown` keeps `fused` (coarse) and gains
`coarse_fused`, `visual_gain`, and `refined`, and the refinement payload carries
`coarse_score`, `coarse_visual_score`, `best_visual_score`, `score_gain`, and
`refined_score`. Candidates that were not refined keep their coarse score untouched and
are never dropped. Ties break on `(-score, video_id, frame_id, keyframe_id)`.

Refinement runs **before** the Top-100 allocation, so local evidence can actually
rerank; refining after truncation would only relabel a fixed list.

*Known asymmetry, stated rather than hidden:* the coarse frame is always in the sample
plan, so `score_gain >= 0` whenever it decoded. A refined candidate therefore never
loses ground to an unrefined one. Since the refined set is chosen as the top-scoring
regions, this can reinforce the existing order but not manufacture a new leader from
below the budget. `rerank_alpha = 0.10` bounds the effect.

## 14. Failure Behaviour

| Situation | Result |
|---|---|
| No MP4 for the video | `applied=false`, `reason=video_unavailable`, coarse kept |
| Container unreadable / no metadata | `applied=false`, `reason=video_metadata_unavailable` |
| Some frames fail to decode | the rest are scored; a warning is attached |
| No frame decodes | `applied=false`, coarse kept |
| Scorer cannot load | `applied=false`, `reason=scorer_unavailable`, warning on the result |
| Scorer raises or returns non-finite | `applied=false`, `reason=scorer_failed`, coarse kept |
| Candidate has no official `frame_idx` | skipped; `submission_frame_idx` stays `None` |
| `scorer_required: true` and no scorer | **raises** — production fails loudly |

One failing video never fails a KIS search. Genuine programming errors still raise.

## 15. Runtime State

The refiner and its scorer are owned by the engine, and the engine is a field of the
frozen `RuntimeDatasetState`, so the refiner belongs to exactly one generation.

`build_runtime_state` now **adopts** `engine.frame_provider` as the state's provider
instead of building a second one, and `verify_engine_identity()` gained a check that
`engine.local_refiner.frame_provider is state.frame_provider`. Mixing a refiner from
generation N with a provider from N+1 is therefore rejected at publication, not
discovered later through wrong pixels.

A request that began under generation N finishes under generation N; result URLs carry
the generation, and `/api/video/decoded_frame/...` returns `409 STALE_RESULT_GENERATION`
for a superseded one.

## 16. API And UI

- `POST /api/video/search` accepts `refine: false` to disable refinement for one query
  (the comparison switch), and returns a `refinement` block (enabled, mode, requested,
  frame output policy, decision, scorer status, warnings) plus `diagnostics`.
- Each result carries `refinement`, `refined_frame_id`, `refined_image`, and
  `submission_frame_id` next to the unchanged `frame_id`.
- `GET /api/video/decoded_frame/<video_id>/<frame_idx>` serves a refined visual frame
  through the same scope and traversal checks as the MP4 route, labelled
  `X-Frame-Role: refined_visual_frame`. It is **not** a submission frame.
- `/api/health` reports `refinement` — configuration plus scorer state — and never
  loads the checkpoint: the state is `not_loaded` until a query needs it.
- The frontend gained a "Local refinement" selector, a per-search status badge, and a
  labelled refined-frame thumbnail that always shows the submission frame beside it.
- No filesystem path is exposed anywhere.

## 17. Diagnostics Without Ground Truth

Per search: `refinement_triggered`, `trigger_reason`, `mode`, `candidates_considered`,
`candidates_selected`, `candidates_refined`, `frames_decoded`, `frames_scored`,
`decode_failures`, `scorer_failures`, `best_differs_from_coarse`,
`mean_visual_score_gain`, `mean_absolute_offset_seconds`, `decode_ms`, `inference_ms`,
`refinement_ms`, `coarse_search_ms`, `total_search_ms`.

Per candidate: `selected_offset_frames`, `selected_offset_seconds`,
`best_is_coarse_frame`, `score_gain`, window bounds, and per-stage timings.

`aggregate_diagnostics()` rolls several searches into `trigger_rate`,
`mean_candidates_refined`, `mean_frames_decoded`, `refinement_ms_p50/p95`,
`fraction_best_differs_from_coarse`, and `mean_absolute_offset_seconds`. A test asserts
that no key named precision/recall/map/accuracy can appear.

## 18. Real L21 Smoke

Real `data` root, scope `existing_videos`, existing cache
`artifacts/aic2026_index_existing_videos` — **reused, not rebuilt** (`cache_hit=true`,
`valid`, `not stale`, fingerprint `199cd0fd…`, 29 videos / 7,800 frames). Real scorer:
`clip` / `openai/clip-vit-base-patch32` / `cpu`, state `ready` (torch 2.13.0+cpu, CUDA
unavailable on this machine). Four exploratory queries, each run with refinement off,
with `mode=always`, and with the shipped `mode=uncertainty`.

| | |
|---|---|
| Searches (always) | 4, trigger rate 1.0 |
| Candidates refined | 20 (5 per query = the budget) |
| Frames decoded / scored | 631 / 631, mean 157.75 per query |
| Decode / scorer failures | 0 / 0 |
| Refinement latency | p50 14,442 ms, p95 14,652 ms (CPU) |
| Split | ~4.2-4.8 s decode, ~9.7-10.0 s CLIP inference |
| Coarse search | 80-90 ms |
| Best frame != coarse frame | 13 of 20 (0.65) |
| Mean absolute temporal offset | 0.909 s |
| Mean visual score gain | 0.0097 |

Per query (top row = `video_id, submission frame_id`):

| Query | Top (off) | Top (on) | Same top row | Uncertainty policy |
|---|---|---|---|---|
| `một người đang đi bộ` | `L21_V018, 5758` | `L21_V018, 5758` | yes | skipped, relative margin 0.0305 > 0.03 |
| `a person riding a motorcycle` | `L21_V018, 15933` | `L21_V018, 15933` | yes | skipped, relative margin 0.0910 |
| `car on the road` | `L21_V019, 13158` | `L21_V019, 13158` | yes | **triggered**, relative margin 0.0105 |
| `people sitting indoors` | `L21_V009, 16024` | `L21_V009, 16024` | yes | **triggered**, relative margin 0.0094 |

Example coarse -> refined visual offsets (submission frame unchanged in every row):

```text
L21_V015  submission 459    coarse 459    -> visual 467    (+8f / +0.27s, gain 0.051)
L21_V002  submission 12381  coarse 12381  -> visual 12389  (+8f / +0.27s, gain 0.052)
L21_V006  submission 8649   coarse 8649   -> visual 8657   (+8f / +0.27s, gain 0.023)
L21_V021  submission 11147  coarse 11147  -> visual 11203  (+56f / +1.87s, gain 0.008)
L21_V018  submission 16260  coarse 16260  -> visual 16212  (-48f / -1.92s, gain 0.002)
L21_V025  submission 12510  coarse 12510  -> visual 12492  (-18f / -0.72s, gain 0.010)
```

The top row never changed in these four queries. That is an observation, not a
judgement: the coarse fused top score is 1.0 after min-max normalization in all four,
and `rerank_alpha * gain` is far too small to displace it. Whether any refined frame is
*better* cannot be stated — there is no ground truth.

Skipping is genuinely cheap: an untriggered uncertainty pass cost 0.08-0.10 ms and
decoded nothing.

## 19. Inspection Artifact

`artifacts/refinement_smoke/summary.json` and `artifacts/refinement_smoke/results.html`
(the directory is gitignored, like every other artifact). The HTML shows, per query and
per candidate, the coarse frame and the refined visual frame side by side with their
IDs, scores, temporal offset, frame counts, and latencies. Nothing is labelled correct,
incorrect, better, or worse — it exists for a human to look at.

Regenerate with `.venv\Scripts\python.exe` and the smoke script in §18's configuration.

## 20. Tests

| File | Covers |
|---|---|
| `tests/test_frame_scorer.py` | the `FrameScorer` contract, determinism, finite-value validation, BGR->RGB, one-prepare-per-request, single batched call, backend sharing, lazy status, no download in unit tests |
| `tests/test_local_refinement.py` | sample-plan bounds, coarse frame always sampled, no duplicates, hard cap, start/end clamping, determinism, per-video fps, and coarse/earlier/later frame selection over real synthetic MP4s |
| `tests/test_refinement_policy.py` | disabled never decodes, always honours the budget, uncertainty triggers below and skips above the threshold, reasons recorded, region dedup, `top_hypotheses`, aggregate rollups |
| `tests/test_local_refinement_integration.py` | engine wiring, score components, determinism, no dropped candidates, Top-100 cap, per-request disable, fallbacks, runtime-state identity, root A/B switch, health without a model load, and the JSON separation of coarse/refined/submission frames |
| `tests/refinement_support.py` | the fakes and the synthetic-MP4 writer |

All offline, deterministic, no network, no AIC data, no model download. Suite:
**579 tests, 0 failures, 1 skipped** (the pre-existing torch lazy-import guard).

## 21. Performance Safety

Hard bounds: `top_hypotheses` -> `candidate_budget` -> `max_frames` per candidate. The
worst case is `candidate_budget * max_frames` decoded frames per query (5 * 32 = 160,
observed 157-159), never a full video and never a hundreds-of-frames surprise. Frames
are downscaled before being held, the in-memory window cache is bounded by
`cache_size_mb`, the model loads once per runtime generation, and the whole path is
skipped without any I/O when the trigger says no.

## 22. Limitations

- **CPU latency.** 14.4 s p50 per refined query on this machine, ~68 % of it CLIP
  inference on 160 frames. A CUDA device or a smaller `max_frames` / `candidate_budget`
  is the lever; `device: auto` already picks CUDA when present.
- `rerank_alpha` and `margin_threshold` are defaults, not tuned values. They cannot be
  tuned honestly until ground truth exists.
- The gain is non-negative by construction (§13), so refinement can reinforce but not
  demote within the refined set.
- Refinement is wired into **KIS only**. Q&A and TRAKE can reuse `FrameScorer`,
  `build_sample_plan`, and `FrameProvider.decode_frames`, but their algorithms are
  Phase 6+. `search_trake(refine_window_s=...)` is still accepted and still unused; it
  is now documented as such rather than silently dead.
- The refined visual frame is not written into the derived frame cache during scoring
  (only when it is displayed), so a repeated identical query re-decodes unless it hits
  the in-memory window cache of the same engine.
- Region merging uses a fixed time window; it does not consider visual similarity.
- The uncertainty heuristic looks at exactly two regions. Cluster density and other
  signals were deliberately not added: a more elaborate rule cannot be justified
  without data.

## 23. Not Started

Q&A per-video hypotheses, TRAKE k-best, TRAKE local/semantic refinement, independent
object/OCR/ASR retrieval channels, Vietnamese query-normalization redesign, submission
validation, and the manual-edit redesign all remain pending for Phase 6+.
