# Phase 8 TRAKE k-best, Adaptive Coverage And Video-Backed Temporal Refinement

Phase 7 made TRAKE structurally correct, and the real smoke immediately showed what that
cost: 65 of 77 video hypotheses were discarded for missing at least one event, leaving 12
complete sequences. Phase 8 raises the number of **complete** sequences the search can
build, without relaxing a single Phase 7 invariant.

> **No accuracy claim.** This repository has no AIC ground truth. "Complete-sequence
> yield" below is a structural count of well-formed sequences; it is **not** accuracy and
> **not** recall. Nothing was tuned against quality, and no sequence is labelled correct.

Phase 8 does **not** implement independent object/OCR/ASR retrieval channels, a
query-normalization redesign, the global submission validator, a manual-edit redesign, a
learned temporal model, or supervised threshold tuning.

## 1. What Phase 7 Left On The Table

| | Phase 7 |
|---|---|
| Sequences per video | exactly 1 |
| Candidate depth | fixed `per_event_top_k = 40` for every event |
| Final ranking | the beam's running score, stale after recovery |
| `score_video_hypothesis` | each event's *earliest* candidate |
| TRAKE local refinement | none; `refine_window_s` inert |
| Real smoke | 77 hypotheses → 12 complete sequences |

Of the 101 missing positions, **59 had no candidate at all** for that event in that
video. A fixed shallow depth, not a broken aligner, was the binding constraint.

## 2. k-best Alignment

`align_video_k_best_beam(events, video, config, *, k)` returns up to `k` distinct
sequences for one video, best first.

The enumeration is real, not re-running the search with perturbed inputs. Every search
state already carries its own `previous` chain, so two states ending at the same
candidate but choosing different earlier events are genuinely different sequences and
both survive. What changed is that the beam retains `beam_width * k` states per event
instead of `beam_width`, and the final step reconstructs the top paths rather than only
the argmax. Paths are deduplicated by their keyframe signature.

Determinism is total: ties break on `(-score, -matched, timestamp, keyframe_id)` at every
stage, so identical inputs give identical output including tie order.

`align_video_beam_dp` remains the single-best entry point and is `k=1` of the same search.

## 3. Exact DP Reference

`align_video_exact_dp` is implemented as a **test oracle**, because the objective turned
out to be Markovian in `(event_index, last present candidate, matched count)`, so the
exact optimum is polynomial rather than exponential — it fell out cleanly instead of
expanding the phase.

It is bounded by `exact_dp_max_states` and returns `None` rather than degrading when the
bound would be exceeded. Nothing in production calls it; the shipped search is
`beam_dp`, its method string is `exact_dp_reference`, and `trake.alignment_method` still
rejects `exact_dp`. Tests assert the wide-beam result matches the exact objective on a
3-event and a 5-event deterministic case.

## 4. Multiple Sequences Per Video, With Diversity

A Top-100 holding twenty sequences that differ by one frame is worth little.
`sequences_are_near_duplicates` is deterministic: two sequences of one video are
near-identical unless at least `min_difference_events` positions use a different frame
**and** some event moved at least `min_time_distance_s`.

```text
[100, 200, 300] vs [100, 201, 300]   near-duplicate  (one event, 0.1 s)
[100, 200, 300] vs [120, 240, 340]   distinct
```

`select_diverse_alignments` keeps the strongest sequences of a video up to
`max_alignments_per_video`. `_select_final_sequences` then fills the final list in
passes: the best sequence of each video first, then a second from each, and so on — one
strong video contributes several readings without starving the others, and never more
than `final_top_k`.

## 5. Adaptive Candidate Expansion

Deep retrieval for every event would be wasteful; the Phase 7 evidence says only *some*
events were blocking completeness. Expansion is therefore selective:

```text
retrieve every event at per_event_top_k (40)
  -> align
  -> while videos_with_full_event_coverage < target_complete_video_hypotheses:
       identify the events reaching FEWER videos than the best-covered event
       re-retrieve ONLY those, at the next depth stage
       realign
```

Stages come from `candidate_depth_expansion` (`[120, 300]`), capped by
`candidate_depth_max` (400) and stopped by `target_complete_video_hypotheses` (12).
Expansion uses the **existing** retrieval path — the same CLIP+BM25 fusion, asked for
more rows. No new channel is introduced; that is Phase 9. Nothing is fabricated: a
candidate either comes back from retrieval or the event stays missing.

Diagnostics: `candidate_expansion_triggered`, `candidate_expansion_stages`,
`events_expanded`, `depth_before`, `depth_after`, `new_candidates_added`,
`initial_candidate_counts`, `expanded_candidate_counts`,
`new_complete_video_hypotheses`, `complete_alignments_before/after_expansion`.

## 6. Pre-alignment Versus Final Scoring

Two scores that used to be conflated are now separate:

| | Purpose | Reads |
|---|---|---|
| `score_video_hypothesis` | choose which videos to attempt | each event's earliest candidate — a coarse pre-filter |
| `alignment_objective` | rank the results | the steps the alignment **actually** holds |

`alignment_objective` recomputes candidate scores, transition penalties, missing
penalties and the coverage bonus from the chosen steps, and every alignment — including
one changed by Phase 7 recovery — is rescored through it. Phase 7 carried the beam's
running score, which stopped describing the sequence the moment recovery replaced a step.

## 7. Event-Local Visual Refinement

`aic2026/trake_refinement.py`. The decisive difference from Phase 5's KIS refinement is
that **each event is scored against its own text**. Using the whole sentence would ask
which frame looks like the whole story, which is not the question.

```text
complete coarse alignment
  -> for each of the first `max_events_per_alignment` events:
       bounded window around ITS coarse mapped frame
       query = THAT event's text
       sample the original MP4, score with CLIP, batched
  -> joint ordered selection across events
  -> rerank the sequence
```

Everything is reused: `LocalFrameRefiner` supplies the sample plan, downscaling, window
cache, batching and fallbacks; `FrameProvider` is the single OpenCV implementation;
`FrameScorer` uses the shared `CLIPBackend`. No second model stack exists.

Refinement runs **after** coarse alignment on a few already-complete sequences, never on
raw per-event candidates — refining the candidate pool would multiply Phase 5's cost by
the retrieval depth.

## 8. Budget

| Setting | Default |
|---|---|
| `refinement.enabled` | `false` (opt-in) |
| `top_alignment_budget` | 3 sequences |
| `max_events_per_alignment` | 4 events |
| `frames_per_event` | 8 |
| `fine_fps` | 2.0 |
| `window_s` | 2.0 |
| `batch_size` | 8 |
| `max_frames_per_query` | **96** (hard ceiling, config-capped at 512) |

`FrameBudget` is shared across the sequences of one query and is decremented as frames
decode; an event that would exceed it is marked `skipped_budget` rather than silently
truncated. Observed cost: 72-96 frames per query, exactly at the ceiling.

## 9. Order Safety Through Joint Selection

Independently picking each event's visual maximum can produce a reversed sequence, which
would display an impossible reading. `local_ordered_refinement` instead runs a small DP
over the frames already sampled inside each event's own window, choosing one frame per
event under non-decreasing time.

Order safety is therefore a property of the choice, not a patch applied afterwards. It is
named `local_ordered_refinement` and is explicitly **not** the global TRAKE alignment DP:
it ranges only over the handful of frames inside each event's window. Events not selected
for refinement participate pinned to their coarse frame, so the selection covers the
whole sequence. `order_violation_detected` and `order_violation_resolved` report what the
independent choice would have done versus what was chosen.

## 10. Frame-ID Policy (Unchanged)

```text
coarse_official_frame_idx   the BTC mapped frame_idx
visual_frame_idx            the refined frame -- evidence, display, local score, reranking
submission_frame_idx        == coarse_official_frame_idx, always
```

`apply_refinement` asserts `frame_ids` is unchanged. A refined frame reaches the UI
through `/api/video/decoded_frame/...`, labelled as evidence. Nothing switches the
submission to a decoded frame until AIC confirms those semantics.

## 11. Reranking

```text
final_sequence_score = coarse_alignment_score
                     + rerank_alpha * clamp(mean(event visual gain), -1, 1)

event visual gain = chosen local visual score - the coarse frame's own visual score
```

Using the *improvement over the coarse frame* rather than the raw CLIP score avoids both
double-counting and a scale mismatch, exactly as in Phase 5. `alpha = 0.10`, untuned.
`coarse_alignment_score`, `visual_gain_aggregate` and `final_sequence_score` are all
reported separately.

## 12. `refine_window_s`

No longer inert. A request value overrides `refinement.window_s` and genuinely selects
the local sampling window; a test asserts a wider value samples more frames. Refinement
itself remains opt-in, and when off the response says `status: "disabled"` with
`frames_decoded: 0` rather than pretending.

## 13. Real L21 Smoke

Real `data` root, scope `existing_videos`, existing cache reused (not rebuilt), 29
videos / 7,800 frames. Scorer `clip` / `openai/clip-vit-base-patch32` / **cpu**, ready.
The same four queries as Phase 7 (three of 3 events, one of 4), in three passes.

| | A: Phase-7-like | B: k-best + expansion | C: B + refinement |
|---|---|---|---|
| Complete sequences | **12** | **187** | 187 |
| Videos with full event coverage | 12 | **66** | 66 |
| Videos contributing >1 sequence | 0 | **62** | 62 |
| Max sequences from one video | 1 | 3 | 3 |
| Events expanded | — | 12 | 12 |
| New candidates added | — | 2,220 | 2,220 |
| Sequences refined | — | — | 12 |
| Events refined | — | — | 39 |
| Frames decoded / scored | 0 | 0 | **312 / 312** |
| Refinement failures | — | — | **0** |
| Order violations detected / resolved | — | — | 0 / 0 |
| Refinement latency p50 / p95 | — | — | 9,851 ms / 12,675 ms |
| Total latency per query | 257-382 ms* | 551-1,486 ms | 8.8-13.9 s |

\* pass A's first query also paid the one-off 12.7 s CLIP text-encoder load.

Per query, complete sequences: A `2, 5, 4, 1` → B `57, 54, 26, 50`. Expansion reached
depth 300 for the events that were short and left well-covered events at 40.

**Structural invariants across all three passes:**

```text
malformed_prediction_count             0
wrong_event_count_prediction_count     0
cross_video_step_count                 0
unordered_submission_sequence_count    0
every returned row has exactly N frames  true
```

Pass A reproducing Phase 7's 12 sequences exactly is a useful check that the baseline
path is unchanged and the gain comes from the new machinery.

**Order violations were 0 on real data.** The mechanism is proven by synthetic tests
where the independent per-event maxima genuinely reverse; it simply did not trigger here.
That is reported rather than presented as evidence the problem does not exist.

## 14. Inspection Artifact

`artifacts/trake_phase8_smoke/summary.json` and `results.html` (the directory is
gitignored). Per query it shows the A/B/C comparison and, for the top sequences, an
event-by-event table with the coarse frame (submitted) beside the refined visual frame
(evidence only), plus the score decomposition, timings, expansion depths and refinement
counters. Several sequences from the same video appear together so they can be compared.
No correct/incorrect labels.

## 15. Tests

| File | Covers |
|---|---|
| `tests/test_trake_kbest.py` | distinct/deterministic/sorted/duplicate-free k-best, exact-DP agreement on 3- and 5-event cases, exact-DP refusal when unbounded, pre-alignment vs final scoring, multiple sequences per video, per-video cap, ranks, Phase 7 invariants under k-best |
| `tests/test_trake_sequence_diversity.py` | near-duplicate rule, configurable thresholds, selection order, per-video cap, determinism under ties, end-to-end dedup, config validation, nested-block flattening |
| `tests/test_trake_candidate_expansion.py` | expansion triggering, non-triggering, ceilings, completing previously incomplete videos, no fabrication, invariants preserved, existing-retrieval-only, empty-result explanation, config validation |
| `tests/test_trake_local_refinement.py` | per-event queries, one batched call per event, `refine_window_s` changing the window, frame/event/sequence budgets, missing MP4, unavailable scorer, ordered-DP selection, reversed choice resolved, submission frames unchanged, score decomposition, no filesystem paths |

Every Phase 7 test is kept. All tests are offline, deterministic, and need no API,
network, GPU, or AIC data. Suite: **811 tests, 0 failures, 1 skipped** (the pre-existing
torch lazy-import guard), up from 750.

## 16. Performance

No full-video decode, no refinement of raw per-event candidates, a hard per-query frame
ceiling, batched scoring, one shared CLIP checkpoint, and no repeated model load. The
cost is visible: `candidate_retrieval_ms`, `alignment_ms`, `refinement_ms`, `total_ms`,
`frames_decoded`, `frames_scored`.

Refinement on CPU costs ~10 s per query for 3 sequences × 4 events × 8 frames. That is
why it is **off by default**; a CUDA device or a smaller budget is the lever.

## 17. Limitations

- **Refinement latency**: ~10 s p50 per query on CPU, dominated by CLIP inference on ~96
  frames. Off by default for exactly that reason.
- `rerank_alpha`, the diversity thresholds, the expansion stages and
  `target_complete_video_hypotheses` are defaults, not tuned values, and cannot be tuned
  honestly without ground truth.
- More complete sequences is a **structural** improvement. Whether any of them is right
  cannot be said here.
- Expansion re-retrieves an event from scratch at the deeper depth rather than fetching
  only the incremental rows.
- `score_video_hypothesis` still uses each event's earliest candidate. It is now
  explicitly only a pre-filter, and final ranking no longer depends on it, but a better
  pre-filter would need quality feedback.
- The k-best beam widens with `k`, so a large `k_best_per_video` costs proportionally
  more alignment time.
- Order violations did not occur on real data; the resolution path is covered only by
  synthetic tests.
- Joint ordered refinement ranges over sampled frames only, not over alternative coarse
  candidates.

## 18. Not Started

Independent object/OCR/ASR retrieval channels, the Vietnamese query-normalization
redesign, the global submission validator, the Phase 12 manual-edit architecture, a
learned temporal model, and any accuracy benchmark all remain pending.
