# Phase 7 TRAKE Structural Correctness And Complete Event Alignment

An official TRAKE row is `video_id, frame_id_1, ..., frame_id_N`, where frame *i* answers
semantic event *i*. Phase 7 makes that shape an invariant of the code rather than a
convention the code happened to follow.

> **No accuracy claim.** This repository has no AIC ground truth. Nothing here was tuned
> against correctness, no sequence is labelled right or wrong, and none of the
> diagnostics is a precision, recall, or accuracy figure.

Phase 7 does **not** implement k-best alignment, exact DP, TRAKE local video refinement,
a learned temporal model, retrieval-channel changes, or the global submission validator.

## 1. The Bug

The DP could skip an event, and the conversion then *removed* the skipped position:

```python
# aic2026/trake.py, pre-Phase-7
frame_ids = tuple(a.candidate.frame_id for a in self.alignments if a.candidate is not None)

# aic2026/engine.py, pre-Phase-7
selected = [a for a in result.alignments if a.candidate is not None]
steps = [TemporalStep(...) for a in selected]
```

Two failures followed from that one filter:

1. **Short rows.** A four-event query could emit three frames. That is not a partial
   answer; it is a malformed row.
2. **Label shift.** `ui/app.py` zipped the already-compacted step list against the full
   event list (`events[index]` over `enumerate(match.steps)`), so after a gap every event
   was displayed against the wrong text.

A third defect sat downstream in `aic2026/metrics.py`:

```python
for frame_id, frame_range in zip(pred.frame_ids, gt.event_ranges):
```

`zip` silently truncates, so a three-frame row answering four events was scored over the
three it happened to cover and looked merely imperfect rather than invalid.

## 2. Event-Preserving Data Model

| Type | Guarantee |
|---|---|
| `TrakeAlignedStep` | one event position; a missing event is an explicit `missing` step, never an absent one |
| `TrakeAlignment` | `len(steps) == len(events)`, `steps[i].event_index == i`, one video throughout |
| `TrakePrediction` | **complete only**: exactly `event_count` frames, no missing indices, one video |
| `TrakeAlignmentReport` | predictions + alignments + discarded + diagnostics |
| `TrakeStructureError` | raised the moment any of those is violated |

`TrakeAlignedStep` carries three identities separately, following the Phase 5 policy:

```text
coarse_official_frame_idx   the BTC mapped frame_idx
visual_frame_idx            a locally refined frame -- always None in Phase 7
submission_frame_idx        what goes in the row == coarse_official_frame_idx
```

Validation is in `__post_init__`, so a malformed structure cannot be constructed at all:
a step cannot hold another event's candidate, a step cannot hold another video's
candidate, a `missing` step cannot carry a candidate, an alignment cannot have the wrong
number of steps or steps out of order, and a prediction cannot carry a missing event, a
foreign video, or the wrong frame count.

## 3. The Complete-Prediction Invariant

```text
len(prediction.frame_ids) == prediction.event_count == len(prediction.steps)
```

Enforced in five places rather than one: the dataclass (`TrakeStructureError`),
`to_complete_prediction` (returns `None` for anything incomplete), `align_trake` (only
complete alignments become predictions), `engine._from_trake` (re-checks before building
the row), and `write_submission(require_row_length=...)` (refuses to serialize).

An incomplete alignment is **discarded**, never reshaped. If nothing is complete, the
result is empty and the diagnostics say why — never a set of short rows.

## 4. Recovery

Skipping stays available *inside* the search, because it is how a partial hypothesis is
represented. After the search, `recover_missing_events` tries to fill each gap:

1. use only that event's own candidates, from the **same video**;
2. bound them by the nearest present event before and after (one-sided when only one
   neighbour exists);
3. apply `min_gap_s` / `max_gap_s` when `recovery_respect_gap` is on;
4. pick the best score, breaking ties by earliest timestamp then keyframe id;
5. mark the step `recovered`, never `aligned`.

Missing positions are visited in ascending order, and each re-reads its neighbours from
the partially recovered sequence, so consecutive gaps stay ordered relative to each
other. If the resulting sequence is not temporally ordered, the whole recovery is
discarded and the original incomplete alignment is kept.

**Nothing is invented.** No sentinel, no frame 0, no borrowing a neighbouring event's
frame, no nearest-timestamp guess that ignores which event is being filled, and no
candidate from another video. If no candidate fits, the event stays missing and the
alignment is discarded.

Status vocabulary:

| Level | Values |
|---|---|
| step | `aligned` · `recovered` · `missing` |
| alignment | `complete` · `complete_with_recovery` · `incomplete` |

## 5. Same-Video And Ordering

Every step of an alignment must belong to the alignment's video; this is checked in
`TrakeAlignment.__post_init__`, in `TrakePrediction.__post_init__`, and again in
`recover_missing_events`, which raises if handed another video's candidates.
`cross_video_step_count` is reported and is 0.

Ordering is judged on **timestamps**, non-decreasing, plus `min_gap_s` / `max_gap_s`
where configured. It is deliberately *not* judged on frame-ID uniqueness: 192 official
videos repeat a `frame_idx`, so two events sharing a frame ID is legitimate data and is
accepted whenever the timestamps advance.

## 6. Method Naming

The search is beam-pruned dynamic programming and is called `beam_dp` everywhere — in
the config, the alignment, the prediction, the API payload, and the UI badge. The entry
point is now `align_video_beam_dp` (with `align_video_dp` retained as an alias).

`trake.alignment_method` accepts `beam_dp` and `beam_pruned_dp`. It **rejects**
`exact_dp` with a message saying the implementation is beam-pruned, not exact. A test
scans the source, config, and UI for any claim of exact DP that is not an explicit denial.

Exact DP and k-best remain Phase 8.

## 7. `refine_window_s`

Kept for API compatibility, and no longer able to look functional. Every TRAKE response
now carries:

```json
{"refinement": {"applied": false, "status": "not_implemented_phase_7"}}
```

plus `refinement_applied`, `refinement_status`, and `refine_window_s_requested` in the
diagnostics. A test asserts the parameter changes nothing about the result.

## 8. Metrics And Export

`trake_r_score` now checks length before scoring: a row whose frame count differs from
the ground-truth event count scores **0**, not partial credit.
`is_structurally_valid_trake_row` exposes the same check.

`write_submission(..., require_row_length=N)` raises `SubmissionStructureError` and
writes nothing on a mismatch. `/api/submission/save` passes `1 + event_count` for TRAKE
and answers `422 MALFORMED_SUBMISSION_ROW`. This is a local safety net; the full schema
validator remains Phase 11.

## 9. UI

The API returns `event_count`, `frame_ids`, `alignment_status`,
`recovered_event_indices`, `missing_event_indices`, `method`, and per-step
`event_index` / `event_label` / `status` / `submission_frame_idx`. The frontend labels
each card from the step's **own** `event_index`, so a label shift is impossible even if
the payload changed shape. A recovered event is visible as a small status suffix rather
than an alarm. Badges show `n/N events`, the alignment status, and the method.

## 10. Diagnostics

Per query: `event_count`, `event_candidate_counts`, `videos_considered`,
`video_hypotheses_considered`, `initial_incomplete_alignments`, `initial_missing_events`,
`recovered_events`, `remaining_missing_events`, `missing_without_candidates`,
`missing_with_rejected_candidates`, `discarded_incomplete_alignments`,
`returned_complete_predictions`, `alignment_method`, `beam_width`, `per_event_top_k`,
`alignment_ms`.

Three structural invariants, which must be zero:

```text
malformed_prediction_count            0
wrong_event_count_prediction_count    0
cross_video_step_count                0
```

None of these is an accuracy metric.

## 11. Real L21 Structural Smoke

Real `data` root, scope `existing_videos`, existing cache
`artifacts/aic2026_index_existing_videos` — **reused, not rebuilt** (`cache_hit=true`,
valid, 29 videos / 7,800 frames). Method `beam_dp`, beam width 8. Four multi-event
queries (three of 3 events, one of 4).

| | |
|---|---|
| Queries | 4 |
| Video hypotheses considered | 77 |
| Initially incomplete alignments | 65 |
| Initial missing events | 101 |
| Events recovered | **0** |
| Remaining missing events | 101 |
| Discarded incomplete alignments | 65 |
| **Returned complete predictions** | **12** |
| `malformed_prediction_count` | **0** |
| `wrong_event_count_prediction_count` | **0** |
| `cross_video_step_count` | **0** |
| Every returned row has exactly N frames | **true** |

Per query: 2, 5, 4, and 1 complete predictions; 9.8 s for the first (one-off CLIP text
encoder load) and 250-385 ms thereafter.

**Why recovery fired zero times**, which matters because "0 recovered" is otherwise
indistinguishable from a broken recovery path: of the 101 missing positions, **59 had no
candidate at all** for that event in that video, and **42 had candidates that all
violated the temporal constraints**. None was recoverable, so none was recovered and
nothing was fabricated. That breakdown is now part of the diagnostics rather than
something a reader has to probe for. The synthetic tests cover the case where a valid
candidate *does* exist and prove recovery fills it.

Sixty-five alignments that would previously have produced short, mislabelled rows were
discarded instead.

## 12. Inspection Artifact

`artifacts/trake_structure_smoke/summary.json` and `results.html` (the directory is
gitignored, like every other artifact). Per query it shows the ordered event texts,
per-event candidate counts, the alignment counts, and for each returned sequence the
video, the frame list, the alignment status, and an event-by-event table with the
submission frame, timestamp, step status, and keyframe thumbnail. Nothing is labelled
correct, incorrect, or better.

## 13. Tests

| File | Covers |
|---|---|
| `tests/test_trake_structure.py` | one step per event, missing events keep their index, labels cannot shift, every structural guard raises, incomplete never becomes a prediction, 4 events → 4 frames, temporal order, duplicate `frame_idx` accepted, method naming, no source claims exact DP |
| `tests/test_trake_recovery.py` | recovery from the same video, previous/next/both-neighbour constraints, one-sided constraints, consecutive gaps, `min_gap`/`max_gap`, disabled recovery, nothing invented, foreign-video refusal, order-breaking recovery rejected wholesale, reporting |
| `tests/test_trake_output_schema.py` | engine rows of exactly N frames for N ∈ {2,3,4}, structural summary all zero, event-preserving matches, provenance payload, `final_top_k`, empty-result safety, `refine_window_s` status, metric length check, writer refusal, HTTP payload, generation, no filesystem paths, config rejects `exact_dp` |

Existing `tests/test_trake_dp.py` still passes unchanged. All tests are offline,
deterministic, and need no API, network, GPU, or AIC data. Suite: **750 tests, 0
failures, 1 skipped** (the pre-existing torch lazy-import guard), up from 687.

## 14. Limitations

- **Recall cost.** Requiring completeness discards 65 of 77 alignments on the real
  smoke. That is correct — a short row is invalid, not partial — but it means TRAKE
  returns fewer sequences than before. Deeper per-event retrieval would raise
  completeness; changing `per_event_top_k` is a tuning decision with no ground truth to
  justify it, so it was left at its existing default.
- `score_video_hypothesis` still ranks videos using each event's *earliest* candidate
  rather than the eventually aligned one. It is a coarse pre-filter, it never reads
  another video's or another event's candidate, and changing it would mean tuning
  against quality — a Phase 8 concern.
- One alignment per video: `alignments_per_video` is still effectively 1. k-best is
  Phase 8.
- Recovery is single-pass and greedy in ascending index order; it does not backtrack to
  reconsider an earlier recovery in light of a later one.
- `refine_window_s` remains inert. TRAKE local refinement is not implemented.
- The submission-length check is local to the writer and the TRAKE endpoint; global
  schema validation is Phase 11.

## 15. Not Started

k-best alignment, exact DP, TRAKE local/semantic refinement, learned temporal scoring,
independent retrieval channels, the query-normalization redesign, the global submission
validator, and the Phase 12 manual-edit architecture all remain pending.
