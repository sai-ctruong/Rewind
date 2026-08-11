# Phase 6 Grounded Multi-Hypothesis Q&A

Phase 6 fixes the most serious Q&A correctness flaw in the repository: one answer was
produced from one video's frames and then attached to prediction rows belonging to other
videos. The unit of answering is now **one video hypothesis**.

> **No accuracy claim.** This repository contains no AIC ground truth. No parameter here
> was tuned against correctness, no answer is labelled right or wrong, and none of the
> diagnostics is a precision, recall, or accuracy figure.

Phase 6 does **not** implement TRAKE k-best, TRAKE local refinement, independent
retrieval channels, a query-normalization redesign, a submission validator, or the
Phase 12 manual-edit architecture.

## 1. The Bug

`AICCompetitionEngine.answer_qa` did this:

```python
candidates = self.search_candidates(ground_query, top_k=requested)   # flat, cross-video
center = candidates[0]                                               # ONE global top-1
records = entry_records(self.entry, video_id=center.video_id)        # ONE video's frames
ans = VqaModule(...).answer(question, selected_records, ...)         # ONE call
preds = [self._from_candidate(c, answer=answer_text) for c in candidates]   # <-- the bug
```

The last line broadcasts `answer_text` — derived exclusively from `center.video_id` — to
every candidate row, including rows for entirely different videos. An AIC Q&A prediction
is `(video_id, frame_id, answer)` and must be grounded in *that* video, so a row for
video B carrying video A's answer is simply wrong. Rows for other videos also had no
evidence at all: nothing was ever collected for them.

Three related defects rode along:

| Defect | Effect |
|---|---|
| `expected_answer_type` never reached normalization | the UI's answer-type selector did nothing: UI → `ui/app.py` (dropped) → `QAInput` (stored, unread) → `normalize_answer` (never told) |
| `normalize_answer("không")` → `"no"` unconditionally | a count of zero silently became a boolean |
| `select_diverse_evidence` seeded `{first, last, nearest}` | asking for one or two frames returned window boundaries, not the strongest evidence |

## 2. The Fix: One Answer Per Video Hypothesis

```text
question + event
  -> coarse retrieval (deep enough to contain several videos)
  -> group candidates BY VIDEO -> ranked QAVideoHypothesis list
  -> take top M videos
     for each video, independently:
        frame hypotheses from THAT video only (temporally diverse)
        optional bounded local refinement (its own budget)
        evidence selection from THAT video only
        load pixels for the selected frames only
        call the backend with a bundle carrying that video_id
        normalize with the expected answer type
        reliability + abstention
        emit rows for THAT video carrying THAT answer
  -> dedup, rank, cap at max_answers
```

An answer can no longer reach another video because it never leaves its hypothesis:
`QAEvidenceBundle.__post_init__` raises if a bundle for video A contains a frame from
video B, and `_answer_one_hypothesis` raises if a backend returns a result whose
`video_id` is not the one it was asked about.

## 3. Data Models (`aic2026/qa.py`)

| Type | Role |
|---|---|
| `QAVideoHypothesis` | one video: rank, retrieval score, support count, its frame hypotheses, optional refinement |
| `QAFrameHypothesis` | one candidate frame: keyframe id, official `frame_idx`, timestamp, score |
| `QAEvidenceFrame` | one piece of evidence: video, frame, source, role, scores, availability, and (transiently) pixels |
| `QAEvidenceBundle` | everything a backend may see for ONE video; validates that nothing crosses videos |
| `QAAnswerResult` | one video's answer: raw, normalized, status, backend, visual flag, warning |
| `QAAnswererStatus` | truthful capability report |
| `VisualQAAnswerer` | the backend protocol |

`QAEvidenceFrame.image_bytes` is `repr=False`, excluded from comparison, and never
serialized. `to_dict()` reports `image_available` (does a source exist) and
`image_loaded` (were pixels actually read) as separate facts.

## 4. Evidence Selection

`select_evidence_frames(frames, count, diversity_s)` is explicit about priority:

1. the strongest frame, tagged `primary`;
2. the strongest remaining frame at least `diversity_s` away from it;
3. the strongest remaining frame on the **other** side, so three frames straddle the
   event (`before` / `primary` / `after`);
4. further slots by score, still respecting the gap, relaxing it only rather than
   returning fewer frames than asked for.

Output is in timestamp order so a backend reads chronologically. `select_diverse_evidence`
remains as a record-based wrapper over the same policy, so older callers get the fixed
behaviour rather than the boundary-frame behaviour.

Per-video frame hypotheses use `select_temporally_diverse`: five adjacent BTC keyframes
of the same second describe one moment and must not consume the whole budget.

## 5. Budgets

Everything is bounded, and every bound is configuration:

| Bound | Default | Meaning |
|---|---|---|
| `top_video_hypotheses` | 8 | videos answered per question |
| `frame_hypotheses_per_video` | 3 | submission-frame candidates per video |
| `evidence_frame_count` | 8 | evidence frames per backend call (hard cap 32) |
| `evidence_temporal_diversity_s` | 1.5 | minimum spacing between chosen frames |
| `max_answers` | 100 | final prediction rows |

Pixels are read **only** for selected evidence frames, and **only** when the backend can
actually look at them: a text-only backend triggers no decoding at all, while
availability is still reported from a cheap `FrameProvider.describe` check.

## 6. Q&A Refinement Budget

Phase 5's KIS budget is 5 regions × 32 frames and cost ~14 s per query on CPU. Running
that for every video hypothesis would cost minutes per question, so Q&A has its own:

| Setting | Default |
|---|---|
| `use_local_refinement` | `false` |
| `refinement_candidate_budget` | 1 region per video |
| `refinement_max_frames` | 12 frames (hard cap 32) |

`_qa_refiner()` builds a refiner from the Phase 5 config with `mode=always` and those
budgets substituted, reusing the engine's own `FrameProvider` and `FrameScorer`. A
refined frame becomes an extra `local_refinement` evidence frame ranked above the coarse
ones — and nothing more: it never changes the submitted frame.

## 7. Backend Contract

```python
class VisualQAAnswerer(Protocol):
    def answer(self, question, evidence: QAEvidenceBundle, *,
               expected_answer_type: str | None = None) -> QAAnswerResult: ...
    def status(self) -> QAAnswererStatus: ...
```

| Backend | `visual_capable` | State on this machine | Notes |
|---|---|---|---|
| `MockTextQAAnswerer` | **false** | `ready` | wraps the historical text heuristics; reasons over captions/OCR/ASR, never pixels |
| `LocalVlmQAAnswerer` | true | **`not_available`** | real adapter for an injected model exposing `generate(images, prompt)`; never downloads weights |
| `ApiVqaAnswerer` | true | **`not_available`** | lazy wrapper over the hosted vision backend; no SDK and no key present here |

`build_qa_answerer("auto")` prefers a hosted backend only when its key is already set,
then an explicitly supplied local model, and otherwise the mock — which reports itself
as non-visual, so the fallback is visible rather than silent. Construction imports
nothing and contacts nothing.

The two pre-Phase-6 stubs (`LocalVlmAnswerer`, `OptionalApiVlmAnswerer`) raised
unconditionally and were registered nowhere. They remain only as deprecated shims that
name their replacement.

## 8. Mock Backend Is Never Called Visual Q&A

The mock reports `visual_capable=False`, `production_ready=False`, and a warning; every
answer it produces carries `visual=false` plus "this is not visual Q&A". `/api/health`
exposes `qa.backend_type` and `qa.visual_capable`, and the UI renders
**"Mock / non-visual fallback"** in the Q&A panel and a `warn`-toned health badge.

## 9. Expected Answer Type

Canonical types: `auto`, `number`, `boolean`, `color`, `short_text`. The UI's historical
spellings (`yes/no`, `text`, `colour`) are accepted and canonicalized rather than
rejected. The chain that used to be dead is now connected end to end:

```text
UI #qa-type -> POST /api/video/vqa expected_answer_type
            -> answer_qa(expected_answer_type=...)
            -> build_answer_prompt(question, type)   (backend prompt)
            -> normalize_answer(text, expected_type=type)
            -> prediction row
```

An unsupported type is a `400 INVALID_ANSWER_TYPE`, not a silent fallback.

## 10. Normalization

| Type | Behaviour |
|---|---|
| `number` | digits extracted, else English/Vietnamese number words: `"three people"` → `3`, `"Bốn"` → `4`, `"mười"` → `10` |
| `boolean` | `yes/có/đúng/true` → `yes`; `no/không/sai/false` → `no` |
| `color` | longest phrase first, so `"xanh dương"` → `blue` beats a bare `"xanh"`; a colour is **never invented** when none was said |
| `short_text` | casefold and collapse whitespace only — a noun phrase survives intact |
| `auto` | historical behaviour, minus the ambiguous bare `không` |

**Vietnamese `không`** is the reason `normalize_answer` takes a type at all: it means
"no" for a boolean question and "zero" for a counting one.

```python
normalize_answer("không", expected_type="number")   # "0"
normalize_answer("không", expected_type="boolean")  # "no"
normalize_answer("không")                           # "không"  <- not guessed at
```

**Unknown answers are checked first.** The real smoke caught this: the mock's fallback
output `"không có mô tả"` ("no description") begins with `không`, so type normalization
turned a refusal into a confident `"no"` for a boolean question and `"0"` for a counting
one. Refusals now short-circuit ahead of type handling and stay refusals, which lets
abstention fire instead.

## 11. Answer Reliability

`answer_reliability_score` is a transparent additive heuristic in [0, 1], deliberately
**not** named a confidence or probability because nothing in it was fitted to data:

```text
  0.10  base
+ 0.25  the backend actually looked at images (visual_capable AND images loaded)
+ 0.10  the backend is production-ready
+ 0.05  per evidence frame, capped at 4 frames  (max 0.20)
+ 0.15  the answer is non-empty and not an "unknown" form
+ 0.10  the answer validates against the requested answer type
+ 0.10  scaled by the coarse retrieval margin of the winning video
```

Every term is observable in the response, so a reader can reconstruct the number. A
non-visual backend can never earn the visual term.

## 12. Abstention And Failure

Four distinct statuses, never collapsed:

| Status | When |
|---|---|
| `answered` | the backend returned a usable answer |
| `abstained` | the answer is empty or an unknown form (and `abstain_enabled`) |
| `backend_failed` | the backend raised; the answer is `""` → `unknown`, never fabricated |
| `visual_unavailable` | a visual backend had no pixels; it was not called at all |

A failure is confined to one hypothesis: the other videos are answered normally, and the
failing video keeps its coarse rows with `unknown` rather than borrowing a neighbour's
answer. `qa.backend.required = true` raises up front instead of degrading silently.

## 13. Frame-ID Policy

The Phase 5 separation is preserved exactly:

| Field | Meaning |
|---|---|
| `coarse_official_frame_idx` | the BTC mapped `frame_idx` |
| `best_visual_frame_idx` | a refined/decoded frame, evidence only |
| `submission_frame_idx` | what goes into the AIC row — the official mapped frame |

Under the default `preserve_coarse` policy `submission_frame_idx == coarse_official_frame_idx`
always. Refined and decoded frames appear as evidence and are served through
`/api/video/decoded_frame/...` with `X-Frame-Role: refined_visual_frame`; no arbitrary
decoded frame index is ever submitted.

## 14. Diagnostics

Per question: `retrieved_video_hypotheses`, `answered_video_hypotheses`,
`visual_hypotheses`, `nonvisual_hypotheses`, `abstentions`, `backend_failures`,
`visual_unavailable`, `evidence_frames_used`, `local_refinement_calls`,
`frame_decode_failures`, `predictions`, `distinct_answer_videos`, plus
`retrieval_ms` / `evidence_selection_ms` / `refinement_ms` / `vqa_ms` / `total_ms`.

Two are the Phase 6 regression guards, computed from the produced rows rather than
asserted from the algorithm, so a refactor that reintroduces copying makes them nonzero:

```text
cross_video_answer_copy_count                  MUST be 0
answer_without_matching_evidence_video_count   MUST be 0
```

None of these is an accuracy metric.

## 15. Runtime State

Q&A takes one `RuntimeDatasetState` snapshot per request, like every other route. The
backend and the frame provider belong to the engine, and the engine is a field of the
frozen state, so a `DATA_ROOT` switch replaces all of them together. Evidence URLs carry
the generation, and a stale one returns `409 STALE_RESULT_GENERATION` — verified by a
test that switches roots and confirms the new generation answers about the new videos
with the new pixels.

## 16. Manual Answer Correction

The correction box used to rewrite the answer on **every** Q&A row, which would have
re-created exactly the cross-video copying this phase removed. It is now scoped to the
**selected video hypothesis**: clicking a hypothesis card selects it, and the correction
applies only to that video's rows. A full per-row edit model remains Phase 12; this is a
small, complete scoping fix rather than a workaround.

## 17. Real L21 Smoke

Real `data` root, scope `existing_videos`, existing cache
`artifacts/aic2026_index_existing_videos` — **reused, not rebuilt** (`cache_hit=true`,
valid, 29 videos / 7,800 frames). Four exploratory questions, `top_video_hypotheses=5`.

### Backend availability

`ANTHROPIC_API_KEY` is not set, the `anthropic` SDK is not installed, and no local VLM is
configured. `build_qa_answerer("auto")` therefore selected **`mock`**:
`visual_capable=false`, `production_ready=false`.

> **REAL VISUAL Q&A REMAINS UNAVAILABLE ON THIS MACHINE.** The pass below is a
> **NON-VISUAL MOCK SMOKE**. Nothing in it demonstrates visual question answering.

### Pass 1 — non-visual mock smoke

| Question | Type | Hypotheses | Answered | Abstained | Evidence frames | Latency |
|---|---|---|---|---|---|---|
| What color is the car? | color | 5 | 0 | 5 | 37 | 7,122 ms* |
| How many people are visible? | number | 5 | 0 | 5 | 33 | 106 ms |
| Is the person standing? | yes/no | 5 | 0 | 5 | 36 | 90 ms |
| What is the person doing? | text | 5 | 0 | 5 | 36 | 76 ms |

\* the first question pays the one-off CLIP **text encoder** load (~7 s); retrieval is
75-90 ms thereafter. Evidence selection was 10-30 ms and the backend call under 1 ms.

Every hypothesis produced the raw answer `"không có mô tả"` ("no description") and
abstained with `answer_reliability_score` ≈ 0.30. That is the honest outcome: the real
AIC dataset carries no generated captions, so a text-only backend has nothing to read.
An earlier run of this same smoke — before the unknown-answer guard of §10 — reported
confident `"0"` and `"no"` answers instead. Finding and removing that fabrication is the
most useful thing this smoke did.

### Pass 2 — structural isolation probe

A deterministic non-production backend that echoes the video it was handed, used only to
prove independence on **real** data. It measures plumbing, never answer quality.

| Check | Result |
|---|---|
| `cross_video_answer_copy_count` (both passes) | **0** |
| `answer_without_matching_evidence_video_count` (both passes) | **0** |
| Every probe answer matches its own video | **true** |
| Distinct answers per question | **5, 5, 5, 5** (one per video hypothesis) |
| Distinct videos answered per question | 5 |
| Evidence frames used (both passes) | 284 |
| Backend failures / frame decode failures | 0 / 0 |

Five distinct answers for five distinct videos, on every question, is precisely what the
pre-Phase-6 code could not produce: it emitted one answer copied across all rows.

## 18. Inspection Artifact

`artifacts/qa_smoke/summary.json` and `artifacts/qa_smoke/results.html` (the directory is
gitignored, like every other artifact). Per question and per video hypothesis the page
shows the submission frame, the evidence thumbnails with their roles and sources, the raw
and normalized answers, the answer status, the backend and whether it was visual, the
reliability score, and the timings. A red banner states that the run was non-visual.
Nothing is labelled correct, incorrect, or better.

## 19. Tests

| File | Covers |
|---|---|
| `tests/test_qa_per_hypothesis.py` | answer isolation (red video → "red", blue video → "blue"), zero cross-video copies, per-video evidence, grouping and budgets, confined backend failures, abstention, dedup, ordering, score decomposition, official submission frames |
| `tests/test_qa_evidence.py` | strongest-first selection for 1/2/3 frames, ordering, budgets, JPEG and MP4 sources, explicit visual-unavailable state, no decoding for a text-only backend, Q&A refinement budget |
| `tests/test_qa_answer_normalization.py` | type canonicalization, English/Vietnamese numbers and booleans, colours, short text, ambiguous `không`, refusals that must not become confident, end-to-end type flow |
| `tests/test_qa_backend_status.py` | mock is non-visual, local VLM unavailable vs injected, API absence and injected client, `auto` selection, reliability heuristic, config validation, health without a model load, HTTP payload, runtime-generation isolation |
| `tests/qa_support.py` | the two-video red/blue fixture and the fake backends |

All offline and deterministic: no network, no API key, no model download, small
synthetic videos and images. Suite: **682 tests, 0 failures, 1 skipped** (the pre-existing
torch lazy-import guard), up from 579.

## 20. Limitations

- **Real visual Q&A is not available here.** The architecture, the contract, and the
  tests are complete, but no image-capable backend is installed, so no visual smoke was
  run and none is claimed.
- The mock backend is useless on real AIC data (no captions), which is now visible as
  honest abstention rather than hidden behind a fabricated answer.
- Q&A refinement is off by default. Turning it on costs roughly
  `top_video_hypotheses × refinement_max_frames` decoded frames plus CPU CLIP inference.
- `answer_reliability_weight`, `video_support_bonus`, and `abstain_threshold` are
  defaults, not tuned values; they cannot be tuned honestly without ground truth.
- Rows for a video all carry that video's single answer. Per-frame answers within one
  video are not produced.
- Video ranking is `best_candidate_score` plus a bounded support bonus — deliberately
  simple, not a learned ranker.
- Manual editing is scoped per video hypothesis, not per row; the full model is Phase 12.
- The `qa.answerer_batch_size` key was removed and now raises a `ConfigError` naming its
  replacement, because nothing ever read it.

## 21. Not Started

TRAKE k-best, TRAKE local/semantic refinement, independent object/OCR/ASR retrieval
channels, the Vietnamese query-normalization redesign, submission validation, and the
Phase 12 manual-edit architecture all remain pending.
