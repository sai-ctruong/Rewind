# Phase 10 Submission Validation And UI Result Safety

A correctness phase. Nothing here tunes retrieval, changes ranking weights, adds a
channel, or touches a TRAKE algorithm. The goal is narrower and harder: a result that
reaches export must be structurally valid, belong to the active runtime generation, use
official submission frame IDs, and be impossible to corrupt by editing something else.

> **Structurally valid is not semantically correct.** This phase can say a file has the
> right shape. It cannot say an answer or a frame is right, and this repository has no
> AIC ground truth. The UI says "Valid format", never "correct".

## 1. What Was Wrong

**Three export paths, no shared rules.** `aic2026/cli.py` called `write_submission` with
no validation at all. `ui/app.py` added only the Phase 7 TRAKE row-length check. Each
place decided for itself what a valid row was.

**The Phase 0 manual-edit bug was still live, and worse than documented:**

```javascript
// ui/index.html, before Phase 10
Object.keys(state.rows).forEach(task => state.rows[task] = state.rows[task].map(
  row => row.map((value, index) => index > 0 && String(value) === old ? edit.value : value)))
```

It iterated **every task** and rewrote **any column past the first** whose value matched.
Editing a KIS frame `100` silently rewrote a Q&A frame `100` and every TRAKE event frame
`100`. The cause is that edits were addressed by *value*.

**Stale results could be exported.** `/api/submission/save` took client-supplied rows and
never checked the runtime generation, so results produced against DATA_ROOT A could be
written after the application had switched to root B.

## 2. One Validator

`aic2026/submission_validation.py`. Every path now goes through it:

```text
engine result -> SubmissionRow -> validate_submission -> write_submission_csv
```

| Type | Role |
|---|---|
| `SubmissionRow` | one official row, already reduced to what gets written |
| `SubmissionIssue` | `code`, `message`, `row`, `expected`, `actual` |
| `SubmissionValidationResult` | valid, rows, counts, errors, warnings, generation |
| `SubmissionValidationError` | raised by `validate_submission_or_raise` and by the writer |

Used by the CLI export, the CLI `validate-submission` command, the UI preflight and the
UI export. There is no second implementation of any rule.

## 3. Schemas

### KIS — `video_id, frame_id`

Video id non-empty and pattern-checked; frame a non-negative integer (an int, or a clean
integer string — `1.5`, `abc`, `""` and `None` are refused); exactly one frame; at most
100 rows.

### Q&A — `video_id, frame_id, answer`

As KIS, plus an answer that exists, is not whitespace, is not a non-answer token
(`unknown`, `không xác định`, …), and comes from a status that may be submitted.

| Status | Exportable | Why |
|---|---|---|
| `answered` | yes | a real backend answered |
| `manual` | yes | a human deliberately typed it |
| `abstained` | **no** | the system reporting it has no answer |
| `backend_failed` | **no** | a failure is not an answer |
| `visual_unavailable` | **no** | nothing was seen |
| `mock_backend` | **no** | a non-visual mock cannot answer a question about a video |

### TRAKE — `video_id, frame_id_1 … frame_id_N`

`len(frame_ids) == event_count`, always. Every frame a non-negative integer, no `None`,
no sentinel. `event_count` is inferred only when it was not supplied *and* every row
agrees; disagreement is an error rather than a guess.

**A repeated frame inside one sequence is allowed.** Phase 3.1 established from all
177,321 official map rows that a repeated `frame_idx` is genuine BTC data.

## 4. Duplicate Policy

| Task | Key |
|---|---|
| KIS | `(video_id, frame_id)` |
| Q&A | `(video_id, frame_id, normalized answer)` |
| TRAKE | `(video_id, tuple(frame_ids))` — the whole sequence |

Exact duplicates are removed by default, **keeping the first occurrence** so rank order
survives, and the count is reported. `remove_duplicates=False` turns them into errors
instead. Validation never reorders rows.

## 5. Submission Frame Policy

Export uses `submission_frame_idx`. A locally refined `best_visual_frame_idx` is evidence
and never reaches a CSV. The adapters read the explicit field from the prediction:

```python
coarse = 100, visual = 125  ->  CSV contains 100, and "125" appears nowhere in the file
```

**Official frame IDs are never reconstructed from internal keyframe IDs.** An internal id
`L21_V001/kf_000123` encodes the keyframe *ordinal*; parsing it would submit 123 instead
of the real 5000. A test asserts the exported file contains the official frame and
neither the ordinal nor `kf_`.

## 6. Runtime-Generation Protection

Every result batch records the generation it was produced under. Export compares it with
the active one and refuses a mismatch with `STALE_RESULT_GENERATION` (HTTP 409). Phase 4
deliberately lets an in-flight request finish against its own snapshot; this stops that
snapshot from becoming a submission after the dataset changed underneath it.

## 7. Manual Edit Identity

`aic2026/result_batch.py`. Edits are addressed by identity, never by value:

```text
result_id + row_id                  KIS, Q&A
result_id + row_id + event_index    TRAKE
```

Two rows sharing a frame number are two different rows, so cross-row corruption is
structurally impossible rather than merely avoided. Separate batches per task make
cross-task contamination impossible too.

Every editable cell keeps `original_value`, `current_value`, `edited`, `edited_at`.
Editing a value back to its original clears the flag.

`POST /api/results/<id>/edit` validates immediately — a negative, non-integer or empty
frame is a 400 before it can reach an export — and the export validator runs again
anyway. Frontend validation is never sufficient on its own.

## 8. Q&A Edits Are Row-Local

Phase 6 scoped the answer correction to a video hypothesis, which still rewrote several
rows at once. It is now row-local by default, with an explicit
**"Apply to all rows in this video hypothesis"** checkbox for the propagating case. There
is no hidden propagation. A human-typed answer is marked `manual`, which makes an
otherwise non-exportable abstention exportable — deliberately, and only by that route.

## 9. Undo And Reset

`POST /api/results/<id>/reset` with a `row_id` restores one row; without one it restores
the whole batch. Both are model-level, so nothing depends on a browser reload. Resetting
one row leaves other edits intact.

## 10. Atomic, UTF-8 Export

Validate → normalize → deduplicate → write to a temporary file beside the target → fsync
→ `os.replace`. A validation failure or a crash mid-write leaves no partial CSV and never
replaces a previously good file. Tests assert both, and that no `.tmp` survives.

UTF-8 throughout, using the `csv` module so an answer containing a comma is quoted rather
than becoming an extra column. Vietnamese round-trips exactly: `đỏ`, `có`, `không`,
`người đang chạy`. No header row — the competition format is bare rows, and a header
would become a submitted row.

## 11. Preflight

`POST /api/submission/preflight` does everything the export does except write. An invalid
batch is a normal **200** answer with `valid: false` and the issue list, not an error
status: asking whether the current rows are exportable is a legitimate question with a
legitimate negative answer. The UI runs it before enabling export and again inside it.

## 12. CLI

```powershell
python -m aic2026.cli validate-submission --task kis --input artifacts/submission.csv
python -m aic2026.cli validate-submission --task trake --input trake.csv --event-count 3
```

Exit codes distinguish the failure kinds: `0` valid, `7` structurally invalid, `2` usage
error (including a TRAKE file whose rows disagree with no `--event-count`), `8` I/O or
encoding error. `--out` on `search` now runs the same validator and writes the sidecar.

## 13. Sidecar Report

Every export writes `submission.csv` and `submission.validation.json` containing the
task, row counts before and after, duplicates removed, truncation, errors and warnings,
the runtime generation, config hash, selected-video hash, result id and manual edit
count — plus a note stating that validation is about format, not correctness. No absolute
paths and no secrets.

## 14. UI Truthfulness

- The control with id `gv-ocr` was labelled **Objects** and sent the backend key
  `objects`. It is now `gv-objects`: id, label and key agree.
- Channel availability comes from Phase 9's real status. An unavailable channel's
  checkbox is disabled, unchecked, and annotated **"(No source data)"** — no
  working-looking control that does nothing.
- Result cards lead with **"Submitted <frame>"**; a refined visual frame is shown
  separately and labelled. Editing the preview cannot change the submitted frame.
- An edited row shows an `edited` badge.
- Preflight reports "Valid FORMAT", never "correct".
- The Q&A panel keeps the Phase 6 non-visual mock warning.

The video dropdown remains a **preview selector**: search is collection-wide and the
control never implied otherwise in the backend. Implementing a real `video_filter` would
be a new retrieval feature, which this phase explicitly must not add.

## 15. Real L21 Smoke

`data`, scope `existing_videos`, cache `artifacts/aic2026_index_channels` reused
(`cache_hit=true`), runtime generation 2.

| | Result |
|---|---|
| **KIS** preflight | valid, 100 rows, 0 duplicates |
| KIS export | 200; every row 2 columns, every frame an integer |
| **Q&A** engine preflight | **invalid — `QA_ANSWER_TOO_LONG`**, export refused 422, no file written |
| Q&A manual export | valid, 20 rows, 3 columns, `đỏ` round-tripped exactly |
| **TRAKE** preflight | valid, 34 sequences, every row 4 columns for 3 events |
| TRAKE event edit | event 1 changed; events 0 and 2 unchanged; other sequences unchanged; count preserved |
| **Manual edit** | natural duplicate frame **10950** across rows; editing one left all 99 others unchanged; edit reached the CSV; reset restored everything |
| **Stale generation** | gen 2 → 3, export **409 STALE_RESULT_GENERATION**, no file written |

Sample rows:

```text
L21_V017,10862
L21_V025,11771,đỏ
L21_V006,19941,20234,20730
```

### What the smoke caught

The first run **exported the Q&A batch successfully**, which was wrong. With
`include_media_text=True` the mock backend echoes a video's entire YouTube description as
its "answer", and the validator accepted it because it was non-empty text that was not
literally `unknown`. Four kilobytes of channel boilerplate reached a CSV.

Two rules were added as a result, and both are now tested:

* `QA_ANSWER_TOO_LONG` — an official answer is a short value (≤ 512 characters). A long
  block of text is a backend dumping its input.
* `mock_backend` status — a backend that reports `visual_capable=false` cannot answer a
  question about a video, so its rows are not exportable as engine answers. Only a
  deliberate human edit (`manual`) makes such a row submittable.

The Q&A CSV in the artifact is therefore labelled **MANUAL STRUCTURAL QA EXPORT TEST**:
the answers were typed by hand to exercise the export path, and no claim is made that any
of them is correct.

## 16. Artifact

`artifacts/phase10_submission_smoke/` (gitignored) contains `summary.json`,
`kis_submission.csv` + `.validation.json`, `kis_edited.csv` + `.validation.json`,
`qa_submission.csv` + `.validation.json`, `trake_submission.csv` + `.validation.json`.
No file labels anything correct or incorrect.

## 17. Tests

| File | Covers |
|---|---|
| `tests/test_submission_validation.py` | all three schemas, frame/video validation, duplicate policies and rank preservation, the 100-row cap, Q&A statuses, over-long answers, non-visual backends, generation and task-mismatch checks |
| `tests/test_submission_export.py` | submission-vs-visual frames for all three tasks, no reconstruction from internal ids, atomicity, no partial or clobbered file, UTF-8 Vietnamese, comma and quote handling, no header, deterministic order, sidecar |
| `tests/test_manual_result_edit.py` | row/event/task edit isolation on deliberately repeated frame values, immediate validation, provenance, reset row and batch, manual answers, the bounded store |
| `tests/test_ui_submission_safety.py` | batch registration, preflight valid and invalid, export and sidecar, refusal of a non-submittable Q&A batch, submission-name rejection, DATA_ROOT switch invalidation, HTTP edit isolation across tasks, reset, edits reaching the CSV, no filesystem paths, channel availability |

**1,026 tests, 0 failures, 1 skipped**, up from 909. All offline, deterministic, no
network, no API, no GPU, no ground truth.

## 18. Limitations

- Result batches are in-memory session state, bounded to 24 batches. A server restart
  loses them; there is no persistence layer, deliberately.
- Undo is "reset row" / "reset batch", not a multi-step edit history.
- Frame bounds are not checked against a video's real length: official frame semantics
  may later allow original-video frames, and inventing bounds from incomplete data would
  reject valid submissions. Only `integer >= 0` with a valid video id is enforced.
- A manually entered `video_id` is not re-checked against the active selection; the id
  comes from the engine result and is not editable in the UI.
- `write_submission` in `aic2026/metrics.py` remains for older callers and tests; the new
  path is `write_submission_csv`.
- The Q&A `mock_backend` rule keys on `backend_visual`, so a future visual backend that
  misreports its capability would be trusted.
- The video dropdown is preview-only; no retrieval filter was added.

## 19. Not Started

Phase 11 final integration, the final diagnostics and release packaging, a real
production visual Q&A backend, and any AIC accuracy benchmark all remain pending.
