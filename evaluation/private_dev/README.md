# Private Development Ground Truth

**Status: CURRENT.** These labels are written by a human, for this project, to make
local experiments measurable. They are **not** AIC ground truth and never become it.

> Every report built from these files carries
> **`PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE`**.
> A number measured here describes this small local set and nothing else.

---

## Why this exists

The repository can measure how much work a query costs but not whether the answer is
right, because no official labels exist. Every semantic question — does refinement help,
does the R1 controller help, is one channel mix better — is unanswerable without labels,
and the honest response so far has been to refuse to answer it.

A small, carefully annotated private set changes that for *development*. It does not
change it for the competition: the organizers' collection, queries and judgements are
different, and a good private score is evidence that a change is worth keeping, not
evidence of a competition result.

## Rules that are enforced in code

| Rule | Where |
|---|---|
| `label_source` must be `private_dev` | `evaluation/ground_truth.py` |
| `annotated_by` must be a human; `system`, `model`, `auto`, `clip`, `vlm`, `generated`, `self`, `prediction` are refused | same, file header *and* every row |
| A row marked `EXAMPLE_ONLY` is parsed for shape and excluded from every scored set | `GroundTruthEntry.is_example` |
| No labels → `GROUND_TRUTH_REQUIRED` | `aic2026.metrics.require_ground_truth` |
| Reports carry the private banner | `evaluation.ground_truth.report_header` |

**Nothing here may be generated from the system's own predictions.** A label produced by
the thing being measured is circular; it would make every experiment agree with itself.

## Files

| File | Task |
|---|---|
| `kis.json` | Textual KIS |
| `qa.json` | Grounded Q&A |
| `trake.json` | Ordered-event TRAKE |

Each holds one task, declares `label_source: private_dev`, and starts with a single
`EXAMPLE_ONLY` template row. Replace or delete the template; adding real rows beside it
is fine.

## Schema

Frame numbers are **official `frame_idx` values from `map-keyframes`** — not seconds, not
keyframe ordinals, not decoder positions. Intervals are inclusive and are never silently
normalized.

### KIS

```json
{
  "query_id": "kis_0001",
  "query": "a person pushes a bicycle across a crossing",
  "video_id": "L21_V004",
  "frame_ranges": [[1200, 1260]],
  "label_source": "private_dev",
  "annotated_by": "your name",
  "split": "development",
  "notes": ""
}
```

Several intervals are allowed when the same moment genuinely recurs. Do not add a second
interval to raise the chance of a hit.

### Q&A

```json
{
  "query_id": "qa_0001",
  "event_description": "a vehicle stops at the intersection",
  "question": "What colour is the vehicle?",
  "video_id": "L21_V002",
  "frame_ranges": [[300, 360]],
  "answers": ["red", "đỏ"],
  "answer_type": "color",
  "label_source": "private_dev",
  "annotated_by": "your name"
}
```

`answers` must list every acceptable surface form. Matching is normalized (case, accents,
punctuation), not exact, but a form nobody wrote down cannot match.

### TRAKE

```json
{
  "query_id": "trake_0001",
  "events": ["a person approaches a vehicle",
             "the person enters the vehicle",
             "the vehicle moves away"],
  "video_id": "L21_V003",
  "event_frame_ranges": [[[100, 108]], [[210, 219]], [[330, 352]]],
  "label_source": "private_dev",
  "annotated_by": "your name"
}
```

One interval list per event, in event order, all from the same video. The event count and
the interval-list count must match exactly — TRAKE is scored per event.

## Annotation protocol

1. **Watch the original MP4.** Only the 29 videos with a local MP4 may be annotated;
   pixels are what makes an interval defensible.
2. **Identify the video and the moment.** If you cannot say why a frame is the answer,
   it is not a label yet.
3. **Find the smallest defensible semantic interval.** The interval is the part where the
   described thing is actually visible — not the surrounding shot.
4. **Record official start and end `frame_idx`.** Use the annotation helper
   (`tools/annotate_private_gt.py`) or read `data/map-keyframes/<video>.csv` directly.
5. **For Q&A, write every acceptable answer**, including the Vietnamese form when it is
   natural, and set `answer_type`.
6. **For TRAKE, label each event independently and keep the order.** Official TRAKE
   intervals are frequently **under ten frames**; a wide interval is a wrong label that
   happens to be easy to hit.

**Do not widen an interval to make retrieval succeed.** The point of a label is to be
right, and a label tuned to the system measures the system against itself.

## Query-creation policy

Queries must resemble the organizers' tasks and must be answerable from what is on
screen:

- **Use observable semantic content.** "A woman in a red jacket crosses in front of a bus"
  is a query; "the video about the market" is a topic.
- **Never leak identifiers.** No video id, file name, timestamp, or verbatim YouTube
  title — unless that text is genuinely visible in the scene.
- **Q&A needs both halves**: an event description that locates the moment, and a question
  whose answer requires seeing it. A question answerable from the event description alone
  is not testing retrieval.
- **TRAKE needs a real ordered sequence** from one video, where the order is part of what
  makes the answer correct.

## Split discipline

Every row carries `split`, one of:

| Split | Use |
|---|---|
| `development` | look at freely; tune against this |
| `holdout` | **do not look at while tuning** |

Reserve roughly 20–30% as `holdout` once enough labels exist. Load one split at a time:

```python
from evaluation.ground_truth import load_private_dev
dev = load_private_dev(split="development")
```

Tuning against the same queries you report on measures memorisation, not quality.

## How many labels

Long-term targets, **not** quotas to fill: ~40–60 KIS, ~20–30 Q&A, ~15–20 TRAKE.

A smaller, honest set beats a padded one. **Never fabricate an annotation to reach a
number** — a wrong label is worse than a missing one, because it produces a confident
result instead of a refusal.

## Scoring

Private labels are scored with the **same** implementation as official ones — `R@1`,
`R@5`, `R@20`, `R@50`, `R@100`, and the Final Score as their mean. There is no separate
private metric, because a separate metric would not be comparable to anything.

```powershell
.venv\Scripts\python.exe tools\evaluate_private_gt.py --config configs\competition.yaml
```

With no real labels it exits refusing, with `GROUND_TRUTH_REQUIRED`. That is correct
behaviour, not a bug.
