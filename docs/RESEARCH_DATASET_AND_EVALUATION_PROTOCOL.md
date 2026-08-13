# Research Dataset and Evaluation Protocol

**Status: CURRENT.** Branch `research/aic2026-metric-budget`. Covers what this project
searches, what it may measure, and the rules that keep a local number from being mistaken
for an AIC result.

> **No official AIC ground truth exists in this repository.** Anything measured here is
> measured against locally annotated labels and is reported as
> **`PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE`**.

---

## 1. Two coverages, deliberately different

The architecture separates what can be *searched* from what can be *seen*. Conflating
them silently deletes searchable data.

| Capability | Requires | Videos here |
|---|---|---|
| **Retrieval coverage** — the coarse index | valid `map-keyframes` + valid CLIP feature | **873** |
| **Visual coverage** — preview, local refinement, visual Q&A | keyframe JPEG or local MP4 | **29** |

844 videos have complete supporting data — map, CLIP, objects, media-info — and no local
MP4. Coarse retrieval never needed one: it scores the BTC CLIP vectors the organizers
supplied. Excluding those videos from the index was a development convenience that cost
96.7% of the searchable collection.

Two scope modes express this:

- `retrieval_ready` — canonical ID ∩ valid map ∩ valid CLIP. Global retrieval.
- `existing_videos` — the above ∩ a local MP4. A *visual* development scope.

Patterns still apply on top of either. Neither hard-codes a video count or a collection
name; both resolve from `DATA_ROOT` at run time, so the selected-ID hash identifies the
real dataset.

### What an absent MP4 actually costs

Nothing in retrieval, and exactly three things elsewhere:

| Capability | Without pixels |
|---|---|
| Coarse KIS / TRAKE retrieval | **works**; official `frame_idx` preserved |
| Frame preview | `available: false`, no exception |
| Local visual refinement | skipped; candidate keeps its coarse score |
| Visual Q&A | no visual call, no fabricated answer |

Tests pin each of these (`tests/test_full_retrieval_scope.py`).

## 2. Why private ground truth is necessary

Without labels the project can measure cost and refuse to measure quality. That is the
honest position, and it makes every design question unanswerable: does refinement help,
does the R1 controller help, is one channel mix better than another.

A small, carefully annotated private set makes *development* decisions measurable. It
does not make competition claims possible.

| | Official AIC GT | Private development GT |
|---|---|---|
| Author | organizers | a human on this project |
| Collection | the full competition collection | 29 locally video-backed videos |
| Queries | organizers' | ours, written to resemble theirs |
| What a score means | the competition result | evidence a change is worth keeping |
| Report label | `OFFICIAL AIC GROUND TRUTH` | `PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE` |

Both are scored by the **same** implementation. A separate private metric would not be
comparable to anything, so there is none.

## 3. Annotation protocol

Only the 29 MP4-backed videos may be annotated: an interval is defensible when someone
watched it.

1. Watch the original MP4.
2. Identify the video and the moment.
3. Choose the **smallest defensible semantic interval** — the part where the described
   thing is actually visible, not the surrounding shot.
4. Record official **`frame_idx`** start and end from `map-keyframes` (not seconds, not
   keyframe ordinals). `tools/annotate_private_gt.py frame-at <video> --seconds N` does
   the lookup.
5. For Q&A, record every acceptable answer form, including the Vietnamese one where
   natural, and set `answer_type`.
6. For TRAKE, label each event independently and preserve the order.

**Do not widen an interval to make retrieval succeed.** Official TRAKE intervals are
frequently **under ten frames**. A wide interval is a wrong label that happens to be easy
to hit, and it will make a bad system look good.

### Query-creation policy

- Use observable semantic content, not topics or file facts.
- Never leak video id, file name, timestamp or a verbatim YouTube title, unless that text
  is visibly in the scene.
- Q&A needs an event description *and* a question whose answer requires seeing the event.
- TRAKE needs a genuinely ordered sequence from one video.

## 4. Schema

Full field lists and worked examples: [`evaluation/private_dev/README.md`](../evaluation/private_dev/README.md).

Enforced in `evaluation/ground_truth.py`:

| Rule | Consequence if broken |
|---|---|
| `label_source: private_dev` on the file | refused |
| `annotated_by` is a human (`system`/`model`/`auto`/`clip`/`vlm`/`generated`/`self`/`prediction` refused), checked on the file **and** every row | refused as circular |
| Frame ranges non-empty, `0 <= start <= end` | refused |
| Q&A answers non-empty | refused |
| TRAKE event count == interval-list count, order preserved | refused |
| A row whose own fields say it is another task | refused |
| Duplicate `query_id` | refused |
| Row marked `EXAMPLE_ONLY` | parsed for shape, **excluded from every scored set** |

Intervals are never silently normalized, and a second interval on a TRAKE event is kept
in the record even though the official scorer consumes one per event.

Frame ids are checked against the local MP4 inventory **only when that inventory is
readable**; when it is not, the helper reports no verdict rather than inventing bounds.

## 5. Split discipline

Every row carries `split`: `development` or `holdout`. Load one at a time:

```python
load_private_dev(split="development")   # tune against this
load_private_dev(split="holdout")       # do not look at while tuning
```

Reserve roughly 20–30% as holdout once enough labels exist. Tuning and reporting on the
same queries measures memorisation.

Long-term targets, not quotas: ~40–60 KIS, ~20–30 Q&A, ~15–20 TRAKE. **Never fabricate an
annotation to reach a number.** A smaller honest set is strictly better, because a wrong
label produces a confident answer where a missing one produces a refusal.

## 6. Official metrics, unchanged

`R@1`, `R@5`, `R@20`, `R@50`, `R@100`; Final Score is their mean.

| Task | A row scores when |
|---|---|
| KIS | video matches **and** the submitted frame is inside a labelled interval |
| Q&A | the above **and** the answer matches (normalized, not exact) |
| TRAKE | video matches; score is the fraction of events whose frame is in that event's interval; a wrong-length row scores 0 |

`aic2026/rank_utility.py` describes the same cutoff geometry for allocation purposes and
is not a second metric.

## 7. Three-axis evaluation

Quality, speed and cost are reported separately and never blended.

| Axis | Fields |
|---|---|
| Quality | `R@1`, `R@5`, `R@20`, `R@50`, `R@100`, `Final Score` |
| Speed | `p50_latency_ms`, `p95_latency_ms`, `warm_mean_ms`, first-query latency |
| Cost | `decoded_frames_per_query`, `image_embeddings_per_query`, `vlm_calls_per_query`, `vlm_images_per_query`, `text_encoder_calls_per_query`, `channel_calls_per_query`, `cost_proxy_per_query`, `peak_rss_mb`, device |

`evaluation/pareto.py` reports **Pareto dominance** — no worse on every compared axis and
strictly better on at least one — and **never** names a winner. Variants whose cache
fingerprints or query counts differ are reported as not comparable rather than compared.

Quality columns exist only when labels were supplied. Without them the report says so.

## 8. No-ground-truth rules

With no real labels the system may report: candidate structure, latency, compute counts,
memory, determinism, failure counts, human-inspection artifacts.

It may **not** report: accuracy, recall, precision, Final Score, "better retrieval",
"improved correctness", SOTA.

`tools/evaluate_private_gt.py` exits **9** with `GROUND_TRUTH_REQUIRED` when the label
files hold only templates. Verified: with the shipped templates it refuses and names the
three example rows it ignored.

## 9. Reproducibility protocol

Three named configurations, defined in `evaluation/experiment_manifest.py` and **not
tuned**:

| Name | Meaning |
|---|---|
| `B0_RELEASE` | the frozen release `7dfe06e` (`aic2026-competition-ready`); run from a worktree at that tag — the difference is code, not config |
| `B0_CLEAN` | R0: dead UI/config removed, caching and prewarm allowed, rankings verified identical |
| `R1_ADAPTIVE` | R1 controller enabled; experimental |

Every run records: git commit and dirty flag, project version, config hash, scope mode,
cache directory and fingerprint, selected-video-IDs hash, query-set hash, GT set hash and
provenance, encoder/scorer/Q&A backend and device, and the hard compute budgets in force.

`comparable(a, b)` refuses to compare two runs whose index, labels, query set or dataset
selection differ, and names which one differs.

**No semantic comparison runs until at least one real human annotation exists.**

## 10. Threats specific to this dataset work

- The private set covers 29 videos while the index covers 873. A KIS query labelled
  against a small subset is easier than the same query against the full collection, so
  private scores are not comparable across index sizes — the manifest records the cache
  fingerprint precisely so that mistake is detectable.
- One annotator means one person's judgement of "smallest defensible interval".
- Q&A answer matching is normalized string matching; an unlisted valid phrasing scores 0.
- The 873-video index has no visual capability for 844 of its videos, so refinement and
  visual Q&A experiments remain confined to the 29-video subset.
