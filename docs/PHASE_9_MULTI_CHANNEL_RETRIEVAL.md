# Phase 9 Multi-Channel Retrieval And Vietnamese Query Normalization

Before Phase 9 the candidate pool was `UNION(CLIP top-k, BM25 top-k)`, and objects and
metadata only *rescored* whatever that union already contained. A frame strongly indicated
by its detector labels, or a video strongly indicated by its title, could never enter the
pool: the signal existed but had no way in. Phase 9 turns each signal into an independent
candidate generator and merges them with provenance intact.

> **No accuracy claim.** This repository has no AIC ground truth. Every number below is
> *candidate coverage* — a structural count of what entered the pool. It is **not**
> recall, precision, or accuracy, and no weight was tuned against imagined quality.

Phase 9 does **not** implement the global submission validator, the manual-edit redesign,
deployment packaging, or supervised weight tuning.

## 1. What The Real Data Audit Found

The first thing Phase 9 did was measure the real cache rather than assume. The result
reframed the whole phase:

| Source | On disk | Loaded in the pre-Phase-9 cache |
|---|---|---|
| CLIP features | 7,800 | **7,800** |
| Object detections | 29 folders / **7,800 JSON files** | **0** |
| Media metadata | **29/29** title + description + tags | **0** |
| BM25 frame text | — | **0 documents with real text** |
| OCR / ASR / frame captions | none | 0 |

The shipped config had `load_objects: false` and `include_media_text: false`, so
`searchable_text()` returned `""` for every record and the BM25 corpus was 7,800 copies of
the empty-document sentinel. **The pipeline was effectively CLIP-only**, and objects and
metadata — which are fully present on disk — were contributing nothing at all.

Getting the object and metadata channels working therefore required a build-time change
and a new cache, not just a query-time one.

## 2. Channel Abstraction

`aic2026/retrieval_channels.py`:

| Type | Role |
|---|---|
| `RetrievalChannel` | protocol: `status()` and `search(query, *, top_k)` |
| `ChannelCandidate` | keyframe, video, frame_idx, raw + normalized score, rank, scope, evidence |
| `ChannelResult` | one channel's answer, its status, and its latency |
| `ChannelStatus` | enabled / available / record_count / index_type / scope / reason |
| `PooledCandidate` | one canonical keyframe plus **every** channel that proposed it |
| `ChannelUnion` | the pool plus coverage diagnostics |
| `RetrievalChannelRegistry` | asks every usable channel and merges the results |

A channel can never introduce an unknown ID: `_EntryChannel._candidate` resolves every
keyframe against `entry.raws` and drops anything it cannot find.

## 3. The Channels

| Channel | Scope | Index | Real L21 status |
|---|---|---|---|
| `clip` | frame | HNSW dense | **available, 7,800** |
| `bm25` | frame | BM25 over frame text | **available, 7,800** (after the rebuild) |
| `objects` | frame | inverted label postings | **available, 7,709** |
| `metadata` | video | per-video token sets | **available, 29** |
| `ocr` | frame | token sets | **unavailable — `no_populated_source_data`** |
| `asr` | frame | token sets | **unavailable — `no_populated_source_data`** |
| `caption` | frame | token sets | **unavailable — `no_populated_source_data`** |

OCR, ASR and frame captions genuinely do not exist in this dataset. They are constructed
so `/health` can report them, they say `available=false` with a reason, and they return
nothing. Nothing is substituted for them: object labels are not OCR, and media metadata is
not a frame caption.

### Objects

An inverted index from normalized label token to `(keyframe_id, best confidence)`, built
once per engine. The score is `term coverage × summed confidence` — deliberately simple
and transparent. Detector confidence orders frames *within* this channel; it is not
treated as a calibrated relevance probability.

### Metadata

Video-level token sets over `media_title`, `media_description`, `media_tags` and
`media_channel`. A hit is scoped `video` and expands into `frames_per_video` (default 8)
evenly spread frames of that video — claiming frame-level grounding from a title would be
a provenance error, and dumping every frame would drown the pool.

## 4. Query Representation

`aic2026/query_normalization.py`. The rule it exists to enforce: **the original query is
never replaced**.

```python
QueryRepresentation(
    original, normalized_whitespace, lowercase, accent_folded,
    tokens_original, tokens_folded,
    negated_tokens, temporal_markers, number_terms,
    expanded_terms, vocabulary_version,
)
```

`dense_query` returns the user's own words, so CLIP keeps seeing natural language;
`lexical_terms` and `object_terms` give the folded and expanded views to BM25, objects and
metadata.

### Accent folding

`NFD` decomposition, drop combining marks, plus the one Vietnamese letter that does not
decompose:

```text
người  -> nguoi      xe máy -> xe may
đường  -> duong      Đà Nẵng -> Da Nang
```

NFC and NFD inputs fold identically, so an upstream normalization difference cannot change
retrieval.

### Bilingual vocabulary

A small, explicit, versioned map (`QUERY_VOCABULARY_VERSION`) from Vietnamese phrases to
the English words detectors actually emit, and the reverse. Longest phrase wins, so
`xe máy` → `motorcycle` beats a bare `xe` → `car`. Every expansion records `term`,
`source` and `origin`. No external API is called and no translation model is used.

### Negation

`không`, `not`, `without` and friends are preserved, marked, and scoped over the following
few tokens. A negated term is excluded from `object_terms`, so `"không có xe"` cannot
become a positive query for vehicles — verified end to end for both the object and
metadata channels.

### Temporal markers

`trước`, `sau đó`, `then`, `next` … are preserved and exposed as `temporal_markers` rather
than filtered out as stopwords. Phase 9 adds no temporal reasoning; it just stops
destroying the signal.

## 5. Union, Normalization And Fusion

Channel score spaces are unrelated — a CLIP cosine, a BM25 sum, a detector confidence, a
metadata overlap ratio. `normalize_channel_scores` puts each channel on `[0, 1]` before
they meet:

* `rank` (default): scale-free and immune to outliers;
* `minmax`: where raw magnitudes matter, with zero-variance handled explicitly (all
  candidates agree → all get 1.0, never 0/0) and negative scores supported.

A non-finite score raises rather than propagating.

Fusion then consumes the *normalized* evidence through the existing `fuse_candidates`,
so the adaptive/weighted/RRF modes are unchanged. Every pooled candidate keeps which
channels found it, at which rank, with which raw and normalized scores, and that
provenance reaches the prediction as `evidence.channels`.

## 6. Double Counting

Media metadata reaches BM25 through `searchable_text` *and* the metadata channel. Rather
than silently counting it twice at full weight, both paths are normalized per channel
before fusion, and the overlap is explicit in the diagnostics (`overlap_with_bm25` per
channel). The provenance is honest: a candidate found by both shows both channels. A clean
frame-only BM25 corpus would require changing `searchable_text`, which is a build-time
schema change with cache consequences; it is documented rather than done silently.

## 7. Cache

`load_objects`, `include_media_text`, `include_ocr`, `include_asr`, `include_captions` were
already inside the cache fingerprint, so enabling objects and metadata correctly
invalidated the old cache. Phase 9 adds `channel_schema_version` to both the build options
and the manifest, so a change in how channel sources are shaped also invalidates.

Query-time settings — channel `enabled` flags, `top_k`, `normalization` — are deliberately
**not** in the fingerprint. Turning a channel off or searching deeper never forces a
rebuild; a test asserts exactly that.

A new cache was built rather than overwriting the working one:

```text
artifacts/aic2026_index_existing_videos   untouched (CLIP-only, still valid)
artifacts/aic2026_index_channels          new: objects + metadata, 189 s build
```

`--allow-stale-cache` was never used.

## 8. Real L21 Inventory

`artifacts/retrieval_channel_inventory.json`, 29 videos / 7,800 frames:

```text
clip            7,800 records
bm25            7,800 documents with real text
objects         7,709 frames with labels, 362 distinct labels
                top: clothing 4822, person 3943, human face 3593, man 3399,
                     woman 2102, tree 1592, skyscraper 1009, car 682 ...
metadata        29/29 videos with title, description and tags
ocr             0
asr             0
frame captions  0
```

## 9. Real KIS Smoke

Union sizes, A = CLIP only, B = CLIP + BM25 (the pre-Phase-9 equivalent), C = every
available channel:

| Query | A | B | C | +C over B | exclusive objects / metadata |
|---|---|---|---|---|---|
| `một người đang đi bộ` | 1,200 | 1,912 | **2,172** | +260 | 91 / 167 |
| `a person riding a motorcycle` | 1,200 | 1,750 | **2,167** | +417 | 417 / 0 |
| `car on the road` | 1,200 | 1,498 | **1,678** | +180 | 0 / 180 |
| `people sitting indoors` | 1,200 | 1,200 | 1,200 | 0 | 0 / 0 |

Totals: **508 candidates introduced exclusively by objects**, **347 exclusively by
metadata**, 857 introduced over the CLIP+BM25 baseline. The fourth query has no vocabulary
or label overlap, so no channel could contribute — reported rather than smoothed over.

Top-100 composition now draws on `clip | bm25 | objects | metadata` where those channels
matched.

## 10. Vietnamese Smoke

| Query | Folded | Union | Paired with | Overlap |
|---|---|---|---|---|
| `một người đi bộ` | `mot nguoi di bo` | 2,007 | `a person walking` (2,327) | 597 (0.30) |
| `xe máy trên đường` | `xe may tren duong` | 2,082 | `motorcycle on the road` (1,935) | 1,109 (0.57) |
| `người ngồi trên ghế` | `nguoi ngoi tren ghe` | 2,211 | `person sitting on a chair` (2,259) | 845 (0.38) |
| `người đi vào cửa` | `nguoi di vao cua` | 2,041 | `nguoi di vao cua` (2,034) | **1,694 (0.83)** |

The accented / unaccented pair overlaps 83 %, which is the point of folding: the same
query written either way reaches nearly the same pool. Vietnamese/English pairs overlap
30–57 %, which is expected and not a defect — CLIP sees the original wording in each case
and the two languages are genuinely different dense queries.

Expansions are visible: `xe máy` → `motorcycle`, `ghế` → `chair`, `cửa` → `door`.

## 11. Real TRAKE Smoke — An Honest Regression

| | A: CLIP only | C: all channels |
|---|---|---|
| Videos with full event coverage (3 queries) | **50** | **42** |
| Complete sequences returned | **140** | **116** |
| malformed / cross-video counters | 0 | 0 |

**Multi-channel retrieval made TRAKE's full-event coverage go down on these queries, and
that is reported rather than buried.** The cause is structural: TRAKE takes a fixed
`per_event_top_k` slice of the *fused* list per event. A more diverse pool spreads those
40 slots across more videos, so fewer videos end up holding all three events within their
slice. Candidate coverage went up; per-event within-video depth went down.

The lever is `trake.per_event_top_k` or `trake.target_complete_video_hypotheses`, and it
was deliberately **not** turned: choosing a value to make this number look better would be
tuning against imagined quality with no ground truth to justify it. Phase 8's adaptive
expansion did not fire because coverage stayed above its target of 12.

## 12. Performance

| | A: CLIP only | C: all channels |
|---|---|---|
| Coarse retrieval per query | 54–58 ms | 82–127 ms |

Roughly 2× for four channels instead of one, and still ~0.1 s — nowhere near the
multi-second regression the phase had to avoid. Every channel index is built once per
engine and reused: a test asserts the object postings and metadata documents are the same
objects across queries. `channel_search_ms`, `fusion_ms` and `total_coarse_ms` are
reported per query.

## 13. Diagnostics

Per query: `candidate_union_size`, `unique_videos`, `exclusive_candidates` per channel,
and per channel `candidates_returned`, `unique_candidates_introduced`, `overlap_with_clip`,
`overlap_with_bm25`, `search_ms`, `available`, `reason`. Plus `channel_search_ms`,
`fusion_ms`, `total_coarse_ms` and the query representation.

All of it is **candidate coverage**. A test asserts no diagnostic key is named recall,
precision, accuracy or map.

## 14. Tests

| File | Covers |
|---|---|
| `tests/test_query_normalization_vi.py` | original preserved, folding, NFC/NFD stability, `đ`, negation preserved and scoped, temporal markers, deterministic bilingual expansion with provenance, unknown words untouched |
| `tests/test_retrieval_channels.py` | honest availability, the union, provenance, ID resolution, rank/minmax normalization incl. zero-variance and negative scores, non-finite rejection, determinism, per-channel toggles, the BM25 probe regression |
| `tests/test_object_channel.py` | **object-only candidates enter the pool; disabled they do not**, Vietnamese→label matching, confidence ordering, thresholds, coverage over confidence, duplicate labels, negation, unavailability, single index build, engine-level proof, survival through fusion |
| `tests/test_metadata_channel.py` | metadata-only videos enter the pool, video scope, bounded spread frames, ordering, negation, absent metadata, metadata is not a frame caption |
| `tests/test_channel_fusion.py` | fusion over normalized evidence, absent channels contribute zero, KIS/Q&A/TRAKE all consume the pool, TRAKE expansion deepens channels, build-time vs query-time cache behaviour, manifest, health |

**909 tests, 0 failures, 1 skipped**, up from 811. All offline, deterministic, no API, no
network, no GPU.

## 15. Limitations

- **TRAKE full-event coverage fell** (50 → 42 videos, 140 → 116 complete sequences) for
  the reason in §11. Untuned by choice.
- BM25 was reported unavailable by an early version of this phase because `BM25Okapi` does
  not retain its corpus; the probe now reads `doc_freqs`. It is worth stating that the
  first real smoke was run with a wrong availability reading, and the numbers above are
  from after the fix.
- Media metadata still reaches both BM25 and the metadata channel. Per-channel
  normalization limits the effect and the overlap is reported, but a clean frame-only BM25
  corpus is a build-time schema change left undone.
- The bilingual vocabulary is ~45 entries chosen by hand. It covers common AIC concepts
  and nothing more; extending it is a deliberate, tested act.
- Negation scoping is a fixed three-token window, not a parser.
- The object channel cannot represent negation at all, so it declines negated terms rather
  than modelling them.
- Metadata frame representatives are evenly spread in time, with no visual selection.
- The new cache is a second artifact; the CLIP-only cache remains valid and unchanged, so
  disk use grew.
- Objects add ~137 frames with no labels above threshold (7,709 of 7,800).

## 16. Not Started

The global submission validator, the Phase 12 manual-edit redesign, deployment packaging,
supervised weight tuning, and any accuracy benchmark all remain pending.
