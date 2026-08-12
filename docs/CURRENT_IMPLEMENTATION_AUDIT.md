# Current Implementation Audit

Audit date: 2026-08-08

Branch: `feat/aic2026-competition-research`

Baseline commit: `61565f7 feat: add AIC 2026 competition pipeline`

## Phase 0 Required Output

### Git Status

```text
## feat/aic2026-competition-research...origin/feat/aic2026-competition-research
```

### Current Branch

```text
feat/aic2026-competition-research
```

### Recent History

```text
61565f7 feat: add AIC 2026 competition pipeline
b692f5c refactor: doi ten kisc_module -> dialogue (bo nhan cuoc thi, nhat quan voi Rewind)
eb21bc1 eval(C1): 51 -> 78 nhan that (KHONG dat 100) + loai 23 nhan TOI VIET SAI
a7c54f1 perf(A9): bo ma tran float trung lap -> -46.6% RAM, va con NHANH HON
2ab3dcc eval: them kiem dinh theo cap - va BAC BO ket luan rerank_pool cua chinh minh
```

### Current Test Result

Commands actually run:

```powershell
.venv\Scripts\python.exe -m compileall -q aic2026 ingestion retrieval evaluation ui tests
.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
compileall: passed
pytest: passed; 1 legacy lazy-import test skipped because torch is installed
```

## Audit Table

| Hang muc | Hien trang code that | Rui ro | File | Viec sua |
|---|---|---|---|---|
| Runtime config | Fixed in commit `6e3ed53`: validated `AppConfig` now drives engine, CLI, UI and benchmark snapshots. | Query-time and build-time configuration share one resolved source and deterministic full config hash. | `aic2026/config.py`, `configs/settings.yaml`, `aic2026/engine.py`, `aic2026/cli.py`, `ui/app.py` | Complete in Phase 1. |
| Cache safety | Fixed in Phase 2: every new cache has an atomic manifest, build fingerprint, data signature, file validation and explicit legacy/stale/corrupt policy. | Incompatible cache is rejected before normal pickle loading; mismatch fields are exposed in CLI/UI/benchmark status. | `aic2026/cache_manifest.py`, `aic2026/engine.py`, `aic2026/cli.py`, `ui/app.py`, `aic2026/benchmark.py` | Complete in commit `feat: add cache manifest and validation`. |
| Dataset alignment | Fixed in Phase 3: exact map/feature equality is required before record creation; mismatch raises `DatasetAlignmentError`. | Production cannot truncate or build a partial mismatch index; diagnostic mode only reports invalid videos. | `aic2026/dataset.py`, `aic2026/dataset_validation.py` | Complete in Phase 3. |
| Dataset validation | Fixed in Phase 3: map/frame, mmap feature, keyframe, object/media JSON and source inventory checks produce atomic structured reports. Phase 3.1 restricts the validation domain to the configured dataset scope. | Builds are gated by `valid_for_index_build`. Under the `L21_*` scope 24 of 29 selected videos are valid; the 5 failures are genuinely missing keyframe packages, and the 844 excluded videos raise no errors. | `aic2026/dataset_validation.py`, `aic2026/cli.py`, `aic2026/engine.py` | Complete in code; the missing `L21_V027`-`L21_V031` keyframes must be downloaded before a real L21 cache rebuild. |
| Dataset scope | Fixed in Phase 3.1: `DatasetScopeConfig` plus `aic2026/dataset_scope.py` select the active video IDs by `fnmatch` pattern; scope is applied before validation and recorded in report, manifest, and benchmark metadata. | A development subset is explicit and reproducible; not-yet-downloaded collections are excluded rather than reported as corrupt data. No AIC batch name is hard-coded in Python. | `aic2026/config.py`, `aic2026/dataset_scope.py`, `aic2026/dataset_validation.py`, `aic2026/dataset.py`, `aic2026/cache_manifest.py`, `aic2026/cli.py` | Complete in Phase 3.1. |
| Data-role model | Fixed in Phase 3.2: `map + CLIP` gate retrieval; keyframe JPEGs are supporting data; per-video `retrieval_valid` / `visual_accessible` / `refinement_ready` / `qa_visual_ready` replace a single valid/invalid verdict. | Not-yet-downloaded keyframe packages no longer invalidate a dataset whose original MP4s are present; `retrieval_valid=true, visual_accessible=false` is a supported state. | `aic2026/dataset_validation.py`, `aic2026/config.py`, `aic2026/dataset.py` | Complete in Phase 3.2. |
| Visual fallback | Fixed in Phase 3.2: `aic2026/frame_provider.py` serves a BTC keyframe JPEG, else decodes the mapped frame from the original MP4, else reports unavailable. Derived frames go only to `dataset.frame_cache_dir`. | Official `frame_idx` is never rewritten by decoder behaviour; official data is never modified; decoding is on demand, never for all Top-100. | `aic2026/frame_provider.py`, `aic2026/engine.py`, `ui/app.py` | Complete in Phase 3.2; Phase 5 refinement will reuse it. |
| Video-backed scope | Fixed in Phase 3.2: scope mode `existing_videos` intersects MP4s on disk with map + CLIP, resolved from `DATA_ROOT` at run time. | The current development subset is reproducible without hard-coded IDs, and the cache fingerprint depends on the resolved IDs, not the mode name. | `aic2026/video_inventory.py`, `aic2026/dataset_scope.py`, `aic2026/cache_manifest.py`, `aic2026/cli.py` | Complete in Phase 3.2. |
| Keyframe identity | Fixed in Phase 3.1: internal ID is `{video_id}/kf_{keyframe_ordinal:06d}`; `frame_idx` is carried separately and is the only submission value. Record schema advanced to v3. | v2's `{video_id}/{frame_idx}` collided on the 192 official videos that repeat a `frame_idx`, silently overwriting `entry.raws` entries. Engine no longer rebuilds the ID from `(video_id, frame_id)` after fusion. | `aic2026/dataset.py`, `aic2026/engine.py`, `aic2026/fusion.py`, `ingestion/schemas.py`, `retrieval/video_engine.py` | Complete in Phase 3.1. |
| DATA_ROOT state | Fixed in Phase 4: one frozen `RuntimeDatasetState` plus a `RuntimeStateManager` own every dataset fact; each request takes one snapshot; activation builds a whole new state and replaces it atomically. | Engine, frame provider, health, video/frame routes, cache status and result URLs can no longer disagree about the active root; a failed switch leaves the previous state serving. | `aic2026/runtime_state.py`, `ui/app.py`, `aic2026/engine.py` | Complete in Phase 4. |
| LocalFrameRefiner usage | **Fixed for KIS in Phase 5**: `AICCompetitionEngine` owns a `LocalFrameRefiner` and calls it inside `search_kis` before Top-100 allocation, under an explicit `always`/`uncertainty`/`disabled` policy with a candidate budget. | Documentation no longer overstates the pipeline: a KIS search really does densely sample bounded MP4 windows and rerank on local visual evidence. Q&A and TRAKE still do **not** refine. | `aic2026/local_refinement.py`, `aic2026/frame_scorer.py`, `aic2026/clip_backend.py`, `aic2026/engine.py`, `ui/app.py` | Complete for KIS in Phase 5; Q&A/TRAKE refinement remains Phase 6+. |
| Visual frame scorer | Fixed in Phase 5: `CLIPFrameScorer` embeds the query once and scores decoded frames in one batched call, sharing the `openai/clip-vit-base-patch32` checkpoint with the text tower through `aic2026/clip_backend.py`. | The refiner no longer needs an injected callable; there is no production fake, and a scorer that cannot load skips refinement with a warning instead of pretending. | `aic2026/frame_scorer.py`, `aic2026/clip_backend.py`, `aic2026/text_encoder.py` | Complete in Phase 5. |
| Frame-ID separation under refinement | Fixed in Phase 5: `coarse_official_frame_idx`, `best_visual_frame_idx` and `submission_frame_idx` are distinct and all reported; `frame_output_policy` defaults to `preserve_coarse`. | A refined frame is evidence only. The submitted row keeps the official mapped `frame_idx` until AIC confirms the semantics of an arbitrary decoded frame. | `aic2026/local_refinement.py`, `aic2026/engine.py`, `ui/app.py`, `ui/index.html` | Complete in Phase 5. |
| `refine_window_s` | Still accepted by `search_trake(..., refine_window_s=6.0)` and still unused, but now documented as such in the docstring rather than silently dead. | The UI control has no effect on TRAKE. Phase 5 wired refinement into KIS only. | `aic2026/engine.py`, `ui/app.py` | Phase 6: wire it into TRAKE refinement or remove the option. |
| MP4 missing fallback | Fixed for KIS in Phase 5: a missing MP4, a decode failure, or an unavailable scorer marks the candidate `applied=false` with a reason and a warning, and the coarse candidate is preserved. | Warnings reach the prediction payload, the search response, and the UI. One failing video never fails a search. | `aic2026/local_refinement.py`, `aic2026/engine.py`, `ui/app.py` | Complete for KIS in Phase 5. |
| Q&A answer count | **Fixed in Phase 6**: `answer_qa` groups the retrieval pool by video, ranks video hypotheses, and answers each one independently from its own evidence. `cross_video_answer_copy_count` and `answer_without_matching_evidence_video_count` are computed from the produced rows and are 0. | An answer can no longer reach another video: a `QAEvidenceBundle` refuses frames from a second video, and a backend answering the wrong video raises. | `aic2026/engine.py`, `aic2026/qa.py` | Complete in Phase 6. |
| Q&A visual backend | Phase 6 defines the `VisualQAAnswerer` contract and three backends with truthful `QAAnswererStatus`. On this machine `auto` resolves to the **mock**, which reports `visual_capable=false`; the local-VLM and API backends report `not_available`. | Health, the search payload, and the UI all label a non-visual backend as such. **Real visual Q&A is still unavailable** — no key, no SDK, no local model. | `aic2026/qa.py`, `ui/app.py`, `ui/index.html` | Contract complete in Phase 6; a real image-capable backend is still not installed. |
| Expected answer type | **Fixed in Phase 6**: UI -> request -> `answer_qa` -> backend prompt -> `normalize_answer(expected_type=...)` -> prediction. An unsupported type is a `400 INVALID_ANSWER_TYPE`. | Vietnamese `khong` is resolved to `0` or `no` by declared type, and left alone when no type is declared. Refusals short-circuit ahead of type handling so "khong co mo ta" can never become a confident "no". | `ui/app.py`, `aic2026/engine.py`, `aic2026/qa.py` | Complete in Phase 6. |
| Q&A evidence selection | **Fixed in Phase 6**: `select_evidence_frames` picks strongest first, then a temporally diverse frame, then the opposite side, and only relaxes the diversity gap rather than under-filling. | The old selector seeded `{first, last, nearest}`, so one- or two-frame evidence returned window boundaries instead of the retrieved evidence. | `aic2026/qa.py` | Complete in Phase 6. |
| Q&A manual answer edit | Scoped in Phase 6: the correction box applies to the SELECTED video hypothesis only. | Rewriting every row would re-create the cross-video answer copying Phase 6 removed. A full per-row edit model is still Phase 12. | `ui/index.html` | Scoped fix in Phase 6; full model pending Phase 12. |
| TRAKE output length | **Fixed in Phase 7**: a `TrakeAlignment` always holds one step per query event, a deterministic recovery pass fills gaps from the same video's candidates for that event, and anything still incomplete is discarded. `TrakePrediction` raises `TrakeStructureError` unless it carries exactly `event_count` frames. | A short or mislabelled TRAKE row is no longer representable. The cost is recall: 65 of 77 alignments were discarded on the real smoke rather than emitted short. | `aic2026/trake.py`, `aic2026/engine.py`, `aic2026/metrics.py`, `ui/app.py`, `ui/index.html` | Complete in Phase 7. |
| TRAKE event labelling | **Fixed in Phase 7**: the API returns each step's own `event_index` / `event_label`, and the UI labels from that rather than from the loop position. | The old code zipped a compacted step list against the full event list, so every event after a gap displayed the wrong text. | `ui/app.py`, `ui/index.html` | Complete in Phase 7. |
| TRAKE metric truncation | **Fixed in Phase 7**: `trake_r_score` checks the frame count against the ground-truth event count before scoring and returns 0 for a mismatch. | `zip` silently truncated, so a 3-frame row answering 4 events scored as partially correct instead of invalid. | `aic2026/metrics.py` | Complete in Phase 7. |
| TRAKE method | Named honestly since Phase 7: `beam_dp` in config, alignment, prediction, API and UI; `exact_dp` is rejected by config validation and a test scans the sources for false claims. | Beam-pruned DP is not exact DP; calling it so would misdescribe the algorithm. | `aic2026/trake.py`, `aic2026/config.py` | Naming complete in Phase 7; exact DP and k-best remain Phase 8. |
| TRAKE `refine_window_s` | Phase 7 makes the inert parameter truthful: every response carries `refinement.applied=false` and `status="not_implemented_phase_7"`. | It is accepted for API compatibility but changes nothing, and a test asserts that. | `aic2026/engine.py`, `ui/app.py` | Truthful in Phase 7; TRAKE refinement remains Phase 8+. |
| TRAKE alignments/video | **Fixed in Phase 8**: `align_video_k_best_beam` enumerates distinct paths from the retained search states, and `select_diverse_alignments` keeps up to `max_alignments_per_video` after a deterministic near-duplicate filter. | Real smoke: 62 videos contributed more than one sequence, and complete sequences went from 12 to 187 on the same four queries. That is structural yield, not accuracy. | `aic2026/trake.py`, `aic2026/engine.py` | Complete in Phase 8. |
| TRAKE candidate depth | **Fixed in Phase 8**: adaptive expansion re-retrieves only the events reaching fewer videos than the best-covered event, through the EXISTING fusion retriever, bounded by `candidate_depth_max` and stopped by `target_complete_video_hypotheses`. | The Phase 7 smoke showed 59 missing positions with no candidate at all at depth 40. Videos with full event coverage went from 12 to 66. No candidate is fabricated. | `aic2026/engine.py`, `aic2026/config.py` | Complete in Phase 8; independent retrieval channels remain Phase 9. |
| TRAKE final scoring | **Fixed in Phase 8**: `alignment_objective` recomputes the score from the steps an alignment actually holds, and `score_video_hypothesis` is now explicitly a pre-filter only. | Phase 7 ranked on the beam's running score, which went stale after recovery replaced a step. | `aic2026/trake.py` | Complete in Phase 8. |
| TRAKE local refinement | **Integrated in Phase 8**: `aic2026/trake_refinement.py` refines a few complete sequences event by event, each against its OWN text, reusing `LocalFrameRefiner` / `FrameProvider` / `FrameScorer` / the shared `CLIPBackend`. | Off by default (~10 s per query on CPU). Bounded by a hard per-query frame ceiling. Refined frames are evidence only: the submitted frame stays the coarse official mapped frame_idx. | `aic2026/trake_refinement.py`, `aic2026/engine.py`, `ui/app.py` | Integrated in Phase 8. |
| TRAKE `refine_window_s` | **Fixed in Phase 8**: it now selects the local refinement window, and a test asserts a wider value samples more frames. | No longer a dead parameter. With refinement off the response reports `status: disabled` and `frames_decoded: 0`. | `aic2026/engine.py`, `aic2026/trake_refinement.py` | Complete in Phase 8. |
| TRAKE exact DP | Phase 8 adds `align_video_exact_dp` as a bounded **test oracle** only; `trake.alignment_method` still rejects `exact_dp` and the shipped search reports `beam_dp`. | The objective is Markovian in (event, last candidate, matched), so the exact optimum is polynomial. Tests assert a wide beam matches it. | `aic2026/trake.py` | Reference complete in Phase 8; k-best exact search is not the production path. |
| Top-100 duplicates | KIS `video_aware_top100` chan duplicate `(video_id, frame_id)`. TRAKE chan duplicate `(video_id, frame_ids)` nhung co the thieu event. Q&A khong co validator truoc export. | Duplicate/schema loi co the lot ra CSV, dac biet sau manual edit. | `aic2026/ranking.py`, `aic2026/metrics.py`, `ui/index.html` | Phase 11: submission validator bat buoc truoc export. |
| Objects/metadata role | **Fixed in Phase 9**: `aic2026/retrieval_channels.py` makes CLIP, BM25, objects and metadata independent generators; the pool is their union with per-channel provenance. | Real smoke: 508 candidates introduced exclusively by objects and 347 by metadata across four KIS queries. A prior audit found the shipped cache had `load_objects: false` and `include_media_text: false`, so the pipeline was effectively CLIP-only and BM25's corpus was 7,800 empty sentinels. | `aic2026/retrieval_channels.py`, `aic2026/engine.py`, `aic2026/config.py` | Complete in Phase 9. |
| Optional text channels | Phase 9 constructs OCR, ASR and frame-caption channels so their absence is REPORTED rather than hidden: all three are `available=false` with `no_populated_source_data` on the real L21 scope. | Nothing is substituted for them: object labels are not OCR, and media metadata is not a frame caption. | `aic2026/retrieval_channels.py`, `ui/app.py` | Honest status in Phase 9; the sources genuinely do not exist. |
| Vietnamese query handling | **Fixed in Phase 9**: `aic2026/query_normalization.py` keeps the original query for CLIP and supplies folded/expanded views to the lexical channels, with a versioned bilingual vocabulary. | Accented and unaccented forms of one query overlap 83% of their candidate pool. Negation and temporal markers are preserved and marked instead of being normalized away. | `aic2026/query_normalization.py`, `aic2026/retrieval_channels.py` | Complete in Phase 9. |
| Metadata fields | Fixed in Phase 3: media title/description/tags/channel/duration are separate from `frame_caption`, OCR, ASR and objects; `llm_caption` is only a frame-caption compatibility alias. | Provenance is preserved in prefixed searchable text; independent metadata candidate generation remains pending Phase 9. | `aic2026/dataset.py`, `ingestion/schemas.py`, `ingestion/build_records.py` | Schema separation complete in Phase 3; channel generation remains Phase 9. |
| Vietnamese normalization module | Delivered in Phase 9 as `aic2026/query_normalization.py`; see the row above. | `fusion.normalize_label` remains for label canonicalization and now shares the same folding rules. | `aic2026/query_normalization.py` | Complete in Phase 9. |
| Submission export | **Fixed in Phase 10**: `aic2026/submission_validation.py` is the single validator for KIS/Q&A/TRAKE, used by the CLI export, the CLI `validate-submission` command, the UI preflight and the UI export. Writes are atomic and UTF-8 with a sidecar report. | The CLI previously validated nothing at all; the UI had only the Phase 7 TRAKE length check. Export now uses `submission_frame_idx` and refuses stale runtime generations. | `aic2026/submission_validation.py`, `aic2026/cli.py`, `ui/app.py` | Complete in Phase 10. |
| Manual frame edit | **Fixed in Phase 10**: `aic2026/result_batch.py` addresses edits by `result_id + row_id` (plus `event_index` for TRAKE), never by value. | The old code rewrote every matching numeric value across every task, so editing a KIS frame corrupted Q&A and TRAKE rows. Verified on a real natural duplicate (frame 10950 in several rows). | `aic2026/result_batch.py`, `ui/index.html`, `ui/app.py` | Complete in Phase 10. |
| Q&A manual answer edit | **Fixed in Phase 10**: row-local by default, with an explicit opt-in checkbox to apply an answer to a whole video hypothesis. | Phase 6 scoped it to the hypothesis, which still rewrote several rows at once. A human-typed answer is marked `manual`, which is what makes it exportable. | `ui/index.html`, `aic2026/result_batch.py` | Complete in Phase 10. |
| UI unsupported-channel controls | **Fixed in Phase 10**: `gv-ocr` renamed to `gv-objects` so id, label and backend key agree, and Phase 9 channel availability disables any control whose source is empty, annotated "(No source data)". | No working-looking checkbox that does nothing. | `ui/index.html`, `ui/app.py` | Complete in Phase 10. |
| Manual frame edit | JS input frame edit thay moi gia tri frame trung nhau trong moi task, va loop qua tat ca task rows. | Sua mot result co the sua nham row/task/video khac co cung frame_id. | `ui/index.html` | Phase 12: result-scoped edit model, backend validate, undo/reset, validator before export. |
| UI checkbox semantics | UI checkbox `gv-ocr` label da doi thanh Objects nhung id van `gv-ocr`; backend chi doc `objects`, bo qua ASR/caption trong index endpoint. | Control ASR/Caption co the vo tac dung; id/ten gay nham. | `ui/index.html`, `ui/app.py` | Phase 12: bo control chua support hoac wire backend dung field. |
| Video dropdown | UI co dropdown video/index nhung search backend hien search collection, khong ap dung selected video filter. | Control vo tac dung lam nguoi dung hieu sai pham vi search. | `ui/index.html`, `ui/app.py`, `aic2026/engine.py` | Phase 12: chon search full collection thi bo dropdown, hoac implement filter that. |
| Diagnostics without labels | Co benchmark logger va artifact contract, nhung chua co diagnostics tong hop khong can ground truth. | Khi chua co labels, kho phat hien structural regression ngoai unit tests. | `evaluation/*` | Phase 13: `diagnostics.py`, `synthetic_scenarios.py`, `runtime_report.py`. |
| Human inspection export | Chua co HTML inspection export theo run id. | Reviewer khong co cong cu xem thu cong nhanh ma khong gan dung/sai. | `evaluation/*`, `artifacts/inspection` | Phase 14: export `results.html`, CSV, config/runtime, human note columns. |
| Documentation truthfulness | README/PHASE_REPORT da ghi blocker labels, nhung van can cap nhat sau cac fix tren; old benchmark docs co tu accuracy cho legacy synthetic/old labels. | De tao an tuong accuracy/novelty khi chua co AIC labels. | `README.md`, `TASKS.md`, `CHATGPT_PROJECT_CONTEXT.md`, `docs/*`, `evaluation/bench_*` | Phase 15: add `NO_GROUND_TRUTH_STATUS.md`, align docs with actual code. |

## Direct Answers To Required Verification Questions

1. Since Phase 5, `LocalFrameRefiner` is called end-to-end by `search_kis`. Q&A and TRAKE still do not call it.
2. `refine_window_s` is accepted by `AICCompetitionEngine.search_trake` and UI, but is still not used: Phase 5 integrated refinement into KIS only.
3. Since Phase 6, Q&A answers each top video hypothesis independently from that video's own evidence; the pre-Phase-6 behaviour of copying one answer across every returned candidate is gone, and two diagnostics assert it stays gone.
4. Since Phase 7, TRAKE cannot output fewer frame IDs than the number of events: missing alignments remain explicit `missing` steps, recovery is attempted from the same video's candidates for that event, and anything still incomplete is discarded rather than shortened.
5. TRAKE is beam-pruned dynamic programming, not exact exhaustive DP, because states are truncated by `beam_width`. Since Phase 7 the config rejects `exact_dp` and the method is reported as `beam_dp` everywhere.
6. Since Phase 8, one video can produce several distinct complete TRAKE sequences (`k_best_per_video` enumerated, `max_alignments_per_video` kept after diversity filtering).
7. KIS Top-100 de-duplicates `(video_id, frame_id)`; TRAKE de-duplicates full sequence; Q&A/export still lacks a shared validator.
8. New caches require schema-v1 manifests; legacy `entry.pkl`-only caches are rejected by default and stale/corrupt fields are reported.
9. The `aic2026:` section in `settings.yaml` is the validated runtime source wired into engine, CLI, UI and benchmark snapshots.
10. UI custom DATA_ROOT does not update all routes; health/list/video serving still use global `DATA_ROOT`.
11. Fixed in Phase 10: manual edits address `result_id + row_id` (plus `event_index` for TRAKE), so a frame edit can no longer affect another row or another task by matching its value.
12. Objects and metadata currently rerank/fuse dense/sparse candidates; they are not independent candidate generators.
13. Loader rejects map/feature count mismatch with `DatasetAlignmentError`; `inspect-data` reports invalid videos without building an index.
14. Q&A default answerer does not visually inspect images unless an API-backed answerer is selected; default no-key path is mock text reasoning.
15. Production CLIP fallback is explicit in encoder status and CLI stderr; UI health exposes encoder warning only after index load, but does not clearly label mock Q&A answerer status.

## File-Level Change Plan

| Area | Files to add/change |
|---|---|
| Config | Add `aic2026/config.py`; change `configs/settings.yaml`, `aic2026/engine.py`, `aic2026/cli.py`, `ui/app.py`, tests `test_config_runtime.py`. |
| Cache | Add `aic2026/cache_manifest.py`; change `aic2026/engine.py`, `aic2026/cli.py`, `ui/app.py`, benchmark logger, tests `test_cache_manifest.py`. |
| Dataset | Change `aic2026/dataset.py`; add inspect-data CLI/report; tests `test_dataset_validation.py`. |
| Refinement | Change `aic2026/local_refinement.py`, `aic2026/engine.py`, `aic2026/ranking.py`, `aic2026/trake.py`, `aic2026/qa.py`; tests `test_refinement_integration.py`. |
| Q&A | Change `aic2026/qa.py`, `aic2026/engine.py`, `retrieval/vqa_module.py`, UI answerer status; tests `test_qa_per_hypothesis.py`. |
| TRAKE | Change `aic2026/trake.py`, `aic2026/engine.py`; tests `test_trake_output_schema.py`, `test_trake_kbest.py`. |
| Retrieval channels | Add `aic2026/retrieval_channels.py`; change engine/fusion/cache manifest/health; tests `test_retrieval_channels.py`. |
| Vietnamese query | Add `aic2026/query_normalization.py`; wire fusion/channels/QA/TRAKE retrieval; tests `test_query_normalization_vi.py`. |
| Submission | Add `aic2026/submission_validation.py`; change `aic2026/metrics.py`, `aic2026/cli.py`, `ui/app.py`; tests `test_submission_validation.py`. |
| UI | Change `ui/app.py`, `ui/index.html`; add result-scoped edit state/API/undo/export validation; tests `test_manual_frame_edit.py`, `test_data_root_state.py`. |
| Diagnostics | Add `evaluation/diagnostics.py`, `evaluation/synthetic_scenarios.py`, `evaluation/runtime_report.py`, `evaluation/inspection_export.py`; tests `test_diagnostics.py`, `test_human_inspection_export.py`. |
| Docs | Update `README.md`, `TASKS.md`, `CHATGPT_PROJECT_CONTEXT.md`; add `docs/NO_GROUND_TRUTH_STATUS.md`. |

## Three Highest Risks

1. **Submission-invalid logic can pass silently**: dataset mismatch truncation, missing TRAKE events, copied Q&A answers, and no export validator can all produce a syntactically plausible but semantically wrong CSV.
2. **Runtime/config/cache mismatch can invalidate benchmark runs**: YAML is not the source of truth, cache has no manifest, and UI custom DATA_ROOT does not propagate to all endpoints.
3. **Documentation can overstate current capability**: since Phase 5 local refinement is genuinely wired into KIS, but it is KIS only, and multi-frame visual Q&A still exists as an interface with a mock answerer rather than a fully integrated visual backend.
## Phase 1 Status Update

- Runtime config: fixed in Phase 1 commit `feat: wire validated runtime configuration`.
- `aic2026/config.py` now provides validated `AppConfig`, deterministic `config_hash`, resolved snapshots and CLI/UI/engine wiring.
- `configs/settings.yaml` now keeps legacy top-level sections for older tests and adds `aic2026:` as the runtime source for the competition pipeline.
- Cache safety: fixed in Phase 2 commit `feat: add cache manifest and validation`; see `docs/PHASE_2_CACHE_MANIFEST.md`.
- Dataset strict validation: fixed in Phase 3; real fast inspection found invalid required keyframes/frame IDs, so the real cache was not rebuilt.

## Phase 3.1 Status Update

- Dataset scope: fixed in Phase 3.1 commit `feat: support scoped AIC development datasets`; see `docs/PHASE_3_1_DATASET_SCOPE_AND_MAPPING.md`.
- The active development scope is `L21_*`, expressed in configuration/CLI only; no AIC batch name is hard-coded in Python.
- Map CSV semantics were inspected from all 873 real files via `tools/inspect_map_schema.py`; duplicate `frame_idx` is valid official data and is now informational, while strictly decreasing `frame_idx` is a hard error.
- Internal keyframe IDs moved to `{video_id}/kf_{keyframe_ordinal:06d}` and `AIC_RECORD_SCHEMA_VERSION` advanced to 3, so record-v2 caches are stale by design.
- Cache fingerprint and manifest now carry `dataset_scope`, `selected_video_count`, and `selected_video_ids_hash`.
- Real `L21_*` inspection: 873 discovered, 29 selected, 844 excluded, 24 valid / 5 invalid, `valid_for_index_build=false` because `L21_V027`-`L21_V031` keyframes are incomplete. No cache was rebuilt.

## Phase 3.2 Status Update

- Video-backed development: fixed in Phase 3.2 commit `feat: support video-backed AIC development`; see `docs/PHASE_3_2_VIDEO_BACKED_DEVELOPMENT.md`.
- Official data roles are now encoded correctly: the video is the competition data, and keyframes/objects/CLIP features/metadata are supporting data.
- Real local inventory: 29 MP4s, all `L21`, all readable, 3.147 GiB, no duplicates; all 29 have map + CLIP, 25 have a JPEG folder, 5 need MP4 fallback for some or all frames.
- `dataset.validation.require_keyframe_images` was removed and replaced by `require_visual_source` (default false); loading the old key raises `ConfigError`.
- Scope mode `existing_videos` plus CLI `--scope-existing-videos` resolve the video-backed subset from disk; the manifest stores the resolved IDs and their hash.
- Real inspection with that scope: 29/29 retrieval-valid, visual-accessible, and refinement-ready; 0 invalid; `valid_for_index_build=true`.
- Real MP4 fallback verified on `L21_V027/kf_000003` (JPEG genuinely absent): exact frame 300 decoded at 1280x720, official `frame_idx` preserved, derived frame written only under `artifacts/`.
- Development cache `artifacts/aic2026_index_existing_videos` built and reported `valid=true, legacy=false, stale=false`. The legacy `artifacts/aic2026_index` was left untouched.
- Advanced local refinement: still pending Phase 5, not implemented.

## Phase 4 Status Update

- Dynamic DATA_ROOT state: **fixed in Phase 4** commit `fix: unify runtime dataset state`; see `docs/PHASE_4_DYNAMIC_DATA_ROOT.md`.
- `aic2026/runtime_state.py` adds a frozen `RuntimeDatasetState` and a locked `RuntimeStateManager`; `app.extensions["aic_runtime_state"]` holds the single authority.
- Module-level `DATA_ROOT` / `AIC_CACHE_DIR` / `SUBMISSION_DIR` remain only as construction-time defaults and are never read while serving a request.
- Activation is atomic: engine, frame provider and state are built first, and a failure leaves the previous state active with `active_state_changed: false`.
- Every published state carries a `generation`; result URLs embed it and a superseded request gets `409 STALE_RESULT_GENERATION`.
- `AICCompetitionEngine.dataset_identity()` plus `verify_engine_identity()` assert that engine, frame provider and routes describe the same root.
- `POST /api/dataset/inspect` is read-only; only `index`/`index_folder` may replace state.
- Video/frame routes reject traversal and out-of-scope IDs, and results contain only logical `/api/...` URLs.
- Cache directories follow the dataset: explicit wins, the configured cache serves the configured root, and any other root derives a distinct directory.
- Real smoke on the 29-video `existing_videos` development set reused the existing cache (`cached: true`) and verified JPEG serving, MP4 fallback, a KIS query, and stale-generation rejection.
- Answers 10 and 11 in the verification list above are superseded: UI custom DATA_ROOT now updates all routes.
- Q&A per-video hypotheses, TRAKE k-best, retrieval channels, submission validation and manual editing all remain pending.
- Q&A per-video hypothesis: pending Phase 6.
- TRAKE complete/k-best output: pending Phase 7-8.

## Phase 5 Status Update

- **LocalFrameRefiner runtime integration: fixed for KIS in Phase 5**, commit
  `feat: integrate query-conditioned local refinement`; see
  `docs/PHASE_5_LOCAL_REFINEMENT.md`.
- Global retrieval is unchanged: the BTC precomputed CLIP index is still the only
  recall layer, nothing was re-embedded, and no index was replaced.
- `aic2026/frame_scorer.py` adds a `FrameScorer` protocol and `CLIPFrameScorer`;
  `aic2026/clip_backend.py` owns one `openai/clip-vit-base-patch32` checkpoint shared by
  the text tower and the image tower, so the two spaces are provably comparable and the
  weights are not loaded twice.
- `FrameProvider` gained `video_metadata`, `decode_frames` (one capture for a whole
  window) and `get_video_frame`; the refiner no longer has its own OpenCV code, and its
  keyframe-record API is unchanged.
- Bounded sampling: `top_hypotheses` -> region dedup -> `candidate_budget` ->
  `max_frames` per candidate. Worst case 5 x 32 frames per query; no full-video decode.
- Trigger policy `always` / `uncertainty` / `disabled`; the uncertainty heuristic is the
  relative gap between the two best candidate regions against `margin_threshold`, fully
  logged in a `RefinementDecision`.
- Reranking is `coarse_fusion_score + rerank_alpha * (best_visual - coarse_visual)`;
  all components are preserved and visible, and untouched candidates keep their coarse
  score.
- Frame-ID safety: `frame_output_policy` defaults to `preserve_coarse`, so the submitted
  `frame_id` remains the official mapped `frame_idx` even when a different local frame
  scores higher. `decoded_frame` exists but is never the default.
- Runtime state: the state adopts the engine's frame provider and
  `verify_engine_identity()` rejects a refiner paired with another generation's provider.
- Real L21 smoke (existing cache **reused, not rebuilt**), real CLIP on CPU: 4 queries,
  20 candidates refined, 631 frames decoded and scored, 0 decode/scorer failures,
  refinement p50 14.4 s, 13 of 20 refined frames differ from the coarse frame, mean
  absolute offset 0.909 s. The shipped uncertainty policy triggered on 2 of 4 queries.
- **No accuracy claim**: there is no AIC ground truth, no threshold was tuned against
  quality, and no result is labelled correct or incorrect.
- Refinement is wired into **KIS only**. Q&A per-video hypotheses, TRAKE k-best and
  TRAKE local refinement remain unimplemented.

## Phase 6 Status Update

- **Q&A one-answer-for-many-predictions bug: fixed in Phase 6**, commit
  `fix: ground QA answers per video hypothesis`; see `docs/PHASE_6_GROUNDED_QA.md`.
- The unit of answering is one **video hypothesis**: candidates are grouped by video,
  ranked, and each video is answered from its own evidence with its own backend call.
- `QAEvidenceBundle` raises if it is handed a frame from another video, and a backend
  that answers about the wrong video raises. Isolation is structural, not conventional.
- `aic2026/qa.py` adds the `VisualQAAnswerer` contract plus `MockTextQAAnswerer`,
  `LocalVlmQAAnswerer` and `ApiVqaAnswerer`, each with a truthful `QAAnswererStatus`.
- **Real visual Q&A is NOT available on this machine**: no `ANTHROPIC_API_KEY`, no
  `anthropic` SDK, no local VLM. `auto` resolves to the non-visual mock, which says so in
  health, in the search payload, and in the UI. No visual smoke was run and none is
  claimed.
- `expected_answer_type` now works end to end, and Vietnamese `khong` is resolved by the
  declared type (`0` for number, `no` for boolean) and left alone without one.
- The real smoke caught a fabrication: `"khong co mo ta"` was becoming a confident `"no"`
  / `"0"`. Unknown answers now short-circuit ahead of type normalization, so the mock
  abstains honestly on all 20 real hypotheses instead.
- Evidence selection is strongest-first with temporal diversity; the old
  `{first, last, nearest}` seeding is gone.
- Q&A has its own refinement budget (off by default; 1 region, 12 frames per video), not
  Phase 5's 5x32 KIS budget.
- Real L21 structural probe (existing cache reused, not rebuilt): 4 questions x 5 video
  hypotheses, **5 distinct answers per question**, `cross_video_answer_copy_count = 0`,
  `answer_without_matching_evidence_video_count = 0`, 284 evidence frames, 0 backend or
  decode failures.
- Manual answer correction is scoped to the selected video hypothesis.
- **No accuracy claim**: no AIC ground truth exists and nothing was tuned against
  correctness.
- TRAKE k-best/refinement, retrieval channels, query-normalization redesign, submission
  validation and the Phase 12 manual-edit architecture all remain pending.

## Phase 7 Status Update

- **TRAKE incomplete-frame output bug: fixed in Phase 7**, commit
  `fix: enforce complete TRAKE event alignments`; see
  `docs/PHASE_7_TRAKE_STRUCTURAL_CORRECTNESS.md`.
- `TrakeAlignment` always holds exactly one step per query event; a skipped event is an
  explicit `missing` step, so an event can no longer vanish and the ones after it can no
  longer shift.
- `TrakePrediction` raises `TrakeStructureError` unless it carries exactly `event_count`
  frames from one video with no missing positions. The invariant is enforced in the
  dataclass, in `to_complete_prediction`, in `align_trake`, in `engine._from_trake`, and
  in `write_submission`.
- Deterministic recovery fills a missing event from that event's own candidates in the
  same video, bounded by its temporal neighbours. It never inserts a sentinel, frame 0, a
  neighbouring event's frame, or a candidate from another video; an order-breaking
  recovery is discarded wholesale.
- Ordering is judged on timestamps, not frame-ID uniqueness, so a repeated official
  `frame_idx` stays valid data.
- `trake_r_score` no longer `zip`-truncates: a row with the wrong frame count scores 0.
- `refine_window_s` is reported as `not_implemented_phase_7` rather than looking
  functional.
- Real L21 structural smoke (existing cache reused, not rebuilt): 4 queries, 77 video
  hypotheses, 65 incomplete alignments **discarded**, **12 complete predictions**, every
  row carrying exactly N frames, and `malformed_prediction_count` /
  `wrong_event_count_prediction_count` / `cross_video_step_count` all **0**. Recovery
  fired 0 times because 59 missing positions had no candidate for that event in that
  video and 42 had candidates that violated the temporal constraints — now visible in the
  diagnostics rather than requiring a probe.
- **No accuracy claim**: no AIC ground truth exists and nothing was tuned against
  correctness.
- k-best, exact DP, TRAKE local refinement and semantic verification all remain pending.

## Phase 8 Status Update

- **One-alignment-per-video limitation: fixed in Phase 8**, commit
  `feat: add k-best TRAKE temporal refinement`; see
  `docs/PHASE_8_TRAKE_KBEST_AND_REFINEMENT.md`.
- `align_video_k_best_beam` enumerates distinct paths from the states the beam already
  retains (widened to `beam_width * k`), deduplicated by keyframe signature and fully
  deterministic. It is not the search re-run with perturbed inputs.
- `align_video_exact_dp` is a bounded **test oracle** proving a wide beam reaches the
  exact objective on small cases. The shipped method is still `beam_dp` and
  `alignment_method` still rejects `exact_dp`.
- Adaptive candidate expansion deepens only the events reaching fewer videos than the
  best-covered event, through the existing CLIP+BM25 retriever, capped by
  `candidate_depth_max`. No new retrieval channel; that is Phase 9.
- `alignment_objective` rescores from the chosen steps, separating the pre-alignment
  video filter from final ranking.
- `aic2026/trake_refinement.py` refines a few complete sequences per query, event by
  event, each against its own text, with a hard `max_frames_per_query` ceiling. Order
  safety comes from a joint ordered DP over the already-sampled frames
  (`local_ordered_refinement`), not from a post-hoc patch.
- `refine_window_s` genuinely selects the local window and is no longer a dead parameter.
- Frame-ID policy unchanged: `submission_frame_idx == coarse_official_frame_idx`, and
  `apply_refinement` asserts the row is untouched.
- Real L21 smoke, same four queries as Phase 7, cache reused: complete sequences
  **12 → 187**, videos with full event coverage **12 → 66**, 62 videos contributing more
  than one sequence, 12 events expanded (depth 40 → 120/300), 2,220 new candidates. With
  refinement on: 12 sequences refined, 39 events, 312 frames decoded and scored, 0
  failures, p50 9.9 s / p95 12.7 s on CPU.
- All Phase 7 structural counters remain **0** across all three passes, and every
  returned row still has exactly N frames.
- **No accuracy claim**: complete-sequence yield is a structural count, not accuracy and
  not recall. Nothing was tuned against quality.
- Independent retrieval channels, the query-normalization redesign, the global submission
  validator and the Phase 12 manual-edit architecture all remain pending.

## Phase 9 Status Update

- **Objects/metadata independent candidate generation: fixed in Phase 9**, commit
  `feat: add multi-channel AIC retrieval`; see
  `docs/PHASE_9_MULTI_CHANNEL_RETRIEVAL.md`.
- **Vietnamese query normalization: fixed in Phase 9** (`aic2026/query_normalization.py`).
- A real-data audit found the shipped cache was built with `load_objects: false` and
  `include_media_text: false`, so `searchable_text()` was empty for every record: the
  BM25 corpus held 7,800 empty sentinels and the pipeline was effectively **CLIP-only**,
  while 7,800 object JSONs and 29 complete media-info files sat unused on disk.
- Seven channels exist. On real L21: clip 7,800 · bm25 7,800 · objects 7,709 (362 labels)
  · metadata 29 available; **ocr, asr and frame captions are genuinely absent** and report
  `available=false` with `no_populated_source_data`.
- Real KIS smoke: **508 candidates introduced exclusively by objects, 347 exclusively by
  metadata**, 857 over the CLIP+BM25 baseline; union up to 2,172 from 1,200.
- Vietnamese: accented vs unaccented forms of one query overlap **83%** of their pool;
  negation is preserved and excluded from positive object matching.
- **Honest regression**: TRAKE full-event coverage fell from 50 to 42 videos and complete
  sequences from 140 to 116, because a more diverse pool spreads the fixed per-event
  `top_k` slice across more videos. Not tuned away, because tuning it would mean tuning
  against imagined quality.
- Coarse retrieval 54-58 ms to 82-127 ms: bounded, indices built once per engine.
- A NEW cache `artifacts/aic2026_index_channels` was built (189 s); the CLIP-only cache is
  untouched and still valid. `--allow-stale-cache` was never used.
- **No accuracy claim**: candidate coverage is a structural count, not recall.
- The global submission validator, the Phase 12 manual-edit architecture and deployment
  packaging all remain pending.

## Phase 10 Status Update

- **Submission validator, export safety, the manual frame-edit cross-row bug, Q&A
  row-local edits and UI channel truthfulness: all fixed in Phase 10**, commit
  `fix: validate submissions and isolate result edits`; see
  `docs/PHASE_10_SUBMISSION_AND_UI_SAFETY.md`.
- `aic2026/submission_validation.py` is the one validator for all three official tasks and
  is used by every export path. The CLI previously validated nothing.
- `aic2026/result_batch.py` addresses manual edits by identity, not value. Verified on a
  real natural duplicate: frame 10950 appeared in several KIS rows, and editing one left
  all 99 others unchanged.
- Export uses `submission_frame_idx`; a refined `best_visual_frame_idx` can never reach a
  CSV, and official frames are never parsed out of internal `kf_` ids.
- Stale runtime generations are refused with `409 STALE_RESULT_GENERATION` and no file is
  written; verified on real data across a generation change.
- Writes are atomic and UTF-8, with a `.validation.json` sidecar. Vietnamese answers
  round-trip exactly and answers containing commas are quoted.
- The real smoke caught a genuine hole: with media text loaded, the non-visual mock echoed
  a whole YouTube description as an "answer" and it passed validation. Two rules were
  added — `QA_ANSWER_TOO_LONG` and a `mock_backend` status that is not exportable — so a
  non-visual backend's output can only be submitted after a deliberate human edit.
- `gv-ocr` renamed to `gv-objects`; unavailable channels are disabled and labelled
  "(No source data)".
- **No accuracy claim**: validation is about FORMAT. Nothing here says an answer or a
  frame is correct, and no AIC ground truth exists.
- Phase 11 final integration, release packaging, a production visual Q&A backend and any
  accuracy benchmark all remain pending.

## Phase 11 Status Update — FINAL AUDIT CLASSIFICATION

Date: 2026-08-12 · Version `0.11.0-aic2026`

This section supersedes the status columns of the audit table above, which is kept as the
historical record. Every item originally raised in Phase 0 is classified here as
**FIXED**, **PARTIAL**, **OPEN** or **BLOCKED_EXTERNAL**. Nothing is marked fixed on the
strength of a plan; each FIXED row names the phase that closed it.

### FIXED

| Item | Closed in | Evidence |
|---|---|---|
| Runtime configuration is the single source of truth | Phase 1 | Validated `AppConfig` + deterministic `config_hash` drive engine, CLI, UI, benchmark. |
| Cache compatibility | Phase 2 | Manifest, fingerprint, data signature; legacy/stale/corrupt rejected; `allow_stale_cache: false` is a readiness FAIL if enabled. |
| Dataset alignment and validation | Phase 3 | Exact map/feature equality enforced; `valid_for_index_build` gates builds. Final pass: 7,800 map rows == 7,800 vectors, 0 invalid videos. |
| Dataset scope | Phase 3.1/3.2 | `existing_videos` resolves 29 of 873 videos from disk; ids and their hash recorded in manifest and profile. |
| Keyframe identity collision | Phase 3.1 | `{video_id}/kf_{ordinal:06d}`; final pass shows 0 duplicate internal ids against 10 duplicate official `frame_idx` values. |
| Visual source fallback | Phase 3.2 | JPEG → MP4 decode → unavailable; official `frame_idx` never rewritten; 0 videos with no visual source. |
| DATA_ROOT runtime state | Phase 4 | One frozen state per activation, atomic replacement, generation-stamped results. |
| `LocalFrameRefiner` actually used (KIS) | Phase 5 | Called by `search_kis` under an explicit trigger policy with a bounded budget. |
| Frame-ID separation under refinement | Phase 5 | Three distinct identities; `preserve_coarse` default; only `submission_frame_idx` reaches a CSV. |
| Q&A answers copied across videos | Phase 6 | Per-hypothesis answering; `cross_video_answer_copy_count: 0` on every real run since. |
| Expected answer type reaching normalization | Phase 6 | UI → request → engine → prompt → `normalize_answer`; unsupported types are `400`. |
| Q&A evidence selection | Phase 6 | Strongest-first with temporal diversity, relaxing the gap instead of under-filling. |
| TRAKE output length | Phase 7 | N events → exactly N frames, always; incomplete alignments discarded, never shortened. |
| TRAKE event labelling | Phase 7 | Each step carries its own `event_index` / `event_label`. |
| TRAKE metric truncation | Phase 7 | Length checked before scoring; a mismatch is 0, not partial credit. |
| Honest algorithm naming | Phase 7 | `beam_dp` everywhere; `exact_dp` rejected by config; a test scans sources for false claims. |
| TRAKE alignments per video | Phase 8 | k-best enumeration + diversity filter. |
| TRAKE candidate depth | Phase 8 | Adaptive expansion of under-covered events only, hard-bounded. |
| TRAKE `refine_window_s` dead parameter | Phase 8 | Now selects the refinement window; a test asserts a wider value samples more frames. |
| Objects/metadata as independent generators | Phase 9 | `retrieval_channels.py`; real smoke showed 508 object-exclusive and 347 metadata-exclusive candidates. |
| Vietnamese query handling | Phase 9 | `query_normalization.py`; final pass confirms accent-folded equivalence, negation marking, safe degenerate input. |
| Submission export validation | Phase 10 | One validator on every export path; atomic UTF-8 CSV plus sidecar report. |
| Manual frame edit corrupting other rows | Phase 10 | Edits addressed by `result_id + row_id`; verified live — editing one row of 100 left the other 99 untouched. |
| Q&A manual answer edit scope | Phase 10 | Row-local by default, with an explicit opt-in to apply to a whole hypothesis. |
| UI unsupported-channel controls | Phase 10 | `gv-objects` renamed to match its backend key; unavailable channels disabled and labelled. |
| Stale-generation exports | Phase 10 | `409 STALE_RESULT_GENERATION`; verified live in Phase 11 after a dataset re-activation, with no file written. |
| Single documented way to start the system | Phase 11 | `aic2026.cli serve`, gated by the readiness preflight. |
| Reproducibility identity | Phase 11 | `SystemProfile` + `identity()` + generated release manifest. |
| Diagnostics without labels | Phase 11 | `competition-check`, `run_competition_smoke.py`, `run_ablation.py` — all structural, none requiring ground truth. |
| Human inspection export | Phase 11 | `artifacts/final_release_smoke/results.html` per run, with no correct/incorrect column. |
| Documentation truthfulness | Phase 11 | README rewritten, `KNOWN_LIMITATIONS.md` added, this classification written, GT guard enforced in code. |

### PARTIAL

| Item | State | What is missing |
|---|---|---|
| Local visual refinement | Implemented and wired for KIS and TRAKE, **disabled by default** | Enabled only under a CUDA device or a larger time budget; whether it *helps* is unknown without ground truth. |
| Q&A end-to-end | Retrieval, grounding and evidence selection complete and verified | The answering step has no visual backend here; engine answers are non-submittable by design. |
| TRAKE optimality | Complete, deterministic, k-best beam search | Beam-pruned, so not provably optimal. Exact DP exists as a bounded test oracle only. |
| Optional text channels | Constructed, enabled and honestly reporting `available: false` | The OCR/ASR/caption sources genuinely do not exist in this data. |
| Dataset coverage | 29 videos / 7,800 frames, fully validated | The other 844 discovered videos have no local MP4; scaling behaviour is unmeasured. |

### OPEN

| Item | Why it is still open |
|---|---|
| UI video dropdown does not filter search | The control implies a scope it does not apply. Either filter or remove it. Cosmetic-to-misleading, not a correctness bug. |
| `evaluation/run_eval.py` legacy harness | Operates on synthetic mock data with self-generated labels. Kept for history; its numbers must never be quoted as system quality. |
| Development server | `serve` uses Flask's development server — fine for one operator on localhost, not a deployment. |
| Browser-level UI verification | No browser backend in this environment; API behaviour is verified, visual/mobile rendering is not. |

### BLOCKED_EXTERNAL

| Item | Blocker | Consequence |
|---|---|---|
| Any accuracy, recall or Final Score for this system | **No official AIC ground truth exists in this repository.** | Semantic metrics refuse to run (`GroundTruthRequired`). Ablation variants are described, never ranked. No parameter has been tuned against quality. |
| Production visual Q&A | No API key, no local VLM checkpoint, no SDK; none may be downloaded automatically. | Engine Q&A answers are non-submittable; a human answer is required and is marked `manual`. |
| Official frame-ID semantics for an arbitrary decoded frame | Unanswered by the organisers. | `frame_output_policy` stays `preserve_coarse`; a refined frame remains evidence only. |
| Full-collection behaviour | The remaining 844 videos' MP4s are not present locally, and downloading more AIC data is out of scope. | Latency, memory and candidate counts are known for 29 videos only. |

### Superseded verification answers

Answers 7, 10, 11, 12, 14 and 15 in "Direct Answers To Required Verification Questions"
above are superseded: a shared submission validator now exists (Phase 10), UI DATA_ROOT
changes propagate to every route (Phase 4), manual edits are identity-addressed
(Phase 10), objects and metadata are independent generators (Phase 9), and the mock Q&A
answerer is labelled as non-visual in health, search payloads, the system profile,
readiness and the UI (Phases 6 and 11). Answers 1–6, 8, 9 and 13 stand, with the Phase 8
update that `refine_window_s` is no longer inert.

