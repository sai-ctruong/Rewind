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
| TRAKE alignments/video | `joint_trake_alignment` tao 1 prediction moi video hypothesis (`align_video_dp` mot lan/video). | Khong tan dung Top-100 bang nhieu sequence trong cung video. | `aic2026/trake.py`, `aic2026/engine.py` | Phase 8: k-best alternatives va diversification. |
| Top-100 duplicates | KIS `video_aware_top100` chan duplicate `(video_id, frame_id)`. TRAKE chan duplicate `(video_id, frame_ids)` nhung co the thieu event. Q&A khong co validator truoc export. | Duplicate/schema loi co the lot ra CSV, dac biet sau manual edit. | `aic2026/ranking.py`, `aic2026/metrics.py`, `ui/index.html` | Phase 11: submission validator bat buoc truoc export. |
| Objects/metadata role | Dense CLIP va sparse BM25 tao union candidate; object/metadata chi tinh score tren union do. Object/metadata chua la independent generator. | Object-only hoac metadata-only frame co the khong bao gio duoc xet. | `aic2026/engine.py`, `aic2026/fusion.py`, `aic2026/dataset.py` | Phase 9: `retrieval_channels.py` voi union candidate tu tung channel. |
| Metadata fields | Fixed in Phase 3: media title/description/tags/channel/duration are separate from `frame_caption`, OCR, ASR and objects; `llm_caption` is only a frame-caption compatibility alias. | Provenance is preserved in prefixed searchable text; independent metadata candidate generation remains pending Phase 9. | `aic2026/dataset.py`, `ingestion/schemas.py`, `ingestion/build_records.py` | Schema separation complete in Phase 3; channel generation remains Phase 9. |
| Vietnamese query handling | `normalize_label` va answer normalization co mot so token khong dau/co dau, nhung chua co module query normalization co struct rieng. | Truy van tieng Viet co dau/khong dau va phrase temporal co the mat tin hieu. | `aic2026/fusion.py`, `aic2026/qa.py` | Phase 10: `query_normalization.py` giu ca original va accent-folded representation. |
| Submission export | `write_submission` chi ghi CSV va cap 100 rows, khong validate schema/task/rank/duplicate/TRAKE event count/Q&A answer empty. | File submission sai co the duoc ghi thanh cong. | `aic2026/metrics.py`, `ui/app.py`, `aic2026/cli.py` | Phase 11: `submission_validation.py`, CLI `validate-submission`, UI/CLI export gate. |
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
6. One video currently produces at most one TRAKE alignment sequence.
7. KIS Top-100 de-duplicates `(video_id, frame_id)`; TRAKE de-duplicates full sequence; Q&A/export still lacks a shared validator.
8. New caches require schema-v1 manifests; legacy `entry.pkl`-only caches are rejected by default and stale/corrupt fields are reported.
9. The `aic2026:` section in `settings.yaml` is the validated runtime source wired into engine, CLI, UI and benchmark snapshots.
10. UI custom DATA_ROOT does not update all routes; health/list/video serving still use global `DATA_ROOT`.
11. UI frame edit can affect multiple rows/tasks because it replaces matching frame values across `state.rows`.
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
