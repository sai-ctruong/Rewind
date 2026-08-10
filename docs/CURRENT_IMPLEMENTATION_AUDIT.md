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
| Keyframe identity | Fixed in Phase 3.1: internal ID is `{video_id}/kf_{keyframe_ordinal:06d}`; `frame_idx` is carried separately and is the only submission value. Record schema advanced to v3. | v2's `{video_id}/{frame_idx}` collided on the 192 official videos that repeat a `frame_idx`, silently overwriting `entry.raws` entries. Engine no longer rebuilds the ID from `(video_id, frame_id)` after fusion. | `aic2026/dataset.py`, `aic2026/engine.py`, `aic2026/fusion.py`, `ingestion/schemas.py`, `retrieval/video_engine.py` | Complete in Phase 3.1. |
| DATA_ROOT state | `index_folder` build engine tu folder moi, nhung health/list/video file/mp4 URL van dung global `DATA_ROOT`. | UI co the search root B nhung phuc vu video/health theo root A. | `ui/app.py` | Phase 4: dua `data_root`, `cache_dir`, config vao state va dung state cho moi endpoint. |
| LocalFrameRefiner usage | `LocalFrameRefiner` ton tai va co test rieng, nhung `AICCompetitionEngine` khong khoi tao hoac goi refiner trong KIS/Q&A/TRAKE. | Documentation noi refinement co the lam nguoi doc hieu nham pipeline that da refine. | `aic2026/local_refinement.py`, `aic2026/engine.py` | Phase 5: interface result day du, goi refiner end-to-end theo config. |
| `refine_window_s` | `search_trake(..., refine_window_s=6.0)` nhan tham so nhung khong dung. | UI/CLI co tham so vo tac dung; benchmark refinement bi gia. | `aic2026/engine.py`, `ui/app.py` | Phase 5: wire window vao refiner hoac bo option cho den khi hoat dong. |
| MP4 missing fallback | Local refiner co keyframe-only fallback khi goi truc tiep, nhung engine khong goi refiner nen missing MP4 khong tham gia KIS/Q&A/TRAKE. | Khong co warning pipeline-level ve gioi han sampling khi thieu MP4. | `aic2026/local_refinement.py`, `aic2026/engine.py`, `ui/app.py` | Phase 5: propagate refinement warning vao prediction/evidence/UI. |
| Q&A answer count | `answer_qa` chon `center = candidates[0]`, tao 1 answer tren video center, sau do gan cung answer cho tat ca candidate predictions. | Answer bi copy xuyen video; evidence video A co the gan cho prediction video B. | `aic2026/engine.py`, `aic2026/qa.py` | Phase 6: group theo video hypothesis, goi answerer rieng cho tung video hypothesis. |
| Q&A visual backend | Mac dinh `default_answerer()` tra `MockVqaAnswerer` neu khong co `ANTHROPIC_API_KEY`; mock suy luan tren text/caption/object, khong nhin anh. | Khong duoc coi la visual Q&A production; can warning ro trong health/UI/output. | `retrieval/vqa_module.py`, `aic2026/qa.py`, `ui/app.py` | Phase 6: answerer status, `answer_reliability_score`, LocalVlmAnswerer interface/test fake backend. |
| Expected answer type | UI gui `expected_answer_type`, engine nhan parameter nhung khong truyen vao `normalize_answer` theo type va khong prompt answerer theo type. | Number/color/boolean normalization khong dung nhu UI ham y. | `ui/app.py`, `aic2026/engine.py`, `aic2026/qa.py` | Phase 6: type-aware prompt/normalization, tests Vietnamese/English numbers and colors. |
| TRAKE output length | DP cho phep skip missing event; `frame_ids` loai alignment `None`, engine cung chi xuat selected non-null steps. | Submission TRAKE co the thieu cot frame cho N event. | `aic2026/trake.py`, `aic2026/engine.py` | Phase 7: validator bat dung N event, fallback candidate hoac loai sequence thieu. |
| TRAKE method | `align_video_dp` cat state bang beam width after moi event. Day la beam-pruned DP, khong phai exact DP day du. | Goi nham exact DP se tao bao cao sai ban chat thuat toan. | `aic2026/trake.py` | Phase 8: tach `align_events_exact_dp`, `align_events_beam_dp`, `k_best_alignments`. |
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

1. `LocalFrameRefiner` is currently called only by its unit tests, not by KIS/Q&A/TRAKE engine paths.
2. `refine_window_s` is accepted by `AICCompetitionEngine.search_trake` and UI, but is not used.
3. Q&A currently creates one answer for the top center candidate's video, then attaches that same answer to every returned candidate.
4. TRAKE can output fewer frame IDs than the number of events because missing alignments are allowed internally and filtered out in `frame_ids`.
5. TRAKE is beam-pruned dynamic programming, not exact exhaustive DP, because states are truncated by `beam_width`.
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
3. **Documentation can overstate current capability**: local refinement and multi-frame visual Q&A exist as modules/interfaces, but are not fully integrated into engine runtime without fallback/mock limitations.
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
- DATA_ROOT dynamic propagation: pending Phase 4, not started.
- Local refinement integration: pending Phase 5.
- Q&A per-video hypothesis: pending Phase 6.
- TRAKE complete/k-best output: pending Phase 7-8.
