# AIC 2026 Repository Audit

> **HISTORICAL.** The Phase-0 audit that started the AIC 2026 work. Its findings are a
> record of that moment; current status lives in
> `docs/CURRENT_IMPLEMENTATION_AUDIT.md`. See `docs/DOCUMENTATION_MAP.md`.


Audit date: 2026-08-04
Branch: `feat/aic2026-competition-research`
Baseline commit: `b692f5c`
Baseline tests: 225 passed with `.venv\Scripts\python.exe -m pytest -q`

The requested `CHATGPT_PROJECT_CONTEXT(2).md` does not exist in the repository. This audit read the actual `CHATGPT_PROJECT_CONTEXT.md` and verified its claims against code and local data.

Local data verification:

- 873 files in `data/clip-features-32`.
- 873 files in `data/map-keyframes`.
- 29 MP4 files in `data/video`.
- The existing full cache reports 177,321 frames and feature dimension 512.
- `evaluation/labels.json` and `labels_en.json` belong to the legacy video-engine benchmark. They are not official AIC KIS/Q&A/TRAKE labels and must not be used to claim AIC accuracy.

## Capability Matrix

| Item | Actually present | Missing or incomplete | Related files | Repair direction |
|---|---|---|---|---|
| Official frame mapping | `frame_idx` from each map CSV is retained and exported | No manual correction/validation utility | `aic2026/dataset.py`, `aic2026/engine.py`, `ui/app.py` | Preserve mapping through ranking, refinement and UI edits |
| Text encoder abstraction | Protocol, hashing encoder and Transformers CLIP implementation exist | Names/status contract, batch API, strict production guard, finite/dtype/dimension checks | `aic2026/engine.py` | Move to `aic2026/text_encoder.py`; expose status and fail production when CLIP is unavailable |
| Real CLIP activation | Production CLIP ViT-B/32 loads from local cache on CPU and reports 512 dimensions | GPU/CUDA is unavailable on this machine | aic2026/text_encoder.py | Benchmark on a fixed AIC development split before quality claims |
| Feature compatibility | Actual AIC feature dimension is 512 | `configs/settings.yaml` still describes ViT-L/14 with 768 dimensions | `configs/settings.yaml`, `aic2026/dataset.py` | Make ViT-B/32 and 512 the AIC competition config; validate at runtime |
| Objects | Labels are read with confidence threshold; raw bounding boxes remain in JSON | Bounding boxes/confidences are discarded; default full cache has objects disabled; no separate object score | `aic2026/dataset.py` | Store structured detections and add confidence-aware object signal |
| Media metadata | Title, description and keywords can be appended to caption text | Default cache disables it; no separate metadata score/evidence; tags variants are incomplete | `aic2026/dataset.py` | Keep per-video metadata and score as an auxiliary signal |
| OCR/ASR/caption | Generic record schema and legacy ingestion support these fields | AIC loader does not load OCR/ASR/captions; UI checkboxes do not activate real signals | `ingestion/schemas.py`, `aic2026/dataset.py`, `ui/index.html` | Load only files that actually exist and report unavailable signals explicitly |
| BM25 | Index text is `objects + OCR + ASR + llm_caption`; in AIC today that means optional objects and media text | With default cache, corpus is effectively empty; BM25 rank is fused but score/evidence is hidden | `ingestion/build_records.py`, `aic2026/dataset.py`, `retrieval/coarse_retriever.py` | Build explicit sparse corpus and expose signal scores |
| Fusion | RRF combines CLIP and BM25 ranks; adaptive BM25 weight heuristic exists | No weighted normalized fusion, structured candidates, object/metadata channels or configurable ablations | `retrieval/coarse_retriever.py`, `retrieval/fusion.py`, `aic2026/engine.py` | Add `aic2026/fusion.py` and config-driven signal routing |
| Top-100 | Output is capped at 100 and official rows are unique when source IDs are unique | It is simply the first 100 fused candidates; no neighborhood expansion or video/temporal diversity | `aic2026/engine.py` | Add `aic2026/ranking.py` with precision head and recall tail |
| Local refinement | TRAKE re-searches indexed keyframes in a time filter | No MP4 decoding, no refinement result/status, no cache or uncertainty trigger | `aic2026/engine.py` | Add `aic2026/local_refinement.py` with MP4 and keyframe-only modes |
| TRAKE | Events are searched independently and constrained to one video with increasing timestamps | Uses recursive chain enumeration, not DP; requires every event; no coverage-aware video hypothesis or gap model | `aic2026/engine.py`, `retrieval/temporal_check.py` | Add polynomial monotonic DP in `aic2026/trake.py` |
| Q&A grounding | Chooses a top candidate, creates an ordered temporal window and may send up to 8 JPEGs | Retrieval blindly concatenates event and question; answer result lacks confidence/normalization; one answer is copied to all ranked candidates | `aic2026/engine.py`, `retrieval/vqa_module.py` | Add `aic2026/qa.py` with diverse evidence, answerer adapters and separate evaluation fields |
| Q&A frame count | JPEG bytes are prepared for at most 8 frames; the default answerer receives the whole keyframe window as records | Prepared images and selected records can differ; no before/middle/after selection contract | `aic2026/engine.py`, `retrieval/vqa_module.py` | Select one ordered evidence set and pass exactly that set to every answerer |
| Official metrics | KIS range hit, Q&A grounded answer hit, TRAKE partial event score, R@k and mean Final Score are implemented | Exact compatibility cannot be claimed without official evaluator/examples; normalization is basic | `aic2026/metrics.py` | Keep formulas isolated and add official-evaluation runners plus fixture tests |
| Benchmark logging | Writes config, query rows and optional summary | Missing environment, predictions, errors, memory/VRAM, decode and cache metrics | `aic2026/benchmark.py` | Expand run artifact contract under `evaluation/` |
| Ground truth | Legacy 51/78-label retrieval files exist | No AIC KIS/Q&A/TRAKE development labels | `evaluation/labels*.json` | Add schema/template and annotation tool; do not report AIC accuracy yet |
| UI | KIS/Q&A/TRAKE views and original frame IDs are shown; CSV export works | Health omits encoder/device/MP4; score evidence, manual correction, timeline and evaluation view are absent | `ui/app.py`, `ui/index.html` | Add competition status and non-blocking evaluation endpoints |
| Documentation | AIC handoff and quick start exist | `TASKS.md` and portions of settings/docs still describe the legacy system and unsupported measurements | `README.md`, `TASKS.md`, `CHATGPT_PROJECT_CONTEXT.md` | Update only after implementation and verified runs |

## Three Largest Risks

1. **Invalid retrieval quality:** production paths can silently use deterministic hashing against real CLIP image vectors. Results then have no cross-modal meaning even though the UI still returns ranked rows.
2. **Misleading multi-signal claims:** object/media data can be read, but the default cache omits both and current scoring does not preserve independent evidence. Existing BM25 may therefore index an empty corpus while appearing enabled.
3. **TRAKE scalability and quality:** recursive Cartesian-style chain enumeration is capped rather than optimized. It can miss the best joint alignment, cannot model missing events/gaps, and does not rank videos by event coverage.

## File-Level Plan

1. Add `aic2026/text_encoder.py`; refactor engine/CLI/UI around an explicit encoder status and production guard.
2. Add `aic2026/fusion.py`; extend dataset records with structured detections/metadata and config-driven ablations.
3. Add `aic2026/ranking.py` and route KIS output through video-aware Top-100 allocation.
4. Add `aic2026/local_refinement.py` with bounded OpenCV decoding and map-keyframe fallback.
5. Add `aic2026/trake.py` with coverage-aware video scoring and monotonic dynamic programming.
6. Add `aic2026/qa.py` with ordered evidence frames, normalization, confidence and answerer interfaces.
7. Add official task runners, ablation, error analysis, latency/environment logging and label templates under `evaluation/`.
8. Extend `ui/app.py` and `ui/index.html` only after backend contracts are covered by offline tests.
9. Update `README.md`, `TASKS.md`, context and related work after code and benchmark artifacts are verified.

## Baseline Gate

The repository now has the real CPU encoder dependencies and cached CLIP model. The install command remains:

```powershell
.venv\Scripts\pip.exe install -r requirements-full.txt
```

Production CLIP smoke now passes with 512-dimensional embeddings and no fallback. No AIC Final Score may be reported until an AIC-format ground-truth file is supplied or annotated.
