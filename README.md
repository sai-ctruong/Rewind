# AIC 2026 Rewind

Competition system for **Textual KIS**, grounded **Q&A** and ordered-event **TRAKE** over
the AIC keyframe collection.

Version `0.11.0-aic2026`.

> **No AIC ground truth exists in this repository.** Nothing here reports accuracy,
> recall or a Final Score for this system, and no parameter has been tuned against
> quality. Every number in this project is structural: candidate counts, complete-sequence
> yield, validated row counts, latency. See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

---

## Quick start

```powershell
cd <repo>
.venv\Scripts\pip.exe install -r requirements-full.txt

# 1. Is this system fit to run?
.venv\Scripts\python.exe -m aic2026.cli --config configs\competition.yaml competition-check --load-engine

# 2. Start it. This is the only supported way to start the system.
.venv\Scripts\python.exe -m aic2026.cli --config configs\competition.yaml serve
```

Open <http://127.0.0.1:5000>. Stop with Ctrl+C.

`serve` runs the readiness preflight first and refuses to start when the verdict is
`NOT_READY`. `competition-check` exits 0 (`READY`), 1 (`READY_WITH_WARNINGS`) or 2
(`NOT_READY`); warnings name real gaps and do not block a run.

Building the index (required once, and whenever the data changes):

```powershell
.venv\Scripts\python.exe -m aic2026.cli --config configs\competition.yaml `
    --rebuild build-index --load-objects --include-media-text
```

`--load-objects --include-media-text` are **not optional**: without them the object and
metadata retrieval channels have no source data and the system is effectively CLIP-only.

Full procedure: [docs/COMPETITION_RELEASE_CHECKLIST.md](docs/COMPETITION_RELEASE_CHECKLIST.md).

## What the system does

**Retrieval** — four independent candidate generators (Phase 9): CLIP dense vectors,
BM25 over searchable text, detected object labels, and media metadata. Each generates its
own candidates and the pool is their union with per-channel provenance, so an
object-only or metadata-only frame can enter the results. OCR, ASR and frame-caption
channels exist and honestly report `available: false` — those sources are empty in this
data. Results are fused (rank-normalized, adaptive), then diversified into a video-aware
Top-100.

**Vietnamese queries** — the original query goes to CLIP unchanged; accent-folded and
lightly expanded views go to the lexical channels. Negation and temporal markers are
preserved and marked rather than normalized away.

**Textual KIS** — optional bounded, query-conditioned local refinement of the original
MP4s (Phase 5). Off by default for latency on CPU. The submitted frame stays the official
mapped `frame_idx`; a refined frame is evidence only.

**Q&A** — each top video hypothesis is answered independently from its own evidence
(Phase 6), so an answer can never be copied onto another video. The shipped backend
without a key or a local VLM is a **non-visual mock** that says so, and whose answers the
submission validator refuses.

**TRAKE** — joint monotonic **beam-pruned** dynamic programming (`beam_dp`, not exact
DP). Every emitted sequence carries exactly one frame per query event; incomplete
alignments are discarded, never shortened (Phase 7). Phase 8 adds k-best sequences per
video, adaptive candidate expansion and opt-in event-local visual refinement.

**Submissions** — one validator for all three tasks, used by every export path (Phase 10).
Atomic UTF-8 CSV with a sidecar validation report, row-scoped manual edits, and refusal to
export results produced before the current dataset activation.

**Reproducibility** — `system-profile` records the version, commit, config hash, cache
fingerprint, dataset hash, schema versions and capability state of a run (Phase 11).

## CLI

| Command | Purpose |
|---|---|
| `--version` | Print the release version. |
| `competition-check` | Structural readiness preflight; PASS/WARN/FAIL per check. |
| `system-profile` | Reproducibility identity of this runtime, as JSON. |
| `serve` | Start the UI/API (refuses `NOT_READY`). |
| `build-index` | Build the cache. Use `--load-objects --include-media-text`. |
| `inspect-data` / `inspect-cache` | Dataset and cache integrity reports. |
| `search` | One-off query from the terminal. |
| `validate-submission` | Structurally validate a CSV. Format only, never correctness. |
| `show-config` | The resolved config plus its hash. |

Tools (all offline, all writing to gitignored `artifacts/`):

```powershell
.venv\Scripts\python.exe tools\run_competition_smoke.py    # end-to-end release smoke
.venv\Scripts\python.exe tools\run_ablation.py --group retrieval   # structural ablation
.venv\Scripts\python.exe tools\build_release_manifest.py   # release + cache inventory
```

## Configuration

`configs/competition.yaml` is the release config — one fixed artifact a competition run is
identified by. `configs/settings.yaml` remains the general development file. Both are
parsed by the same validated loader (`aic2026.config.load_app_config`); there is no
release-only parser and no release-only field.

Safety settings that should not be changed casually: `production_mode: true`,
`allow_hashing_fallback: false`, `allow_stale_cache: false`,
`frame_output_policy: preserve_coarse`.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Offline and deterministic: no network, no GPU, no model download, no ground truth
required.

## Verified environment

torch 2.13.0+cpu · transformers 5.14.1 · `openai/clip-vit-base-patch32` cached locally ·
CPU inference · 512-dimensional normalized embeddings · 29-video / 7,800-frame development
scope.

## Documentation

| Document | Contents |
|---|---|
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | What this system cannot do or has not proven. Read this first. |
| [docs/COMPETITION_RELEASE_CHECKLIST.md](docs/COMPETITION_RELEASE_CHECKLIST.md) | Pre-session, per-query and pre-submission procedure. |
| [PHASE_REPORT.md](PHASE_REPORT.md) | Chronological history of every phase. |
| [docs/CURRENT_IMPLEMENTATION_AUDIT.md](docs/CURRENT_IMPLEMENTATION_AUDIT.md) | Item-by-item audit with final FIXED / PARTIAL / OPEN status. |
| `docs/PHASE_*.md` | Design and verification report for each phase. |
| [docs/AIC2026_COMPETITION.md](docs/AIC2026_COMPETITION.md) | Task and submission-format notes. |

## Evaluation gate

There are no AIC-format labels here. To evaluate, annotate a development set from
`evaluation/labels/template.jsonl` and run `evaluation.official_eval.evaluate_labels`.
Without labels, semantic metrics raise `GroundTruthRequired` by design; the legacy
`evaluation/labels*.json` files annotate the bundled demo clips and must never be
reported as AIC results.
