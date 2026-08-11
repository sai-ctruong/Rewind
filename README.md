# AIC 2026 Rewind

Competition system for Textual KIS, grounded Q&A and ordered-event TRAKE over the AIC keyframe collection.

## Current Pipeline

- Real openai/clip-vit-base-patch32 text encoder with strict 512-dimensional validation.
- Explicit hashing fallback for unit/smoke tests only; production mode refuses it.
- CLIP, confidence-aware objects, media metadata and BM25 score fusion with ablations.
- Video-aware diversified Top-100 ranking.
- Bounded query-conditioned local refinement of the original MP4s, wired into **Textual
  KIS only** (Phase 5). Q&A and TRAKE do not refine. The official submission frame stays
  the coarse mapped `frame_idx`; see `docs/PHASE_5_LOCAL_REFINEMENT.md`.
- Joint monotonic dynamic programming for TRAKE.
- Ordered diverse multi-frame Q&A evidence, answer normalization and confidence.
- Official-style R-score/R@k/Final Score evaluation artifacts and competition UI.

See AUDIT.md and PHASE_REPORT.md for verified details and limitations.

## Install And Verify

~~~powershell
cd C:\Users\ad\Downloads\codepython\project\KISC_module
.venv\Scripts\pip.exe install -r requirements-full.txt
.venv\Scripts\python.exe -m pytest -q
~~~

The full dependency command is required for meaningful retrieval accuracy. Without it, CLI/UI show a fallback warning.

Build the full multi-signal index:

~~~powershell
.venv\Scripts\python.exe -m aic2026.cli --production-mode --data-root data --cache-dir artifacts\aic2026_multisignal_index --rebuild build-index --load-objects --include-media-text --verify-keyframes
~~~

Run the UI:

~~~powershell
.venv\Scripts\python.exe -m ui.app
~~~

Open http://127.0.0.1:5000 and stop with Ctrl+C.

## Verified Runtime

- torch 2.13.0+cpu
- transformers 5.14.1
- openai/clip-vit-base-patch32 cached locally
- CPU inference, 512-dimensional normalized embeddings
- Production smoke returned five ranked rows with production_ready=true and no fallback

## Evaluation Gate

The repository does not contain AIC-format development labels. Do not report AIC accuracy from the legacy evaluation/labels*.json. Use evaluation/labels/template.jsonl to annotate a development set, then run evaluation from the UI.

Suggested development minimum, not an official benchmark: 50 KIS, 30 Q&A and 30 TRAKE queries.