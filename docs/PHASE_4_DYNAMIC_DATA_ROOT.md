# Phase 4 Dynamic DATA_ROOT And Runtime Dataset State

Phase 4 fixes the runtime-state bug recorded in the Phase 0 audit: the application
could serve one request using an engine built from one dataset root while every other
surface still described a different one. It does not implement advanced local
refinement, Q&A per-video hypotheses, TRAKE k-best, independent retrieval channels,
query-normalization redesign, a submission validator, manual-edit redesign, or paper
benchmarking.

## 1. The Bug

`ui/app.py` kept dataset facts in a mutable dict plus three module-level constants
(`DATA_ROOT`, `AIC_CACHE_DIR`, `SUBMISSION_DIR`). `POST /api/video/index_folder`
rebuilt the engine from the requested folder and then assigned only two dict keys:

```python
state["engine"] = engine
state["load"] = load
```

`state["config"]` was never updated. So after loading root B:

| Surface | Root actually used |
|---|---|
| search engine | **B** |
| `_frame_provider()` (keyed on `config.dataset.root`) | A |
| `/api/video/file/<video_id>` | A |
| `/api/video/list` | A |
| `/api/health` `data_root` | A |
| `_cache_status()` | A |
| `video_url` existence check | A |

It also routed root A's `cache_dir` at root B's data. Two further routes ignored
configuration entirely: `/api/video/save` wrote to the global `AIC_CACHE_DIR` and
`/api/submission/save` to the global `SUBMISSION_DIR`.

Because dataset roots share video IDs (`L21_V001` exists in every copy), the mismatch
did not error — it silently served plausible-looking pixels from the wrong dataset.

## 2. RuntimeDatasetState

`aic2026/runtime_state.py` defines one **frozen** dataclass holding every
dataset-dependent fact. Being immutable is the point: there is no way to update half
of it.

```python
generation: int                     created_at_utc: str
app_config: AppConfig               config_path: str
config_hash: str
data_root: str                      cache_dir: str
frame_cache_dir: str
dataset_scope: dict                 resolved_video_ids: tuple[str, ...]
selected_video_ids_hash: str
frame_provider: FrameProvider
engine: AICCompetitionEngine | None      # None before activation
load: AICLoadResult | None
cache_status: dict | None           dataset_status: dict | None
video_inventory_summary: dict
```

`engine`, `load`, and `dataset_status` are explicitly optional, because a freshly
started app has a valid dataset root and cache status but no index loaded yet.

Helpers: `runtime_summary()`, `identity()`, `knows_video()`, and
`verify_engine_identity()`.

## 3. Engine Identity Is Verified, Not Assumed

`AICCompetitionEngine.dataset_identity()` exposes the engine's own `data_root`,
`cache_dir`, `config_hash`, indexed video IDs, and frame count.

`verify_engine_identity()` runs on **every** state construction and publication, and
raises `RuntimeStateError` if the engine's data root differs from the state's, or if
the frame provider serves a different root than the engine. That is exactly the
invariant the old code violated, now checked rather than trusted.

## 4. RuntimeStateManager

A deliberately small manager — a lock plus immutable snapshots is the whole
correctness story for a single-process app.

| Method | Behaviour |
|---|---|
| `get_state()` | returns one snapshot; a handler calls it **once** |
| `replace_state(new)` | verifies identity, then publishes atomically |
| `build_and_replace(...)` | builds first, publishes only on full success |
| `next_generation()` | peek without publishing |
| `status()` | active runtime summary |

It lives in `app.extensions["aic_runtime_state"]`, not in Flask's `g` (which is
per-request and would not persist).

## 5. Atomic Replacement

```text
old_state stays active
   -> validate root / config / scope
   -> construct engine (may raise)
   -> construct FrameProvider
   -> construct complete new state (verifies identity)
   -> atomic replace
new_state active
```

Nothing on the live state is mutated before the final step. Any failure leaves the
previous state fully intact, and the response says so explicitly with
`active_state_changed: false` plus the unchanged `active_state` summary.

## 6. Generation

Every published state carries a monotonically increasing `generation`, starting at 1
for the initial (engine-less) state. It appears in health, video list, search
responses, every result object, and the `X-Runtime-Generation` response header.

Result URLs embed it:

```text
/api/video/frame/L21_V023/kf_000114?generation=2
/api/video/file/L21_V023?generation=2
```

Requesting with a superseded generation returns `409 STALE_RESULT_GENERATION` instead
of serving an identically named frame from the *new* dataset. Requests that omit the
parameter are still served against the active state, so nothing existing breaks.

## 7. Snapshot Semantics

Each handler starts with `state = snapshot()` and uses only that object. A request that
began under generation N finishes under generation N even if a switch lands midway —
which is fine. What can no longer happen is one request mixing an engine from
generation N with a frame provider or data root from N+1.

## 8. Cache Directory Follows The Dataset

`resolve_cache_dir(app_config, data_root, explicit=...)`:

1. an explicit `cache_dir` always wins;
2. if the root equals the configured root, the configured cache is used — so the real
   development cache is still found;
3. otherwise a distinct directory is derived from a digest of the canonical root plus
   the scope payload.

Switching roots therefore never silently reuses another root's cache. The Phase 2
manifest remains the final safety net: forcing root A's cache onto root B is rejected
with `409 STALE_CACHE` and the active state is untouched.

## 9. Inspect Versus Activate

Two clearly separated operations:

| Endpoint | Effect |
|---|---|
| `POST /api/dataset/inspect` | **read-only**; returns a report for any root and always answers `active_state_changed: false` |
| `POST /api/video/index_folder` | activates a root: builds a whole new state and replaces the active one |
| `POST /api/video/index` | activates the root the active state already points at |

Inspecting root B while root A is active leaves root A serving every route.

## 10. Path Security

`safe_video_path(state, video_id)`:

- rejects `..`, `/`, `\`, and NUL outright;
- requires the ID to be part of the **active** selection (`VIDEO_NOT_IN_ACTIVE_SCOPE`);
- resolves the candidate and confirms it is inside the active video directory.

Result JSON contains only logical API URLs; no filesystem path is ever returned to the
frontend. Submission names are restricted to alphanumerics, `-`, and `_`.

## 11. Error Model

```text
RUNTIME_STATE_UNINITIALIZED   400   no engine activated yet
DATA_ROOT_INVALID             400   missing path, or a directory that is not an AIC root
STATE_BUILD_FAILED            400   the new state could not be built
DATASET_INVALID               422   the root is an AIC layout but the data is invalid
FRAME_UNAVAILABLE             422   no keyframe JPEG and no decodable MP4
CORRUPT_CACHE_MANIFEST        422
LEGACY_CACHE / STALE_CACHE    409
STALE_RESULT_GENERATION       409   the request targets a superseded dataset
VIDEO_NOT_IN_ACTIVE_SCOPE     404   unknown or out-of-scope video ID
FRAME_NOT_FOUND               404
```

`500` is reserved for genuine internal bugs.

## 12. Compatibility Constants

`DATA_ROOT`, `AIC_CACHE_DIR`, and `SUBMISSION_DIR` still exist in `ui/app.py` but are
now consulted **only** while constructing the initial state inside `create_app`. No
request handler reads them, so they can never override runtime state.

`create_app(config_path=None, app_config=None, initial_data_root=None,
initial_cache_dir=None)`: an explicit `app_config` wins, otherwise `config_path` is
loaded, and the initial roots override the dataset paths.

## 13. Startup For The Current Development Dataset

Startup never rebuilds. It reports cache status and waits for an explicit activation.

```powershell
$env:AIC_CONFIG = "configs\settings.yaml"
.venv\Scripts\python.exe -c @'
from dataclasses import replace
from aic2026.config import DatasetScopeConfig, load_app_config
import ui.app as appmod

base = load_app_config("configs/settings.yaml")
config = replace(base, dataset=replace(
    base.dataset,
    root="data",
    cache_dir="artifacts/aic2026_index_existing_videos",
    scope=DatasetScopeConfig(mode="existing_videos"),
))
appmod.create_app(app_config=config).run(port=5000, threaded=True)
'@
```

Then click **Dataset Root** (or `POST /api/video/index`) to activate. No AIC batch name
is hard-coded anywhere.

## 14. Real Smoke Result

Run against the real `data` root with `scope mode = existing_videos` and the existing
`artifacts/aic2026_index_existing_videos`, on 2026-08-10. **The cache was reused, not
rebuilt** (`cached: true`).

| Check | Result |
|---|---|
| Startup state | generation 1, engine not loaded, cache already `valid: true` |
| After activation | generation 2, `engine_loaded: true` |
| Data root | `c:/users/ad/downloads/codepython/project/kisc_module/data` |
| Scope | `existing_videos`, 29 selected |
| Selected IDs hash | `9c6062381f9139a87c8049516e6b4634e37cb289a95c51cc41dd6ac2370691fe` |
| Cache | `valid: true`, `legacy: false`, `stale: false`, fingerprint `199cd0fd…` |
| Dataset capabilities | retrieval-valid 29, visual-accessible 29, refinement-ready 29, JPEG-backed 24, MP4-fallback 5, invalid 0 |
| Video inventory | 29 MP4s, `L21`: 29, 3,378,944,552 bytes |
| Video list | 29 entries, generation 2 |
| JPEG frame `L21_V001/kf_000001` | `200`, `X-Frame-Source: keyframe_jpeg`, `X-Frame-Id: 0`, 135,232 bytes |
| MP4 fallback `L21_V027/kf_000003` | `200`, `X-Frame-Source: video_decode`, `X-Frame-Id: 300`, 171,428 bytes |
| KIS query | `200`, generation 2, 5 results (e.g. `L21_V023, 14346`) |
| Result URLs | all logical `/api/...` and all carrying `generation=2` |
| Following a result image / video URL | `200` / `200` |
| Stale generation request | `409 STALE_RESULT_GENERATION` |

No accuracy benchmark was run: no AIC ground truth exists.

## 15. Limitations

- Scope for the *initial* state comes from config or `create_app` arguments; the
  frontend has one root input, and per-request scope overrides
  (`scope_mode`, `include_patterns`, `exclude_patterns`) are available on the activate
  and inspect endpoints but are not surfaced as UI controls.
- Generation checking is opt-in per request: URLs the app generates carry it, but a
  hand-written URL without the parameter is still served against the active state.
- The evaluation worker thread captures the engine from the snapshot at submit time; a
  dataset switch mid-run does not cancel it, and its results describe the generation it
  started with.
- Derived frames are not cleared on a state switch, by design: the frame cache key
  already validates the source MP4's path, size, and mtime.
- Concurrency is snapshot-based, not transactional; a long request may finish against a
  superseded dataset, which is reported through the generation rather than prevented.

## 16. Not Started

Phase 5 has not been started. Advanced local refinement, Q&A per-video hypotheses,
TRAKE k-best, independent retrieval channels, query normalization, submission
validation, and manual-edit redesign all remain pending.
