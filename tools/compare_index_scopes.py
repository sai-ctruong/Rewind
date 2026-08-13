"""Structural comparison of the 29-video visual scope against the 873-video global scope.

Runs the SAME fixed fixture through both indexes and reports what changed structurally:
result counts, candidate union size, how many distinct videos and how many pixel-less
videos are represented, latency, memory, query-cache counters and channel counters.

It compares NOTHING semantic. The searchable collection is different, so the rankings are
different; without ground truth that is a fact, not an improvement, and this script
refuses to describe it as one.

It also checks that a result from a video with no MP4 behaves correctly: it appears, it
reports itself visually unavailable, it does not crash preview or refinement, and it
keeps its official mapped frame_idx.

    .venv\\Scripts\\python.exe tools/compare_index_scopes.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_ROOT))

from aic2026.config import config_hash, load_app_config  # noqa: E402
from aic2026.engine import AICCompetitionEngine  # noqa: E402
from aic2026.video_inventory import support_coverage  # noqa: E402
from evaluation.ground_truth import load_private_dev  # noqa: E402

DEFAULT_FIXTURE = _ROOT / "tests" / "fixtures" / "final_smoke_queries.json"
DEFAULT_OUTPUT = _ROOT / "artifacts" / "full_retrieval_smoke"


def rss_mb() -> Optional[float]:
    try:
        import psutil
    except ImportError:  # pragma: no cover - optional
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def probe_non_visual(engine: AICCompetitionEngine, visual: set[str]) -> dict:
    """Prove a pixel-less video behaves safely: indexed, submittable, visually honest.

    Picks a real indexed video that has no MP4 and no keyframe JPEG, then checks each
    capability separately. Nothing here is allowed to raise; a missing capability must
    report itself rather than crash.
    """
    target = next(
        (
            raw
            for raw in engine.entry.raws.values()
            if raw.video_id not in visual and raw.frame_idx is not None
        ),
        None,
    )
    if target is None:
        return {"skipped": "no pixel-less video in this index"}

    out: dict[str, Any] = {
        "video_id": target.video_id,
        "keyframe_id": None,
        "official_frame_idx": int(target.frame_idx),
    }
    for keyframe_id, raw in engine.entry.raws.items():
        if raw is target:
            out["keyframe_id"] = keyframe_id
            break

    # Preview / frame request: structured unavailability, never an exception.
    try:
        frame = engine.frame_provider.get_frame(target)
        out["frame_request"] = {
            "raised": False,
            "available": bool(frame.available),
            "image_bytes": frame.image_bytes is not None,
            "reason": getattr(frame, "reason", None) or getattr(frame, "source", None),
        }
    except Exception as exc:  # noqa: BLE001 - a raise here would BE the failure
        out["frame_request"] = {"raised": True, "error": f"{type(exc).__name__}: {exc}"}

    # Local refinement over a pool containing only this candidate.
    try:
        pool = [c for c in engine.search_candidates("a", top_k=200) if c.video_id == target.video_id]
        refinement = engine._refine_candidates("a", pool) if pool else None
        applied = 0 if refinement is None else sum(
            1 for item in refinement.by_keyframe().values() if item.applied
        )
        out["local_refinement"] = {
            "raised": False,
            "candidates": len(pool),
            "applied": applied,
            "frames_decoded": 0 if refinement is None else int(
                (refinement.diagnostics or {}).get("frames_decoded", 0)
            ),
        }
    except Exception as exc:  # noqa: BLE001
        out["local_refinement"] = {"raised": True, "error": f"{type(exc).__name__}: {exc}"}
    return out


def run_scope(name: str, config_path: str, fixture: dict, top_k: int, visual: set[str]) -> dict:
    memory = {"before_engine_mb": rss_mb()}
    config = load_app_config(config_path)
    started = time.perf_counter()
    engine, load = AICCompetitionEngine.from_data_root(app_config=config)
    startup_ms = (time.perf_counter() - started) * 1000.0
    memory["after_engine_mb"] = rss_mb()
    print(f"[{name}] engine ready in {round(startup_ms)}ms", file=sys.stderr)

    indexed = {raw.video_id for raw in engine.entry.raws.values()}
    rows, latency = [], []
    non_visual_examples: list[dict[str, Any]] = []
    for index, query in enumerate(fixture.get("kis", [])):
        begin = time.perf_counter()
        outcome = engine.search_kis_detailed(query, top_k=top_k)
        elapsed = (time.perf_counter() - begin) * 1000.0
        if index:
            latency.append(elapsed)
        diagnostics = outcome.diagnostics()
        channels = diagnostics.get("channels", {})
        videos = {p.video_id for p in outcome.predictions}
        pixel_less = sorted(videos - visual)
        rows.append({
            "query": query,
            "results": len(outcome.predictions),
            "candidate_union_size": channels.get("candidate_union_size"),
            "distinct_videos": len(videos),
            "videos_without_mp4": len(pixel_less),
            "channels_contributing": sorted(
                key for key, info in (channels.get("channels") or {}).items()
                if info.get("candidates_returned")
            ),
            "total_ms": round(elapsed, 1),
            "cost": diagnostics.get("cost", {}).get("channels"),
        })
        for prediction in outcome.predictions:
            if prediction.video_id in visual or len(non_visual_examples) >= 3:
                continue
            non_visual_examples.append({
                "video_id": prediction.video_id,
                "submitted_frame_idx": str(prediction.frame_id),
                "refinement_applied": bool((prediction.refinement or {}).get("applied")),
            })

    trake_rows = []
    for events in fixture.get("trake", []):
        begin = time.perf_counter()
        outcome = engine.search_trake_detailed(list(events), max_results=top_k)
        elapsed = (time.perf_counter() - begin) * 1000.0
        videos = {p.video_id for p in outcome.predictions}
        trake_rows.append({
            "events": len(events),
            "returned": len(outcome.predictions),
            "distinct_videos": len(videos),
            "videos_without_mp4": len(sorted(videos - visual)),
            "structural": outcome.structural_summary(),
            "total_ms": round(elapsed, 1),
        })

    qa_rows = []
    for item in fixture.get("qa", []):
        begin = time.perf_counter()
        predictions, info = engine.answer_qa(
            item.get("event", ""), item.get("question", ""), top_k=20,
            expected_answer_type=item.get("expected_answer_type"),
        )
        elapsed = (time.perf_counter() - begin) * 1000.0
        hypotheses = info["diagnostics"].get("hypotheses", []) or []
        qa_rows.append({
            "question": item.get("question"),
            "hypotheses": info["diagnostics"]["retrieved_video_hypotheses"],
            "hypotheses_without_mp4": len([
                h for h in hypotheses if h.get("video_id") not in visual
            ]),
            "vlm_calls": info["diagnostics"]["cost"]["qa"]["vlm_calls"],
            "backend_visual_capable": info["diagnostics"]["vlm_budget"]["backend_visual_capable"],
            "total_ms": round(elapsed, 1),
        })

    memory["after_queries_mb"] = rss_mb()
    fallback = probe_non_visual(engine, visual) if (indexed - visual) else None
    return {
        "scope": name,
        "fallback": fallback,
        "config": config_path,
        "config_hash": config_hash(config),
        "cache_dir": str(config.dataset.cache_dir),
        "cache_fingerprint": load.cache_fingerprint,
        "cache_valid": load.cache_valid,
        "indexed_videos": len(indexed),
        "indexed_frames": int(engine.entry.num_indexed),
        "videos_without_mp4_in_index": len(indexed - visual),
        "startup_ms": round(startup_ms, 1),
        "kis": rows,
        "trake": trake_rows,
        "qa": qa_rows,
        "warm_kis_ms": [round(value, 1) for value in latency],
        "memory": memory,
        "query_cache": engine.query_cache_status()["query_embeddings"],
        "channel_status": {
            key: {"usable": info.get("usable"), "records": info.get("records")}
            for key, info in engine.channel_status().items()
        },
        "non_visual_examples": non_visual_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visual-config", default="configs/competition.yaml")
    parser.add_argument("--full-config", default="configs/competition_full_retrieval.yaml")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    visual = {item.video_id for item in support_coverage("data") if item.video or item.keyframe_jpeg}

    results = {
        "visual_29": run_scope("visual_29", args.visual_config, fixture, args.top_k, visual),
        "retrieval_ready_full": run_scope(
            "retrieval_ready_full", args.full_config, fixture, args.top_k, visual
        ),
    }

    left, right = results["visual_29"], results["retrieval_ready_full"]
    comparison = {
        "indexed_videos": {left["scope"]: left["indexed_videos"], right["scope"]: right["indexed_videos"]},
        "indexed_frames": {left["scope"]: left["indexed_frames"], right["scope"]: right["indexed_frames"]},
        "distinct_cache_fingerprints": left["cache_fingerprint"] != right["cache_fingerprint"],
        "mean_candidate_union": {
            side["scope"]: round(
                sum(row["candidate_union_size"] or 0 for row in side["kis"]) / max(1, len(side["kis"])), 1
            )
            for side in (left, right)
        },
        "mean_distinct_videos_per_kis_query": {
            side["scope"]: round(
                sum(row["distinct_videos"] for row in side["kis"]) / max(1, len(side["kis"])), 1
            )
            for side in (left, right)
        },
        "kis_results_from_videos_without_mp4": {
            side["scope"]: sum(row["videos_without_mp4"] for row in side["kis"]) for side in (left, right)
        },
        "warm_kis_mean_ms": {
            side["scope"]: round(sum(side["warm_kis_ms"]) / max(1, len(side["warm_kis_ms"])), 1)
            for side in (left, right)
        },
        "memory_after_queries_mb": {
            side["scope"]: side["memory"]["after_queries_mb"] for side in (left, right)
        },
        "note": (
            "The searchable collection differs, so the rankings differ. Without ground "
            "truth that is a difference, not an improvement, and neither side is called "
            "better."
        ),
    }

    # Ground-truth state belongs in the summary: a reader must be able to see, in the
    # same file as the numbers, that no semantic evaluation was possible.
    gt = load_private_dev()
    gt_state = {
        "real_human_labels": gt.counts(),
        "real_label_total": len(gt.real_entries),
        "example_rows_ignored": len(gt.example_entries),
        "banner": gt.banner,
        "semantic_evaluation": "REFUSED",
        "refusal_code": "GROUND_TRUTH_REQUIRED",
        "note": (
            "No accuracy, recall, precision or Final Score appears in this artifact "
            "because no real human ground truth exists."
        ),
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_29_vs_full.json").write_text(
        json.dumps({"scopes": results, "comparison": comparison}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    full = results["retrieval_ready_full"]
    visual_scope = results["visual_29"]
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "disclaimer": (
                    "Structural comparison of two index scopes. The searchable collection "
                    "differs, so results differ; without ground truth that is a fact and "
                    "not an improvement. Nothing here is a quality measurement."
                ),
                "full_cache": {
                    "path": full["cache_dir"],
                    "fingerprint": full["cache_fingerprint"],
                    "valid": full["cache_valid"],
                    "config_hash": full["config_hash"],
                    "selected_videos": full["indexed_videos"],
                    "indexed_records": full["indexed_frames"],
                    "videos_without_mp4": full["videos_without_mp4_in_index"],
                    "channels": full["channel_status"],
                },
                "visual_cache": {
                    "path": visual_scope["cache_dir"],
                    "fingerprint": visual_scope["cache_fingerprint"],
                    "selected_videos": visual_scope["indexed_videos"],
                    "indexed_records": visual_scope["indexed_frames"],
                    "channels": visual_scope["channel_status"],
                },
                "structural_comparison": comparison,
                "non_mp4_examples": full["non_visual_examples"],
                "fallback": full.get("fallback"),
                "ground_truth": gt_state,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
