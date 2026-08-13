"""No-ground-truth research smoke: B0_CLEAN vs ADAPTIVE_BUDGET on local data.

Runs one fixed fixture through both configurations and reports what differs
STRUCTURALLY: ranking identity, latency, cache hits, channel calls, decoded frames,
image embeddings, budget stages, TRAKE event allocations, VLM calls.

It does not say which configuration is better. Without official ground truth a changed
ranking is a difference, not an improvement, and this script refuses to describe it as
one. Quality columns appear only when a ground-truth file is supplied.

    .venv\\Scripts\\python.exe tools/run_research_smoke.py --config configs/competition.yaml
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_ROOT))

from aic2026.config import config_hash, load_app_config  # noqa: E402
from aic2026.engine import AICCompetitionEngine  # noqa: E402
from evaluation.pareto import build_report, variant_from_costs, write_report  # noqa: E402

DEFAULT_FIXTURE = _ROOT / "tests" / "fixtures" / "final_smoke_queries.json"
DEFAULT_OUTPUT = _ROOT / "artifacts" / "research_smoke"

VARIANTS: dict[str, dict[str, Any]] = {
    "B0_CLEAN": {},
    "ADAPTIVE_BUDGET": {"adaptive_budget": {"enabled": True}},
}


def run_variant(name: str, config_path: str, overrides: dict, fixture: dict, top_k: int) -> dict:
    config = load_app_config(config_path, overrides)
    started = time.perf_counter()
    engine, load = AICCompetitionEngine.from_data_root(app_config=config)
    startup_ms = (time.perf_counter() - started) * 1000.0
    print(f"[{name}] engine ready in {round(startup_ms)}ms", file=sys.stderr)

    kis_rows, kis_costs, kis_latency = [], [], []
    for index, query in enumerate(fixture.get("kis", [])):
        begin = time.perf_counter()
        outcome = engine.search_kis_detailed(query, top_k=top_k)
        elapsed = (time.perf_counter() - begin) * 1000.0
        diagnostics = outcome.diagnostics()
        if index:  # the first query pays the one-off model load
            kis_latency.append(elapsed)
        if outcome.cost is not None:
            kis_costs.append(outcome.cost)
        budget = diagnostics.get("adaptive_budget") or {}
        grid = budget.get("official_grid") or {}
        progressive = budget.get("progressive_video") or {}
        kis_rows.append(
            {
                "query": query,
                "rows": [[p.video_id, str(p.frame_id)] for p in outcome.predictions],
                "total_ms": round(elapsed, 1),
                "cost": diagnostics.get("cost"),
                "budget_enabled": bool(budget.get("enabled")),
                "uncertainty": (budget.get("uncertainty") or {}).get("uncertainty"),
                "grid_vectors_read": grid.get("vectors_read", 0),
                "grid_frames_decoded": grid.get("frames_decoded", 0),
                "progressive_frames_scored": progressive.get("frames_scored", 0),
                "progressive_stop_reason": progressive.get("stop_reason")
                or progressive.get("skipped_reason"),
                "allocation": budget.get("allocation"),
                "channel_policy": (diagnostics.get("channels") or {}).get("channel_policy"),
            }
        )

    trake_rows, trake_costs, trake_latency = [], [], []
    for events in fixture.get("trake", []):
        begin = time.perf_counter()
        outcome = engine.search_trake_detailed(list(events), max_results=top_k)
        elapsed = (time.perf_counter() - begin) * 1000.0
        trake_latency.append(elapsed)
        budget = outcome.diagnostics.get("adaptive_budget") or {}
        trake_rows.append(
            {
                "events": list(events),
                "rows": [p.row() for p in outcome.predictions],
                "returned": len(outcome.predictions),
                "structural": outcome.structural_summary(),
                "total_ms": round(elapsed, 1),
                "cost": outcome.diagnostics.get("cost"),
                "event_budget": budget.get("frame_budget_by_event"),
                "weakest_event": budget.get("weakest_event_index"),
                "query_execution": outcome.diagnostics.get("query_execution"),
            }
        )

    qa_rows, qa_latency = [], []
    for item in fixture.get("qa", []):
        begin = time.perf_counter()
        predictions, info = engine.answer_qa(
            item.get("event", ""), item.get("question", ""), top_k=20,
            expected_answer_type=item.get("expected_answer_type"),
        )
        elapsed = (time.perf_counter() - begin) * 1000.0
        qa_latency.append(elapsed)
        qa_rows.append(
            {
                "question": item.get("question"),
                "rows": [[p.video_id, str(p.frame_id), p.answer] for p in predictions],
                "vlm_budget": info["diagnostics"].get("vlm_budget"),
                "cost": info["diagnostics"].get("cost"),
                "total_ms": round(elapsed, 1),
            }
        )

    return {
        "name": name,
        "config_hash": config_hash(config),
        "cache_fingerprint": load.cache_fingerprint,
        "startup_ms": round(startup_ms, 1),
        "kis": kis_rows,
        "trake": trake_rows,
        "qa": qa_rows,
        "latency": {
            "kis_warm_ms": kis_latency,
            "trake_ms": trake_latency,
            "qa_ms": qa_latency,
        },
        "query_cache": engine.query_cache_status(),
        "_costs": kis_costs + trake_costs,
    }


def compare(left: dict, right: dict) -> dict:
    """Structural differences only. Never a verdict."""
    out: dict[str, Any] = {}
    for task in ("kis", "trake", "qa"):
        pairs = list(zip(left[task], right[task]))
        identical = sum(1 for a, b in pairs if a["rows"] == b["rows"])
        out[task] = {
            "queries": len(pairs),
            "identical_ranking": identical,
            "changed_ranking": len(pairs) - identical,
        }
    def total(variant, field):
        return sum(
            int(((row.get("cost") or {}).get(field[0]) or {}).get(field[1], 0) or 0)
            for group in ("kis", "trake")
            for row in variant[group]
        )
    out["cost"] = {
        "text_encoder_calls": {
            left["name"]: total(left, ("text", "encoder_calls")),
            right["name"]: total(right, ("text", "encoder_calls")),
        },
        "frames_decoded": {
            left["name"]: total(left, ("video", "frames_decoded")),
            right["name"]: total(right, ("video", "frames_decoded")),
        },
        "image_embeddings": {
            left["name"]: total(left, ("image_embedding", "computed")),
            right["name"]: total(right, ("image_embedding", "computed")),
        },
        "vlm_calls": {
            left["name"]: sum(int((row["cost"] or {}).get("qa", {}).get("vlm_calls", 0)) for row in left["qa"]),
            right["name"]: sum(int((row["cost"] or {}).get("qa", {}).get("vlm_calls", 0)) for row in right["qa"]),
        },
    }
    out["note"] = (
        "A changed ranking is neither success nor failure: no ground truth exists to "
        "say which ordering is better. Only the counts above are measurements."
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    results = {
        name: run_variant(name, args.config, overrides, fixture, args.top_k)
        for name, overrides in VARIANTS.items()
    }

    variants = []
    for name, payload in results.items():
        latency = payload["latency"]["kis_warm_ms"] + payload["latency"]["trake_ms"]
        variants.append(
            variant_from_costs(
                name,
                costs=payload["_costs"],
                latencies_ms=latency,
                structural={
                    "trake_sequences": sum(row["returned"] for row in payload["trake"]),
                    "malformed": sum(
                        row["structural"]["malformed_prediction_count"] for row in payload["trake"]
                    ),
                },
                config_hash=payload["config_hash"],
                cache_fingerprint=payload["cache_fingerprint"],
            )
        )
        payload.pop("_costs")

    report = build_report(variants)
    out_dir = Path(args.output)
    paths = write_report(report, out_dir, stem="b0_clean_vs_adaptive")
    comparison = compare(results["B0_CLEAN"], results["ADAPTIVE_BUDGET"])
    (out_dir / "summary.json").write_text(
        json.dumps({"variants": results, "comparison": comparison}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "comparison": comparison,
        "pareto": report["pareto_fronts"],
        "has_quality_axis": report["has_quality_axis"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
