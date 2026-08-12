"""Structural ablation runner.

Runs the same fixed queries through configuration variants and records what each variant
did STRUCTURALLY: how many candidates each channel contributed, how large the fused
union was, how many complete TRAKE sequences came out, and how long it took.

It deliberately does NOT rank the variants. Deciding that one variant retrieves *better*
than another requires official ground truth, which this repository does not have; the
runner refuses to emit a "best variant" and says so in its output. When labels arrive,
`evaluation.official_eval.evaluate_labels` supplies the quality half and these same
variants can be scored without changing this file's structure.

    .venv\\Scripts\\python.exe tools/run_ablation.py --config configs/competition.yaml --group retrieval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_ROOT))

from aic2026.config import config_hash, load_app_config  # noqa: E402
from aic2026.engine import AICCompetitionEngine  # noqa: E402
from aic2026.system_profile import build_system_profile  # noqa: E402

DEFAULT_FIXTURE = _ROOT / "tests" / "fixtures" / "final_smoke_queries.json"

# Each variant is a config override applied to the SAME base config, so two variants
# differ by exactly the thing being ablated and nothing else.
VARIANTS: dict[str, dict[str, dict[str, Any]]] = {
    "retrieval": {
        "clip_only": {
            "retrieval_channels": {
                "channels": {
                    "bm25": {"enabled": False},
                    "objects": {"enabled": False},
                    "metadata": {"enabled": False},
                }
            }
        },
        "clip_sparse": {
            "retrieval_channels": {
                "channels": {"objects": {"enabled": False}, "metadata": {"enabled": False}}
            }
        },
        "clip_objects": {
            "retrieval_channels": {
                "channels": {"bm25": {"enabled": False}, "metadata": {"enabled": False}}
            }
        },
        "all_channels": {},
        "rrf_fusion": {"fusion": {"method": "rrf", "adaptive": False}},
    },
    "trake": {
        "beam_dp_k1": {"trake": {"k_best_per_video": 1, "max_alignments_per_video": 1}},
        "beam_dp_k4": {},
        "no_recovery": {"trake": {"recover_missing_events": False}},
        "no_expansion": {"trake": {"candidate_depth_expansion": [], "candidate_depth_max": 40}},
    },
}


def kis_diagnostics(engine, queries: list[str], top_k: int) -> dict[str, Any]:
    rows = []
    for query in queries:
        started = time.perf_counter()
        outcome = engine.search_kis_detailed(query, top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000.0
        diagnostics = outcome.diagnostics()
        channels = diagnostics.get("channels", {})
        rows.append(
            {
                "query": query,
                "results": len(outcome.predictions),
                "candidate_union_size": channels.get("candidate_union_size"),
                "per_channel_candidates": {
                    name: info.get("candidates_returned")
                    for name, info in (channels.get("channels") or {}).items()
                },
                "exclusive_candidates": channels.get("exclusive_candidates", {}),
                "total_ms": round(elapsed, 1),
                "top3": [[p.video_id, p.frame_id] for p in outcome.predictions[:3]],
            }
        )
    # The first query pays the one-off text-encoder load. Averaging it in would make a
    # variant look slower or faster for a reason that has nothing to do with the ablation.
    warm = [r["total_ms"] for r in rows[1:]]
    return {
        "queries": len(rows),
        "runs": rows,
        "mean_union_size": (
            round(sum(r["candidate_union_size"] or 0 for r in rows) / len(rows), 1) if rows else None
        ),
        "cold_first_query_ms": rows[0]["total_ms"] if rows else None,
        "mean_warm_ms": round(sum(warm) / len(warm), 1) if warm else None,
    }


def trake_diagnostics(engine, sequences: list[list[str]], top_k: int) -> dict[str, Any]:
    rows = []
    for events in sequences:
        started = time.perf_counter()
        outcome = engine.search_trake_detailed(list(events), max_results=top_k)
        elapsed = (time.perf_counter() - started) * 1000.0
        diagnostics = outcome.diagnostics
        rows.append(
            {
                "events": list(events),
                "returned": len(outcome.predictions),
                "videos_with_full_event_coverage": diagnostics.get("videos_with_full_event_coverage"),
                "unique_sequences_generated": diagnostics.get("unique_sequences_generated"),
                "expansion_triggered": diagnostics.get("candidate_expansion_triggered"),
                "structural": outcome.structural_summary(),
                "total_ms": round(elapsed, 1),
            }
        )
    return {
        "queries": len(rows),
        "runs": rows,
        "total_sequences": sum(r["returned"] for r in rows),
        "mean_ms": round(sum(r["total_ms"] for r in rows) / len(rows), 1) if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--group", choices=sorted(VARIANTS), default="retrieval")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(_ROOT / "artifacts" / "ablation"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--variant", action="append", help="Run only these variants.")
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    wanted = args.variant or list(VARIANTS[args.group])
    report: dict[str, Any] = {
        "group": args.group,
        "config": args.config,
        "fixture": Path(args.fixture).name,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "disclaimer": (
            "STRUCTURAL ablation. Variants are described, never ranked: choosing a best "
            "variant is a quality judgement and requires official AIC ground truth, which "
            "this repository does not have. Candidate counts are candidate coverage, NOT "
            "recall."
        ),
        "variants": {},
    }

    for name in wanted:
        overrides = VARIANTS[args.group][name]
        config = load_app_config(args.config, overrides)
        profile = build_system_profile(config, config_path=args.config)
        started = time.perf_counter()
        engine, load = AICCompetitionEngine.from_data_root(app_config=config)
        startup_ms = (time.perf_counter() - started) * 1000.0
        print(f"[{args.group}/{name}] engine ready in {round(startup_ms)}ms", file=sys.stderr)

        entry: dict[str, Any] = {
            "overrides": overrides,
            "config_hash": config_hash(config),
            "cache_fingerprint": profile.cache_fingerprint,
            "cache_valid": load.cache_valid,
            "channels": {
                channel: {"usable": info.get("usable"), "records": info.get("records")}
                for channel, info in engine.channel_status().items()
            },
            "startup_ms": round(startup_ms, 1),
        }
        if args.group == "retrieval":
            entry["kis"] = kis_diagnostics(engine, list(fixture.get("kis", [])), args.top_k)
        else:
            entry["trake"] = trake_diagnostics(engine, list(fixture.get("trake", [])), args.top_k)
        report["variants"][name] = entry

    # Comparability guard: a difference only means something if the two runs saw the same
    # index. A changed cache fingerprint means the variants are not comparable at all.
    fingerprints = {entry["cache_fingerprint"] for entry in report["variants"].values()}
    report["comparable"] = len(fingerprints) == 1
    if not report["comparable"]:
        report["comparability_warning"] = (
            "Variants ran against different caches; their diagnostics are NOT comparable."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.group}_ablation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "group": args.group,
        "variants": list(report["variants"]),
        "comparable": report["comparable"],
        "output": str(target),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
