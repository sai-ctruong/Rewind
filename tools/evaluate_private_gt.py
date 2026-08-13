"""Score the system against PRIVATE development ground truth.

Uses the official metric implementation unchanged — `R@1`, `R@5`, `R@20`, `R@50`,
`R@100` and the Final Score as their mean — because a private-only metric would not be
comparable to anything. What differs is the label provenance, and that is stated on
every line of output:

    PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE

With no human labels the run REFUSES with `GROUND_TRUTH_REQUIRED`. Template rows marked
`EXAMPLE_ONLY` are parsed for their shape and never counted, so a file full of examples
refuses exactly like an empty one.

    .venv\\Scripts\\python.exe tools/evaluate_private_gt.py --config configs/competition.yaml
    .venv\\Scripts\\python.exe tools/evaluate_private_gt.py --split holdout
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

from aic2026.config import load_app_config  # noqa: E402
from aic2026.engine import AICCompetitionEngine  # noqa: E402
from aic2026.metrics import (  # noqa: E402
    TOP_KS,
    GroundTruthRequired,
    RankedAnswer,
    evaluate_query,
    mean_report,
)
from evaluation.experiment_manifest import build_manifest  # noqa: E402
from evaluation.ground_truth import (  # noqa: E402
    PRIVATE_DEV_DIR,
    SPLITS,
    GroundTruthSet,
    load_private_dev,
    report_header,
)
from evaluation.pareto import build_report, variant_from_costs, write_report  # noqa: E402

DEFAULT_OUTPUT = _ROOT / "artifacts" / "private_dev_eval"


def require_labels(gt: GroundTruthSet) -> GroundTruthSet:
    """Refuse rather than score nothing. An example row is not a label."""
    if not gt.has_real_labels:
        raise GroundTruthRequired(
            detail=(
                f"{gt.path} contains {len(gt.example_entries)} template row(s) and no "
                "human annotation. Templates are excluded from scoring by design; "
                "annotate at least one query (see evaluation/private_dev/README.md)."
            )
        )
    return gt


def predict(engine: AICCompetitionEngine, entry, *, top_k: int) -> tuple[list[RankedAnswer], Any, float]:
    """Run the task this entry describes. No label is ever shown to the engine."""
    started = time.perf_counter()
    if entry.task == "kis":
        outcome = engine.search_kis_detailed(entry.query, top_k=top_k)
        rows = [
            RankedAnswer(p.video_id, (str(p.frame_id),)) for p in outcome.predictions
        ]
        return rows, outcome.cost, (time.perf_counter() - started) * 1000.0
    if entry.task == "qa":
        predictions, info = engine.answer_qa(
            entry.event_text, entry.question, top_k=top_k,
            expected_answer_type=entry.answer_type or None,
        )
        rows = [
            RankedAnswer(p.video_id, (str(p.frame_id),), p.answer) for p in predictions
        ]
        cost = info["diagnostics"].get("cost")
        return rows, cost, (time.perf_counter() - started) * 1000.0
    outcome = engine.search_trake_detailed(list(entry.events), max_results=top_k)
    rows = [
        RankedAnswer(p.video_id, tuple(str(value) for value in p.row()[1:]))
        for p in outcome.predictions
    ]
    return rows, None, (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--gt-dir", default=str(PRIVATE_DEV_DIR))
    parser.add_argument("--split", choices=list(SPLITS) + ["all"], default="development")
    parser.add_argument("--variant", default="B0_CLEAN", help="Name for this configuration.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    gt = load_private_dev(args.gt_dir, split=None if args.split == "all" else args.split)
    try:
        require_labels(gt)
    except GroundTruthRequired as exc:
        print(json.dumps({
            "error_code": exc.error_code,
            "error": str(exc),
            "banner": gt.banner,
            "real_labels": gt.counts(),
            "example_rows_ignored": len(gt.example_entries),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 9

    config = load_app_config(args.config)
    engine, load = AICCompetitionEngine.from_data_root(app_config=config)

    per_task: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    costs = []
    latencies: list[float] = []
    for entry in gt.real_entries:
        predictions, cost, elapsed = predict(engine, entry, top_k=args.top_k)
        latencies.append(elapsed)
        if cost is not None:
            costs.append(cost)
        report = evaluate_query(entry.task, predictions, entry.to_metric_gt())
        per_task.setdefault(entry.task, []).append(report)
        rows.append({
            "query_id": entry.query_id,
            "task": entry.task,
            "video_id": entry.video_id,
            "returned": len(predictions),
            "latency_ms": round(elapsed, 1),
            **{key: round(value, 6) for key, value in report.items()},
        })

    summary = {task: mean_report(reports) for task, reports in per_task.items()}
    overall = mean_report([report for reports in per_task.values() for report in reports])

    manifest = build_manifest(config, name=args.variant, gt=gt, load=load)
    variant = variant_from_costs(
        args.variant,
        costs=costs,
        latencies_ms=latencies,
        quality={key: round(float(overall[key]), 6) for key in [f"R@{k}" for k in TOP_KS] + ["Final Score"]},
        config_hash=manifest["config_hash"],
        cache_fingerprint=load.cache_fingerprint or "",
        ground_truth=gt.provenance(),
    )
    pareto = build_report([variant], ground_truth=gt.provenance())

    payload = {
        "banner": gt.banner,
        "warning": (
            "PRIVATE DEVELOPMENT GT — NOT OFFICIAL AIC SCORE. These numbers describe a "
            "small locally annotated set, not the AIC benchmark."
        ),
        **report_header(gt),
        "manifest": manifest,
        "split": args.split,
        "per_query": rows,
        "per_task": summary,
        "overall": overall,
        "three_axis": pareto,
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.variant}_{args.split}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(pareto, out_dir, stem=f"{args.variant}_{args.split}_three_axis")

    print(gt.banner, file=sys.stderr)
    print(json.dumps({
        "banner": gt.banner,
        "split": args.split,
        "queries": len(rows),
        "per_task": summary,
        "overall": overall,
        "output": str(out_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
