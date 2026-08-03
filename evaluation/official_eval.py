"""End-to-end official-style evaluation for AIC-format JSONL labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from aic2026.benchmark import BenchmarkLogger, QueryLog, environment_snapshot, time_call
from aic2026.metrics import RankedAnswer, evaluate_query, mean_report, parse_ground_truth_row


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_labels(
    labels: Sequence[dict], predictor: Callable[[dict], Sequence[Sequence[object]]], *, run_name: str = "baseline",
    out_dir: str | Path = "evaluation/benchmarks/aic2026", environment: dict | None = None,
) -> tuple[dict, Path]:
    reports, logs, predictions, errors = [], [], [], []
    extra = {"video_correct": [], "frame_correct": [], "answer_correct": [], "end_to_end_correct": []}
    for index, label in enumerate(labels):
        task = str(label["task"]).lower()
        rows, latency = time_call(predictor, label)
        has_answer = task in {"qa", "q&a", "vqa"}
        ranked = [RankedAnswer.from_row(row, has_answer=has_answer) for row in rows[:100]]
        gt = parse_ground_truth_row(task, label)
        report = evaluate_query(task, ranked, gt)
        reports.append(report)
        query_id = str(label.get("query_id", index))
        logs.append(QueryLog(task, query_id, str(label.get("query", "")), latency, [list(map(str, row)) for row in rows]))
        predictions.append({"query_id": query_id, "task": task, "rows": [list(row) for row in rows]})
        if report["R@100"] < 1.0:
            errors.append({"query_id": query_id, "category": classify_error(task, ranked, gt, label), "report": report})
    summary = mean_report(reports)
    logger = BenchmarkLogger(out_dir)
    run_dir = logger.write_run(
        run_name, {"label_count": len(labels)}, logs, summary,
        predictions=predictions, errors=errors,
        environment=environment or environment_snapshot(query_count=len(labels)),
    )
    return summary, run_dir


def classify_error(task: str, predictions, gt, label: dict) -> str:
    if not predictions or all(pred.video_id != gt.video_id for pred in predictions):
        return "wrong video"
    if task == "trake":
        return "temporal order error"
    if task in {"qa", "q&a", "vqa"}:
        return "Q&A hallucination"
    return "right video wrong frame"


def require_labels(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Please provide an AIC ground-truth file or annotate at least 50 KIS, 30 Q&A, and 30 TRAKE development queries."
        )
    return path
