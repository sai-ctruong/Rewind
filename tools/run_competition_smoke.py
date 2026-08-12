"""Reproducible end-to-end smoke for the three AIC tasks.

Runs a fixed query fixture through the real system and records what happened: system
profile, readiness, per-task structural diagnostics, latency, memory, and a real
submission round trip (serialize -> read back -> validate again).

Everything reported here is STRUCTURAL. There is no AIC ground truth in this repository,
so nothing is labelled correct or incorrect and no accuracy is computed.

    .venv\\Scripts\\python.exe tools/run_competition_smoke.py --config configs/competition.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # noqa: E402 - allow running as a plain script
    sys.path.insert(0, str(_ROOT))

from aic2026.config import load_app_config  # noqa: E402
from aic2026.engine import AICCompetitionEngine  # noqa: E402
from aic2026.submission_validation import (  # noqa: E402
    read_submission_csv,
    submission_rows_for,
    validate_submission,
    validation_report_path,
    write_submission_csv,
    write_validation_report,
)
from aic2026.system_profile import build_system_profile, evaluate_readiness  # noqa: E402

DEFAULT_FIXTURE = _ROOT / "tests" / "fixtures" / "final_smoke_queries.json"
DEFAULT_OUTPUT = _ROOT / "artifacts" / "final_release_smoke"


def rss_mb() -> Optional[float]:
    """Resident memory, or None if psutil is unavailable. No new dependency is added."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - optional
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def summarize(values: list[float]) -> dict[str, Any]:
    """p50/p95 only when there are enough samples to mean anything."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    out: dict[str, Any] = {
        "count": len(ordered),
        "min_ms": round(ordered[0], 1),
        "median_ms": round(statistics.median(ordered), 1),
        "max_ms": round(ordered[-1], 1),
    }
    if len(ordered) >= 5:
        out["p50_ms"] = round(statistics.median(ordered), 1)
        # Nearest-rank, rounding UP: with few samples a rounded-down index reports a p95
        # below the slowest observed query, which understates the tail it exists to show.
        rank = max(1, math.ceil(0.95 * len(ordered)))
        out["p95_ms"] = round(ordered[rank - 1], 1)
    return out


def roundtrip(task: str, predictions, out_dir: Path, name: str, event_count=None) -> dict[str, Any]:
    """serialize -> read back -> validate again -> serialize, and prove it is stable."""
    rows = submission_rows_for(task, predictions)
    first = validate_submission(task, rows, event_count=event_count)
    result: dict[str, Any] = {
        "task": task,
        "preflight_valid": first.valid,
        "rows": first.row_count,
        "duplicates_removed": first.duplicates_removed,
        "errors": [issue.code for issue in first.errors][:5],
    }
    if not first.valid:
        result["exported"] = False
        return result
    path = write_submission_csv(first, out_dir / f"{name}.csv")
    write_validation_report(
        first, validation_report_path(path), metadata={"source": "final_smoke", "task": task}
    )
    parsed = read_submission_csv(path)
    second = validate_submission(task, parsed, event_count=event_count)
    reserialized = write_submission_csv(second, out_dir / f"{name}.roundtrip.csv")
    result.update(
        {
            "exported": True,
            "csv_rows": len(parsed),
            "revalidated_valid": second.valid,
            "revalidated_rows": second.row_count,
            "byte_identical_roundtrip": path.read_bytes() == reserialized.read_bytes(),
            "csv": path.name,
        }
    )
    (out_dir / f"{name}.roundtrip.csv").unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    run_dir = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = run_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    memory = {"before_engine_mb": rss_mb()}
    config = load_app_config(args.config)

    started = time.perf_counter()
    engine, load = AICCompetitionEngine.from_data_root(app_config=config)
    startup_ms = (time.perf_counter() - started) * 1000.0
    memory["after_engine_mb"] = rss_mb()

    profile = build_system_profile(config, config_path=args.config, engine=engine)
    readiness = evaluate_readiness(config, config_path=args.config, engine=engine, profile=profile)
    print(f"readiness: {readiness.status}", file=sys.stderr)

    report: dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": args.config,
        "fixture": Path(args.fixture).name,
        "disclaimer": (
            "Structural integration smoke. No AIC ground truth exists, so no result is "
            "labelled correct and no accuracy is reported."
        ),
        "system_profile": profile.to_dict(),
        "readiness": readiness.to_dict(),
        "cache": {
            "dir": str(config.dataset.cache_dir),
            "hit": load.cache_hit,
            "valid": load.cache_valid,
            "stale": load.cache_stale,
            "fingerprint": load.cache_fingerprint,
            "videos": len({raw.video_id for raw in engine.entry.raws.values()}),
            "frames": int(engine.entry.num_indexed),
        },
        "timing": {"startup_ms": round(startup_ms, 1)},
        "tasks": {},
    }

    # ------------------------------------------------------------------ KIS
    kis_queries = list(fixture.get("kis", []))
    kis_rows: list[dict[str, Any]] = []
    kis_latency: list[float] = []
    first_query_ms = None
    model_load_ms = None
    for index, query in enumerate(kis_queries):
        begin = time.perf_counter()
        outcome = engine.search_kis_detailed(query, top_k=args.top_k)
        elapsed = (time.perf_counter() - begin) * 1000.0
        if index == 0:
            # The first query pays the one-off CLIP text-encoder load. It is excluded
            # from the warm distribution and reported separately, because folding a
            # multi-second model load into "retrieval latency" would misdescribe both.
            first_query_ms = elapsed
        else:
            kis_latency.append(elapsed)
        diagnostics = outcome.diagnostics()
        channels = diagnostics.get("channels", {}).get("channels", {})
        kis_rows.append(
            {
                "query": query,
                "results": len(outcome.predictions),
                "candidate_union_size": diagnostics.get("channels", {}).get("candidate_union_size"),
                "channels_contributing": sorted(
                    name for name, info in channels.items() if info.get("candidates_returned")
                ),
                "exclusive": diagnostics.get("channels", {}).get("exclusive_candidates", {}),
                "refinement_triggered": diagnostics.get("refinement_triggered", False),
                "frames_decoded": diagnostics.get("frames_decoded", 0),
                "coarse_search_ms": diagnostics.get("coarse_search_ms"),
                "total_ms": round(elapsed, 1),
                "top": [
                    [p.video_id, p.frame_id] for p in outcome.predictions[:3]
                ],
            }
        )
        print(f"[kis] {query!r} -> {len(outcome.predictions)} rows, {round(elapsed)}ms", file=sys.stderr)

    last = engine.search_kis(kis_queries[0], top_k=args.top_k) if kis_queries else []
    report["tasks"]["kis"] = {
        "queries": len(kis_queries),
        "runs": kis_rows,
        "latency": summarize(kis_latency),
        "submission": roundtrip("kis", last, submission_dir, "kis_submission"),
    }
    warm = summarize(kis_latency)
    report["timing"]["first_query_ms"] = None if first_query_ms is None else round(first_query_ms, 1)
    report["timing"]["warm_query"] = warm
    # An estimate, and labelled as one: the difference between the cold first query and
    # the warm median is dominated by the one-off CLIP text-encoder load.
    report["timing"]["model_load_estimate_ms"] = (
        None
        if first_query_ms is None or not warm.get("median_ms")
        else round(first_query_ms - warm["median_ms"], 1)
    )

    # ------------------------------------------------------------------- Q&A
    qa_rows: list[dict[str, Any]] = []
    qa_latency: list[float] = []
    qa_predictions = []
    for item in fixture.get("qa", []):
        begin = time.perf_counter()
        predictions, info = engine.answer_qa(
            item.get("event", ""),
            item.get("question", ""),
            top_k=20,
            expected_answer_type=item.get("expected_answer_type"),
        )
        elapsed = (time.perf_counter() - begin) * 1000.0
        qa_latency.append(elapsed)
        qa_predictions = predictions
        diagnostics = info["diagnostics"]
        qa_rows.append(
            {
                "question": item.get("question"),
                "expected_answer_type": info.get("expected_answer_type"),
                "hypotheses": diagnostics["retrieved_video_hypotheses"],
                "answered": diagnostics["answered_video_hypotheses"],
                "abstentions": diagnostics["abstentions"],
                "backend": info["backend"]["backend_type"],
                "backend_visual_capable": info["backend"]["visual_capable"],
                "cross_video_answer_copy_count": diagnostics["cross_video_answer_copy_count"],
                "answer_without_matching_evidence_video_count": diagnostics[
                    "answer_without_matching_evidence_video_count"
                ],
                "evidence_frames_used": diagnostics["evidence_frames_used"],
                "retrieval_ms": diagnostics["retrieval_ms"],
                "vqa_ms": diagnostics["vqa_ms"],
                "total_ms": round(elapsed, 1),
            }
        )
        print(
            f"[qa] {item.get('question')!r} -> {diagnostics['answered_video_hypotheses']}"
            f"/{diagnostics['retrieved_video_hypotheses']} answered, {round(elapsed)}ms",
            file=sys.stderr,
        )
    report["tasks"]["qa"] = {
        "queries": len(qa_rows),
        "runs": qa_rows,
        "latency": summarize(qa_latency),
        # Engine-produced Q&A rows are expected to be non-submittable here: the backend
        # is a non-visual mock. That refusal IS the correct outcome, not a failure.
        "submission": roundtrip("qa", qa_predictions, submission_dir, "qa_engine_submission"),
        "note": (
            "No production visual Q&A backend is available on this machine. Retrieval, "
            "per-video hypothesis grounding and evidence selection are exercised; "
            "engine-produced ANSWERS are correctly refused by the submission validator."
        ),
    }

    # ----------------------------------------------------------------- TRAKE
    trake_rows: list[dict[str, Any]] = []
    trake_latency: list[float] = []
    trake_last = []
    trake_event_count = None
    for events in fixture.get("trake", []):
        begin = time.perf_counter()
        outcome = engine.search_trake_detailed(list(events), max_results=args.top_k)
        elapsed = (time.perf_counter() - begin) * 1000.0
        trake_latency.append(elapsed)
        trake_last = outcome.predictions
        trake_event_count = len(events)
        diagnostics = outcome.diagnostics
        structural = outcome.structural_summary()
        trake_rows.append(
            {
                "events": list(events),
                "event_count": len(events),
                "initial_candidate_counts": diagnostics.get("initial_candidate_counts"),
                "expansion_triggered": diagnostics.get("candidate_expansion_triggered"),
                "events_expanded": diagnostics.get("events_expanded"),
                "videos_with_full_event_coverage": diagnostics.get("videos_with_full_event_coverage"),
                "k_best_alignments_generated": diagnostics.get("k_best_alignments_generated"),
                "unique_sequences_generated": diagnostics.get("unique_sequences_generated"),
                "returned": len(outcome.predictions),
                "frames_decoded": diagnostics.get("frames_decoded", 0),
                "structural": structural,
                "unordered_submission_sequence_count": diagnostics.get(
                    "unordered_submission_sequence_count"
                ),
                "total_ms": round(elapsed, 1),
            }
        )
        print(
            f"[trake] {len(events)} events -> {len(outcome.predictions)} sequences, "
            f"{round(elapsed)}ms",
            file=sys.stderr,
        )
    report["tasks"]["trake"] = {
        "queries": len(trake_rows),
        "runs": trake_rows,
        "latency": summarize(trake_latency),
        "submission": roundtrip(
            "trake", trake_last, submission_dir, "trake_submission", event_count=trake_event_count
        ),
    }

    memory["after_queries_mb"] = rss_mb()
    report["memory"] = memory
    report["structural_invariants"] = {
        "malformed_prediction_count": sum(
            row["structural"]["malformed_prediction_count"] for row in trake_rows
        ),
        "wrong_event_count_prediction_count": sum(
            row["structural"]["wrong_event_count_prediction_count"] for row in trake_rows
        ),
        "cross_video_step_count": sum(
            row["structural"]["cross_video_step_count"] for row in trake_rows
        ),
        "unordered_submission_sequence_count": sum(
            row["unordered_submission_sequence_count"] or 0 for row in trake_rows
        ),
        "cross_video_answer_copy_count": sum(
            row["cross_video_answer_copy_count"] for row in qa_rows
        ),
        "answer_without_matching_evidence_video_count": sum(
            row["answer_without_matching_evidence_video_count"] for row in qa_rows
        ),
    }

    (run_dir / "system_profile.json").write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "queries.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "performance.json").write_text(
        json.dumps(
            {
                "system": {
                    "platform": profile.platform,
                    "python": profile.python_version,
                    "device": profile.kis.get("scorer_device"),
                    "note": "CPU-only environment; timings are hardware specific.",
                },
                "dataset": {
                    "videos": report["cache"]["videos"],
                    "frames": report["cache"]["frames"],
                    "cache_valid": report["cache"]["valid"],
                },
                "timing": report["timing"],
                "kis_latency": report["tasks"]["kis"]["latency"],
                "qa_latency": report["tasks"]["qa"]["latency"],
                "trake_latency": report["tasks"]["trake"]["latency"],
                "memory": memory,
                "frames_decoded": sum(row.get("frames_decoded", 0) for row in kis_rows)
                + sum(row.get("frames_decoded", 0) for row in trake_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "results.html").write_text(render_html(report), encoding="utf-8")

    print(json.dumps({
        "readiness": readiness.status,
        "kis_submission": report["tasks"]["kis"]["submission"],
        "qa_submission_valid": report["tasks"]["qa"]["submission"]["preflight_valid"],
        "trake_submission": report["tasks"]["trake"]["submission"],
        "structural_invariants": report["structural_invariants"],
        "timing": report["timing"],
        "memory": memory,
    }, indent=2))
    return 0


def render_html(report: dict) -> str:
    import html as html_module

    esc = html_module.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>AIC 2026 final release smoke</title><style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:24px;max-width:1150px;color:#111}",
        "table{border-collapse:collapse;width:100%;margin:8px 0 20px}",
        "td,th{border:1px solid #ccc;padding:6px 8px;text-align:left;vertical-align:top}",
        ".note{background:#fffbe6;border:1px solid #e8d48b;padding:10px;margin:10px 0}",
        "code{background:#f4f4f4;padding:1px 4px}",
        "</style></head><body>",
        "<h1>AIC 2026 final release smoke</h1>",
        f"<p class='note'><b>Structural only.</b> {esc(report['disclaimer'])}</p>",
        f"<p>version <code>{esc(report['system_profile']['project_version'])}</code> · commit "
        f"<code>{esc(str(report['system_profile']['git_commit']))}</code> · config "
        f"<code>{esc(report['config'])}</code> · readiness "
        f"<b>{esc(report['readiness']['status'])}</b></p>",
        f"<pre>{esc(json.dumps(report['structural_invariants'], indent=2))}</pre>",
        f"<pre>{esc(json.dumps(report['timing'], indent=2))}</pre>",
        "<h2>KIS</h2><table><tr><th>Query</th><th>Rows</th><th>Union</th>"
        "<th>Channels</th><th>Top (video, submitted frame)</th><th>ms</th></tr>",
    ]
    for row in report["tasks"]["kis"]["runs"]:
        parts.append(
            f"<tr><td>{esc(row['query'])}</td><td>{row['results']}</td>"
            f"<td>{row['candidate_union_size']}</td>"
            f"<td>{esc(', '.join(row['channels_contributing']))}</td>"
            f"<td><code>{esc(json.dumps(row['top']))}</code></td>"
            f"<td>{row['total_ms']}</td></tr>"
        )
    parts.append("</table>")

    parts.append(
        f"<h2>Q&amp;A</h2><p class='note'>{esc(report['tasks']['qa']['note'])}</p>"
        "<table><tr><th>Question</th><th>Type</th><th>Answered / hypotheses</th>"
        "<th>Backend</th><th>Cross-video copies</th><th>ms</th></tr>"
    )
    for row in report["tasks"]["qa"]["runs"]:
        parts.append(
            f"<tr><td>{esc(str(row['question']))}</td><td>{esc(str(row['expected_answer_type']))}</td>"
            f"<td>{row['answered']} / {row['hypotheses']}</td>"
            f"<td>{esc(row['backend'])} ({'visual' if row['backend_visual_capable'] else 'NON-VISUAL'})</td>"
            f"<td>{row['cross_video_answer_copy_count']}</td><td>{row['total_ms']}</td></tr>"
        )
    parts.append("</table>")

    parts.append(
        "<h2>TRAKE</h2><table><tr><th>Events</th><th>Coverage</th><th>Sequences</th>"
        "<th>Expansion</th><th>Structural</th><th>ms</th></tr>"
    )
    for row in report["tasks"]["trake"]["runs"]:
        parts.append(
            f"<tr><td>{esc(' &rarr; '.join(row['events']))}</td>"
            f"<td>{row['videos_with_full_event_coverage']}</td><td>{row['returned']}</td>"
            f"<td>{row['expansion_triggered']} {esc(json.dumps(row['events_expanded']))}</td>"
            f"<td><code>{esc(json.dumps(row['structural']))}</code></td>"
            f"<td>{row['total_ms']}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Submission round trip</h2><pre>")
    parts.append(esc(json.dumps({
        task: report["tasks"][task]["submission"] for task in ("kis", "qa", "trake")
    }, indent=2)))
    parts.append("</pre></body></html>")
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
