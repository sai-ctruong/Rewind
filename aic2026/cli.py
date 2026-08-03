"""Command-line helpers for AIC 2026 competition mode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import BenchmarkLogger, QueryLog, environment_snapshot, time_call
from .engine import AICCompetitionEngine
from .metrics import write_submission


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIC 2026 Rewind competition CLI")
    p.add_argument("--data-root", default="data", help="AIC DATA_ROOT")
    p.add_argument("--cache-dir", default="artifacts/aic2026_index")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--production-mode", action="store_true", help="Require the real CLIP encoder")
    p.add_argument("--no-hashing-fallback", action="store_true")
    p.add_argument("--device", help="Encoder device, for example cpu or cuda")
    sub = p.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-index")
    build.add_argument("--limit-videos", type=int)
    build.add_argument("--limit-frames-per-video", type=int)
    build.add_argument("--load-objects", action="store_true")
    build.add_argument("--index-kind", choices=["flat", "hnsw"], default="flat")
    build.add_argument("--include-media-text", action="store_true")
    build.add_argument("--verify-keyframes", action="store_true")

    search = sub.add_parser("search")
    search.add_argument("--task", choices=["kis", "qa", "trake"], default="kis")
    search.add_argument("--query", required=True)
    search.add_argument("--question", default="")
    search.add_argument("--events", nargs="*")
    search.add_argument("--out")
    search.add_argument("--top-k", type=int, default=100)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "build-index":
        _, load = AICCompetitionEngine.from_data_root(
            args.data_root,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild,
            limit_videos=args.limit_videos,
            limit_frames_per_video=args.limit_frames_per_video,
            load_objects=args.load_objects,
            index_kind=args.index_kind,
            include_media_text=args.include_media_text,
            verify_keyframes=args.verify_keyframes,
            production_mode=args.production_mode,
            allow_hashing_fallback=not args.no_hashing_fallback,
            device=args.device,
        )
        print(json.dumps({
            "cache_hit": load.cache_hit,
            "build_seconds": round(load.build_seconds, 3),
            "stats": None if load.stats is None else load.stats.__dict__,
        }, ensure_ascii=False, indent=2))
        return 0

    engine, load = AICCompetitionEngine.from_data_root(
        args.data_root,
        cache_dir=args.cache_dir,
        rebuild=args.rebuild,
        production_mode=args.production_mode,
        allow_hashing_fallback=not args.no_hashing_fallback,
        device=args.device,
    )
    if args.task == "kis":
        preds, latency = time_call(engine.search_kis, args.query, top_k=args.top_k)
    elif args.task == "qa":
        (preds, info), latency = time_call(engine.answer_qa, args.query, args.question, top_k=args.top_k)
    else:
        events = args.events or [x.strip() for x in args.query.split(";") if x.strip()]
        (preds, matches), latency = time_call(engine.search_trake, events, max_results=args.top_k)
    rows = [p.row() for p in preds]
    status = engine.encoder_status()
    print(json.dumps({"encoder": status}, ensure_ascii=False), file=sys.stderr)
    print("\n".join(", ".join(r) for r in rows))
    if args.out:
        write_submission(rows, args.out)
    logger = BenchmarkLogger()
    logger.write_run(
        "cli",
        {"task": args.task, "query": args.query, "cache_hit": load.cache_hit, "encoder": status},
        [QueryLog(args.task, "cli", args.query, latency, rows, {"n": len(rows)})],
        predictions=[{"query_id": "cli", "task": args.task, "rows": rows}],
        environment=environment_snapshot(
            encoder=status,
            dataset=None if load.stats is None else load.stats.__dict__,
            query_count=1,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())