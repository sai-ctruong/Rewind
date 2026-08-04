"""Command-line helpers for AIC 2026 competition mode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkLogger, QueryLog, environment_snapshot, time_call
from .config import AppConfig, config_hash, config_to_dict, load_app_config
from .engine import AICCompetitionEngine
from .metrics import write_submission


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIC 2026 Rewind competition CLI")
    p.add_argument("--config", default="configs/settings.yaml", help="Runtime config YAML")
    p.add_argument("--data-root", default=None, help="Override AIC DATA_ROOT")
    p.add_argument("--cache-dir", default=None, help="Override cache directory")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--production-mode", action=argparse.BooleanOptionalAction, default=None, help="Override production mode")
    p.add_argument("--hashing-fallback", action=argparse.BooleanOptionalAction, default=None, help="Allow hashing fallback when real CLIP is unavailable")
    p.add_argument("--device", default=None, help="Encoder device, for example cpu or cuda")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show-config")

    build = sub.add_parser("build-index")
    build.add_argument("--limit-videos", type=int)
    build.add_argument("--limit-frames-per-video", type=int)
    build.add_argument("--load-objects", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--index-kind", choices=["flat", "hnsw"], default=None)
    build.add_argument("--include-media-text", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--verify-keyframes", action=argparse.BooleanOptionalAction, default=None)

    search = sub.add_parser("search")
    search.add_argument("--task", choices=["kis", "qa", "trake"], default="kis")
    search.add_argument("--query", required=True)
    search.add_argument("--question", default="")
    search.add_argument("--events", nargs="*")
    search.add_argument("--out")
    search.add_argument("--top-k", type=int, default=None)
    return p


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if value is None:
        return
    node = target
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    _set_nested(overrides, ("dataset", "root"), args.data_root)
    _set_nested(overrides, ("dataset", "cache_dir"), args.cache_dir)
    _set_nested(overrides, ("runtime", "production_mode"), args.production_mode)
    _set_nested(overrides, ("runtime", "device"), args.device)
    _set_nested(overrides, ("encoder", "allow_hashing_fallback"), args.hashing_fallback)
    if args.cmd == "build-index":
        _set_nested(overrides, ("dataset", "load_objects"), args.load_objects)
        _set_nested(overrides, ("dataset", "include_media_text"), args.include_media_text)
        _set_nested(overrides, ("dataset", "verify_keyframes"), args.verify_keyframes)
        _set_nested(overrides, ("dataset", "index_kind"), args.index_kind)
    return overrides


def _load_cli_config(args: argparse.Namespace) -> AppConfig:
    return load_app_config(args.config, cli_overrides(args))


def _config_status(config: AppConfig, path: str | Path) -> dict[str, Any]:
    return {"config_path": str(path), "config_hash": config_hash(config), "config": config_to_dict(config)}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    app_config = _load_cli_config(args)
    status = _config_status(app_config, args.config)
    if args.cmd == "show-config":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-index":
        _, load = AICCompetitionEngine.from_data_root(
            app_config=app_config,
            rebuild=args.rebuild,
            limit_videos=args.limit_videos,
            limit_frames_per_video=args.limit_frames_per_video,
        )
        print(json.dumps({
            "cache_hit": load.cache_hit,
            "build_seconds": round(load.build_seconds, 3),
            "stats": None if load.stats is None else load.stats.__dict__,
            "config_hash": status["config_hash"],
        }, ensure_ascii=False, indent=2))
        return 0

    engine, load = AICCompetitionEngine.from_data_root(app_config=app_config, rebuild=args.rebuild)
    top_k = args.top_k
    if args.task == "kis":
        preds, latency = time_call(engine.search_kis, args.query, top_k=top_k)
    elif args.task == "qa":
        (preds, info), latency = time_call(engine.answer_qa, args.query, args.question, top_k=top_k)
    else:
        events = args.events or [x.strip() for x in args.query.split(";") if x.strip()]
        (preds, matches), latency = time_call(engine.search_trake, events, max_results=top_k)
    rows = [p.row() for p in preds]
    encoder_status = engine.encoder_status()
    print(json.dumps({"encoder": encoder_status, "config_hash": status["config_hash"]}, ensure_ascii=False), file=sys.stderr)
    print("\n".join(", ".join(r) for r in rows))
    if args.out:
        write_submission(rows, args.out, max_rows=app_config.submission.max_predictions)
    logger = BenchmarkLogger(app_config.evaluation.output_dir)
    logger.write_run(
        "cli",
        status["config"],
        [QueryLog(args.task, "cli", args.query, latency, rows, {"n": len(rows), "config_hash": status["config_hash"]})],
        {"config_hash": status["config_hash"]},
        predictions=[{"query_id": "cli", "task": args.task, "rows": rows}],
        environment=environment_snapshot(
            encoder=encoder_status,
            dataset=None if load.stats is None else load.stats.__dict__,
            query_count=1,
            config_hash=status["config_hash"],
        ),
        config_hash=status["config_hash"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())