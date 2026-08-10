"""Command-line helpers for AIC 2026 competition mode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkLogger, QueryLog, environment_snapshot, time_call
from .cache_manifest import (
    CacheManifestError,
    LegacyCacheError,
    StaleCacheError,
    cache_build_options_from_config,
    detect_code_version,
    inspect_cache,
)
from .config import AppConfig, ConfigError, config_hash, config_to_dict, load_app_config
from .dataset_scope import DatasetScopeError
from .dataset_validation import (
    DatasetAlignmentError,
    DatasetError,
    DatasetValidationError,
    inspect_aic_dataset,
    write_dataset_report,
)
from .engine import AICCompetitionEngine
from .metrics import write_submission


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIC 2026 Rewind competition CLI")
    p.add_argument("--config", default="configs/settings.yaml", help="Runtime config YAML")
    p.add_argument("--data-root", default=None, help="Override AIC DATA_ROOT")
    p.add_argument("--cache-dir", default=None, help="Override cache directory")
    p.add_argument(
        "--video-include",
        action="append",
        default=None,
        metavar="PATTERN",
        help=(
            "Override dataset.scope.include_patterns with a video-ID glob, for example "
            '"L21_*". Repeat the option to include several patterns.'
        ),
    )
    p.add_argument(
        "--video-exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Override dataset.scope.exclude_patterns; applied after the include patterns.",
    )
    p.add_argument(
        "--scope-existing-videos",
        action="store_true",
        help=(
            "Restrict the dataset to videos whose original MP4 is on disk AND that have "
            "map-keyframes plus CLIP features. Resolved from DATA_ROOT at run time, and "
            "combinable with --video-include/--video-exclude."
        ),
    )
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--production-mode", action=argparse.BooleanOptionalAction, default=None, help="Override production mode")
    p.add_argument("--hashing-fallback", action=argparse.BooleanOptionalAction, default=None, help="Allow hashing fallback when real CLIP is unavailable")
    p.add_argument("--allow-stale-cache", action=argparse.BooleanOptionalAction, default=None, help="Explicitly allow stale or legacy cache outside production")
    p.add_argument("--device", default=None, help="Encoder device, for example cpu or cuda")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show-config")
    sub.add_parser("inspect-cache")

    inspect_data = sub.add_parser("inspect-data")
    inspect_data.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    inspect_data.add_argument("--deep", action="store_true")
    inspect_data.add_argument("--limit-videos", type=int)
    inspect_data.add_argument("--output")
    inspect_data.add_argument("--summary-only", action="store_true")

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
    # Scope precedence: explicit CLI override > YAML > default. Passing neither option
    # leaves the YAML scope untouched; settings.yaml is never rewritten.
    _set_nested(overrides, ("dataset", "scope", "include_patterns"), args.video_include)
    _set_nested(overrides, ("dataset", "scope", "exclude_patterns"), args.video_exclude)
    if getattr(args, "scope_existing_videos", False):
        _set_nested(overrides, ("dataset", "scope", "mode"), "existing_videos")
    _set_nested(overrides, ("runtime", "production_mode"), args.production_mode)
    _set_nested(overrides, ("runtime", "device"), args.device)
    _set_nested(overrides, ("encoder", "allow_hashing_fallback"), args.hashing_fallback)
    _set_nested(overrides, ("cache", "allow_stale_cache"), args.allow_stale_cache)
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


def _cache_metadata(load) -> dict[str, Any]:
    manifest = load.cache_manifest
    return {
        "cache_hit": load.cache_hit,
        "cache_valid": load.cache_valid,
        "cache_stale": load.cache_stale,
        "cache_fingerprint": load.cache_fingerprint,
        "cache_manifest_schema": None if manifest is None else manifest.schema_version,
        "data_signature": None if manifest is None else manifest.data_signature,
        "code_version": load.cache_code_version,
    }


def _print_cache_error(exc: Exception, code: str) -> None:
    print(json.dumps({"error": str(exc), "error_code": code}, ensure_ascii=False, indent=2), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        app_config = _load_cli_config(args)
    except ConfigError as exc:
        _print_cache_error(exc, "INVALID_CONFIG")
        return 2
    except DatasetScopeError as exc:
        _print_cache_error(exc, "INVALID_DATASET_SCOPE")
        return 2
    status = _config_status(app_config, args.config)
    if args.cmd == "show-config":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "inspect-data":
        try:
            try:
                report = inspect_aic_dataset(
                    Path(app_config.dataset.root),
                    app_config=app_config,
                    strict=args.strict,
                    deep=args.deep,
                    limit_videos=args.limit_videos,
                )
            except DatasetValidationError as exc:
                report = exc.report
            report_path = None
            if args.output:
                report_path = str(
                    write_dataset_report(
                        report,
                        args.output,
                        summary_only=args.summary_only,
                    )
                )
            payload = report.summary(report_path=report_path)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if report.valid_for_index_build else 5
        except DatasetScopeError as exc:
            _print_cache_error(exc, "EMPTY_DATASET_SCOPE")
            return 5
        except DatasetError as exc:
            _print_cache_error(exc, "DATASET_INSPECTION_ERROR")
            return 6

    if args.cmd == "inspect-cache":
        expected = cache_build_options_from_config(app_config)
        report = inspect_cache(
            app_config.dataset.cache_dir,
            expected,
            validate_data_signature=app_config.cache.validate_data_signature,
            code_version_policy=app_config.cache.code_version_policy,
            current_code_version=detect_code_version(),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["corrupt"]:
            return 4
        if report["stale"] or report["legacy"]:
            return 3
        return 0

    try:
        if args.cmd == "build-index":
            _, load = AICCompetitionEngine.from_data_root(
                app_config=app_config,
                rebuild=args.rebuild,
                limit_videos=args.limit_videos,
                limit_frames_per_video=args.limit_frames_per_video,
            )
            print(json.dumps({
                "cache_hit": load.cache_hit,
                "cache_valid": load.cache_valid,
                "cache_legacy": load.cache_legacy,
                "cache_stale": load.cache_stale,
                "cache_stale_reason": load.cache_stale_reason,
                "cache_manifest_path": load.cache_manifest_path,
                "cache_fingerprint": load.cache_fingerprint,
                "cache_mismatches": load.cache_mismatches,
                "cache_warnings": load.cache_warnings,
                "build_seconds": round(load.build_seconds, 3),
                "stats": None if load.stats is None else load.stats.__dict__,
                "manifest": None if load.cache_manifest is None else load.cache_manifest.to_dict(),
                "config_hash": status["config_hash"],
            }, ensure_ascii=False, indent=2))
            return 0

        engine, load = AICCompetitionEngine.from_data_root(
            app_config=app_config, rebuild=args.rebuild
        )
    except (DatasetAlignmentError, DatasetValidationError) as exc:
        _print_cache_error(exc, "DATASET_INVALID")
        return 5
    except DatasetScopeError as exc:
        _print_cache_error(exc, "EMPTY_DATASET_SCOPE")
        return 5
    except DatasetError as exc:
        _print_cache_error(exc, "DATASET_ERROR")
        return 6
    except LegacyCacheError as exc:
        _print_cache_error(exc, "LEGACY_CACHE")
        return 3
    except StaleCacheError as exc:
        _print_cache_error(exc, "STALE_CACHE")
        return 3
    except CacheManifestError as exc:
        _print_cache_error(exc, "CORRUPT_CACHE_MANIFEST")
        return 4

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
    cache_meta = _cache_metadata(load)
    logger.write_run(
        "cli",
        status["config"],
        [QueryLog(args.task, "cli", args.query, latency, rows, {
            "n": len(rows), "config_hash": status["config_hash"], **cache_meta
        })],
        {"config_hash": status["config_hash"], **cache_meta},
        predictions=[{"query_id": "cli", "task": args.task, "rows": rows}],
        environment=environment_snapshot(
            encoder=encoder_status,
            dataset=None if load.stats is None else load.stats.__dict__,
            query_count=1,
            config_hash=status["config_hash"],
        ),
        config_hash=status["config_hash"],
        cache_manifest=load.cache_manifest,
        cache_status=cache_meta,
        dataset_report=None if load.stats is None else load.stats.dataset_report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())