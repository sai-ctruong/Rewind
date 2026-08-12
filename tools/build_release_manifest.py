"""Generate the release manifest and a cache/artifact inventory.

Output is gitignored (`artifacts/`): it is environment-specific, so it is generated on
demand rather than committed. Nothing is deleted; obsolete caches are only identified.

    .venv\\Scripts\\python.exe tools/build_release_manifest.py --config configs/competition.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_ROOT))

from aic2026.cache_manifest import CACHE_MANIFEST_FILENAME  # noqa: E402
from aic2026.config import load_app_config  # noqa: E402
from aic2026.system_profile import build_system_profile, evaluate_readiness  # noqa: E402
from aic2026.version import PROJECT_VERSION, git_commit, git_is_dirty  # noqa: E402

ARTIFACTS = _ROOT / "artifacts"


def directory_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return round(total / (1024 * 1024), 1)


def cache_inventory(active_cache: str) -> list[dict[str, Any]]:
    """Every cache directory on disk, classified. Nothing is removed."""
    active = Path(active_cache).resolve(strict=False)
    out: list[dict[str, Any]] = []
    if not ARTIFACTS.is_dir():
        return out
    for path in sorted(ARTIFACTS.iterdir()):
        if not path.is_dir() or not (path / "entry").is_dir():
            continue
        manifest_path = path / CACHE_MANIFEST_FILENAME
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                manifest = {"error": "unreadable manifest"}
        is_active = path.resolve(strict=False) == active
        out.append(
            {
                "path": str(path.relative_to(_ROOT)).replace("\\", "/"),
                "role": "active competition cache" if is_active else "other cache",
                "has_manifest": manifest_path.is_file(),
                "schema_version": manifest.get("schema_version"),
                "channel_schema_version": manifest.get("channel_schema_version"),
                "cache_fingerprint": manifest.get("cache_fingerprint"),
                "scope": (manifest.get("dataset_scope") or {}).get("mode"),
                "selected_video_count": manifest.get("selected_video_count"),
                "load_objects": manifest.get("load_objects"),
                "include_media_text": manifest.get("include_media_text"),
                "size_mb": directory_size_mb(path),
                "safe_to_remove_manually": not is_active,
            }
        )
    return out


def artifact_inventory(active_cache: str) -> list[dict[str, Any]]:
    """Categorize everything under artifacts/. Guidance only; nothing is deleted."""
    active = Path(active_cache).resolve(strict=False)
    categories = {
        "video_frame_cache": "derived disposable (frames decoded from MP4s)",
        "submissions": "generated submissions",
    }
    out: list[dict[str, Any]] = []
    if not ARTIFACTS.is_dir():
        return out
    for path in sorted(ARTIFACTS.iterdir()):
        if not path.is_dir():
            continue
        if (path / "entry").is_dir():
            category = (
                "required runtime (ACTIVE cache)"
                if path.resolve(strict=False) == active
                else "other cache (inactive)"
            )
        elif path.name in categories:
            category = categories[path.name]
        elif "smoke" in path.name or path.name.endswith("_report"):
            category = "smoke/report (regenerable)"
        else:
            category = "other"
        out.append(
            {
                "path": str(path.relative_to(_ROOT)).replace("\\", "/"),
                "category": category,
                "size_mb": directory_size_mb(path),
            }
        )
    return out


def test_count() -> int | None:
    """Collected test count, or None if pytest could not be asked."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--collect-only"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(result.stdout.splitlines()):
        if "test" in line and "collected" in line:
            for token in line.split():
                if token.isdigit():
                    return int(token)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--output", default=str(ARTIFACTS / "release_manifest.json"))
    parser.add_argument("--tests", action=argparse.BooleanOptionalAction, default=False,
                        help="Collect the test count (slow).")
    args = parser.parse_args()

    config = load_app_config(args.config)
    profile = build_system_profile(config, config_path=args.config)
    readiness = evaluate_readiness(config, config_path=args.config, profile=profile)

    manifest = {
        "version": PROJECT_VERSION,
        "git_commit": git_commit(short=False),
        "git_dirty": git_is_dirty(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": args.config,
        "config_hash": profile.config_hash,
        "cache_fingerprint": profile.cache_fingerprint,
        "selected_video_ids_hash": profile.selected_video_ids_hash,
        "selected_video_count": profile.selected_video_count,
        "record_schema_version": profile.record_schema_version,
        "cache_schema_version": profile.cache_schema_version,
        "channel_schema_version": profile.channel_schema_version,
        "submission_validation_version": profile.submission_validation_version,
        "readiness_status": readiness.status,
        "warnings": [item.name for item in readiness.warnings],
        "failures": [item.name for item in readiness.failures],
        "test_count": test_count() if args.tests else None,
        "cache_inventory": cache_inventory(str(config.dataset.cache_dir)),
        "artifact_inventory": artifact_inventory(str(config.dataset.cache_dir)),
        "note": (
            "Structural release identity. Readiness reports whether the system can run "
            "and produce well-formed submissions; it makes no claim about answer quality, "
            "and no AIC ground truth exists in this repository."
        ),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
