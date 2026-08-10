"""Inventory the original AIC videos on disk and their supporting data.

Writes two artifacts:

    artifacts/video_inventory.json         what MP4s exist, sizes, collections
    artifacts/video_support_coverage.json  per video: map / CLIP / JPEG / objects / media

    .venv\\Scripts\\python.exe tools/inspect_video_inventory.py --data-root data --probe-readable

Read-only: it never writes to DATA_ROOT and never decodes more than one frame per file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic2026.video_inventory import (  # noqa: E402 -- after the sys.path bootstrap above
    discover_videos,
    existing_video_ids_with_retrieval_support,
    summarize_coverage,
    support_coverage,
)


def _write(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--inventory-output", default="artifacts/video_inventory.json")
    parser.add_argument("--coverage-output", default="artifacts/video_support_coverage.json")
    parser.add_argument(
        "--probe-readable",
        action="store_true",
        help="Open each MP4 and decode one frame to confirm it is usable.",
    )
    args = parser.parse_args(argv)

    inventory = discover_videos(args.data_root, probe_readable=args.probe_readable)
    inventory_payload = inventory.to_dict()
    _write(inventory_payload, Path(args.inventory_output))

    coverage = support_coverage(args.data_root)
    video_coverage = [item for item in coverage if item.video]
    coverage_payload = {
        "data_root": inventory.data_root,
        "summary_all_discovered_ids": summarize_coverage(coverage),
        "summary_videos_on_disk": summarize_coverage(video_coverage),
        "existing_video_scope_ids": list(
            existing_video_ids_with_retrieval_support(args.data_root)
        ),
        "videos": [item.to_dict() for item in video_coverage],
        "all_discovered": [item.to_dict() for item in coverage],
    }
    _write(coverage_payload, Path(args.coverage_output))

    print(
        json.dumps(
            {
                "video_root": inventory.video_root,
                "video_count": inventory_payload["video_count"],
                "collections": inventory_payload["collections"],
                "duplicate_ids": inventory_payload["duplicate_ids"],
                "unreadable_count": inventory_payload["unreadable_count"],
                "unreadable": inventory_payload["unreadable"],
                "total_bytes": inventory_payload["total_bytes"],
                "total_gib": round(inventory_payload["total_bytes"] / 1024**3, 3),
                "coverage_videos_on_disk": coverage_payload["summary_videos_on_disk"],
                "existing_video_scope_count": len(coverage_payload["existing_video_scope_ids"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
