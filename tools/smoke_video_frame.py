"""Smoke-test the MP4 visual fallback against the real dataset.

Picks a mapped keyframe whose BTC JPEG is missing but whose original MP4 exists, and
proves the frame can be decoded on demand. Also forces the video path for a video that
*does* have JPEGs, via `prefer_keyframe_jpeg=False`, without deleting anything.

    .venv\\Scripts\\python.exe tools/smoke_video_frame.py --data-root data

Read-only with respect to DATA_ROOT: derived frames are written only under
`artifacts/video_frame_cache/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic2026.frame_provider import FrameProvider  # noqa: E402 -- after sys.path bootstrap
from aic2026.video_inventory import (  # noqa: E402
    existing_video_ids_with_retrieval_support,
    video_path_for,
)
from ingestion.schemas import RawKeyframe  # noqa: E402


def _map_rows(data_root: Path, video_id: str) -> list[dict[str, str]]:
    path = data_root / "map-keyframes" / f"{video_id}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _keyframe_jpeg(data_root: Path, video_id: str, ordinal: int) -> Path | None:
    folder = data_root / "keyframes" / video_id
    for stem in (f"{ordinal:03d}", f"{ordinal:04d}", str(ordinal)):
        for extension in (".jpg", ".jpeg", ".png"):
            candidate = folder / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    return None


def _record(data_root: Path, video_id: str, row: dict[str, str]) -> RawKeyframe:
    ordinal = int(row["n"])
    jpeg = _keyframe_jpeg(data_root, video_id, ordinal)
    return RawKeyframe(
        id=f"{video_id}/kf_{ordinal:06d}",
        video_id=video_id,
        timestamp=float(row["pts_time"]),
        image_path=None if jpeg is None else str(jpeg),
        source_video=str(video_path_for(data_root, video_id)),
        frame_idx=int(row["frame_idx"]),
        keyframe_ordinal=ordinal,
    )


def _valid_jpeg(payload: bytes | None) -> dict[str, Any]:
    if not payload:
        return {"valid": False, "reason": "no bytes"}
    facts: dict[str, Any] = {"bytes": len(payload), "jpeg_magic": payload[:2] == b"\xff\xd8"}
    try:
        import cv2
        import numpy as np

        array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        facts["decodable"] = array is not None
        facts["shape"] = None if array is None else list(array.shape)
    except Exception as exc:  # pragma: no cover - diagnostic path
        facts["decodable"] = False
        facts["error"] = str(exc)
    facts["valid"] = bool(facts.get("jpeg_magic") and facts.get("decodable"))
    return facts


def _case(provider: FrameProvider, record: RawKeyframe, *, prefer_jpeg: bool, label: str) -> dict[str, Any]:
    result = provider.get_frame(record, prefer_keyframe_jpeg=prefer_jpeg)
    return {
        "case": label,
        "keyframe_id": record.id,
        "keyframe_ordinal": record.keyframe_ordinal,
        "official_frame_idx": record.frame_idx,
        "prefer_keyframe_jpeg": prefer_jpeg,
        "btc_jpeg_on_disk": record.image_path is not None,
        "result": result.to_dict(),
        "official_frame_idx_preserved": result.frame_idx == record.frame_idx,
        "image": _valid_jpeg(result.image_bytes),
        "cache_path": result.cache_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="artifacts/video_frame_smoke.json")
    parser.add_argument("--cache-dir", default="artifacts/video_frame_cache")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    provider = FrameProvider(data_root, cache_dir=args.cache_dir)
    candidates = existing_video_ids_with_retrieval_support(data_root)
    if not candidates:
        print(json.dumps({"error": "No video has both map and CLIP support."}), file=sys.stderr)
        return 2

    fallback_case: dict[str, Any] | None = None
    forced_case: dict[str, Any] | None = None
    for video_id in candidates:
        rows = _map_rows(data_root, video_id)
        if not rows:
            continue
        for row in rows:
            record = _record(data_root, video_id, row)
            if fallback_case is None and record.image_path is None:
                # The preferred real test: a mapped keyframe with no BTC JPEG at all.
                fallback_case = _case(provider, record, prefer_jpeg=True, label="jpeg_missing_natural_fallback")
                break
        if forced_case is None:
            forced = _record(data_root, video_id, rows[min(1, len(rows) - 1)])
            if forced.image_path is not None:
                forced_case = _case(provider, forced, prefer_jpeg=False, label="forced_video_decode")
        if fallback_case is not None and forced_case is not None:
            break

    cases = [case for case in (fallback_case, forced_case) if case is not None]
    repeat = None
    if fallback_case is not None:
        rows = _map_rows(data_root, fallback_case["keyframe_id"].split("/")[0])
        record = _record(
            data_root,
            fallback_case["keyframe_id"].split("/")[0],
            rows[fallback_case["keyframe_ordinal"] - 1],
        )
        repeat = _case(provider, record, prefer_jpeg=True, label="second_request_uses_derived_cache")
        cases.append(repeat)

    payload = {
        "data_root": str(data_root.resolve(strict=False).as_posix()),
        "derived_frame_cache_dir": str(Path(args.cache_dir).as_posix()),
        "video_backed_candidates": len(candidates),
        "cases": cases,
        "all_official_frame_idx_preserved": all(case["official_frame_idx_preserved"] for case in cases),
        "all_images_valid": all(case["image"].get("valid") for case in cases),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_images_valid"] and payload["all_official_frame_idx_preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
