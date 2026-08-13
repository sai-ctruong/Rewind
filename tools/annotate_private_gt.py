"""Local annotation helper for private development ground truth.

Research tooling, deliberately outside the competition runtime: it is not importable by
the engine, it is not wired into the UI, and it never runs during a competition session.

What it does is the boring part of annotating — turning a timestamp into the official
`frame_idx` the label has to carry, and writing schema-correct JSON. It reads
`map-keyframes` and the MP4 only.

What it will NOT do, by construction:

* suggest an answer, an interval, or a query from any model;
* run CLIP, a VLM, or the retrieval engine;
* touch the network;
* modify the competition UI or any production path.

The human watches the video and decides. This just records the decision.

    # what frame_idx is 12.5 s into L21_V004?
    .venv\\Scripts\\python.exe tools/annotate_private_gt.py frame-at L21_V004 --seconds 12.5

    # what timestamp does official frame 1200 correspond to?
    .venv\\Scripts\\python.exe tools/annotate_private_gt.py time-at L21_V004 --frame 1200

    # which videos can be annotated at all?
    .venv\\Scripts\\python.exe tools/annotate_private_gt.py videos

    # append a label (all values supplied by the human)
    .venv\\Scripts\\python.exe tools/annotate_private_gt.py add-kis \\
        --query-id kis_0001 --video L21_V004 --query "a person pushes a bicycle" \\
        --range 1200 1260 --annotated-by "your name"

    # check everything parses and report how many REAL labels exist
    .venv\\Scripts\\python.exe tools/annotate_private_gt.py validate
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(_ROOT))

from evaluation.ground_truth import (  # noqa: E402
    LABEL_SOURCE_PRIVATE_DEV,
    PRIVATE_DEV_DIR,
    PRIVATE_DEV_FILES,
    SPLITS,
    GroundTruthSchemaError,
    load_private_dev,
)

FORBIDDEN_ANNOTATOR_HINT = (
    "annotated_by must name a human. 'system', 'model', 'auto', 'clip', 'vlm' and "
    "friends are refused: a label produced by the system being measured is circular."
)


def map_rows(data_root: Path, video_id: str) -> list[dict[str, Any]]:
    """Official keyframe map for one video: ordinal, pts_time, fps, frame_idx."""
    path = data_root / "map-keyframes" / f"{video_id}.csv"
    if not path.is_file():
        raise SystemExit(f"No map-keyframes file for {video_id!r} at {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def annotatable_videos(data_root: Path) -> list[str]:
    """Videos with a local MP4: the only ones a human can honestly annotate."""
    video_dir = data_root / "video"
    if not video_dir.is_dir():
        return []
    return sorted(path.stem for path in video_dir.glob("*.mp4"))


def frame_at(data_root: Path, video_id: str, seconds: float) -> dict[str, Any]:
    """The mapped keyframe nearest a timestamp, with its OFFICIAL frame_idx.

    Reported as the nearest *mapped* keyframe, never as an interpolated frame number: a
    label has to name a frame the official mapping actually contains.
    """
    rows = map_rows(data_root, video_id)
    best = min(rows, key=lambda row: abs(float(row["pts_time"]) - float(seconds)))
    return {
        "video_id": video_id,
        "requested_seconds": float(seconds),
        "nearest_mapped_frame_idx": int(float(best["frame_idx"])),
        "nearest_mapped_pts_time": float(best["pts_time"]),
        "keyframe_ordinal": int(float(best["n"])),
        "fps": float(best["fps"]),
        "note": "frame_idx is the official value to put in a label.",
    }


def time_at(data_root: Path, video_id: str, frame_idx: int) -> dict[str, Any]:
    rows = map_rows(data_root, video_id)
    best = min(rows, key=lambda row: abs(int(float(row["frame_idx"])) - int(frame_idx)))
    return {
        "video_id": video_id,
        "requested_frame_idx": int(frame_idx),
        "nearest_mapped_frame_idx": int(float(best["frame_idx"])),
        "pts_time": float(best["pts_time"]),
        "keyframe_ordinal": int(float(best["n"])),
    }


def load_file(task: str) -> tuple[Path, dict[str, Any]]:
    path = PRIVATE_DEV_DIR / PRIVATE_DEV_FILES[task]
    if not path.is_file():
        raise SystemExit(f"Missing {path}; create it from evaluation/private_dev/README.md")
    return path, json.loads(path.read_text(encoding="utf-8"))


def append_entry(task: str, entry: dict[str, Any]) -> Path:
    """Append one human-written row, refusing to overwrite an existing query_id."""
    path, payload = load_file(task)
    existing = {str(row.get("query_id")) for row in payload.get("entries") or ()}
    if entry["query_id"] in existing:
        raise SystemExit(
            f"query_id {entry['query_id']!r} already exists in {path.name}; pick another "
            "or edit the file by hand."
        )
    payload.setdefault("entries", []).append(entry)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def base_entry(args: argparse.Namespace, task: str) -> dict[str, Any]:
    if not str(args.annotated_by).strip():
        raise SystemExit(FORBIDDEN_ANNOTATOR_HINT)
    return {
        "query_id": args.query_id,
        "video_id": args.video,
        "label_source": LABEL_SOURCE_PRIVATE_DEV,
        "annotated_by": args.annotated_by,
        "split": args.split,
        "notes": args.notes or "",
    }


def cmd_videos(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    videos = annotatable_videos(data_root)
    print(json.dumps({
        "annotatable_videos": videos,
        "count": len(videos),
        "note": (
            "Only videos with a local MP4 can be annotated: a defensible interval "
            "requires watching the pixels."
        ),
    }, indent=2))
    return 0


def cmd_frame_at(args: argparse.Namespace) -> int:
    print(json.dumps(frame_at(Path(args.data_root), args.video, args.seconds), indent=2))
    return 0


def cmd_time_at(args: argparse.Namespace) -> int:
    print(json.dumps(time_at(Path(args.data_root), args.video, args.frame), indent=2))
    return 0


def cmd_add_kis(args: argparse.Namespace) -> int:
    entry = base_entry(args, "kis")
    entry.update({"query": args.query, "frame_ranges": [list(pair) for pair in args.range]})
    path = append_entry("kis", entry)
    print(json.dumps({"written": str(path), "query_id": entry["query_id"]}, indent=2))
    return 0


def cmd_add_qa(args: argparse.Namespace) -> int:
    entry = base_entry(args, "qa")
    entry.update(
        {
            "event_description": args.event,
            "question": args.question,
            "frame_ranges": [list(pair) for pair in args.range],
            "answers": list(args.answer),
            "answer_type": args.answer_type,
        }
    )
    path = append_entry("qa", entry)
    print(json.dumps({"written": str(path), "query_id": entry["query_id"]}, indent=2))
    return 0


def cmd_add_trake(args: argparse.Namespace) -> int:
    if len(args.event) != len(args.range):
        raise SystemExit(
            f"{len(args.event)} event(s) but {len(args.range)} interval(s); TRAKE is "
            "scored per event, so they must correspond one to one."
        )
    entry = base_entry(args, "trake")
    entry.update(
        {
            "events": list(args.event),
            "event_frame_ranges": [[list(pair)] for pair in args.range],
        }
    )
    path = append_entry("trake", entry)
    print(json.dumps({"written": str(path), "query_id": entry["query_id"]}, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Parse every file and report how many REAL labels exist. Templates do not count."""
    try:
        gt = load_private_dev(args.directory)
    except GroundTruthSchemaError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 3
    provenance = gt.provenance()
    data_root = Path(args.data_root)
    known = set(annotatable_videos(data_root))
    unknown = sorted({entry.video_id for entry in gt.real_entries} - known) if known else []
    payload = {
        "valid": True,
        "banner": gt.banner,
        "real_labels": provenance["counts"],
        "real_label_total": len(gt.real_entries),
        "example_rows_ignored": provenance["example_rows_ignored"],
        "content_hash": gt.content_hash(),
        # Only checked when the MP4 inventory is actually readable; an absent data root
        # produces no verdict rather than an invented one.
        "videos_without_local_mp4": unknown,
        "ready_for_evaluation": bool(gt.real_entries),
    }
    if not gt.real_entries:
        payload["note"] = (
            "No human labels yet. Semantic evaluation will refuse with "
            "GROUND_TRUTH_REQUIRED, which is correct."
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def interval(value: str) -> tuple[int, int]:
    start, _, end = value.partition(":")
    return (int(start), int(end or start))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("videos", help="List videos that have a local MP4.").set_defaults(func=cmd_videos)

    at = sub.add_parser("frame-at", help="Official frame_idx nearest a timestamp.")
    at.add_argument("video")
    at.add_argument("--seconds", type=float, required=True)
    at.set_defaults(func=cmd_frame_at)

    ta = sub.add_parser("time-at", help="Timestamp of an official frame_idx.")
    ta.add_argument("video")
    ta.add_argument("--frame", type=int, required=True)
    ta.set_defaults(func=cmd_time_at)

    def shared(target: argparse.ArgumentParser) -> None:
        target.add_argument("--query-id", required=True)
        target.add_argument("--video", required=True)
        target.add_argument("--annotated-by", required=True, help=FORBIDDEN_ANNOTATOR_HINT)
        target.add_argument("--split", choices=list(SPLITS), default="development")
        target.add_argument("--notes", default="")

    kis = sub.add_parser("add-kis", help="Append one KIS label.")
    shared(kis)
    kis.add_argument("--query", required=True)
    kis.add_argument("--range", nargs=2, type=int, action="append", required=True,
                     metavar=("START", "END"), help="Official frame_idx bounds; repeatable.")
    kis.set_defaults(func=cmd_add_kis)

    qa = sub.add_parser("add-qa", help="Append one Q&A label.")
    shared(qa)
    qa.add_argument("--event", required=True)
    qa.add_argument("--question", required=True)
    qa.add_argument("--answer", action="append", required=True, help="Repeatable.")
    qa.add_argument("--answer-type", default="text",
                    choices=["number", "yes/no", "color", "text", "auto"])
    qa.add_argument("--range", nargs=2, type=int, action="append", required=True,
                    metavar=("START", "END"))
    qa.set_defaults(func=cmd_add_qa)

    trake = sub.add_parser("add-trake", help="Append one TRAKE sequence label.")
    shared(trake)
    trake.add_argument("--event", action="append", required=True, help="Repeatable, in order.")
    trake.add_argument("--range", nargs=2, type=int, action="append", required=True,
                       metavar=("START", "END"), help="One per event, same order.")
    trake.set_defaults(func=cmd_add_trake)

    validate = sub.add_parser("validate", help="Parse the files and count REAL labels.")
    validate.add_argument("--directory", default=str(PRIVATE_DEV_DIR))
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
