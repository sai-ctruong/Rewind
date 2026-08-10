"""Describe the real `map-keyframes` CSV schema straight from DATA_ROOT.

Phase 3.1 needed evidence, not assumptions, before deciding whether duplicate
`frame_idx` values mean corrupt data. This script reads every map CSV under a data
root and writes `artifacts/map_schema_report.json`.

    .venv\\Scripts\\python.exe tools/inspect_map_schema.py --data-root data

It never writes to DATA_ROOT and never loads feature arrays.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

MAX_EXAMPLES = 8


def _rows_around(rows: list[dict[str, Any]], index: int, *, before: int = 1, after: int = 1) -> list[dict[str, Any]]:
    return rows[max(0, index - before) : min(len(rows), index + after + 1)]


def inspect_map_schema(data_root: Path) -> dict[str, Any]:
    map_dir = Path(data_root) / "map-keyframes"
    if not map_dir.is_dir():
        raise SystemExit(f"No map-keyframes directory under {data_root}")

    schemas: Counter[tuple[str, ...]] = Counter()
    files_sampled = 0
    total_rows = 0
    videos_with_duplicates = 0
    videos_with_equal_consecutive = 0
    videos_with_decreasing = 0
    videos_with_non_sequential_ordinal: list[str] = []
    rows_matching_truncation = 0
    rows_matching_rounding = 0
    truncation_mismatch_examples: list[dict[str, Any]] = []
    duplicate_examples: list[dict[str, Any]] = []
    non_monotonic_examples: list[dict[str, Any]] = []
    decreasing_examples: list[dict[str, Any]] = []

    for path in sorted(map_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            raw_rows = list(reader)
        schemas[columns] += 1
        files_sampled += 1
        total_rows += len(raw_rows)
        if not {"n", "pts_time", "fps", "frame_idx"} <= set(columns):
            continue

        rows = [
            {
                "n": int(row["n"]),
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "frame_idx": int(row["frame_idx"]),
            }
            for row in raw_rows
        ]
        if [row["n"] for row in rows] != list(range(1, len(rows) + 1)):
            videos_with_non_sequential_ordinal.append(path.stem)

        seen: set[int] = set()
        duplicate_positions: list[int] = []
        for index, row in enumerate(rows):
            if row["frame_idx"] in seen:
                duplicate_positions.append(index)
            seen.add(row["frame_idx"])
        equal_positions = [
            index
            for index in range(1, len(rows))
            if rows[index]["frame_idx"] == rows[index - 1]["frame_idx"]
        ]
        decreasing_positions = [
            index
            for index in range(1, len(rows))
            if rows[index]["frame_idx"] < rows[index - 1]["frame_idx"]
        ]

        if duplicate_positions:
            videos_with_duplicates += 1
            if len(duplicate_examples) < MAX_EXAMPLES:
                index = duplicate_positions[0]
                duplicate_examples.append(
                    {
                        "video_id": path.stem,
                        "duplicate_frame_idx": rows[index]["frame_idx"],
                        "rows": _rows_around(rows, index),
                    }
                )
        if equal_positions:
            videos_with_equal_consecutive += 1
            if len(non_monotonic_examples) < MAX_EXAMPLES:
                index = equal_positions[0]
                non_monotonic_examples.append(
                    {"video_id": path.stem, "kind": "equal_consecutive", "rows": _rows_around(rows, index, before=1, after=0)}
                )
        if decreasing_positions:
            videos_with_decreasing += 1
            if len(decreasing_examples) < MAX_EXAMPLES:
                index = decreasing_positions[0]
                decreasing_examples.append(
                    {"video_id": path.stem, "kind": "strictly_decreasing", "rows": _rows_around(rows, index, before=1, after=0)}
                )

        for row in rows:
            product = row["pts_time"] * row["fps"]
            if int(product) == row["frame_idx"]:
                rows_matching_truncation += 1
            elif len(truncation_mismatch_examples) < 10:
                truncation_mismatch_examples.append({**row, "video_id": path.stem, "int_pts_time_times_fps": int(product)})
            if round(product) == row["frame_idx"]:
                rows_matching_rounding += 1

    if rows_matching_truncation == total_rows and videos_with_decreasing == 0:
        conclusion = (
            "CASE B. There is exactly one map schema and the parser already reads the correct "
            "column. `frame_idx` equals int(pts_time * fps) in every row of the collection, so "
            "duplicate values are a genuine artifact of the official truncation-derived mapping "
            "(two keyframes one source frame apart at the start of a video both truncate to 0). "
            "No video anywhere in the collection has a strictly decreasing frame_idx. Therefore "
            "duplicate and equal-consecutive frame_idx must not invalidate a video, while a "
            "strictly decreasing frame_idx stays a hard error, and the internal keyframe ID must "
            "be derived from the keyframe ordinal `n` rather than from `frame_idx`."
        )
    else:
        conclusion = (
            "Inconclusive against the Phase 3.1 CASE A/B/C hypotheses; inspect "
            "truncation_mismatch_examples and strictly_decreasing_examples before changing policy."
        )

    return {
        "data_root": Path(data_root).resolve(strict=False).as_posix(),
        "files_sampled": files_sampled,
        "total_rows": total_rows,
        "schemas_found": [{"columns": list(columns), "file_count": count} for columns, count in schemas.most_common()],
        "columns": sorted({column for columns in schemas for column in columns}),
        "keyframe_ordinal_column": "n",
        "official_frame_index_column": "frame_idx",
        "timestamp_column": "pts_time",
        "additional_columns": ["fps"],
        "videos_with_non_sequential_ordinal": videos_with_non_sequential_ordinal,
        "videos_with_duplicate_frame_idx": videos_with_duplicates,
        "videos_with_equal_consecutive_frame_idx": videos_with_equal_consecutive,
        "videos_with_strictly_decreasing_frame_idx": videos_with_decreasing,
        "rows_where_frame_idx_equals_truncated_pts_time_times_fps": rows_matching_truncation,
        "rows_where_frame_idx_equals_rounded_pts_time_times_fps": rows_matching_rounding,
        "truncation_mismatch_examples": truncation_mismatch_examples,
        "duplicate_examples": duplicate_examples,
        "non_monotonic_examples": non_monotonic_examples,
        "strictly_decreasing_examples": decreasing_examples,
        "conclusion": conclusion,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="artifacts/map_schema_report.json")
    args = parser.parse_args(argv)
    report = inspect_map_schema(Path(args.data_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("examples")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
