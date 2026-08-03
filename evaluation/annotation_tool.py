"""Validate and append human AIC development annotations."""
from __future__ import annotations

import json
from pathlib import Path


def validate(row: dict) -> None:
    for key in ("query_id", "query", "task", "video_id"):
        if key not in row:
            raise ValueError(f"Missing annotation field: {key}")
    if row["task"] == "trake" and not row.get("events"):
        raise ValueError("TRAKE annotation requires event intervals.")
    if row["task"] != "trake" and not {"start", "end"} <= row.keys():
        raise ValueError("KIS/Q&A annotation requires start and end.")


def append_annotation(path: str | Path, row: dict) -> None:
    validate(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
