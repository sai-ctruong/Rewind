"""Stage 1 refinement: rescore the OFFICIAL keyframe grid before touching an MP4.

Every mapped keyframe already has a BTC CLIP vector inside the index and an official
`frame_idx`. Scoring a handful of a candidate's neighbours against the query embedding
therefore costs a few dot products — no decode, no JPEG read, no image encoder — and
produces a local score curve around the candidate.

Two properties make this the right first stage:

* it is roughly a hundred times cheaper than decoding the same neighbourhood from the
  MP4, so it can run where full refinement cannot be afforded;
* every frame it can promote is an explicit `map-keyframes` record, so the result is
  submission-safe by construction. An arbitrary decoded frame is not, and this stage
  never produces one.

What it returns is evidence and a shape: the best mapped neighbour, the peak margin, and
how stable the curve is. Deciding what to do with that is the controller's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class GridPoint:
    """One official mapped keyframe scored against the query."""

    keyframe_id: str
    frame_idx: Optional[int]
    timestamp: float
    offset: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "offset": self.offset,
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class GridCurve:
    """The local score curve around one coarse candidate."""

    keyframe_id: str
    video_id: str
    points: tuple[GridPoint, ...] = ()
    coarse_score: float = 0.0
    vectors_read: int = 0

    @property
    def best(self) -> Optional[GridPoint]:
        return max(self.points, key=lambda item: item.score) if self.points else None

    @property
    def peak_margin(self) -> float:
        """Gap between the best and second-best point, relative to the best.

        0 when the neighbourhood is flat — several frames look equally good, so the
        curve says nothing about which one to prefer.
        """
        if len(self.points) < 2:
            return 0.0
        ordered = sorted((item.score for item in self.points), reverse=True)
        best = ordered[0]
        return 0.0 if abs(best) < 1e-9 else max(0.0, (best - ordered[1]) / abs(best))

    @property
    def temporal_stability(self) -> float:
        """1 when the peak sits at the coarse frame; falls off as the peak moves away."""
        peak = self.best
        if peak is None:
            return 1.0
        return 1.0 / (1.0 + abs(int(peak.offset)))

    @property
    def slope(self) -> float:
        """Mean absolute change between neighbouring points: a flat curve scores 0."""
        if len(self.points) < 2:
            return 0.0
        ordered = sorted(self.points, key=lambda item: item.offset)
        deltas = [abs(b.score - a.score) for a, b in zip(ordered, ordered[1:])]
        return sum(deltas) / len(deltas)

    @property
    def improves_on_coarse(self) -> bool:
        peak = self.best
        return bool(peak is not None and peak.score > self.coarse_score)

    def to_dict(self) -> dict[str, Any]:
        peak = self.best
        return {
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "coarse_score": round(self.coarse_score, 6),
            "points": [item.to_dict() for item in self.points],
            "vectors_read": self.vectors_read,
            "best_offset": None if peak is None else peak.offset,
            "best_keyframe_id": None if peak is None else peak.keyframe_id,
            "best_frame_idx": None if peak is None else peak.frame_idx,
            "best_score": None if peak is None else round(peak.score, 6),
            "peak_margin": round(self.peak_margin, 6),
            "temporal_stability": round(self.temporal_stability, 6),
            "slope": round(self.slope, 6),
            "improves_on_coarse": self.improves_on_coarse,
        }


@dataclass
class GridRefinementResult:
    curves: list[GridCurve] = field(default_factory=list)
    vectors_read: int = 0
    candidates_examined: int = 0
    frames_decoded: int = 0  # always 0: this stage never decodes
    skipped_reason: str = ""

    def by_keyframe(self) -> dict[str, GridCurve]:
        return {curve.keyframe_id: curve for curve in self.curves}

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_examined": self.candidates_examined,
            "vectors_read": self.vectors_read,
            "frames_decoded": self.frames_decoded,
            "curves": [curve.to_dict() for curve in self.curves],
            **({"skipped_reason": self.skipped_reason} if self.skipped_reason else {}),
            "note": (
                "Official mapped keyframes only, scored from vectors already in the "
                "index. No MP4 decode and no image embedding. Every point carries an "
                "official frame_idx, so any of them is submission-safe."
            ),
        }


def _offsets(count: int) -> list[int]:
    """Symmetric neighbour offsets, nearest first: -1, +1, -2, +2, ..."""
    out: list[int] = []
    for step in range(1, int(count) + 1):
        out.extend((-step, step))
    return out


class OfficialGridRefiner:
    """Score a candidate's official neighbours using indexed vectors only."""

    def __init__(self, entry, *, neighbors: int = 4, max_candidates: int = 20) -> None:
        self.entry = entry
        self.neighbors = max(1, int(neighbors))
        self.max_candidates = max(1, int(max_candidates))

    def refine(
        self,
        query_vector: np.ndarray,
        candidates: Sequence[Any],
        *,
        budget_candidates: Optional[int] = None,
    ) -> GridRefinementResult:
        """Build a local score curve for the strongest candidates.

        `query_vector` is the embedding the coarse search already computed; nothing is
        re-encoded here.
        """
        result = GridRefinementResult()
        index = getattr(self.entry, "index", None)
        if index is None or not hasattr(index, "neighbor_rows"):
            result.skipped_reason = "index does not expose the official grid API"
            return result
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            result.skipped_reason = "query vector unavailable"
            return result

        limit = min(self.max_candidates, int(budget_candidates or self.max_candidates))
        offsets = _offsets(self.neighbors)
        for candidate in list(candidates)[:limit]:
            keyframe_id = str(getattr(candidate, "keyframe_id", "") or "")
            if not keyframe_id:
                continue
            neighbours = index.neighbor_rows(keyframe_id, offsets)
            own_row = index.row_of(keyframe_id)
            if own_row is None:
                continue
            result.candidates_examined += 1
            # Offsets travel WITH their rows: near a video boundary some requested
            # offsets do not exist, and zipping two lists would mislabel the survivors.
            located: list[tuple[int, int]] = [(0, own_row)] + list(neighbours)
            all_rows = [row for _, row in located]
            try:
                vectors = index.vectors_for_rows(all_rows)
            except Exception:  # noqa: BLE001 - a missing vector must not fail a search
                continue
            if getattr(vectors, "size", 0) == 0:
                continue
            scores = np.asarray(vectors, dtype=np.float32) @ vector
            result.vectors_read += len(all_rows)

            points: list[GridPoint] = []
            for position, (offset, row) in enumerate(located):
                neighbour_id = index.ids[row]
                raw = self.entry.raws.get(neighbour_id)
                points.append(
                    GridPoint(
                        keyframe_id=neighbour_id,
                        # The official mapped frame. Never derived from the internal id.
                        frame_idx=None if raw is None or raw.frame_idx is None else int(raw.frame_idx),
                        timestamp=float(index.timestamps[row]),
                        offset=int(offset),
                        score=float(scores[position]),
                    )
                )
            result.curves.append(
                GridCurve(
                    keyframe_id=keyframe_id,
                    video_id=str(getattr(candidate, "video_id", "")),
                    points=tuple(points),
                    coarse_score=float(scores[0]),
                    vectors_read=len(all_rows),
                )
            )
        return result


__all__ = [
    "GridCurve",
    "GridPoint",
    "GridRefinementResult",
    "OfficialGridRefiner",
]
