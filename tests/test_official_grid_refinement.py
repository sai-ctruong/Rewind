"""R1 stage 1: rescoring the official keyframe grid costs no decode and stays submittable.

The point of this stage is that it buys local temporal evidence for roughly the price of
a few dot products, and that everything it can surface is an official `map-keyframes`
record. Both are tested here, along with the guarantee that it never invents a frame.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aic2026.official_grid import GridCurve, GridPoint, OfficialGridRefiner
from tests.release_support import build_engine


class FakeIndex:
    """A three-video grid whose vectors are known, so scores are predictable."""

    def __init__(self, vectors: dict[str, np.ndarray], video_of: dict[str, str]) -> None:
        self.ids = list(vectors)
        self.video_ids = [video_of[key] for key in self.ids]
        self.timestamps = [float(i) for i in range(len(self.ids))]
        self._vectors = vectors
        self.reconstruct_calls = 0

    def row_of(self, keyframe_id):
        return self.ids.index(keyframe_id) if keyframe_id in self.ids else None

    def rows_for_video(self, video_id):
        return [row for row, video in enumerate(self.video_ids) if video == video_id]

    def neighbor_rows(self, keyframe_id, offsets):
        row = self.row_of(keyframe_id)
        if row is None:
            return []
        rows = self.rows_for_video(self.video_ids[row])
        position = rows.index(row)
        out = []
        for offset in offsets:
            target = position + int(offset)
            if 0 <= target < len(rows) and rows[target] != row:
                out.append((int(offset), rows[target]))
        return out

    def vectors_for_rows(self, rows, encoder="clip"):
        self.reconstruct_calls += 1
        return np.stack([self._vectors[self.ids[row]] for row in rows]).astype(np.float32)


def make_entry():
    """Video V has five frames; frame 2 is the coarse hit and frame 3 is the true peak."""
    vectors = {}
    video_of = {}
    scores = {0: 0.2, 1: 0.5, 2: 0.7, 3: 0.95, 4: 0.3}
    for ordinal, value in scores.items():
        key = f"V/kf_{ordinal:06d}"
        vectors[key] = np.array([value, np.sqrt(max(0.0, 1 - value**2))], dtype=np.float32)
        video_of[key] = "V"
    # A second video so neighbour lookup cannot leak across videos.
    for ordinal in range(2):
        key = f"W/kf_{ordinal:06d}"
        vectors[key] = np.array([0.99, 0.14], dtype=np.float32)
        video_of[key] = "W"
    index = FakeIndex(vectors, video_of)
    raws = {
        key: SimpleNamespace(video_id=video_of[key], frame_idx=100 + 10 * i, source_video=None)
        for i, key in enumerate(vectors)
    }
    return SimpleNamespace(index=index, raws=raws)


def candidate(keyframe_id: str, video_id: str = "V", score: float = 1.0):
    return SimpleNamespace(keyframe_id=keyframe_id, video_id=video_id, score=score, timestamp=0.0)


QUERY = np.array([1.0, 0.0], dtype=np.float32)


# --------------------------------------------------------------------- the curve


def test_it_scores_the_candidate_and_its_neighbours() -> None:
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=2).refine(QUERY, [candidate("V/kf_000002")])
    curve = result.curves[0]
    assert curve.keyframe_id == "V/kf_000002"
    assert len(curve.points) == 5  # itself plus four neighbours
    assert {point.offset for point in curve.points} == {0, -1, 1, -2, 2}


def test_it_finds_the_local_peak() -> None:
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=2).refine(QUERY, [candidate("V/kf_000002")])
    curve = result.curves[0]
    assert curve.best.keyframe_id == "V/kf_000003"
    assert curve.improves_on_coarse is True
    assert curve.best.offset == 1


def test_every_point_carries_an_official_frame_idx() -> None:
    """This is what makes the stage submission-safe: no derived or decoded frames."""
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=2).refine(QUERY, [candidate("V/kf_000002")])
    for point in result.curves[0].points:
        assert point.frame_idx is not None
        assert point.frame_idx == entry.raws[point.keyframe_id].frame_idx


def test_it_never_decodes_a_frame() -> None:
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=3).refine(QUERY, [candidate("V/kf_000002")])
    assert result.frames_decoded == 0
    assert result.to_dict()["frames_decoded"] == 0
    assert "No MP4 decode" in result.to_dict()["note"]


def test_neighbours_never_cross_a_video_boundary() -> None:
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=4).refine(QUERY, [candidate("V/kf_000000")])
    assert {point.keyframe_id.split("/")[0] for point in result.curves[0].points} == {"V"}


def test_edge_candidate_has_fewer_neighbours_not_an_error() -> None:
    """At a video boundary the missing offsets vanish; the survivors keep their labels."""
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=3).refine(QUERY, [candidate("V/kf_000000")])
    curve = result.curves[0]
    assert sorted(point.offset for point in curve.points) == [0, 1, 2, 3]
    for point in curve.points:
        assert point.keyframe_id == f"V/kf_{point.offset:06d}"


def test_the_real_index_labels_boundary_neighbours_correctly(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    index = engine.entry.index
    first = index.ids[0]
    pairs = index.neighbor_rows(first, [-2, -1, 1, 2])
    # The first frame of a video has no earlier neighbour, and nothing pretends it does.
    assert all(offset > 0 for offset, _ in pairs)


# ------------------------------------------------------------------ curve shape


def test_a_flat_neighbourhood_reports_no_margin() -> None:
    points = tuple(
        GridPoint(keyframe_id=f"V/kf_{i}", frame_idx=i, timestamp=float(i), offset=i, score=0.5)
        for i in range(3)
    )
    curve = GridCurve(keyframe_id="V/kf_0", video_id="V", points=points, coarse_score=0.5)
    assert curve.peak_margin == 0.0
    assert curve.slope == 0.0


def test_a_sharp_peak_reports_a_margin_and_stability() -> None:
    entry = make_entry()
    curve = OfficialGridRefiner(entry, neighbors=2).refine(QUERY, [candidate("V/kf_000002")]).curves[0]
    assert curve.peak_margin > 0.0
    assert 0.0 < curve.temporal_stability <= 1.0
    assert curve.slope > 0.0


def test_stability_is_highest_when_the_coarse_frame_is_the_peak() -> None:
    entry = make_entry()
    curve = OfficialGridRefiner(entry, neighbors=1).refine(QUERY, [candidate("V/kf_000003")]).curves[0]
    assert curve.best.offset == 0
    assert curve.temporal_stability == 1.0


# ------------------------------------------------------------------- budgeting


def test_candidate_budget_is_respected() -> None:
    entry = make_entry()
    candidates = [candidate(f"V/kf_{i:06d}") for i in range(5)]
    result = OfficialGridRefiner(entry, neighbors=1, max_candidates=10).refine(
        QUERY, candidates, budget_candidates=2
    )
    assert result.candidates_examined == 2
    assert len(result.curves) == 2


def test_vectors_read_is_reported() -> None:
    entry = make_entry()
    result = OfficialGridRefiner(entry, neighbors=2).refine(QUERY, [candidate("V/kf_000002")])
    assert result.vectors_read == 5
    assert result.to_dict()["vectors_read"] == 5


def test_an_index_without_the_grid_api_is_skipped_not_crashed() -> None:
    entry = SimpleNamespace(index=SimpleNamespace(), raws={})
    result = OfficialGridRefiner(entry).refine(QUERY, [candidate("V/kf_000000")])
    assert result.curves == []
    assert "does not expose" in result.skipped_reason


def test_an_unusable_query_vector_is_skipped() -> None:
    entry = make_entry()
    result = OfficialGridRefiner(entry).refine(np.array([], dtype=np.float32), [candidate("V/kf_000000")])
    assert "query vector unavailable" in result.skipped_reason


# ---------------------------------------------------------------- real index API


def test_the_real_index_exposes_the_public_grid_api(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    index = engine.entry.index
    first = index.ids[0]
    assert index.row_of(first) == 0
    assert index.row_of("nope") is None
    assert index.rows_for_video(index.video_ids[0])
    neighbours = index.neighbor_rows(first, [-1, 1])
    assert all(index.video_ids[row] == index.video_ids[0] for _, row in neighbours)
    vectors = index.vectors_for_rows([0])
    assert vectors.shape[0] == 1


def test_the_stage_runs_against_a_real_index(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    pool = engine.search_candidates("a", top_k=5)
    vector = engine.encode_query("a")
    result = OfficialGridRefiner(engine.entry, neighbors=2).refine(vector, pool)
    assert result.frames_decoded == 0
    assert result.candidates_examined >= 1
    for curve in result.curves:
        for point in curve.points:
            assert point.frame_idx is not None
