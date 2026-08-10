"""Phase 3.1: what each keyframe identifier means, and which invariant protects it.

Three concepts are distinct and this module pins each one down:

keyframe_ordinal
    1-based position in the map CSV (`n`). Same position as the CLIP feature row and
    the keyframe image file. Unique within a video.
frame_idx
    official source-video frame index from the map CSV. The AIC submission value.
    NOT unique: 192 of the 873 official videos repeat one (see
    `artifacts/map_schema_report.json`, produced by `tools/inspect_map_schema.py`).
keyframe_id
    internal `{video_id}/kf_{keyframe_ordinal:06d}`. Globally unique, never parsed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import aic2026.dataset as dataset_module
from aic2026.config import app_config_from_dict
from aic2026.dataset import AICDatasetLoader, iter_official_rows, official_frame_id
from aic2026.dataset_validation import DatasetAlignmentError, inspect_aic_dataset
from aic2026.engine import AICCompetitionEngine

VIDEO_ID = "L21_V006"
ORDINAL_COLOR_STEP = 60

# Verbatim shape of the real L21_V006 head: two keyframes one source frame apart both
# truncate to official frame_idx 0, because frame_idx == int(pts_time * fps).
OFFICIAL_DUPLICATE_ROWS = [
    {"n": 1, "pts_time": 0.0, "fps": 30.0, "frame_idx": 0},
    {"n": 2, "pts_time": 0.0333333, "fps": 30.0, "frame_idx": 0},
    {"n": 3, "pts_time": 4.03333, "fps": 30.0, "frame_idx": 120},
]

DISTINCT_ROWS = [
    {"n": 1, "pts_time": 0.0, "fps": 30.0, "frame_idx": 100},
    {"n": 2, "pts_time": 1.0, "fps": 30.0, "frame_idx": 130},
    {"n": 3, "pts_time": 2.0, "fps": 30.0, "frame_idx": 160},
]


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def make_dataset(
    root: Path,
    *,
    rows: list[dict],
    features: np.ndarray | None = None,
    image_ordinals: list[int] | None = None,
    video_id: str = VIDEO_ID,
) -> Path:
    for relative in ("map-keyframes", "clip-features-32", f"keyframes/{video_id}"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    body = ["n,pts_time,fps,frame_idx"]
    body += [f"{r['n']},{r['pts_time']},{r['fps']},{r['frame_idx']}" for r in rows]
    (root / "map-keyframes" / f"{video_id}.csv").write_text("\n".join(body) + "\n", encoding="utf-8")
    if features is None:
        # Row i is the one-hot vector for position i, so a retrieved row proves which
        # feature row it came from.
        features = np.eye(len(rows), 3, dtype=np.float32)
    np.save(root / "clip-features-32" / f"{video_id}.npy", features)
    for ordinal in image_ordinals if image_ordinals is not None else [int(r["n"]) for r in rows]:
        # The red channel encodes the ordinal, so an image can be traced back to it.
        # Scaled well clear of JPEG quantization noise.
        Image.new("RGB", (8, 8), (ordinal * ORDINAL_COLOR_STEP, 0, 0)).save(
            root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
        )
    return root


def make_config(root: Path, *, cache_dir: Path | None = None, feature_dim: int = 3):
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir or root.parent / "cache"),
                    "validation": {"expected_feature_dim": feature_dim},
                },
                "encoder": {"feature_dim": feature_dim},
            }
        }
    )


def build(root: Path, monkeypatch=None):
    return AICDatasetLoader(root, app_config=make_config(root)).build_entry()


def issue_codes(report, video_id: str = VIDEO_ID) -> set[str]:
    return {issue.code for video in report.videos if video.video_id == video_id for issue in video.issues}


def issue_severity(report, code: str, video_id: str = VIDEO_ID) -> str | None:
    for video in report.videos:
        if video.video_id != video_id:
            continue
        for issue in video.issues:
            if issue.code == code:
                return issue.severity
    return None


# ------------------------------------------------------- row / feature / image order


def test_feature_row_i_maps_to_map_row_i(tmp_path, monkeypatch) -> None:
    root = make_dataset(tmp_path / "data", rows=DISTINCT_ROWS)
    captured: dict = {}
    original = dataset_module._build_aic_index

    def capture(records, *, kind="flat"):
        captured["records"] = records
        return original(records, kind=kind)

    monkeypatch.setattr(dataset_module, "_build_aic_index", capture)
    build(root)
    records = captured["records"]
    assert len(records) == 3
    for index, record in enumerate(records):
        expected = np.eye(3, 3, dtype=np.float32)[index]
        assert np.array_equal(record.clip_embedding, expected)
        assert record.timestamp == pytest.approx(DISTINCT_ROWS[index]["pts_time"])


def test_keyframe_ordinal_i_maps_to_keyframe_image_i(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=DISTINCT_ROWS)
    entry, _ = build(root)
    for row in DISTINCT_ROWS:
        ordinal = int(row["n"])
        raw = entry.raws[f"{VIDEO_ID}/kf_{ordinal:06d}"]
        assert Path(raw.image_path).name == f"{ordinal:03d}.jpg"
        # The image really is the one written for that ordinal, not for the frame_idx.
        red = Image.open(raw.image_path).convert("RGB").getpixel((0, 0))[0]
        assert abs(red - ordinal * ORDINAL_COLOR_STEP) <= 4


def test_official_frame_idx_is_independent_of_the_ordinal(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=DISTINCT_ROWS)
    entry, _ = build(root)
    ordinals = [entry.raws[key].keyframe_ordinal for key in entry.index.ids]
    frame_indices = [entry.raws[key].frame_idx for key in entry.index.ids]
    assert ordinals == [1, 2, 3]
    assert frame_indices == [100, 130, 160]


# ------------------------------------------------------------------ ID uniqueness


def test_two_keyframes_with_the_same_frame_idx_get_unique_internal_ids(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    entry, stats = build(root)
    assert entry.index.ids == [
        f"{VIDEO_ID}/kf_000001",
        f"{VIDEO_ID}/kf_000002",
        f"{VIDEO_ID}/kf_000003",
    ]
    # Nothing was silently dropped by an ID collision.
    assert len(entry.raws) == stats.frames == 3
    first, second = entry.raws[f"{VIDEO_ID}/kf_000001"], entry.raws[f"{VIDEO_ID}/kf_000002"]
    assert first.frame_idx == second.frame_idx == 0
    assert first.keyframe_ordinal == 1 and second.keyframe_ordinal == 2


def test_repeated_keyframe_ordinal_is_a_hard_error(tmp_path) -> None:
    rows = [dict(DISTINCT_ROWS[0]), dict(DISTINCT_ROWS[1], n=1), dict(DISTINCT_ROWS[2])]
    root = make_dataset(tmp_path / "data", rows=rows, image_ordinals=[1, 3])
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert "KEYFRAME_ORDINAL_DUPLICATE" in issue_codes(report)
    assert not report.valid_for_index_build
    assert report.duplicate_internal_keyframe_id_count == 1


# ------------------------------------------------------------------- submission


def test_submission_rows_use_frame_idx_and_never_the_ordinal(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    entry, _ = build(root)
    engine = AICCompetitionEngine(
        entry, text_encoder=TinyTextEncoder(), query_templates=("{q}",), bm25_weight=0.0
    )
    rows = [prediction.row() for prediction in engine.search_kis("anything", top_k=3)]
    assert rows, "expected at least one prediction"
    for video_id, frame_id in rows:
        assert video_id == VIDEO_ID
        assert frame_id in {"0", "120"}
    # Ordinals 1..3 must never leak out as submission frames; 120 proves the third
    # keyframe reports its official frame index, not its ordinal.
    assert {frame_id for _, frame_id in rows} & {"1", "2", "3"} == set()


def test_iter_official_rows_reads_frame_idx_from_the_record(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    entry, _ = build(root)
    assert list(iter_official_rows(entry, entry.index.ids)) == [
        (VIDEO_ID, "0"),
        (VIDEO_ID, "0"),
        (VIDEO_ID, "120"),
    ]


def test_official_frame_id_never_parses_the_internal_id(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    entry, _ = build(root)
    # The v2 fallback returned the ID suffix. With ordinal-based IDs that suffix is
    # `kf_000003`, so any surviving fallback would be caught here rather than silently
    # emitting an ordinal as an official frame.
    assert official_frame_id(entry, f"{VIDEO_ID}/kf_000003") == "120"
    with pytest.raises(KeyError):
        official_frame_id(entry, f"{VIDEO_ID}/kf_999999")


def test_search_results_keep_their_internal_identity_through_fusion(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    entry, _ = build(root)
    engine = AICCompetitionEngine(
        entry, text_encoder=TinyTextEncoder(), query_templates=("{q}",), bm25_weight=0.0
    )
    candidates = engine.search_candidates("anything", top_k=3)
    keyframe_ids = [candidate.keyframe_id for candidate in candidates]
    # Both frame_idx=0 keyframes survive as separate candidates; rebuilding the ID from
    # (video_id, frame_id) would have aliased them onto one row.
    assert len(keyframe_ids) == len(set(keyframe_ids)) == 3
    assert set(keyframe_ids) <= set(entry.index.ids)


# ---------------------------------------------------------- verified map policies


def test_duplicate_official_frame_idx_is_informational_not_invalid(tmp_path) -> None:
    """Regression for the verified CASE B policy: repeats are official BTC data."""
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert issue_severity(report, "FRAME_IDX_DUPLICATE") == "info"
    assert report.videos[0].duplicate_frame_indices == [0]
    assert report.duplicate_official_frame_idx_count == 1
    assert report.duplicate_internal_keyframe_id_count == 0
    assert report.valid_for_index_build


def test_equal_consecutive_frame_idx_is_informational(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert issue_severity(report, "FRAME_IDX_NON_MONOTONIC") == "info"
    assert report.videos[0].non_monotonic_frame_indices == [0]
    assert report.videos[0].decreasing_frame_indices == []


def test_strictly_decreasing_frame_idx_remains_a_hard_error(tmp_path) -> None:
    """No official video decreases; if one appears, keyframe order is unusable."""
    rows = [dict(DISTINCT_ROWS[0], frame_idx=160), dict(DISTINCT_ROWS[1], frame_idx=100)]
    root = make_dataset(
        tmp_path / "data", rows=rows, features=np.eye(2, 3, dtype=np.float32)
    )
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert issue_severity(report, "FRAME_IDX_DECREASING") == "error"
    assert report.videos[0].decreasing_frame_indices == [100]
    assert not report.valid_for_index_build


def test_duplicate_frame_idx_dataset_still_builds_an_index(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", rows=OFFICIAL_DUPLICATE_ROWS)
    entry, stats = build(root)
    assert stats.dataset_validated and stats.frames == 3
    assert entry.num_indexed == 3


def test_map_feature_count_mismatch_remains_a_hard_error(tmp_path) -> None:
    root = make_dataset(
        tmp_path / "data",
        rows=OFFICIAL_DUPLICATE_ROWS,
        features=np.eye(2, 3, dtype=np.float32),
    )
    with pytest.raises(DatasetAlignmentError, match="map rows .*3.*feature vectors .*2"):
        build(root)


@pytest.mark.parametrize(
    ("frame_idx", "code"),
    [(-5, "FRAME_IDX_NEGATIVE"), ("not-an-int", "FRAME_IDX_INVALID")],
)
def test_invalid_official_frame_index_remains_an_error(tmp_path, frame_idx, code) -> None:
    rows = [dict(DISTINCT_ROWS[0], frame_idx=frame_idx), dict(DISTINCT_ROWS[1])]
    root = make_dataset(
        tmp_path / str(code), rows=rows, features=np.eye(2, 3, dtype=np.float32)
    )
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert code in issue_codes(report)
    assert issue_severity(report, code) == "error"
    assert not report.valid_for_index_build
