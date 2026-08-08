from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import aic2026.dataset as dataset_module
from aic2026.cache_manifest import (
    cache_build_options_from_config,
    cache_fingerprint,
    read_cache_manifest,
)
from aic2026.cli import main as cli_main
from aic2026.config import AppConfig, ConfigError, app_config_from_dict
from aic2026.dataset import AICDatasetLoader
from aic2026.dataset_validation import (
    DatasetAlignmentError,
    DatasetIssue,
    DatasetValidationError,
    inspect_aic_dataset,
    write_dataset_report,
)
from aic2026.engine import AICCompetitionEngine
from ingestion.build_records import searchable_text
from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION, KeyframeRecord


VIDEO_ID = "L01_V001"
BASE_ROWS = [
    {"n": 1, "pts_time": 0.0, "fps": 30.0, "frame_idx": 100},
    {"n": 2, "pts_time": 1.0, "fps": 30.0, "frame_idx": 130},
]


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def _write_map(root: Path, video_id: str, rows: list[dict]) -> None:
    path = root / "map-keyframes" / f"{video_id}.csv"
    body = ["n,pts_time,fps,frame_idx"]
    body.extend(
        f"{row['n']},{row['pts_time']},{row['fps']},{row['frame_idx']}" for row in rows
    )
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def make_dataset(
    root: Path,
    *,
    video_id: str = VIDEO_ID,
    rows: list[dict] | None = None,
    features: np.ndarray | None = None,
    write_images: bool = True,
) -> Path:
    rows = [dict(row) for row in (rows or BASE_ROWS)]
    features = (
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        if features is None
        else features
    )
    for relative in ("map-keyframes", "clip-features-32", f"keyframes/{video_id}"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _write_map(root, video_id, rows)
    np.save(root / "clip-features-32" / f"{video_id}.npy", features)
    if write_images:
        for fallback, row in enumerate(rows, start=1):
            try:
                ordinal = int(row["n"])
            except (TypeError, ValueError):
                ordinal = fallback
            if ordinal <= 0:
                ordinal = fallback
            Image.new("RGB", (8, 8), (ordinal * 50 % 255, 10, 20)).save(
                root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
            )
    return root


def make_config(
    root: Path,
    *,
    cache_dir: Path | None = None,
    feature_dim: int = 2,
    expected_feature_dim: int | None = 2,
    load_objects: bool = False,
    include_media_text: bool = True,
    strict_objects: bool = False,
    supported_dtypes: tuple[str, ...] = ("float16", "float32"),
) -> AppConfig:
    validation: dict = {
        "strict_alignment": True,
        "require_keyframe_images": True,
        "strict_objects": strict_objects,
        "supported_feature_dtypes": list(supported_dtypes),
    }
    if expected_feature_dim is not None:
        validation["expected_feature_dim"] = expected_feature_dim
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir or root.parent / "cache"),
                    "load_objects": load_objects,
                    "include_media_text": include_media_text,
                    "validation": validation,
                },
                "encoder": {"feature_dim": feature_dim},
            }
        }
    )


def inspect(root: Path, config: AppConfig, *, strict: bool = False, deep: bool = False, limit: int | None = None):
    return inspect_aic_dataset(
        root, app_config=config, strict=strict, deep=deep, limit_videos=limit
    )


def issue_codes(report, video_id: str = VIDEO_ID) -> set[str]:
    return {
        issue.code
        for video in report.videos
        if video.video_id == video_id
        for issue in video.issues
    }


def write_cli_config(path: Path, config: AppConfig) -> Path:
    data = {
        "aic2026": {
            "dataset": {
                "root": config.dataset.root,
                "cache_dir": config.dataset.cache_dir,
                "load_objects": config.dataset.load_objects,
                "include_media_text": config.dataset.include_media_text,
                "validation": {
                    "strict_alignment": True,
                    "require_keyframe_images": True,
                    "strict_objects": config.dataset.validation.strict_objects,
                    "expected_feature_dim": config.encoder.feature_dim,
                },
            },
            "encoder": {"feature_dim": config.encoder.feature_dim},
        }
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _object_payload(*, score="0.8", box=None) -> dict:
    return {
        "detection_class_entities": ["Coffee cup"],
        "detection_scores": [score],
        "detection_boxes": [box or ["0.1", "0.2", "0.8", "0.9"]],
    }


def _write_objects(root: Path, payload: dict | str, *, video_id: str = VIDEO_ID) -> None:
    folder = root / "objects" / video_id
    folder.mkdir(parents=True, exist_ok=True)
    for ordinal in (1, 2):
        path = folder / f"{ordinal:03d}.json"
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_dataset_passes(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    report = inspect(root, make_config(root))
    assert report.valid_for_index_build
    assert report.total_map_rows == report.total_feature_vectors == 2


def test_map_rows_greater_than_features_raise_strict_error(tmp_path) -> None:
    rows = BASE_ROWS + [{"n": 3, "pts_time": 2.0, "fps": 30, "frame_idx": 160}]
    root = make_dataset(tmp_path / "data", rows=rows)
    config = make_config(root)
    with pytest.raises(DatasetAlignmentError, match="map rows .*3.*feature vectors .*2"):
        AICDatasetLoader(root, app_config=config).build_entry()


def test_features_greater_than_map_rows_raise_strict_error(tmp_path) -> None:
    root = make_dataset(
        tmp_path / "data",
        features=np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )
    with pytest.raises(DatasetAlignmentError):
        AICDatasetLoader(root, app_config=make_config(root)).build_entry()


def test_non_strict_mismatch_records_issue_and_invalid_video(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", features=np.array([[1, 0]], dtype=np.float32))
    report = inspect(root, make_config(root), strict=False)
    assert "MAP_FEATURE_COUNT_MISMATCH" in issue_codes(report)
    assert report.invalid_video_count == 1
    assert not report.valid_for_index_build


def test_strict_inspection_raises_with_complete_report(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", features=np.ones((1, 2), dtype=np.float32))
    with pytest.raises(DatasetValidationError) as caught:
        inspect(root, make_config(root), strict=True)
    assert "MAP_FEATURE_COUNT_MISMATCH" in issue_codes(caught.value.report)
    assert caught.value.report.invalid_video_count == 1


def test_equal_map_and_feature_counts_pass(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    assert inspect(root, make_config(root), strict=True).valid_for_index_build


@pytest.mark.parametrize(
    ("frame_values", "code"),
    [
        ([100, 100], "FRAME_IDX_DUPLICATE"),
        ([-1, 130], "FRAME_IDX_NEGATIVE"),
        (["abc", 130], "FRAME_IDX_INVALID"),
        ([130, 100], "FRAME_IDX_NON_MONOTONIC"),
    ],
)
def test_frame_idx_validation(tmp_path, frame_values, code) -> None:
    rows = [dict(BASE_ROWS[index], frame_idx=value) for index, value in enumerate(frame_values)]
    root = make_dataset(tmp_path / code, rows=rows)
    assert code in issue_codes(inspect(root, make_config(root)))


def test_missing_required_map_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "map-keyframes" / f"{VIDEO_ID}.csv").unlink()
    report = inspect(root, make_config(root))
    assert {"MAP_MISSING", "REQUIRED_SOURCE_MISSING"} <= issue_codes(report)


def test_missing_required_feature_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "clip-features-32" / f"{VIDEO_ID}.npy").unlink()
    report = inspect(root, make_config(root))
    assert {"FEATURE_MISSING", "REQUIRED_SOURCE_MISSING"} <= issue_codes(report)


def test_missing_keyframe_image_is_error(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "keyframes" / VIDEO_ID / "002.jpg").unlink()
    assert "KEYFRAME_IMAGE_MISSING" in issue_codes(inspect(root, make_config(root)))


def test_orphan_keyframe_image_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    Image.new("RGB", (8, 8)).save(root / "keyframes" / VIDEO_ID / "999.jpg")
    report = inspect(root, make_config(root))
    assert {"KEYFRAME_IMAGE_ORPHAN", "KEYFRAME_COUNT_MISMATCH"} <= issue_codes(report)


def test_missing_keyframe_folder_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    shutil.rmtree(root / "keyframes" / VIDEO_ID)
    report = inspect(root, make_config(root))
    assert {"KEYFRAME_FOLDER_MISSING", "REQUIRED_SOURCE_MISSING"} <= issue_codes(report)


def test_feature_ndim_must_be_two(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", features=np.array([1, 2], dtype=np.float32))
    assert "FEATURE_INVALID_NDIM" in issue_codes(inspect(root, make_config(root)))


def test_feature_dimension_mismatch(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", features=np.ones((2, 3), dtype=np.float32))
    assert "FEATURE_DIM_MISMATCH" in issue_codes(inspect(root, make_config(root)))


def test_feature_dtype_unsupported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", features=np.ones((2, 2), dtype=np.int32))
    assert "FEATURE_DTYPE_UNSUPPORTED" in issue_codes(inspect(root, make_config(root)))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_feature_non_finite_values(tmp_path, value) -> None:
    features = np.array([[1.0, 0.0], [value, 1.0]], dtype=np.float32)
    root = make_dataset(tmp_path / str(value), features=features)
    report = inspect(root, make_config(root))
    assert "FEATURE_NON_FINITE" in issue_codes(report)
    assert report.videos[0].feature_non_finite_count == 1


def test_zero_feature_vector_is_error(tmp_path) -> None:
    features = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    root = make_dataset(tmp_path / "data", features=features)
    assert "FEATURE_ZERO_VECTOR" in issue_codes(inspect(root, make_config(root)))


def test_collection_feature_dimensions_must_match(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", video_id="L01_V001")
    make_dataset(root, video_id="L01_V002", features=np.ones((2, 3), dtype=np.float32))
    report = inspect(root, make_config(root))
    assert report.feature_dimensions == [2, 3]
    assert all("FEATURE_DIM_MISMATCH" in issue_codes(report, video_id) for video_id in ("L01_V001", "L01_V002"))


def test_missing_optional_media_info_and_mp4_remain_valid(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    report = inspect(root, make_config(root))
    assert report.valid_for_index_build
    assert report.missing_sources["media_info"] == [VIDEO_ID]
    assert report.missing_sources["video"] == [VIDEO_ID]


def test_corrupt_media_json_is_warning(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "media-info").mkdir()
    (root / "media-info" / f"{VIDEO_ID}.json").write_text("{", encoding="utf-8")
    report = inspect(root, make_config(root))
    assert "MEDIA_JSON_CORRUPT" in issue_codes(report)
    assert report.valid_for_index_build


def test_missing_objects_non_strict_is_warning(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    config = make_config(root, load_objects=True, strict_objects=False)
    report = inspect(root, config)
    assert "OBJECT_FILE_MISSING" in issue_codes(report)
    assert report.valid_for_index_build


def test_missing_objects_strict_is_invalid(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    config = make_config(root, load_objects=True, strict_objects=True)
    report = inspect(root, config)
    assert "OBJECT_FILE_MISSING" in issue_codes(report)
    assert not report.valid_for_index_build


def test_corrupt_object_json_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    _write_objects(root, "{")
    report = inspect(root, make_config(root, load_objects=True))
    assert "OBJECT_JSON_CORRUPT" in issue_codes(report)
    assert report.videos[0].corrupt_object_files == 2


def test_invalid_object_confidence_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    _write_objects(root, _object_payload(score="1.5"))
    report = inspect(root, make_config(root, load_objects=True))
    assert "OBJECT_SCORE_INVALID" in issue_codes(report)


def test_invalid_object_box_is_reported(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    _write_objects(root, _object_payload(box=["0.8", "0.2", "0.1", "0.9"]))
    report = inspect(root, make_config(root, load_objects=True))
    assert "OBJECT_BOX_INVALID" in issue_codes(report)


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("map", "ORPHAN_MAP"),
        ("feature", "ORPHAN_FEATURE"),
        ("keyframe", "ORPHAN_KEYFRAME_FOLDER"),
        ("object", "ORPHAN_OBJECT_FOLDER"),
        ("media", "ORPHAN_MEDIA_INFO"),
        ("video", "ORPHAN_VIDEO"),
    ],
)
def test_orphan_sources_are_reported(tmp_path, source, code) -> None:
    root = make_dataset(tmp_path / "data")
    orphan = "L99_V999"
    if source == "map":
        _write_map(root, orphan, BASE_ROWS)
    elif source == "feature":
        np.save(root / "clip-features-32" / f"{orphan}.npy", np.ones((2, 2), dtype=np.float32))
    elif source == "keyframe":
        (root / "keyframes" / orphan).mkdir()
    elif source == "object":
        (root / "objects" / orphan).mkdir(parents=True)
    elif source == "media":
        (root / "media-info").mkdir()
        (root / "media-info" / f"{orphan}.json").write_text("{}", encoding="utf-8")
    else:
        (root / "video").mkdir()
        (root / "video" / f"{orphan}.mp4").write_bytes(b"")
    report = inspect(root, make_config(root))
    assert code in issue_codes(report, orphan)
    if source in {"object", "media", "video"}:
        assert report.valid_for_index_build


def test_record_uses_official_frame_idx_not_ordinal(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    entry, _ = AICDatasetLoader(root, app_config=make_config(root)).build_entry()
    assert entry.index.ids == [f"{VIDEO_ID}/100", f"{VIDEO_ID}/130"]
    assert entry.raws[f"{VIDEO_ID}/100"].frame_idx == 100


def test_media_metadata_is_separate_from_frame_caption(tmp_path, monkeypatch) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "media-info").mkdir()
    (root / "media-info" / f"{VIDEO_ID}.json").write_text(
        json.dumps({"title": "Video title", "description": "Video description", "keywords": ["tag"], "fps": 30}),
        encoding="utf-8",
    )
    captured = {}
    original = dataset_module._build_aic_index

    def capture(records, *, kind="flat"):
        captured["records"] = records
        return original(records, kind=kind)

    monkeypatch.setattr(dataset_module, "_build_aic_index", capture)
    AICDatasetLoader(root, app_config=make_config(root)).build_entry()
    record = captured["records"][0]
    assert record.media_title == "Video title"
    assert record.media_description == "Video description"
    assert record.media_fps == 30
    assert record.frame_caption == ""
    assert record.llm_caption is None


def test_searchable_text_preserves_provenance_without_double_counting() -> None:
    record = KeyframeRecord(
        id="V/1",
        video_id="V",
        timestamp=0,
        clip_embedding=np.ones(2, dtype=np.float32),
        objects=["cup"],
        media_title="Morning",
        media_description="Kitchen",
        media_tags=("home",),
        frame_caption="A red cup",
        ocr_text="SALE",
        asr_text="hello",
    )
    text = searchable_text(record)
    for prefix in ("objects:", "media_title:", "media_description:", "media_tags:", "frame_caption:", "ocr:", "asr:"):
        assert prefix in text
    assert text.count("A red cup") == 1


def test_legacy_record_searchable_text_is_backward_compatible() -> None:
    record = object.__new__(KeyframeRecord)
    record.objects = ["legacy"]
    record.ocr_text = "ocr"
    record.asr_text = None
    record.llm_caption = "old caption"
    restored = pickle.loads(pickle.dumps(record))
    text = searchable_text(restored)
    assert "frame_caption: old caption" in text


def test_inspect_data_does_not_build_cache(tmp_path, capsys) -> None:
    root = make_dataset(tmp_path / "data")
    cache = tmp_path / "cache"
    config_path = write_cli_config(tmp_path / "settings.yaml", make_config(root, cache_dir=cache))
    assert cli_main(["--config", str(config_path), "inspect-data"]) == 0
    assert not cache.exists()
    assert json.loads(capsys.readouterr().out)["valid_for_index_build"] is True


def test_cli_valid_dataset_exits_zero(tmp_path, capsys) -> None:
    root = make_dataset(tmp_path / "data")
    path = write_cli_config(tmp_path / "settings.yaml", make_config(root))
    assert cli_main(["--config", str(path), "inspect-data", "--summary-only"]) == 0
    assert json.loads(capsys.readouterr().out)["invalid_video_count"] == 0


def test_cli_invalid_dataset_exits_five(tmp_path, capsys) -> None:
    root = make_dataset(tmp_path / "data", features=np.ones((1, 2), dtype=np.float32))
    path = write_cli_config(tmp_path / "settings.yaml", make_config(root))
    assert cli_main(["--config", str(path), "inspect-data", "--strict"]) == 5
    assert json.loads(capsys.readouterr().out)["invalid_video_count"] == 1


def test_json_report_is_utf8_and_json_safe(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    report = inspect(root, make_config(root))
    report.issues.append(DatasetIssue("info", "UNICODE", None, None, None, "dữ liệu tiếng Việt", "đủ", "liệu"))
    path = write_dataset_report(report, tmp_path / "bao-cao.json")
    payload = path.read_text(encoding="utf-8")
    assert "dữ liệu tiếng Việt" in payload
    assert isinstance(json.loads(payload)["videos"], list)


def test_atomic_report_write_leaves_no_temp_file(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    output = tmp_path / "report.json"
    write_dataset_report(inspect(root, make_config(root)), output)
    assert output.is_file()
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_deep_false_does_not_decode_images(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "keyframes" / VIDEO_ID / "001.jpg").write_bytes(b"not-an-image")
    assert inspect(root, make_config(root), deep=False).valid_for_index_build


def test_deep_true_detects_corrupt_image(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / "keyframes" / VIDEO_ID / "001.jpg").write_bytes(b"not-an-image")
    report = inspect(root, make_config(root), deep=True)
    assert "KEYFRAME_IMAGE_CORRUPT" in issue_codes(report)
    assert not report.valid_for_index_build


def test_limit_videos_is_deterministic(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", video_id="L01_V002")
    make_dataset(root, video_id="L01_V001")
    report = inspect(root, make_config(root), limit=1)
    assert [video.video_id for video in report.videos] == ["L01_V001"]


def test_build_index_gate_writes_no_partial_cache(tmp_path) -> None:
    root = make_dataset(tmp_path / "data", features=np.ones((1, 2), dtype=np.float32))
    cache = tmp_path / "cache"
    config = make_config(root, cache_dir=cache)
    with pytest.raises(DatasetAlignmentError):
        AICCompetitionEngine.from_data_root(
            app_config=config,
            text_encoder=TinyTextEncoder(),
            rebuild=True,
        )
    assert not (cache / "entry" / "entry.pkl").exists()
    assert not (cache / "cache_manifest.json").exists()


def test_cache_manifest_uses_record_schema_v2_and_dataset_report(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    cache = tmp_path / "cache"
    config = make_config(root, cache_dir=cache)
    _, load = AICCompetitionEngine.from_data_root(
        app_config=config,
        text_encoder=TinyTextEncoder(),
        rebuild=True,
    )
    manifest = read_cache_manifest(cache / "cache_manifest.json")
    assert manifest.record_schema_version == AIC_RECORD_SCHEMA_VERSION == 2
    assert manifest.files["dataset_report"] == "dataset_report.json"
    assert load.stats is not None and load.stats.dataset_validated


def test_record_schema_version_changes_cache_fingerprint(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    options = cache_build_options_from_config(make_config(root))
    old = replace(options, record_schema_version=AIC_RECORD_SCHEMA_VERSION - 1)
    assert cache_fingerprint(old) != cache_fingerprint(options)


def test_expected_feature_dimension_is_in_cache_fingerprint(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    options = cache_build_options_from_config(make_config(root))
    assert options.index_params["expected_feature_dim"] == 2


@pytest.mark.parametrize(
    ("validation", "message"),
    [
        ({"strict_alignment": False}, "strict_alignment"),
        ({"require_keyframe_images": False}, "require_keyframe_images"),
        ({"supported_feature_dtypes": ["float64"]}, "supported_feature_dtypes"),
        ({"expected_feature_dim": 3}, "expected_feature_dim"),
        ({"max_image_decode_checks_per_video": 0}, "max_image_decode_checks_per_video"),
    ],
)
def test_dataset_validation_config_is_checked(validation, message) -> None:
    with pytest.raises(ConfigError, match=message):
        app_config_from_dict(
            {
                "aic2026": {
                    "dataset": {"validation": validation},
                    "encoder": {"feature_dim": 2},
                }
            }
        )


def test_timestamp_and_fps_diagnostics(tmp_path) -> None:
    rows = [dict(BASE_ROWS[0]), dict(BASE_ROWS[1], pts_time=-1, fps=0)]
    root = make_dataset(tmp_path / "data", rows=rows)
    codes = issue_codes(inspect(root, make_config(root)))
    assert {"TIMESTAMP_INVALID", "MAP_FPS_INVALID"} <= codes


def test_timestamp_non_monotonic_is_error(tmp_path) -> None:
    rows = [dict(BASE_ROWS[0], pts_time=2), dict(BASE_ROWS[1], pts_time=1)]
    root = make_dataset(tmp_path / "data", rows=rows)
    assert "TIMESTAMP_NON_MONOTONIC" in issue_codes(inspect(root, make_config(root)))


def test_summary_only_output_still_writes_report_file(tmp_path, capsys) -> None:
    root = make_dataset(tmp_path / "data")
    config_path = write_cli_config(tmp_path / "settings.yaml", make_config(root))
    output = tmp_path / "summary.json"
    assert cli_main([
        "--config", str(config_path), "inspect-data", "--summary-only", "--output", str(output)
    ]) == 0
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["valid_for_index_build"] is True
    assert json.loads(capsys.readouterr().out)["report_path"] == str(output)
