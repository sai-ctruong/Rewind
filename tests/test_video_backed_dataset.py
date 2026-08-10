"""Phase 3.2: the capability model, the existing-video scope, and video-aware serving.

Official AIC data roles drive these expectations: the **video** is the competition
data, while keyframes / objects / CLIP features / metadata are supporting data. So
map + CLIP gate retrieval, and a missing keyframe JPEG does not.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from aic2026.cache_manifest import (
    CACHE_MANIFEST_FILENAME,
    StaleCacheError,
    cache_build_options_from_config,
    cache_fingerprint,
    read_cache_manifest,
)
from aic2026.cli import main as cli_main
from aic2026.config import DatasetScopeConfig, app_config_from_dict
from aic2026.dataset_scope import (
    hash_selected_video_ids,
    resolve_dataset_scope,
    select_video_ids,
)
from aic2026.dataset_validation import inspect_aic_dataset
from aic2026.engine import AICCompetitionEngine
from aic2026.video_inventory import (
    discover_videos,
    existing_video_ids_with_retrieval_support,
    summarize_coverage,
    support_coverage,
)
import ui.app as appmod

cv2 = pytest.importorskip("cv2", reason="OpenCV is required to generate MP4 fixtures")

FPS = 10.0
WIDTH = 64
HEIGHT = 48


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def write_video(root: Path, video_id: str, *, frame_count: int = 30) -> Path:
    path = root / "video" / f"{video_id}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():  # pragma: no cover - depends on local codec support
        pytest.skip("No MP4 encoder is available in this OpenCV build")
    try:
        for index in range(frame_count):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 8) % 256
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:  # pragma: no cover
        pytest.skip("OpenCV produced no usable MP4 in this environment")
    return path


def write_support(
    root: Path,
    video_id: str,
    *,
    map_csv: bool = True,
    clip: bool = True,
    jpeg: bool = True,
) -> None:
    if map_csv:
        (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,10.0,0\n2,1.0,10.0,10\n", encoding="utf-8"
        )
    if clip:
        (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
        np.save(
            root / "clip-features-32" / f"{video_id}.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
    if jpeg:
        folder = root / "keyframes" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        for ordinal in (1, 2):
            Image.new("RGB", (8, 8), (ordinal * 60, 0, 0)).save(folder / f"{ordinal:03d}.jpg")


def make_config(root: Path, *, cache_dir: Path | None = None, mode="patterns", include=("*",), exclude=()):
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir or root.parent / "cache"),
                    # Keep derived frames inside tmp_path; tests never write to the repo.
                    "frame_cache_dir": str(root.parent / "video_frame_cache"),
                    "scope": {
                        "mode": mode,
                        "include_patterns": list(include),
                        "exclude_patterns": list(exclude),
                    },
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
            }
        }
    )


def report_for(root: Path, video_id: str, report):
    return next(item for item in report.videos if item.video_id == video_id)


# --------------------------------------------------------------- validity model


def test_map_clip_and_jpeg_is_retrieval_valid_and_visually_accessible(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V001")
    video = report_for(root, "L21_V001", inspect_aic_dataset(root, app_config=make_config(root)))
    assert video.retrieval_valid and video.visual_accessible
    assert video.visual_source == "keyframe_jpeg"
    assert video.refinement_ready is False


def test_map_clip_and_mp4_without_jpeg_is_retrieval_valid_and_visually_accessible(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V028", jpeg=False)
    write_video(root, "L21_V028")
    report = inspect_aic_dataset(root, app_config=make_config(root))
    video = report_for(root, "L21_V028", report)
    assert video.retrieval_valid and video.visual_accessible
    assert video.visual_source == "video_decode"
    assert video.refinement_ready and video.qa_visual_ready
    assert report.valid_for_index_build


def test_map_clip_without_jpeg_or_mp4_is_retrieval_valid_but_not_visual(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V050", jpeg=False)
    report = inspect_aic_dataset(root, app_config=make_config(root))
    video = report_for(root, "L21_V050", report)
    assert video.retrieval_valid is True
    assert video.visual_accessible is False
    assert video.visual_source == "none"
    assert video.refinement_ready is False
    # Retrieval-only is a legitimate state, so the index build is not blocked.
    assert report.valid_for_index_build


def test_mp4_without_map_or_clip_is_not_retrieval_valid(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V001")
    write_video(root, "L21_V099")
    report = inspect_aic_dataset(root, app_config=make_config(root))
    video = report_for(root, "L21_V099", report)
    assert video.retrieval_valid is False
    assert video.visual_accessible is True
    assert "VIDEO_PRESENT_BUT_RETRIEVAL_SUPPORT_MISSING" in report.issue_counts


def test_missing_jpeg_no_longer_invalidates_global_retrieval(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V028", jpeg=False)
    write_video(root, "L21_V028")
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert report.valid_for_index_build
    assert report.retrieval_valid_video_count == 1
    assert "KEYFRAME_JPEG_UNAVAILABLE" in report.issue_counts
    codes = {issue.code for issue in report.issues if issue.severity == "error"}
    assert codes == set()


def test_report_counts_the_capability_model(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V001")  # jpeg only
    write_support(root, "L21_V028", jpeg=False)  # mp4 fallback
    write_video(root, "L21_V028")
    write_support(root, "L21_V050", jpeg=False)  # retrieval only
    report = inspect_aic_dataset(root, app_config=make_config(root))
    assert report.retrieval_valid_video_count == 3
    assert report.visual_accessible_video_count == 2
    assert report.qa_visual_ready_video_count == 2
    assert report.refinement_ready_video_count == 1
    assert report.keyframe_jpeg_backed_video_count == 1
    assert report.video_fallback_video_count == 1
    assert report.no_visual_source_video_count == 1


def test_partial_jpeg_coverage_reports_the_video_as_the_visual_source(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V027")
    (root / "keyframes" / "L21_V027" / "002.jpg").unlink()
    write_video(root, "L21_V027")
    video = report_for(root, "L21_V027", inspect_aic_dataset(root, app_config=make_config(root)))
    assert video.keyframe_jpeg_available is True
    assert video.missing_keyframes == 1
    assert video.visual_source == "video_decode"


# ----------------------------------------------------------------- inventory


def test_inventory_lists_real_files_and_collections(tmp_path) -> None:
    root = tmp_path / "data"
    write_video(root, "L21_V028")
    write_video(root, "L22_V001")
    inventory = discover_videos(root, probe_readable=True)
    assert inventory.video_ids == ("L21_V028", "L22_V001")
    assert inventory.collections == {"L21": 1, "L22": 1}
    assert inventory.duplicate_ids == []
    assert inventory.total_bytes > 0
    assert all(item.readable for item in inventory.videos)


def test_inventory_flags_unreadable_containers(tmp_path) -> None:
    root = tmp_path / "data"
    (root / "video").mkdir(parents=True)
    (root / "video" / "L21_V404.mp4").write_bytes(b"not an mp4")
    inventory = discover_videos(root, probe_readable=True)
    assert inventory.to_dict()["unreadable_count"] == 1
    assert inventory.readable_video_ids == ()


def test_support_coverage_summary(tmp_path) -> None:
    root = tmp_path / "data"
    write_support(root, "L21_V001")
    write_video(root, "L21_V001")
    write_support(root, "L21_V028", jpeg=False)
    write_video(root, "L21_V028")
    write_video(root, "L21_V099")  # no support data at all
    write_support(root, "L21_V050")  # support data with no video
    summary = summarize_coverage(support_coverage(root))
    assert summary["video"] == 3
    assert summary["video_map_clip"] == 2
    assert summary["video_map_clip_jpeg"] == 1
    assert summary["video_needing_mp4_fallback"] == 1
    assert summary["video_without_map"] == 1
    assert summary["map_clip_without_video"] == 1


# ------------------------------------------------------- existing-video scope


def build_mixed_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_support(root, "L21_V001")
    write_video(root, "L21_V001")
    write_support(root, "L21_V028", jpeg=False)
    write_video(root, "L21_V028")
    write_support(root, "L21_V050")  # support data only, no MP4
    write_video(root, "L21_V099")  # MP4 only, no support data
    return root


def test_existing_video_scope_resolves_to_real_intersection(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    assert existing_video_ids_with_retrieval_support(root) == ("L21_V001", "L21_V028")


def test_existing_video_scope_excludes_mp4_without_map_or_clip(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    resolved = resolve_dataset_scope(DatasetScopeConfig(mode="existing_videos"), root)
    assert resolved.source_video_ids == ("L21_V001", "L21_V028")
    discovered = ["L21_V001", "L21_V028", "L21_V050", "L21_V099"]
    assert select_video_ids(discovered, resolved) == ("L21_V001", "L21_V028")


def test_existing_video_scope_still_honours_patterns(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    resolved = resolve_dataset_scope(
        DatasetScopeConfig(mode="existing_videos", exclude_patterns=("L21_V028",)), root
    )
    discovered = ["L21_V001", "L21_V028", "L21_V050", "L21_V099"]
    assert select_video_ids(discovered, resolved) == ("L21_V001",)


def test_unresolved_existing_video_scope_fails_loudly(tmp_path) -> None:
    from aic2026.dataset_scope import DatasetScopeError

    with pytest.raises(DatasetScopeError, match="must be resolved against DATA_ROOT"):
        select_video_ids(["L21_V001"], DatasetScopeConfig(mode="existing_videos"))


def test_adding_a_video_changes_the_selected_ids_hash(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    before = hash_selected_video_ids(existing_video_ids_with_retrieval_support(root))
    write_support(root, "L21_V050")
    write_video(root, "L21_V050")
    after = hash_selected_video_ids(existing_video_ids_with_retrieval_support(root))
    assert before != after


def test_selected_ids_hash_is_deterministic(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    first = hash_selected_video_ids(existing_video_ids_with_retrieval_support(root))
    second = hash_selected_video_ids(existing_video_ids_with_retrieval_support(root))
    assert first == second == hash_selected_video_ids(["L21_V028", "L21_V001"])


def test_inspection_with_existing_video_scope_selects_only_video_backed_ids(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    report = inspect_aic_dataset(root, app_config=make_config(root, mode="existing_videos"))
    assert report.selected_video_ids == ["L21_V001", "L21_V028"]
    assert report.scope["mode"] == "existing_videos"
    assert report.discovered_video_count == 4
    assert report.excluded_video_count == 2
    assert report.valid_for_index_build


# ---------------------------------------------------------------------- cache


def test_existing_video_scope_differs_from_a_wildcard_scope_when_sets_differ(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    wildcard = cache_fingerprint(make_config(root, include=["L21_*"]))
    existing = cache_fingerprint(make_config(root, mode="existing_videos"))
    assert wildcard != existing


def test_manifest_stores_resolved_ids_hash_and_mode(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    cache = tmp_path / "cache_existing"
    config = make_config(root, cache_dir=cache, mode="existing_videos")
    _, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    manifest = read_cache_manifest(cache / CACHE_MANIFEST_FILENAME)
    assert manifest.dataset_scope["mode"] == "existing_videos"
    assert manifest.selected_video_ids == ["L21_V001", "L21_V028"]
    assert manifest.selected_video_ids_hash == hash_selected_video_ids(
        ["L21_V001", "L21_V028"]
    )
    assert load.stats.scope_mode == "existing_videos"
    assert load.stats.refinement_ready_videos == 2


def test_downloading_another_video_makes_the_existing_video_cache_stale(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    cache = tmp_path / "cache_existing"
    config = make_config(root, cache_dir=cache, mode="existing_videos")
    AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    # A newly downloaded MP4 joins the resolved set, so the old cache no longer
    # describes the dataset.
    write_video(root, "L21_V050")
    with pytest.raises(StaleCacheError, match="selected_video_ids_hash"):
        AICCompetitionEngine.from_data_root(
            app_config=config, text_encoder=TinyTextEncoder()
        )


def test_query_time_ranking_change_does_not_alter_the_fingerprint(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    config = make_config(root, mode="existing_videos")
    changed = replace(config, ranking=replace(config.ranking, final_top_k=11))
    assert cache_fingerprint(config) == cache_fingerprint(changed)


def test_derived_frame_cache_is_not_part_of_the_manifest(tmp_path) -> None:
    root = build_mixed_root(tmp_path)
    cache = tmp_path / "cache_existing"
    config = make_config(root, cache_dir=cache, mode="existing_videos")
    AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    manifest = read_cache_manifest(cache / CACHE_MANIFEST_FILENAME)
    assert not any("frame_cache" in key or "frame_cache" in value for key, value in manifest.files.items())


# ------------------------------------------------------------------------ CLI


def write_cli_config(path: Path, root: Path, cache: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "aic2026": {
                    "dataset": {
                        "root": str(root),
                        "cache_dir": str(cache),
                        "frame_cache_dir": str(cache.parent / "video_frame_cache"),
                        "validation": {"expected_feature_dim": 2},
                    },
                    "encoder": {"feature_dim": 2},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_cli_scope_existing_videos_flag(tmp_path, capsys) -> None:
    root = build_mixed_root(tmp_path)
    config_path = write_cli_config(tmp_path / "settings.yaml", root, tmp_path / "cache")
    assert cli_main(["--config", str(config_path), "--scope-existing-videos", "inspect-data"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"]["mode"] == "existing_videos"
    assert payload["selected_video_count"] == 2
    assert payload["discovered_video_count"] == 4
    assert payload["refinement_ready_video_count"] == 2


def test_cli_without_the_flag_keeps_the_full_scope(tmp_path, capsys) -> None:
    root = build_mixed_root(tmp_path)
    config_path = write_cli_config(tmp_path / "settings.yaml", root, tmp_path / "cache")
    assert cli_main(["--config", str(config_path), "inspect-data"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"]["mode"] == "patterns"
    assert payload["selected_video_count"] == 4


# ------------------------------------------------------------------- UI/API


def build_ui(tmp_path: Path, *, jpeg: bool, video: bool):
    root = tmp_path / "data"
    write_support(root, "L21_V028", jpeg=jpeg)
    if video:
        write_video(root, "L21_V028")
    cache = tmp_path / "cache"
    config = make_config(root, cache_dir=cache)
    engine, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    app = appmod.create_app(app_config=config)
    app.testing = True
    client = app.test_client()
    with app.app_context():
        pass
    # Prime the app state with the already-built engine, mirroring an indexed session.
    client.post("/api/video/index", json={})
    return client, engine


def test_frame_route_serves_the_keyframe_jpeg_when_present(tmp_path) -> None:
    client, _ = build_ui(tmp_path, jpeg=True, video=True)
    response = client.get("/api/video/frame/L21_V028/kf_000001")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["X-Frame-Source"] == "keyframe_jpeg"


def test_frame_route_decodes_the_video_when_the_jpeg_is_absent(tmp_path) -> None:
    client, _ = build_ui(tmp_path, jpeg=False, video=True)
    response = client.get("/api/video/frame/L21_V028/kf_000002")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["X-Frame-Source"] == "video_decode"
    # The official submission frame is echoed unchanged.
    assert response.headers["X-Frame-Id"] == "10"


def test_frame_route_reports_unavailable_instead_of_500(tmp_path) -> None:
    client, _ = build_ui(tmp_path, jpeg=False, video=False)
    response = client.get("/api/video/frame/L21_V028/kf_000002")
    assert response.status_code == 422
    body = response.get_json()
    assert body["error_code"] == "FRAME_UNAVAILABLE"
    assert body["frame"]["source"] == "none"
    assert body["frame"]["available"] is False


def test_frame_route_writes_derived_frames_only_to_the_configured_cache(tmp_path) -> None:
    client, _ = build_ui(tmp_path, jpeg=False, video=True)
    assert client.get("/api/video/frame/L21_V028/kf_000002").status_code == 200
    derived = list((tmp_path / "video_frame_cache").rglob("*.jpg"))
    assert derived, "the decoded frame should be cached under the configured directory"
    assert not list((tmp_path / "data").rglob("frame_*.jpg"))


def test_frame_cache_dir_inside_data_root_is_rejected(tmp_path) -> None:
    from aic2026.config import ConfigError

    with pytest.raises(ConfigError, match="must not live inside DATA_ROOT"):
        app_config_from_dict(
            {
                "aic2026": {
                    "dataset": {
                        "root": str(tmp_path / "data"),
                        "frame_cache_dir": str(tmp_path / "data" / "keyframes"),
                    }
                }
            }
        )


def test_unknown_frame_id_is_404(tmp_path) -> None:
    client, _ = build_ui(tmp_path, jpeg=True, video=True)
    response = client.get("/api/video/frame/L21_V028/kf_999999")
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "FRAME_NOT_FOUND"


def test_search_results_expose_image_availability_without_decoding(tmp_path) -> None:
    client, _ = build_ui(tmp_path, jpeg=False, video=True)
    body = client.post("/api/video/search", json={"query": "anything", "topk": 5}).get_json()
    assert body["results"], "expected search results"
    for item in body["results"]:
        assert item["image_available"] is True
        assert item["image_source"] == "video_decode"
        assert item["video_available"] is True
