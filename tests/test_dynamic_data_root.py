"""Phase 4 regression: switching DATA_ROOT must move the WHOLE application.

Roots A and B deliberately contain the same video ID `L21_V001` with different frame
content. That is what makes this a real test of the Phase 0 bug: before Phase 4 the
engine moved to root B while the frame route, video route, health, and cache status
stayed on root A, and because the IDs matched, the mix-up served plausible-looking
pixels from the wrong dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ui.app as appmod
from aic2026.cache_manifest import canonical_data_root
from aic2026.config import app_config_from_dict

cv2 = pytest.importorskip("cv2", reason="OpenCV is required to read served frames")

VIDEO_ID = "L21_V001"
STATE_KEY = appmod.STATE_EXTENSION_KEY


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def make_root(root: Path, *, red: int, extra_video: str | None = None) -> Path:
    """Build a minimal dataset whose keyframe JPEGs encode `red` in every pixel."""
    ids = [VIDEO_ID] + ([extra_video] if extra_video else [])
    for video_id in ids:
        for relative in ("map-keyframes", "clip-features-32", f"keyframes/{video_id}"):
            (root / relative).mkdir(parents=True, exist_ok=True)
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,1.0,30.0,30\n", encoding="utf-8"
        )
        np.save(
            root / "clip-features-32" / f"{video_id}.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        for ordinal in (1, 2):
            Image.new("RGB", (16, 16), (red, 0, 0)).save(
                root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
            )
        (root / "video").mkdir(parents=True, exist_ok=True)
        (root / "video" / f"{video_id}.mp4").write_bytes(f"mp4-{root.name}-{video_id}".encode())
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path):
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir),
                    "frame_cache_dir": str(frame_cache_dir),
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
            }
        }
    )


@pytest.fixture()
def roots(tmp_path):
    """Root A red=250, root B red=40, same video ID in both."""
    root_a = make_root(tmp_path / "root_a", red=250)
    root_b = make_root(tmp_path / "root_b", red=40, extra_video="L21_V002")
    return root_a, root_b


@pytest.fixture()
def client(tmp_path, roots):
    root_a, _ = roots
    config = make_config(root_a, tmp_path / "cache_a", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    return app.test_client(), app


def served_red(response) -> int:
    array = cv2.imdecode(np.frombuffer(response.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert array is not None
    return int(array[0, 0, 2])


def activate(client, root: Path, cache_dir: Path | None = None, **extra):
    body = {"path": str(root), "rebuild": True, **extra}
    if cache_dir is not None:
        body["cache_dir"] = str(cache_dir)
    return client.post("/api/video/index_folder", json=body)


def patch_encoder(monkeypatch):
    """Keep the engine offline and deterministic."""
    original = appmod.AICCompetitionEngine.from_data_root

    def offline(*args, **kwargs):
        kwargs.setdefault("text_encoder", TinyTextEncoder())
        return original(*args, **kwargs)

    monkeypatch.setattr(appmod.AICCompetitionEngine, "from_data_root", staticmethod(offline))


@pytest.fixture()
def offline_client(client, monkeypatch):
    patch_encoder(monkeypatch)
    return client


# ------------------------------------------------------------------- baseline


def test_create_app_publishes_one_initial_state(client) -> None:
    _, app = client
    manager = app.extensions[STATE_KEY]
    state = manager.get_state()
    assert state.generation == 1
    assert state.engine_loaded is False
    assert state.resolved_video_ids == (VIDEO_ID,)


def test_health_reports_the_active_runtime(offline_client, roots) -> None:
    test_client, _ = offline_client
    root_a, _ = roots
    body = test_client.get("/api/health").get_json()
    assert body["runtime"]["generation"] == 1
    assert body["runtime"]["data_root"] == canonical_data_root(root_a)
    assert body["runtime"]["engine_loaded"] is False
    assert body["runtime"]["selected_video_count"] == 1


# --------------------------------------------------------------- A/B switching


def test_root_a_then_root_b_moves_the_whole_application(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, root_b = roots
    manager = app.extensions[STATE_KEY]

    assert activate(test_client, root_a, tmp_path / "cache_a").status_code == 200
    state_a = manager.get_state()
    assert state_a.generation == 2
    assert state_a.data_root == canonical_data_root(root_a)

    # Root A serves its own pixels.
    frame_a = test_client.get(f"/api/video/frame/{VIDEO_ID}/kf_000001")
    assert frame_a.status_code == 200
    assert served_red(frame_a) > 200

    health_a = test_client.get("/api/health").get_json()
    assert health_a["runtime"]["data_root"] == canonical_data_root(root_a)
    assert test_client.get("/api/video/list").get_json()["videos"] == [VIDEO_ID]
    video_a = test_client.get(f"/api/video/file/{VIDEO_ID}")
    assert video_a.status_code == 200 and b"root_a" in video_a.data

    # Switch.
    assert activate(test_client, root_b, tmp_path / "cache_b").status_code == 200
    state_b = manager.get_state()
    assert state_b.generation == 3
    assert state_b.data_root == canonical_data_root(root_b)

    # Every surface follows, including the identically named video ID.
    frame_b = test_client.get(f"/api/video/frame/{VIDEO_ID}/kf_000001")
    assert frame_b.status_code == 200
    assert served_red(frame_b) < 100, "served root A pixels after switching to root B"

    health_b = test_client.get("/api/health").get_json()
    assert health_b["runtime"]["data_root"] == canonical_data_root(root_b)
    assert health_b["runtime"]["generation"] == 3

    listed = test_client.get("/api/video/list").get_json()
    assert listed["videos"] == [VIDEO_ID, "L21_V002"]
    assert listed["folder"] == str(root_b)

    video_b = test_client.get(f"/api/video/file/{VIDEO_ID}")
    assert video_b.status_code == 200
    assert b"root_b" in video_b.data and b"root_a" not in video_b.data

    # Engine, frame provider, and cache identity all moved together.
    assert canonical_data_root(state_b.engine.app_config.dataset.root) == canonical_data_root(root_b)
    assert canonical_data_root(state_b.frame_provider.data_root) == canonical_data_root(root_b)
    assert state_b.cache_dir == str(tmp_path / "cache_b")
    assert state_b.cache_status["valid"] is True


def test_search_after_switch_uses_root_b(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, root_b = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    activate(test_client, root_b, tmp_path / "cache_b")
    body = test_client.post("/api/video/search", json={"query": "anything", "topk": 5}).get_json()
    assert body["generation"] == 3
    assert body["results"], "expected results"
    for item in body["results"]:
        assert item["generation"] == 3
        assert "generation=3" in item["image"]
        assert item["video_url"] is None or "generation=3" in item["video_url"]
        # Logical API URLs only; no filesystem path leaks to the frontend.
        assert item["image"].startswith("/api/")
        assert str(root_a) not in str(item)
        assert str(root_b) not in str(item)


def test_frame_provider_instance_changes_with_the_root(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, root_b = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_a, tmp_path / "cache_a")
    provider_a = manager.get_state().frame_provider
    activate(test_client, root_b, tmp_path / "cache_b")
    provider_b = manager.get_state().frame_provider
    assert provider_a is not provider_b
    assert canonical_data_root(provider_a.data_root) == canonical_data_root(root_a)
    assert canonical_data_root(provider_b.data_root) == canonical_data_root(root_b)


# ------------------------------------------------------- failures keep state


def test_invalid_root_does_not_mutate_the_active_state(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, _ = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_a, tmp_path / "cache_a")
    before = manager.get_state()

    response = test_client.post(
        "/api/video/index_folder", json={"path": str(tmp_path / "missing")}
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["error_code"] == "DATA_ROOT_INVALID"
    assert body["active_state_changed"] is False
    assert manager.get_state() is before


def test_directory_without_dataset_layout_keeps_state(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, _ = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_a, tmp_path / "cache_a")
    before = manager.get_state()
    empty = tmp_path / "empty"
    empty.mkdir()
    response = test_client.post("/api/video/index_folder", json={"path": str(empty)})
    assert response.status_code == 400
    assert response.get_json()["active_state_changed"] is False
    assert manager.get_state() is before


def test_missing_path_is_a_client_error(offline_client) -> None:
    test_client, _ = offline_client
    response = test_client.post("/api/video/index_folder", json={"path": "   "})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "DATA_ROOT_INVALID"


# ------------------------------------------------------ inspect vs activate


def test_inspection_of_root_b_does_not_activate_it(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, root_b = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_a, tmp_path / "cache_a")
    before = manager.get_state()

    response = test_client.post("/api/dataset/inspect", json={"path": str(root_b)})
    assert response.status_code == 200
    body = response.get_json()
    assert body["active_state_changed"] is False
    assert body["summary"]["selected_video_count"] == 2  # root B has two videos
    assert body["active_state"]["data_root"] == canonical_data_root(root_a)

    after = manager.get_state()
    assert after is before
    assert after.data_root == canonical_data_root(root_a)
    # And the frame route still serves root A.
    assert served_red(test_client.get(f"/api/video/frame/{VIDEO_ID}/kf_000001")) > 200


def test_inspection_of_an_invalid_root_is_a_client_error(offline_client, tmp_path) -> None:
    test_client, _ = offline_client
    response = test_client.post("/api/dataset/inspect", json={"path": str(tmp_path / "nope")})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "DATA_ROOT_INVALID"


# -------------------------------------------------------------- generation


def test_stale_generation_is_rejected_after_a_switch(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, root_b = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    stale_url = f"/api/video/frame/{VIDEO_ID}/kf_000001?generation=2"
    assert test_client.get(stale_url).status_code == 200

    activate(test_client, root_b, tmp_path / "cache_b")
    response = test_client.get(stale_url)
    assert response.status_code == 409
    assert response.get_json()["error_code"] == "STALE_RESULT_GENERATION"
    # The current generation still works.
    assert test_client.get(
        f"/api/video/frame/{VIDEO_ID}/kf_000001?generation=3"
    ).status_code == 200


def test_stale_generation_is_rejected_on_the_video_route(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, root_b = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    activate(test_client, root_b, tmp_path / "cache_b")
    response = test_client.get(f"/api/video/file/{VIDEO_ID}?generation=2")
    assert response.status_code == 409
    assert response.get_json()["error_code"] == "STALE_RESULT_GENERATION"


def test_requests_without_a_generation_still_work(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, _ = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    assert test_client.get(f"/api/video/frame/{VIDEO_ID}/kf_000001").status_code == 200


# ------------------------------------------------------------------- cache


def test_switching_root_without_explicit_cache_derives_a_new_one(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, root_b = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_a, tmp_path / "cache_a")
    cache_a = manager.get_state().cache_dir
    # No cache_dir given for root B: it must not reuse root A's cache.
    assert activate(test_client, root_b).status_code == 200
    cache_b = manager.get_state().cache_dir
    assert cache_b != cache_a
    assert manager.get_state().cache_status["valid"] is True


def test_reusing_root_a_cache_for_root_b_is_rejected(offline_client, roots, tmp_path) -> None:
    """The manifest is the final safety net even if a cache dir is forced."""
    test_client, app = offline_client
    root_a, root_b = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_a, tmp_path / "cache_a")
    before = manager.get_state()
    response = test_client.post(
        "/api/video/index_folder",
        json={"path": str(root_b), "cache_dir": str(tmp_path / "cache_a"), "rebuild": False},
    )
    assert response.status_code == 409
    assert response.get_json()["error_code"] == "STALE_CACHE"
    assert manager.get_state() is before


def test_valid_root_b_cache_loads_without_rebuild(offline_client, roots, tmp_path) -> None:
    test_client, app = offline_client
    root_a, root_b = roots
    manager = app.extensions[STATE_KEY]
    activate(test_client, root_b, tmp_path / "cache_b")
    activate(test_client, root_a, tmp_path / "cache_a")
    # Now load root B again from its existing cache, no rebuild.
    response = test_client.post(
        "/api/video/index_folder",
        json={"path": str(root_b), "cache_dir": str(tmp_path / "cache_b"), "rebuild": False},
    )
    assert response.status_code == 200
    assert response.get_json()["cached"] is True
    assert manager.get_state().data_root == canonical_data_root(root_b)


# ----------------------------------------------------------------- security


def test_video_route_rejects_path_traversal(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, _ = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"do not serve me")
    for attempt in ("..%2Fsecret", "..", "%2e%2e%2fsecret"):
        response = test_client.get(f"/api/video/file/{attempt}")
        assert response.status_code in {404, 409}
        assert b"do not serve me" not in response.data


def test_video_route_rejects_unknown_video_id(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, _ = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    response = test_client.get("/api/video/file/L99_V999")
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "VIDEO_NOT_IN_ACTIVE_SCOPE"


def test_video_id_from_the_other_root_is_rejected_when_not_selected(
    offline_client, roots, tmp_path
) -> None:
    """`L21_V002` only exists in root B; root A must never serve it."""
    test_client, _ = offline_client
    root_a, _ = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    response = test_client.get("/api/video/file/L21_V002")
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "VIDEO_NOT_IN_ACTIVE_SCOPE"


def test_health_does_not_leak_absolute_paths_into_results(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, _ = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    body = test_client.post("/api/video/search", json={"query": "anything"}).get_json()
    for item in body["results"]:
        for value in (item["image"], item["video_url"]):
            if value:
                assert value.startswith("/api/")


# --------------------------------------------------------------- regression


def test_uninitialized_engine_returns_a_structured_error(client) -> None:
    test_client, _ = client
    response = test_client.post("/api/video/search", json={"query": "x"})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "RUNTIME_STATE_UNINITIALIZED"


def test_save_uses_the_active_cache_dir_not_a_global(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, root_b = roots
    activate(test_client, root_a, tmp_path / "cache_a")
    activate(test_client, root_b, tmp_path / "cache_b")
    body = test_client.post("/api/video/save", json={}).get_json()
    assert body["dir"] == str(tmp_path / "cache_b" / "entry")
    assert (tmp_path / "cache_b" / "entry").is_dir()


def test_jpeg_fallback_still_works_after_a_switch(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, root_b = roots
    activate(test_client, root_b, tmp_path / "cache_b")
    response = test_client.get(f"/api/video/frame/{VIDEO_ID}/kf_000002")
    assert response.status_code == 200
    assert response.headers["X-Frame-Source"] == "keyframe_jpeg"
    assert response.headers["X-Runtime-Generation"] == "2"


def test_missing_visual_source_reports_unavailable_not_500(offline_client, roots, tmp_path) -> None:
    test_client, _ = offline_client
    root_a, _ = roots
    # Remove the JPEGs; the placeholder MP4 is not decodable, so nothing is available.
    for path in (root_a / "keyframes" / VIDEO_ID).glob("*.jpg"):
        path.unlink()
    activate(test_client, root_a, tmp_path / "cache_a")
    response = test_client.get(f"/api/video/frame/{VIDEO_ID}/kf_000001")
    assert response.status_code == 422
    assert response.get_json()["error_code"] == "FRAME_UNAVAILABLE"
