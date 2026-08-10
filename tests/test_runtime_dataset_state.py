"""Phase 4: the runtime dataset state model and its manager.

Offline and deterministic: small generated datasets under tmp_path, no network, no
model download, no GPU.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from aic2026.cache_manifest import canonical_data_root
from aic2026.config import DatasetScopeConfig, app_config_from_dict
from aic2026.engine import AICCompetitionEngine
from aic2026.runtime_state import (
    STALE_RESULT_GENERATION,
    VIDEO_NOT_IN_ACTIVE_SCOPE,
    RuntimeDatasetState,
    RuntimeStateError,
    RuntimeStateManager,
    build_runtime_state,
    check_generation,
    derived_cache_dir,
    resolve_cache_dir,
    safe_video_path,
)

VIDEO_ID = "L21_V001"


class TinyTextEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def make_dataset(root, *, video_id: str = VIDEO_ID, with_video: bool = False):
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
        Image.new("RGB", (8, 8), (ordinal * 60, 0, 0)).save(
            root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
        )
    if with_video:
        (root / "video").mkdir(parents=True, exist_ok=True)
        (root / "video" / f"{video_id}.mp4").write_bytes(b"placeholder")
    return root


def make_config(root, cache_dir=None, frame_cache_dir=None):
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir or root.parent / "cache"),
                    "frame_cache_dir": str(frame_cache_dir or root.parent / "frames"),
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
            }
        }
    )


def initial_state(tmp_path, **kwargs) -> RuntimeDatasetState:
    root = make_dataset(tmp_path / "data", **kwargs)
    return build_runtime_state(
        app_config=make_config(root),
        config_path="<test>",
        generation=1,
    )


# --------------------------------------------------------------------- identity


def test_state_carries_config_data_and_cache_identity(tmp_path) -> None:
    state = initial_state(tmp_path)
    identity = state.identity()
    assert identity["generation"] == 1
    assert identity["data_root"] == canonical_data_root(tmp_path / "data")
    assert identity["cache_dir"] == str(tmp_path / "cache")
    assert identity["config_hash"] == state.config_hash
    assert state.resolved_video_ids == (VIDEO_ID,)
    assert state.selected_video_ids_hash
    assert state.engine_loaded is False


def test_state_is_frozen(tmp_path) -> None:
    state = initial_state(tmp_path)
    with pytest.raises(Exception):
        state.generation = 99  # type: ignore[misc]


def test_frame_provider_follows_the_state_root(tmp_path) -> None:
    state = initial_state(tmp_path)
    assert canonical_data_root(state.frame_provider.data_root) == state.data_root
    assert str(state.frame_provider.cache.cache_dir) == str(tmp_path / "frames")


def test_state_reports_the_video_inventory(tmp_path) -> None:
    state = initial_state(tmp_path, with_video=True)
    assert state.video_inventory_summary["available_count"] == 1
    assert state.video_inventory_summary["collections"] == {"L21": 1}


def test_state_without_engine_still_reports_cache_status(tmp_path) -> None:
    state = initial_state(tmp_path)
    assert state.cache_status is not None
    assert state.cache_status["valid"] is False
    assert state.dataset_status is None


# ---------------------------------------------------------------------- manager


def test_manager_returns_a_stable_snapshot(tmp_path) -> None:
    first = initial_state(tmp_path)
    manager = RuntimeStateManager(first)
    snapshot = manager.get_state()
    manager.replace_state(replace(first, generation=2))
    # The snapshot taken before the swap keeps describing generation 1.
    assert snapshot.generation == 1
    assert manager.get_state().generation == 2
    assert snapshot is not manager.get_state()


def test_successful_replacement_increments_generation(tmp_path) -> None:
    manager = RuntimeStateManager(initial_state(tmp_path))
    assert manager.get_state().generation == 1
    root = tmp_path / "data"
    manager.build_and_replace(app_config=make_config(root), data_root=root)
    assert manager.get_state().generation == 2
    manager.build_and_replace(app_config=make_config(root), data_root=root)
    assert manager.get_state().generation == 3


def test_failed_build_keeps_the_old_state_active(tmp_path) -> None:
    manager = RuntimeStateManager(initial_state(tmp_path))
    before = manager.get_state()
    with pytest.raises(Exception):
        manager.build_and_replace(
            app_config=make_config(tmp_path / "data"),
            data_root=tmp_path / "does-not-exist",
            engine=object(),  # forces verify_engine_identity to fail
        )
    after = manager.get_state()
    assert after is before
    assert after.generation == 1


def test_next_generation_does_not_publish(tmp_path) -> None:
    manager = RuntimeStateManager(initial_state(tmp_path))
    assert manager.next_generation() == 2
    assert manager.get_state().generation == 1


def test_manager_status_matches_the_active_state(tmp_path) -> None:
    manager = RuntimeStateManager(initial_state(tmp_path))
    assert manager.status() == manager.get_state().runtime_summary()


# ------------------------------------------------------------- engine identity


def test_engine_identity_mismatch_is_rejected(tmp_path) -> None:
    """The exact Phase 0 bug, caught at state construction."""
    root_a = make_dataset(tmp_path / "a")
    root_b = make_dataset(tmp_path / "b")
    engine, _ = AICCompetitionEngine.from_data_root(
        app_config=make_config(root_a, cache_dir=tmp_path / "cache_a"),
        text_encoder=TinyTextEncoder(),
        rebuild=True,
    )
    with pytest.raises(RuntimeStateError, match="engine was built for data root"):
        build_runtime_state(
            app_config=make_config(root_b, cache_dir=tmp_path / "cache_b"),
            config_path="<test>",
            generation=2,
            data_root=root_b,
            engine=engine,
        )


def test_engine_identity_match_is_accepted(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    config = make_config(root, cache_dir=tmp_path / "cache")
    engine, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    state = build_runtime_state(
        app_config=config,
        config_path="<test>",
        generation=2,
        data_root=root,
        engine=engine,
        load=load,
    )
    assert state.engine_loaded
    assert state.dataset_status["retrieval_valid_count"] == 1
    assert engine.dataset_identity()["data_root"] == str(root)


# ------------------------------------------------------------------ cache dirs


def test_configured_cache_is_kept_for_the_configured_root(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    config = make_config(root, cache_dir=tmp_path / "configured")
    assert resolve_cache_dir(config, root) == str(tmp_path / "configured")


def test_a_different_root_derives_a_distinct_cache_dir(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    other = make_dataset(tmp_path / "other")
    config = make_config(root, cache_dir=tmp_path / "configured")
    derived = resolve_cache_dir(config, other)
    assert derived != str(tmp_path / "configured")
    assert derived == derived_cache_dir(config, other, config.dataset.scope)


def test_explicit_cache_dir_always_wins(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    config = make_config(root, cache_dir=tmp_path / "configured")
    assert resolve_cache_dir(config, root, explicit=tmp_path / "explicit") == str(
        tmp_path / "explicit"
    )


def test_derived_cache_dir_depends_on_root_and_scope(tmp_path) -> None:
    root = make_dataset(tmp_path / "data")
    other = make_dataset(tmp_path / "other")
    config = make_config(root)
    scope_a = DatasetScopeConfig(include_patterns=("L21_*",))
    scope_b = DatasetScopeConfig(include_patterns=("L22_*",))
    assert derived_cache_dir(config, other, scope_a) != derived_cache_dir(config, root, scope_a)
    assert derived_cache_dir(config, other, scope_a) != derived_cache_dir(config, other, scope_b)
    assert derived_cache_dir(config, other, scope_a) == derived_cache_dir(config, other, scope_a)


# ------------------------------------------------------------------ generation


def test_generation_check_accepts_the_active_generation(tmp_path) -> None:
    state = initial_state(tmp_path)
    check_generation(state, None)
    check_generation(state, "")
    check_generation(state, 1)
    check_generation(state, "1")


def test_generation_check_rejects_a_superseded_generation(tmp_path) -> None:
    state = replace(initial_state(tmp_path), generation=5)
    with pytest.raises(RuntimeStateError) as caught:
        check_generation(state, 4)
    assert caught.value.error_code == STALE_RESULT_GENERATION


def test_generation_check_rejects_garbage(tmp_path) -> None:
    state = initial_state(tmp_path)
    with pytest.raises(RuntimeStateError) as caught:
        check_generation(state, "not-a-number")
    assert caught.value.error_code == STALE_RESULT_GENERATION


# -------------------------------------------------------------------- security


@pytest.mark.parametrize(
    "video_id",
    [
        "..",
        "../secret",
        "..\\secret",
        "sub/dir",
        "sub\\dir",
        "L21_V001/../../etc",
        "",
    ],
)
def test_path_traversal_is_rejected(tmp_path, video_id) -> None:
    state = initial_state(tmp_path, with_video=True)
    with pytest.raises(RuntimeStateError) as caught:
        safe_video_path(state, video_id)
    assert caught.value.error_code == VIDEO_NOT_IN_ACTIVE_SCOPE


def test_unknown_video_id_is_rejected(tmp_path) -> None:
    state = initial_state(tmp_path, with_video=True)
    with pytest.raises(RuntimeStateError, match="not part of the active dataset"):
        safe_video_path(state, "L99_V999")


def test_known_video_resolves_inside_the_active_root(tmp_path) -> None:
    state = initial_state(tmp_path, with_video=True)
    path = safe_video_path(state, VIDEO_ID)
    assert path.name == f"{VIDEO_ID}.mp4"
    assert path.is_file()
    assert (tmp_path / "data" / "video").resolve() in path.parents
