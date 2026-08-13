"""A video without pixels is searchable, and honest about what it cannot do.

844 of the 873 videos in this collection have map, CLIP, objects and metadata but no
local MP4. Coarse retrieval needs none of those pixels, so they belong in the index. What
must never happen is the opposite failure: a system that indexes them and then pretends
it can preview, refine, or visually answer questions about them.
"""
from __future__ import annotations

import pytest

from aic2026.cache_manifest import cache_fingerprint
from aic2026.config import DatasetScopeConfig
from aic2026.dataset_scope import resolve_dataset_scope, select_video_ids
from aic2026.engine import AICCompetitionEngine
from aic2026.video_inventory import (
    existing_video_ids_with_retrieval_support,
    retrieval_ready_video_ids,
    support_coverage,
)
from tests.release_support import (
    TinyTextEncoder,
    make_config,
    make_data,
    make_retrieval_only_video,
)

VISUAL = "L01_V001"
NO_PIXELS = "L02_V009"


def mixed_root(tmp_path):
    """One video with pixels, one with supporting data only."""
    root = make_data(tmp_path / "data", video_ids=(VISUAL,))
    (root / "video").mkdir(parents=True, exist_ok=True)
    (root / "video" / f"{VISUAL}.mp4").write_bytes(b"not a real mp4")
    make_retrieval_only_video(root, NO_PIXELS)
    return root


def build(tmp_path, *, mode: str, cache_name: str = "cache"):
    root = mixed_root(tmp_path)
    config = make_config(
        root,
        tmp_path / cache_name,
        dataset={"scope": {"mode": mode, "include_patterns": ["*"]}},
        ranking={"min_frame_gap": 0, "final_top_k": 100},
    )
    engine, load = AICCompetitionEngine.from_data_root(
        app_config=config, text_encoder=TinyTextEncoder(), rebuild=True
    )
    return engine, config, load, root


# ------------------------------------------------------------------- 1. selection


def test_retrieval_ready_selection_does_not_require_an_mp4(tmp_path) -> None:
    root = mixed_root(tmp_path)
    ready = set(retrieval_ready_video_ids(root))
    visual = set(existing_video_ids_with_retrieval_support(root))
    assert ready == {VISUAL, NO_PIXELS}
    assert visual == {VISUAL}
    assert NO_PIXELS in ready and NO_PIXELS not in visual


def test_the_no_pixel_video_really_has_no_pixels(tmp_path) -> None:
    root = mixed_root(tmp_path)
    coverage = {item.video_id: item for item in support_coverage(root)}
    entry = coverage[NO_PIXELS]
    assert entry.map and entry.clip and entry.retrieval_supported
    assert entry.video is False and entry.keyframe_jpeg is False


def test_scope_resolution_selects_it(tmp_path) -> None:
    root = mixed_root(tmp_path)
    resolved = resolve_dataset_scope(
        DatasetScopeConfig(include_patterns=("*",), mode="retrieval_ready"), root
    )
    assert NO_PIXELS in select_video_ids([VISUAL, NO_PIXELS], resolved)


# --------------------------------------------------------------- 2. it is searchable


def test_a_video_without_an_mp4_can_be_retrieved(tmp_path) -> None:
    engine, _, _, _ = build(tmp_path, mode="retrieval_ready")
    videos = {raw.video_id for raw in engine.entry.raws.values()}
    assert NO_PIXELS in videos
    candidates = engine.search_candidates("a", top_k=50)
    assert NO_PIXELS in {candidate.video_id for candidate in candidates}


def test_it_can_appear_in_kis_results_with_an_official_frame(tmp_path) -> None:
    engine, _, _, _ = build(tmp_path, mode="retrieval_ready")
    predictions = engine.search_kis("a", top_k=100)
    rows = [p for p in predictions if p.video_id == NO_PIXELS]
    assert rows, "a retrieval-ready video must be able to reach the result list"
    for prediction in rows:
        # The submitted frame is the official mapped frame_idx, unchanged.
        assert int(prediction.frame_id) in {100, 130}


def test_it_can_appear_in_trake_sequences(tmp_path) -> None:
    engine, _, _, _ = build(tmp_path, mode="retrieval_ready")
    outcome = engine.search_trake_detailed(["a", "b"], max_results=50)
    assert outcome.structural_summary()["malformed_prediction_count"] == 0
    videos = {p.video_id for p in outcome.predictions}
    # Either video may win the alignment; what matters is that nothing crashed and no
    # sequence mixed videos.
    assert videos <= {VISUAL, NO_PIXELS}
    assert outcome.structural_summary()["cross_video_step_count"] == 0


def test_the_visual_scope_excludes_it(tmp_path) -> None:
    engine, _, _, _ = build(tmp_path, mode="existing_videos", cache_name="visual_cache")
    assert NO_PIXELS not in {raw.video_id for raw in engine.entry.raws.values()}


# ------------------------------------------------- 3. refinement degrades cleanly


def test_refinement_skips_a_pixel_less_candidate_without_crashing(tmp_path) -> None:
    from tests.refinement_support import FakeFrameScorer

    root = mixed_root(tmp_path)
    config = make_config(
        root,
        tmp_path / "cache",
        dataset={"scope": {"mode": "retrieval_ready", "include_patterns": ["*"]}},
        ranking={"min_frame_gap": 0, "final_top_k": 100},
        refinement={"enabled": True, "mode": "always", "candidate_budget": 4},
    )
    engine, _ = AICCompetitionEngine.from_data_root(
        app_config=config,
        text_encoder=TinyTextEncoder(),
        frame_scorer=FakeFrameScorer(15),
        rebuild=True,
    )
    outcome = engine.search_kis_detailed("a", top_k=50)
    assert outcome.predictions, "refinement must not empty the result list"
    for prediction in outcome.predictions:
        if prediction.video_id != NO_PIXELS:
            continue
        refinement = prediction.refinement or {}
        # Not refined, and the coarse frame survives untouched.
        assert refinement.get("applied") in (False, None)
        assert int(prediction.frame_id) in {100, 130}


def test_frame_provider_reports_unavailable_rather_than_raising(tmp_path) -> None:
    engine, _, _, _ = build(tmp_path, mode="retrieval_ready")
    raw = next(r for r in engine.entry.raws.values() if r.video_id == NO_PIXELS)
    result = engine.frame_provider.get_frame(raw)
    assert result.available is False
    assert result.image_bytes is None


# ------------------------------------------------------- 4. Q&A fabricates nothing


def test_qa_on_a_pixel_less_video_produces_no_exportable_answer(tmp_path) -> None:
    from aic2026.submission_validation import submission_rows_for, validate_submission

    engine, _, _, _ = build(tmp_path, mode="retrieval_ready")
    predictions, info = engine.answer_qa("a", "what colour?", top_k=20)
    for prediction in predictions:
        if prediction.video_id != NO_PIXELS or not prediction.qa:
            continue
        assert prediction.qa.get("backend_visual") is not True
    # Whatever the mock produced, none of it is submittable.
    if predictions:
        assert not validate_submission("qa", submission_rows_for("qa", predictions)).valid
    assert info["diagnostics"]["cost"]["qa"]["vlm_calls"] == 0


def test_qa_evidence_for_a_pixel_less_video_is_marked_unavailable(tmp_path) -> None:
    engine, _, _, _ = build(tmp_path, mode="retrieval_ready")
    _, info = engine.answer_qa("a", "what colour?", top_k=20)
    for hypothesis in info["diagnostics"].get("hypotheses", []) or []:
        if hypothesis.get("video_id") != NO_PIXELS:
            continue
        for frame in hypothesis.get("evidence", []) or []:
            assert not frame.get("image_available", False)


# ------------------------------------------------------------ 5. cache identity


def test_the_two_scopes_produce_different_cache_identities(tmp_path) -> None:
    """A different video selection must never share a cache fingerprint."""
    root = mixed_root(tmp_path)
    full = make_config(
        root, tmp_path / "full", dataset={"scope": {"mode": "retrieval_ready", "include_patterns": ["*"]}}
    )
    visual = make_config(
        root, tmp_path / "visual", dataset={"scope": {"mode": "existing_videos", "include_patterns": ["*"]}}
    )
    assert cache_fingerprint(full) != cache_fingerprint(visual)


def test_the_full_scope_indexes_more_frames(tmp_path) -> None:
    full, _, _, _ = build(tmp_path / "a", mode="retrieval_ready")
    visual, _, _, _ = build(tmp_path / "b", mode="existing_videos")
    assert int(full.entry.num_indexed) > int(visual.entry.num_indexed)


def test_adding_an_mp4_does_not_change_the_retrieval_selection(tmp_path) -> None:
    """Retrieval identity depends on map + CLIP. Downloading a video changes what you
    can SEE, not what is searchable, so the selected-IDs hash must not move."""
    from aic2026.dataset_scope import hash_selected_video_ids

    root = mixed_root(tmp_path)
    before = retrieval_ready_video_ids(root)
    (root / "video" / f"{NO_PIXELS}.mp4").write_bytes(b"newly downloaded")
    after = retrieval_ready_video_ids(root)
    assert before == after
    assert hash_selected_video_ids(before) == hash_selected_video_ids(after)
    # ...while the VISUAL scope does change, which is the whole point of keeping them apart.
    assert NO_PIXELS in set(existing_video_ids_with_retrieval_support(root))


def test_removing_an_mp4_does_not_change_the_retrieval_selection(tmp_path) -> None:
    from aic2026.dataset_scope import hash_selected_video_ids

    root = mixed_root(tmp_path)
    before = retrieval_ready_video_ids(root)
    (root / "video" / f"{VISUAL}.mp4").unlink()
    after = retrieval_ready_video_ids(root)
    assert hash_selected_video_ids(before) == hash_selected_video_ids(after)
    assert existing_video_ids_with_retrieval_support(root) == ()


def test_adding_map_and_clip_does_change_the_retrieval_identity(tmp_path) -> None:
    """Supporting data is what retrieval identity is made of."""
    from aic2026.dataset_scope import hash_selected_video_ids

    root = mixed_root(tmp_path)
    before = hash_selected_video_ids(retrieval_ready_video_ids(root))
    make_retrieval_only_video(root, "L03_V001")
    after = hash_selected_video_ids(retrieval_ready_video_ids(root))
    assert before != after


def test_removing_a_clip_feature_removes_the_video_from_retrieval(tmp_path) -> None:
    root = mixed_root(tmp_path)
    (root / "clip-features-32" / f"{NO_PIXELS}.npy").unlink()
    assert NO_PIXELS not in set(retrieval_ready_video_ids(root))


def test_removing_a_map_removes_the_video_from_retrieval(tmp_path) -> None:
    root = mixed_root(tmp_path)
    (root / "map-keyframes" / f"{NO_PIXELS}.csv").unlink()
    assert NO_PIXELS not in set(retrieval_ready_video_ids(root))


def test_the_four_capabilities_stay_independent(tmp_path) -> None:
    """`retrieval_supported`, `video`, `keyframe_jpeg` and `media_info` are separate
    facts. Collapsing them into one boolean is exactly the bug this scope work fixed."""
    root = mixed_root(tmp_path)
    coverage = {item.video_id: item for item in support_coverage(root)}
    visual, blind = coverage[VISUAL], coverage[NO_PIXELS]

    assert visual.retrieval_supported and blind.retrieval_supported
    assert visual.video is True and blind.video is False
    assert visual.keyframe_jpeg is True and blind.keyframe_jpeg is False
    assert visual.media_info and blind.media_info
    # Retrieval capability is not derived from any visual flag.
    assert blind.retrieval_supported is (blind.map and blind.clip)


def test_the_manifest_records_the_scope_and_its_resolved_ids(tmp_path) -> None:
    _, _, load, _ = build(tmp_path, mode="retrieval_ready")
    manifest = load.cache_manifest
    assert manifest is not None
    assert manifest.dataset_scope["mode"] == "retrieval_ready"
    assert set(manifest.selected_video_ids) == {VISUAL, NO_PIXELS}
    assert manifest.selected_video_ids_hash
    assert manifest.cache_fingerprint
