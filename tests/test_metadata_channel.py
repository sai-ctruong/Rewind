"""Metadata as an INDEPENDENT, video-scoped candidate generator.

Media title, description and tags describe a VIDEO, never a specific frame. So a metadata
hit may introduce candidates the dense pool never saw, but it must carry `scope="video"`
provenance and expand into a bounded, spread set of frames instead of dumping the whole
video into the pool.
"""
from __future__ import annotations

import pytest

from aic2026.query_normalization import normalize_query
from aic2026.retrieval_channels import (
    CHANNEL_METADATA,
    SCOPE_VIDEO,
    MetadataChannel,
)
from tests.test_retrieval_channels import make_entry, raw


def video(video_id: str, frames: int, *, title="", description="", tags=(), start=0.0):
    return {
        f"{video_id}/{index}": raw(
            f"{video_id}/{index}",
            video_id,
            start + float(index),
            frame_idx=index,
            title=title,
            description=description,
            tags=tags,
        )
        for index in range(frames)
    }


# ------------------------------------------------------------------ retrieval


def test_a_metadata_only_video_can_enter_the_pool() -> None:
    entry = make_entry(
        {
            **video("NEWS", 4, title="60 Giây - bản tin giao thông"),
            **video("OTHER", 4, title="a cooking programme"),
        }
    )
    result = MetadataChannel(entry).search(normalize_query("giao thông"), top_k=20)
    assert result.candidates
    assert {item.video_id for item in result.candidates} == {"NEWS"}


def test_metadata_candidates_are_video_scoped() -> None:
    entry = make_entry(video("NEWS", 4, title="bản tin giao thông"))
    result = MetadataChannel(entry).search(normalize_query("giao thông"), top_k=20)
    assert all(item.scope == SCOPE_VIDEO for item in result.candidates)
    assert result.status.scope == SCOPE_VIDEO
    # Provenance names the matched terms, not a frame-level claim.
    assert all(item.evidence for item in result.candidates)


def test_description_and_tags_are_searchable_too() -> None:
    entry = make_entry(
        {
            **video("A", 2, description="phóng sự về xe máy ở thành phố"),
            **video("B", 2, tags=("motorcycle", "traffic")),
            **video("C", 2, title="unrelated"),
        }
    )
    channel = MetadataChannel(entry)
    from_description = channel.search(normalize_query("phóng sự"), top_k=20)
    assert {item.video_id for item in from_description.candidates} == {"A"}
    from_tags = channel.search(normalize_query("traffic"), top_k=20)
    assert {item.video_id for item in from_tags.candidates} == {"B"}


def test_frames_per_video_is_bounded_and_spread() -> None:
    entry = make_entry(video("NEWS", 50, title="giao thông"))
    result = MetadataChannel(entry, frames_per_video=5).search(
        normalize_query("giao thông"), top_k=100
    )
    assert len(result.candidates) == 5
    timestamps = sorted(item.timestamp for item in result.candidates)
    # Spread across the video rather than clustered at its start.
    assert timestamps[0] < timestamps[-1]
    assert timestamps[-1] - timestamps[0] > 10.0


def test_a_short_video_returns_all_its_frames() -> None:
    entry = make_entry(video("NEWS", 3, title="giao thông"))
    result = MetadataChannel(entry, frames_per_video=8).search(
        normalize_query("giao thông"), top_k=100
    )
    assert len(result.candidates) == 3


def test_top_k_bounds_the_channel_overall() -> None:
    entry = make_entry(
        {
            **video("A", 20, title="giao thông"),
            **video("B", 20, title="giao thông", start=100.0),
        }
    )
    result = MetadataChannel(entry, frames_per_video=8).search(
        normalize_query("giao thông"), top_k=5
    )
    assert len(result.candidates) <= 5


def test_videos_are_ordered_by_term_coverage() -> None:
    entry = make_entry(
        {
            **video("BOTH", 2, title="xe máy giao thông"),
            **video("ONE", 2, title="giao thông"),
        }
    )
    result = MetadataChannel(entry, frames_per_video=2).search(
        normalize_query("xe máy giao thông"), top_k=20
    )
    assert result.candidates[0].video_id == "BOTH"


def test_absent_metadata_reports_unavailable_without_failing() -> None:
    entry = make_entry(video("PLAIN", 3))
    channel = MetadataChannel(entry)
    status = channel.status()
    assert status.available is False
    assert status.record_count == 0
    assert channel.search(normalize_query("anything"), top_k=10).candidates == ()


def test_a_negated_term_does_not_retrieve_a_video() -> None:
    entry = make_entry(
        {
            **video("CARS", 2, title="phóng sự về xe hơi"),
            **video("PEOPLE", 2, title="phóng sự về người đi bộ"),
        }
    )
    channel = MetadataChannel(entry, frames_per_video=2)
    retrieved = {
        item.video_id
        for item in channel.search(
            normalize_query("không có xe hơi nhưng có người"), top_k=20
        ).candidates
    }
    assert "PEOPLE" in retrieved
    assert "CARS" not in retrieved


def test_metadata_is_not_treated_as_a_frame_caption() -> None:
    # The loader may place media text in `caption_by_id`; the metadata channel reads it,
    # but the CAPTION channel must not claim it as frame-level evidence.
    from aic2026.retrieval_channels import CHANNEL_CAPTION, TextFieldChannel

    entry = make_entry(
        video("NEWS", 2),
        captions={"NEWS/0": "media_title: bản tin giao thông"},
    )
    metadata = MetadataChannel(entry, frames_per_video=2)
    assert metadata.status().available is True
    caption = TextFieldChannel(entry, CHANNEL_CAPTION, lambda item: getattr(item, "frame_caption", ""))
    # No record carries a real frame caption, so the caption channel stays unavailable.
    assert caption.status().available is False


def test_the_index_is_built_once() -> None:
    entry = make_entry(video("NEWS", 30, title="giao thông"))
    channel = MetadataChannel(entry)
    documents = channel._documents
    channel.search(normalize_query("giao thông"), top_k=10)
    channel.search(normalize_query("giao thông"), top_k=10)
    assert channel._documents is documents


def test_results_are_deterministic() -> None:
    entry = make_entry(
        {
            **video("A", 6, title="giao thông"),
            **video("B", 6, title="giao thông", start=50.0),
        }
    )
    channel = MetadataChannel(entry, frames_per_video=3)
    first = channel.search(normalize_query("giao thông"), top_k=20)
    second = channel.search(normalize_query("giao thông"), top_k=20)
    assert [item.keyframe_id for item in first.candidates] == [
        item.keyframe_id for item in second.candidates
    ]


def test_a_vietnamese_query_matches_folded_metadata() -> None:
    # The stored title is accented; the query is not. Folding makes them meet.
    entry = make_entry(video("NEWS", 2, title="bản tin giao thông"))
    channel = MetadataChannel(entry, frames_per_video=2)
    assert channel.search(normalize_query("giao thong"), top_k=10).candidates
    assert channel.search(normalize_query("giao thông"), top_k=10).candidates
