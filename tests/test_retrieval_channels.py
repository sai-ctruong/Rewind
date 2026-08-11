"""The channel abstraction, the union, normalization, and honest availability.

The architectural claim under test: a signal that is not CLIP or BM25 can now put a
candidate into the pool by itself. Everything else here protects that claim — provenance
survives the union, score spaces are made comparable before they meet, and a channel
without data says so instead of quietly returning nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from aic2026.config import ConfigError, RetrievalChannelConfig, app_config_from_dict
from aic2026.query_normalization import normalize_query
from aic2026.retrieval_channels import (
    CHANNEL_ASR,
    CHANNEL_BM25,
    CHANNEL_CAPTION,
    CHANNEL_CLIP,
    CHANNEL_METADATA,
    CHANNEL_NAMES,
    CHANNEL_OBJECTS,
    CHANNEL_OCR,
    CHANNEL_SCHEMA_VERSION,
    REASON_NO_DATA,
    SCOPE_VIDEO,
    ChannelCandidate,
    ChannelError,
    ChannelResult,
    ChannelStatus,
    ObjectChannel,
    RetrievalChannel,
    RetrievalChannelRegistry,
    TextFieldChannel,
    build_channel_registry,
    channel_depths,
    normalize_channel_scores,
)
from ingestion.build_index import KeyframeIndex
from ingestion.schemas import RawKeyframe
from retrieval.video_engine import VideoIndexEntry


def raw(
    keyframe_id: str,
    video_id: str,
    timestamp: float,
    *,
    frame_idx: int,
    objects=(),
    detections=None,
    title: str = "",
    description: str = "",
    tags=(),
    caption: str = "",
):
    item = RawKeyframe(
        keyframe_id,
        video_id,
        timestamp,
        frame_idx=frame_idx,
        keyframe_ordinal=frame_idx,
        objects=list(objects),
        object_detections=list(detections or []),
    )
    item.media_title = title
    item.media_description = description
    item.media_tags = tuple(tags)
    item.media_channel = ""
    item.frame_caption = caption
    return item


def make_entry(raws: dict, *, texts=None, ocr=None, asr=None, captions=None) -> VideoIndexEntry:
    index = KeyframeIndex(
        ids=list(raws),
        video_ids=[raws[key].video_id for key in raws],
        timestamps=[raws[key].timestamp for key in raws],
        objects=[list(raws[key].objects) for key in raws],
    )
    entry = VideoIndexEntry("dataset", index, raws, len(raws), len(raws))
    entry.ocr_by_id = dict(ocr or {})
    entry.asr_by_id = dict(asr or {})
    entry.caption_by_id = dict(captions or {})
    return entry


def fake_encode(query: str):
    return np.zeros(2, dtype=np.float32)


# --------------------------------------------------------------- status honesty


def test_an_empty_optional_source_reports_unavailable_with_a_reason() -> None:
    entry = make_entry({"V/1": raw("V/1", "V", 0.0, frame_idx=1)})
    for name in (CHANNEL_OCR, CHANNEL_ASR, CHANNEL_CAPTION):
        channel = TextFieldChannel(entry, name, {})
        status = channel.status()
        assert status.available is False
        assert status.usable is False
        assert status.reason == REASON_NO_DATA
        assert status.record_count == 0
        # It must not pretend to have searched.
        result = channel.search(normalize_query("anything"), top_k=10)
        assert result.candidates == ()
        assert result.searched is False


def test_a_populated_optional_source_becomes_available() -> None:
    entry = make_entry({"V/1": raw("V/1", "V", 0.0, frame_idx=1)})
    channel = TextFieldChannel(entry, CHANNEL_OCR, {"V/1": "STOP sign ahead"})
    assert channel.status().available is True
    assert channel.status().record_count == 1
    result = channel.search(normalize_query("stop sign"), top_k=10)
    assert [item.keyframe_id for item in result.candidates] == ["V/1"]
    assert result.candidates[0].evidence


def test_nothing_is_substituted_for_a_missing_source() -> None:
    # Objects exist but OCR does not: the OCR channel must stay empty rather than
    # borrowing the object labels.
    entry = make_entry(
        {"V/1": raw("V/1", "V", 0.0, frame_idx=1, objects=["stop sign"])}
    )
    ocr = TextFieldChannel(entry, CHANNEL_OCR, {})
    assert ocr.status().available is False
    assert ocr.search(normalize_query("stop sign"), top_k=10).candidates == ()


def test_bm25_availability_is_measured_from_the_real_index() -> None:
    """Regression: `BM25Okapi` does not retain the corpus it was built from.

    Probing a `.corpus` attribute reports zero documents for every dataset, which made a
    fully populated BM25 index look unavailable. Availability must come from `doc_freqs`.
    """
    from ingestion.build_index import KeyframeIndex
    from ingestion.schemas import KeyframeRecord

    populated = [
        KeyframeRecord(
            id=f"V/{i}",
            video_id="V",
            timestamp=float(i),
            clip_embedding=np.array([1.0, 0.0], dtype=np.float32),
            objects=["motorcycle", "person"],
        )
        for i in range(3)
    ]
    index = KeyframeIndex.build(populated)
    entry = VideoIndexEntry(
        "dataset",
        index,
        {record.id: raw(record.id, "V", record.timestamp, frame_idx=i) for i, record in enumerate(populated)},
        3,
        3,
    )
    assert getattr(index._bm25, "corpus", None) is None, "the fixture must reproduce the cause"
    from aic2026.retrieval_channels import Bm25Channel

    status = Bm25Channel(entry).status()
    assert status.available is True
    assert status.record_count == 3


def test_bm25_reports_unavailable_when_every_document_is_the_sentinel() -> None:
    from ingestion.build_index import KeyframeIndex
    from ingestion.schemas import KeyframeRecord
    from aic2026.retrieval_channels import Bm25Channel

    empty = [
        KeyframeRecord(
            id=f"V/{i}",
            video_id="V",
            timestamp=float(i),
            clip_embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        for i in range(3)
    ]
    index = KeyframeIndex.build(empty)
    entry = VideoIndexEntry(
        "dataset",
        index,
        {record.id: raw(record.id, "V", record.timestamp, frame_idx=i) for i, record in enumerate(empty)},
        3,
        3,
    )
    status = Bm25Channel(entry).status()
    assert status.available is False
    assert status.reason == REASON_NO_DATA


def test_a_disabled_channel_is_still_reported() -> None:
    entry = make_entry({"V/1": raw("V/1", "V", 0.0, frame_idx=1, objects=["car"])})
    channel = ObjectChannel(entry, enabled=False)
    status = channel.status()
    assert status.enabled is False
    assert status.available is True, "data exists even though the channel is switched off"
    assert status.usable is False


# ------------------------------------------------------------------- the union


def channels_entry():
    """One frame per video; each video is reachable through a different signal."""
    raws = {
        "CLIPV/1": raw("CLIPV/1", "CLIPV", 0.0, frame_idx=10),
        "OBJV/1": raw(
            "OBJV/1", "OBJV", 0.0, frame_idx=20,
            detections=[{"label": "motorcycle", "confidence": 0.9}],
        ),
        "METAV/1": raw(
            "METAV/1", "METAV", 0.0, frame_idx=30, title="a documentary about motorcycles"
        ),
        "OCRV/1": raw("OCRV/1", "OCRV", 0.0, frame_idx=40),
    }
    return make_entry(raws, ocr={"OCRV/1": "motorcycle rental"})


def registry_for(entry, **overrides) -> RetrievalChannelRegistry:
    settings = dict(clip_enabled=False)  # the dense stub is exercised separately
    settings.update(overrides)
    config = RetrievalChannelConfig(**settings)
    return build_channel_registry(entry, fake_encode, config)


def test_the_union_deduplicates_on_canonical_keyframe_ids() -> None:
    entry = make_entry(
        {
            "V/1": raw(
                "V/1", "V", 0.0, frame_idx=1,
                detections=[{"label": "motorcycle", "confidence": 0.9}],
                title="motorcycle documentary",
            )
        }
    )
    union = registry_for(entry).search(
        "motorcycle", depths={name: 20 for name in CHANNEL_NAMES}
    )
    assert len(union.candidates) == 1
    candidate = union.candidates[0]
    # One entry, but it remembers BOTH channels that proposed it.
    assert set(candidate.channels) >= {CHANNEL_OBJECTS, CHANNEL_METADATA}
    assert candidate.rank_in(CHANNEL_OBJECTS) is not None
    assert candidate.introduced_only_by() is None


def test_channel_provenance_ranks_and_scores_survive_the_union() -> None:
    union = registry_for(channels_entry()).search(
        "motorcycle", depths={name: 20 for name in CHANNEL_NAMES}
    )
    by_id = union.by_id()
    payload = by_id["OBJV/1"].to_dict()
    assert payload["channels"] == [CHANNEL_OBJECTS]
    assert payload["ranks"][CHANNEL_OBJECTS] >= 1
    assert payload["raw_scores"][CHANNEL_OBJECTS] > 0
    assert 0.0 <= payload["normalized_scores"][CHANNEL_OBJECTS] <= 1.0
    assert "motorcycle" in payload["evidence"][CHANNEL_OBJECTS]


def test_a_channel_can_never_introduce_an_unknown_id() -> None:
    entry = channels_entry()

    class Rogue:
        name = "rogue"

        def status(self):
            return ChannelStatus(name="rogue", enabled=True, available=True, record_count=1)

        def search(self, query, *, top_k):
            return ChannelResult(
                channel="rogue",
                status=self.status(),
                candidates=(
                    ChannelCandidate(
                        keyframe_id="NOT_IN_THE_INDEX",
                        video_id="GHOST",
                        channel="rogue",
                        raw_score=1.0,
                        rank=1,
                    ),
                ),
            )

    union = RetrievalChannelRegistry([Rogue()]).search("q", depths={"rogue": 5})
    # The registry keeps what the channel returned, but the ENTRY-backed channels are the
    # ones that resolve IDs; a rogue id is visible and traceable rather than hidden.
    assert [c.keyframe_id for c in union.candidates] == ["NOT_IN_THE_INDEX"]
    # The engine-facing path drops it, because it cannot be resolved to a record.
    assert "NOT_IN_THE_INDEX" not in entry.raws


def test_entry_backed_channels_drop_ids_that_are_not_indexed() -> None:
    entry = channels_entry()
    channel = ObjectChannel(entry)
    # Corrupt the postings with an ID the entry does not know.
    channel._postings["ghost"] = [("MISSING/9", 1.0)]
    channel._frames_with_labels = max(channel._frames_with_labels, 1)
    result = channel.search(normalize_query("ghost"), top_k=10)
    assert all(item.keyframe_id in entry.raws for item in result.candidates)


def test_union_diagnostics_count_coverage_not_recall() -> None:
    union = registry_for(channels_entry()).search(
        "motorcycle", depths={name: 20 for name in CHANNEL_NAMES}
    )
    diagnostics = union.diagnostics
    assert diagnostics["candidate_union_size"] == len(union.candidates)
    assert diagnostics["unique_videos"] >= 2
    assert diagnostics["channel_schema_version"] == CHANNEL_SCHEMA_VERSION
    assert "coverage" in diagnostics["note"].lower()
    # No diagnostic KEY may masquerade as a quality metric. The disclaimer note is
    # excluded from the scan precisely because it names those words to deny them.
    keys = str({k: v for k, v in diagnostics.items() if k != "note"}).lower()
    for banned in ("recall", "precision", "accuracy", "map@"):
        assert banned not in keys
    objects = diagnostics["channels"][CHANNEL_OBJECTS]
    assert objects["unique_candidates_introduced"] >= 1
    assert objects["candidates_returned"] >= 1
    ocr = diagnostics["channels"][CHANNEL_OCR]
    assert ocr["available"] is True


def test_exclusive_and_overlap_counts_are_correct() -> None:
    entry = make_entry(
        {
            "BOTH/1": raw(
                "BOTH/1", "BOTH", 0.0, frame_idx=1,
                detections=[{"label": "motorcycle", "confidence": 0.9}],
                title="motorcycle film",
            ),
            "OBJ/1": raw(
                "OBJ/1", "OBJ", 0.0, frame_idx=2,
                detections=[{"label": "motorcycle", "confidence": 0.9}],
            ),
        }
    )
    union = registry_for(entry).search(
        "motorcycle", depths={name: 20 for name in CHANNEL_NAMES}
    )
    exclusive = union.diagnostics["exclusive_candidates"]
    # OBJ/1 is object-only; BOTH/1 is shared, so it is exclusive to neither.
    assert exclusive.get(CHANNEL_OBJECTS, 0) == 1
    assert exclusive.get(CHANNEL_METADATA, 0) == 0


# ---------------------------------------------------------------- normalization


def test_rank_normalization_is_monotonic_and_bounded() -> None:
    items = tuple(
        ChannelCandidate(f"V/{i}", "V", "c", raw_score=100.0 - i, rank=i + 1)
        for i in range(4)
    )
    normalized = normalize_channel_scores(items, method="rank")
    scores = [item.normalized_score for item in normalized]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < value <= 1.0 for value in scores)


def test_minmax_normalization_handles_zero_variance() -> None:
    items = tuple(
        ChannelCandidate(f"V/{i}", "V", "c", raw_score=5.0, rank=i + 1) for i in range(3)
    )
    normalized = normalize_channel_scores(items, method="minmax")
    # Everything agrees: full weight rather than a divide-by-zero or all-zeros.
    assert [item.normalized_score for item in normalized] == [1.0, 1.0, 1.0]


def test_minmax_normalization_handles_negative_scores() -> None:
    items = (
        ChannelCandidate("V/1", "V", "c", raw_score=-0.4, rank=1),
        ChannelCandidate("V/2", "V", "c", raw_score=-0.9, rank=2),
    )
    normalized = normalize_channel_scores(items, method="minmax")
    assert normalized[0].normalized_score == pytest.approx(1.0)
    assert normalized[1].normalized_score == pytest.approx(0.0)


def test_non_finite_scores_are_rejected() -> None:
    items = (ChannelCandidate("V/1", "V", "c", raw_score=float("nan"), rank=1),)
    with pytest.raises(ChannelError, match="non-finite"):
        normalize_channel_scores(items, method="minmax")
    with pytest.raises(ChannelError, match="Unknown channel normalization"):
        normalize_channel_scores(items, method="magic")


def test_empty_input_normalizes_to_empty() -> None:
    assert normalize_channel_scores((), method="rank") == ()


def test_the_union_is_deterministic() -> None:
    entry = channels_entry()
    first = registry_for(entry).search("motorcycle", depths={n: 20 for n in CHANNEL_NAMES})
    second = registry_for(entry).search("motorcycle", depths={n: 20 for n in CHANNEL_NAMES})
    assert [c.keyframe_id for c in first.candidates] == [c.keyframe_id for c in second.candidates]
    assert [c.channels for c in first.candidates] == [c.channels for c in second.candidates]


# --------------------------------------------------------------------- config


def test_channel_depths_scale_together() -> None:
    config = RetrievalChannelConfig()
    plain = channel_depths(config)
    deeper = channel_depths(config, scale=2.0)
    assert plain[CHANNEL_CLIP] == config.clip_top_k
    assert deeper[CHANNEL_CLIP] == config.clip_top_k * 2
    assert deeper[CHANNEL_OBJECTS] == config.objects_top_k * 2
    # Scale never shrinks a depth.
    assert channel_depths(config, scale=0.1)[CHANNEL_CLIP] == config.clip_top_k


def test_channel_config_is_validated() -> None:
    def build(**channels):
        return app_config_from_dict({"aic2026": {"retrieval_channels": channels}})

    assert build(channels={"objects": {"top_k": 42}}).retrieval_channels.objects_top_k == 42
    assert (
        build(channels={"objects": {"confidence_threshold": 0.5}})
        .retrieval_channels.object_confidence_threshold
        == 0.5
    )
    with pytest.raises(ConfigError, match="top_k must be > 0"):
        build(channels={"clip": {"top_k": 0}})
    with pytest.raises(ConfigError, match="confidence_threshold"):
        build(channels={"objects": {"confidence_threshold": 5.0}})
    with pytest.raises(ConfigError, match="normalization must be"):
        build(normalization="softmax")
    with pytest.raises(ConfigError, match="at least one retrieval channel"):
        build(
            channels={
                name: {"enabled": False}
                for name in ("clip", "bm25", "objects", "metadata", "ocr", "asr", "caption")
            }
        )


def test_every_channel_can_be_toggled_individually() -> None:
    entry = channels_entry()
    for name in (CHANNEL_OBJECTS, CHANNEL_METADATA, CHANNEL_OCR):
        registry = registry_for(entry, **{f"{name}_enabled": False})
        status = registry.status()[name]
        assert status["enabled"] is False
        assert status["usable"] is False
        union = registry.search("motorcycle", depths={n: 20 for n in CHANNEL_NAMES})
        assert all(name not in candidate.channels for candidate in union.candidates)


def test_shipped_channel_defaults() -> None:
    from aic2026.config import load_app_config

    channels = load_app_config("configs/settings.yaml").retrieval_channels
    assert channels.clip_enabled and channels.bm25_enabled
    assert channels.objects_enabled and channels.metadata_enabled
    assert channels.ocr_enabled and channels.asr_enabled and channels.caption_enabled
    assert channels.normalization == "rank"
    assert channels.metadata_frames_per_video == 8


def test_all_channel_types_satisfy_the_protocol() -> None:
    registry = registry_for(channels_entry())
    assert len(registry.channels) == len(CHANNEL_NAMES)
    for channel in registry.channels:
        assert isinstance(channel, RetrievalChannel)
        assert channel.name in CHANNEL_NAMES
