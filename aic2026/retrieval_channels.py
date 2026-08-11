"""Independent retrieval channels: every signal can introduce its own candidates.

Before Phase 9 the candidate pool was `UNION(CLIP top-k, BM25 top-k)` and objects and
metadata only *rescored* whatever that union already contained. A frame strongly
indicated by its detector labels, or a video strongly indicated by its title, could
therefore never enter the pool at all — the signal existed but had no way in.

Here each signal is a `RetrievalChannel` that answers a query with its own candidates.
The pool becomes the union of every enabled, available channel, and each candidate
remembers which channels found it, at which rank, with which raw and normalized scores.
A candidate introduced only by objects stays visibly object-introduced all the way to the
response.

Two honesty rules:

* A channel whose source data is not populated reports `available=false` with a reason.
  It does not return empty results as though it had looked. On the real L21 development
  scope OCR, ASR and frame captions are genuinely absent, and they say so.
* Every candidate must resolve to a canonical indexed keyframe. A channel cannot invent
  an ID.

No accuracy is claimed anywhere: this repository has no AIC ground truth, so the
diagnostics below count candidate *coverage*, never recall.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .query_normalization import (
    QueryRepresentation,
    label_tokens,
    normalize_label,
    normalize_query,
)

CHANNEL_CLIP = "clip"
CHANNEL_BM25 = "bm25"
CHANNEL_OBJECTS = "objects"
CHANNEL_METADATA = "metadata"
CHANNEL_OCR = "ocr"
CHANNEL_ASR = "asr"
CHANNEL_CAPTION = "caption"

CHANNEL_NAMES = (
    CHANNEL_CLIP,
    CHANNEL_BM25,
    CHANNEL_OBJECTS,
    CHANNEL_METADATA,
    CHANNEL_OCR,
    CHANNEL_ASR,
    CHANNEL_CAPTION,
)

SCOPE_FRAME = "frame"
SCOPE_VIDEO = "video"

REASON_NO_DATA = "no_populated_source_data"
REASON_DISABLED = "disabled_by_configuration"

# Bumped when a channel's build-time index content or schema changes, so a cache built
# with different channel sources is not silently reused.
CHANNEL_SCHEMA_VERSION = 1


class ChannelError(RuntimeError):
    """Raised when a channel returns something that cannot be trusted."""


@dataclass(frozen=True)
class ChannelStatus:
    """Whether a channel can actually contribute, and why not when it cannot."""

    name: str
    enabled: bool = True
    available: bool = False
    record_count: int = 0
    index_type: str = "none"
    reason: Optional[str] = None
    scope: str = SCOPE_FRAME

    @property
    def usable(self) -> bool:
        return self.enabled and self.available

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "available": self.available,
            "usable": self.usable,
            "records": int(self.record_count),
            "index_type": self.index_type,
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChannelCandidate:
    """One candidate from one channel, always resolving to a canonical keyframe."""

    keyframe_id: str
    video_id: str
    channel: str
    raw_score: float
    rank: int
    frame_idx: Optional[int] = None
    timestamp: float = 0.0
    normalized_score: float = 0.0
    scope: str = SCOPE_FRAME
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "channel": self.channel,
            "rank": int(self.rank),
            "raw_score": round(float(self.raw_score), 6),
            "normalized_score": round(float(self.normalized_score), 6),
            "scope": self.scope,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ChannelResult:
    """What one channel returned for one query, plus how long it took."""

    channel: str
    status: ChannelStatus
    candidates: tuple[ChannelCandidate, ...] = ()
    search_ms: float = 0.0
    warning: Optional[str] = None

    @property
    def searched(self) -> bool:
        return self.status.usable

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "status": self.status.to_dict(),
            "candidates": len(self.candidates),
            "search_ms": round(float(self.search_ms), 3),
            "warning": self.warning,
        }


@runtime_checkable
class RetrievalChannel(Protocol):
    """A source that can independently propose candidates."""

    name: str

    def status(self) -> ChannelStatus: ...

    def search(self, query: QueryRepresentation, *, top_k: int) -> ChannelResult: ...


def normalize_channel_scores(
    candidates: Sequence[ChannelCandidate], *, method: str = "rank"
) -> tuple[ChannelCandidate, ...]:
    """Put one channel's scores on a comparable [0, 1] scale.

    Channel score spaces are unrelated — a CLIP cosine, a BM25 sum, and a detector
    confidence cannot be added directly. `rank` is the default because it is stable
    regardless of the underlying scale and cannot be destabilised by an outlier;
    `minmax` is available where the raw magnitudes are meaningful. Both handle the
    zero-variance case explicitly rather than dividing by zero.
    """
    items = list(candidates)
    if not items:
        return ()
    if method == "minmax":
        values = [float(item.raw_score) for item in items]
        if not all(math.isfinite(value) for value in values):
            raise ChannelError("Channel returned a non-finite score.")
        low, high = min(values), max(values)
        if high - low <= 1e-12:
            # Every candidate agrees: give them all full weight rather than 0/0.
            return tuple(replace(item, normalized_score=1.0) for item in items)
        return tuple(
            replace(item, normalized_score=(float(item.raw_score) - low) / (high - low))
            for item in items
        )
    if method != "rank":
        raise ChannelError(f"Unknown channel normalization method {method!r}.")
    total = len(items)
    return tuple(
        replace(item, normalized_score=1.0 - (index / total))
        for index, item in enumerate(items)
    )


@dataclass
class PooledCandidate:
    """A canonical keyframe plus every channel that proposed it."""

    keyframe_id: str
    video_id: str
    frame_idx: Optional[int] = None
    timestamp: float = 0.0
    by_channel: dict[str, ChannelCandidate] = field(default_factory=dict)

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_channel))

    def rank_in(self, channel: str) -> Optional[int]:
        item = self.by_channel.get(channel)
        return None if item is None else int(item.rank)

    def raw_score(self, channel: str) -> float:
        item = self.by_channel.get(channel)
        return 0.0 if item is None else float(item.raw_score)

    def normalized_score(self, channel: str) -> float:
        item = self.by_channel.get(channel)
        return 0.0 if item is None else float(item.normalized_score)

    def introduced_only_by(self) -> Optional[str]:
        """The channel that alone proposed this candidate, if there is exactly one."""
        return self.channels[0] if len(self.by_channel) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "channels": list(self.channels),
            "ranks": {name: item.rank for name, item in sorted(self.by_channel.items())},
            "raw_scores": {
                name: round(float(item.raw_score), 6)
                for name, item in sorted(self.by_channel.items())
            },
            "normalized_scores": {
                name: round(float(item.normalized_score), 6)
                for name, item in sorted(self.by_channel.items())
            },
            "evidence": {
                name: list(item.evidence)
                for name, item in sorted(self.by_channel.items())
                if item.evidence
            },
        }


@dataclass(frozen=True)
class ChannelUnion:
    """The pooled candidates plus per-channel coverage diagnostics."""

    candidates: tuple[PooledCandidate, ...] = ()
    results: tuple[ChannelResult, ...] = ()
    query: Optional[QueryRepresentation] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def by_id(self) -> dict[str, PooledCandidate]:
        return {item.keyframe_id: item for item in self.candidates}


class RetrievalChannelRegistry:
    """Holds the channels and merges their results into one pool."""

    def __init__(self, channels: Sequence[RetrievalChannel], *, normalization: str = "rank"):
        self.channels = list(channels)
        self.normalization = normalization

    def status(self) -> dict[str, Any]:
        return {channel.name: channel.status().to_dict() for channel in self.channels}

    def usable_channels(self) -> list[RetrievalChannel]:
        return [channel for channel in self.channels if channel.status().usable]

    def search(
        self,
        query: str | QueryRepresentation,
        *,
        depths: Mapping[str, int],
        default_top_k: int = 200,
    ) -> ChannelUnion:
        """Ask every usable channel, then union the results by canonical keyframe ID."""
        representation = (
            query if isinstance(query, QueryRepresentation) else normalize_query(query)
        )
        results: list[ChannelResult] = []
        pooled: dict[str, PooledCandidate] = {}
        for channel in self.channels:
            status = channel.status()
            if not status.usable:
                results.append(ChannelResult(channel=channel.name, status=status))
                continue
            top_k = int(depths.get(channel.name, default_top_k))
            if top_k <= 0:
                results.append(
                    ChannelResult(
                        channel=channel.name,
                        status=replace(status, enabled=False, reason="top_k<=0"),
                    )
                )
                continue
            result = channel.search(representation, top_k=top_k)
            normalized = normalize_channel_scores(result.candidates, method=self.normalization)
            result = replace(result, candidates=normalized)
            results.append(result)
            for candidate in normalized:
                entry = pooled.get(candidate.keyframe_id)
                if entry is None:
                    entry = PooledCandidate(
                        keyframe_id=candidate.keyframe_id,
                        video_id=candidate.video_id,
                        frame_idx=candidate.frame_idx,
                        timestamp=candidate.timestamp,
                    )
                    pooled[candidate.keyframe_id] = entry
                # One entry per channel: a channel proposing an ID twice keeps its best.
                existing = entry.by_channel.get(candidate.channel)
                if existing is None or candidate.rank < existing.rank:
                    entry.by_channel[candidate.channel] = candidate

        candidates = tuple(
            sorted(pooled.values(), key=lambda item: (item.video_id, item.keyframe_id))
        )
        return ChannelUnion(
            candidates=candidates,
            results=tuple(results),
            query=representation,
            diagnostics=channel_diagnostics(candidates, results),
        )


def channel_diagnostics(
    candidates: Sequence[PooledCandidate], results: Sequence[ChannelResult]
) -> dict[str, Any]:
    """Structural candidate-coverage counters. NOT recall, and never named as such."""
    per_channel: dict[str, dict[str, Any]] = {}
    exclusive: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        only = candidate.introduced_only_by()
        if only is not None:
            exclusive[only] += 1
    clip_ids = {c.keyframe_id for c in candidates if CHANNEL_CLIP in c.by_channel}
    bm25_ids = {c.keyframe_id for c in candidates if CHANNEL_BM25 in c.by_channel}
    for result in results:
        ids = {
            candidate.keyframe_id
            for candidate in candidates
            if result.channel in candidate.by_channel
        }
        per_channel[result.channel] = {
            "enabled": result.status.enabled,
            "available": result.status.available,
            "searched": result.searched,
            "candidates_returned": len(result.candidates),
            "unique_candidates_introduced": int(exclusive.get(result.channel, 0)),
            "overlap_with_clip": len(ids & clip_ids) if result.channel != CHANNEL_CLIP else len(ids),
            "overlap_with_bm25": len(ids & bm25_ids) if result.channel != CHANNEL_BM25 else len(ids),
            "search_ms": round(float(result.search_ms), 3),
            "reason": result.status.reason,
        }
    return {
        "candidate_union_size": len(candidates),
        "unique_videos": len({candidate.video_id for candidate in candidates}),
        "channels": per_channel,
        "exclusive_candidates": {name: int(count) for name, count in sorted(exclusive.items())},
        "channel_schema_version": CHANNEL_SCHEMA_VERSION,
        "note": (
            "Candidate coverage only. No AIC ground truth exists, so none of these "
            "counts is recall, precision, or accuracy."
        ),
    }


# --------------------------------------------------------------------- channels


class _EntryChannel:
    """Shared plumbing: resolve rows of the canonical index to channel candidates."""

    name = "base"
    scope = SCOPE_FRAME

    def __init__(self, entry, *, enabled: bool = True):
        self.entry = entry
        self.enabled = bool(enabled)

    def _candidate(
        self,
        keyframe_id: str,
        rank: int,
        raw_score: float,
        *,
        evidence: Sequence[str] = (),
        scope: Optional[str] = None,
    ) -> Optional[ChannelCandidate]:
        raw = self.entry.raws.get(keyframe_id)
        if raw is None:
            # A channel may never invent an ID: an unknown one is dropped, not passed on.
            return None
        return ChannelCandidate(
            keyframe_id=keyframe_id,
            video_id=raw.video_id,
            channel=self.name,
            raw_score=float(raw_score),
            rank=int(rank),
            frame_idx=None if raw.frame_idx is None else int(raw.frame_idx),
            timestamp=float(raw.timestamp),
            scope=scope or self.scope,
            evidence=tuple(evidence),
        )


class ClipChannel(_EntryChannel):
    """Dense CLIP retrieval, wrapping the existing engine behaviour unchanged.

    It receives the ORIGINAL query text: CLIP was trained on natural language, and
    handing it an accent-folded or expanded string would degrade it.
    """

    name = CHANNEL_CLIP

    def __init__(self, entry, encode_query: Callable[[str], Any], *, enabled: bool = True):
        super().__init__(entry, enabled=enabled)
        self._encode_query = encode_query

    def status(self) -> ChannelStatus:
        index = getattr(self.entry.index, "_clip_index", None)
        count = int(getattr(self.entry, "num_indexed", 0) or 0)
        return ChannelStatus(
            name=self.name,
            enabled=self.enabled,
            available=index is not None and count > 0,
            record_count=count,
            index_type="hnsw_dense",
            reason=None if index is not None and count > 0 else REASON_NO_DATA,
        )

    def search(self, query: QueryRepresentation, *, top_k: int) -> ChannelResult:
        import time

        started = time.perf_counter()
        vector = self._encode_query(query.dense_query)
        depth = max(1, min(int(top_k), len(self.entry.index.ids)))
        pairs = self.entry.index.dense_search(vector, "clip", depth)
        candidates: list[ChannelCandidate] = []
        for rank, (row, score) in enumerate(pairs, start=1):
            keyframe_id = self.entry.index.ids[row]
            candidate = self._candidate(keyframe_id, rank, score, evidence=("dense_clip",))
            if candidate is not None:
                candidates.append(candidate)
        return ChannelResult(
            channel=self.name,
            status=self.status(),
            candidates=tuple(candidates),
            search_ms=(time.perf_counter() - started) * 1000.0,
        )


class Bm25Channel(_EntryChannel):
    """Sparse frame-scoped textual retrieval over whatever text the records carry.

    Availability is measured from the corpus, not assumed: on a dataset built without
    objects, captions, OCR or ASR every document is the empty sentinel, and this channel
    correctly reports that it has no populated source data.
    """

    name = CHANNEL_BM25

    def __init__(self, entry, *, enabled: bool = True, non_empty_documents: Optional[int] = None):
        super().__init__(entry, enabled=enabled)
        self._non_empty = non_empty_documents

    # The index builder substitutes this token for a document with no text at all, so a
    # corpus made only of sentinels carries nothing retrievable.
    EMPTY_DOCUMENT_TOKEN = "∅"

    def _document_count(self) -> int:
        """How many documents hold real text.

        Measured from `doc_freqs`, because `BM25Okapi` does not retain the corpus it was
        built from: reading a `.corpus` attribute reports zero for every dataset and would
        make a populated BM25 index look unavailable.
        """
        if self._non_empty is not None:
            return int(self._non_empty)
        bm25 = getattr(self.entry.index, "_bm25", None)
        if bm25 is None:
            return 0
        frequencies = getattr(bm25, "doc_freqs", None)
        if frequencies is not None:
            return sum(
                1
                for document in frequencies
                if any(token != self.EMPTY_DOCUMENT_TOKEN for token in document)
            )
        corpus = getattr(bm25, "corpus", None)
        if corpus is None:
            return 0
        return sum(
            1
            for document in corpus
            if any(token != self.EMPTY_DOCUMENT_TOKEN for token in document)
        )

    def status(self) -> ChannelStatus:
        count = self._document_count()
        return ChannelStatus(
            name=self.name,
            enabled=self.enabled,
            available=count > 0,
            record_count=count,
            index_type="bm25",
            reason=None if count > 0 else REASON_NO_DATA,
        )

    def search(self, query: QueryRepresentation, *, top_k: int) -> ChannelResult:
        import time

        started = time.perf_counter()
        depth = max(1, min(int(top_k), len(self.entry.index.ids)))
        # Sparse matching may use folded and expanded terms; only CLIP needs the
        # original wording.
        text = " ".join(dict.fromkeys((query.lowercase, query.accent_folded) + query.object_terms))
        pairs = self.entry.index.sparse_search(text, depth, None)
        candidates: list[ChannelCandidate] = []
        for rank, (row, score) in enumerate(pairs, start=1):
            if float(score) <= 0.0:
                continue
            keyframe_id = self.entry.index.ids[row]
            candidate = self._candidate(keyframe_id, rank, score, evidence=("bm25_frame_text",))
            if candidate is not None:
                candidates.append(candidate)
        return ChannelResult(
            channel=self.name,
            status=self.status(),
            candidates=tuple(candidates),
            search_ms=(time.perf_counter() - started) * 1000.0,
        )


class ObjectChannel(_EntryChannel):
    """Independent retrieval over detector labels, via an inverted index.

    This is the channel that changes the architecture: a frame whose only strong signal
    is its object labels can now enter the pool on its own, instead of needing CLIP or
    BM25 to have found it first.

    The score is deliberately simple and transparent — matched-term coverage weighted by
    detector confidence. Detector confidence is NOT treated as a calibrated relevance
    probability; it only orders frames within this channel.
    """

    name = CHANNEL_OBJECTS

    def __init__(
        self,
        entry,
        *,
        enabled: bool = True,
        confidence_threshold: float = 0.25,
    ):
        super().__init__(entry, enabled=enabled)
        self.confidence_threshold = float(confidence_threshold)
        self._postings: dict[str, list[tuple[str, float]]] = {}
        self._frames_with_labels = 0
        self._build()

    def _build(self) -> None:
        """One inverted index at construction; queries never rescan the records."""
        postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for keyframe_id, raw in self.entry.raws.items():
            detections = list(getattr(raw, "object_detections", None) or [])
            if not detections:
                detections = [
                    {"label": label, "confidence": 1.0}
                    for label in (getattr(raw, "objects", None) or ())
                ]
            best: dict[str, float] = {}
            for detection in detections:
                confidence = float(detection.get("confidence", 0.0) or 0.0)
                if not math.isfinite(confidence) or confidence < self.confidence_threshold:
                    continue
                for token in label_tokens(str(detection.get("label", ""))):
                    if not token:
                        continue
                    best[token] = max(best.get(token, 0.0), confidence)
            if best:
                self._frames_with_labels += 1
            for token, confidence in best.items():
                postings[token].append((keyframe_id, confidence))
        self._postings = {token: sorted(rows) for token, rows in postings.items()}

    def status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            enabled=self.enabled,
            available=self._frames_with_labels > 0,
            record_count=self._frames_with_labels,
            index_type="inverted_labels",
            reason=None if self._frames_with_labels > 0 else REASON_NO_DATA,
        )

    def search(self, query: QueryRepresentation, *, top_k: int) -> ChannelResult:
        import time

        started = time.perf_counter()
        # Negated terms are excluded: "khong co xe" must not become a positive query for
        # vehicles. The object channel cannot model negation, so it declines to use it.
        terms = tuple(dict.fromkeys(term for term in query.object_terms if term in self._postings))
        if not terms:
            return ChannelResult(
                channel=self.name,
                status=self.status(),
                search_ms=(time.perf_counter() - started) * 1000.0,
                warning=None if query.object_terms else "No usable object terms in the query.",
            )
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, set[str]] = defaultdict(set)
        for term in terms:
            for keyframe_id, confidence in self._postings.get(term, ()):
                scores[keyframe_id] += float(confidence)
                matched[keyframe_id].add(term)
        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -(len(matched[item[0]]) / len(terms)) * item[1],
                -len(matched[item[0]]),
                item[0],
            ),
        )[: max(1, int(top_k))]
        candidates: list[ChannelCandidate] = []
        for rank, (keyframe_id, confidence_sum) in enumerate(ranked, start=1):
            coverage = len(matched[keyframe_id]) / len(terms)
            candidate = self._candidate(
                keyframe_id,
                rank,
                coverage * confidence_sum,
                evidence=tuple(sorted(matched[keyframe_id])),
            )
            if candidate is not None:
                candidates.append(candidate)
        return ChannelResult(
            channel=self.name,
            status=self.status(),
            candidates=tuple(candidates),
            search_ms=(time.perf_counter() - started) * 1000.0,
        )


class MetadataChannel(_EntryChannel):
    """Independent VIDEO-level retrieval over media title, description and tags.

    Metadata describes the video, not a frame, so a metadata hit is scoped `video` and
    is expanded into a bounded, evenly spread set of frame hypotheses from that video.
    Claiming frame-level grounding from a title would be a provenance error, and dumping
    every frame of a matching video would drown the pool.
    """

    name = CHANNEL_METADATA
    scope = SCOPE_VIDEO

    def __init__(
        self,
        entry,
        *,
        enabled: bool = True,
        frames_per_video: int = 8,
    ):
        super().__init__(entry, enabled=enabled)
        self.frames_per_video = max(1, int(frames_per_video))
        self._documents: dict[str, tuple[str, ...]] = {}
        self._frames_by_video: dict[str, list[str]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        by_video: dict[str, set[str]] = defaultdict(set)
        for keyframe_id, raw in sorted(
            self.entry.raws.items(), key=lambda item: float(item[1].timestamp)
        ):
            self._frames_by_video[raw.video_id].append(keyframe_id)
            for field_name in ("media_title", "media_description", "media_channel"):
                value = getattr(raw, field_name, "") or ""
                if value:
                    by_video[raw.video_id].update(label_tokens(value))
            for tag in getattr(raw, "media_tags", None) or ():
                by_video[raw.video_id].update(label_tokens(str(tag)))
        # Metadata may also live on the entry's per-frame caption map when the loader
        # placed it there; it is read but never treated as a frame caption.
        captions = getattr(self.entry, "caption_by_id", None) or {}
        for keyframe_id, text in captions.items():
            raw = self.entry.raws.get(keyframe_id)
            if raw is not None and text:
                by_video[raw.video_id].update(label_tokens(str(text)))
        self._documents = {video: tuple(sorted(tokens)) for video, tokens in by_video.items() if tokens}

    def status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            enabled=self.enabled,
            available=bool(self._documents),
            record_count=len(self._documents),
            index_type="video_token_sets",
            scope=SCOPE_VIDEO,
            reason=None if self._documents else REASON_NO_DATA,
        )

    def _representative_frames(self, video_id: str) -> list[str]:
        """A bounded, evenly spread sample of one video's frames."""
        frames = self._frames_by_video.get(video_id, [])
        if len(frames) <= self.frames_per_video:
            return list(frames)
        step = len(frames) / self.frames_per_video
        return [frames[min(len(frames) - 1, int(index * step))] for index in range(self.frames_per_video)]

    def search(self, query: QueryRepresentation, *, top_k: int) -> ChannelResult:
        import time

        started = time.perf_counter()
        terms = set(query.lexical_terms) - set(query.negated_terms)
        if not terms:
            return ChannelResult(
                channel=self.name,
                status=self.status(),
                search_ms=(time.perf_counter() - started) * 1000.0,
            )
        scored: list[tuple[float, str, tuple[str, ...]]] = []
        for video_id, tokens in self._documents.items():
            overlap = tuple(sorted(terms & set(tokens)))
            if not overlap:
                continue
            scored.append((len(overlap) / len(terms), video_id, overlap))
        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates: list[ChannelCandidate] = []
        rank = 0
        for score, video_id, overlap in scored:
            for keyframe_id in self._representative_frames(video_id):
                if len(candidates) >= max(1, int(top_k)):
                    break
                rank += 1
                candidate = self._candidate(
                    keyframe_id,
                    rank,
                    score,
                    evidence=overlap,
                    scope=SCOPE_VIDEO,
                )
                if candidate is not None:
                    candidates.append(candidate)
            if len(candidates) >= max(1, int(top_k)):
                break
        return ChannelResult(
            channel=self.name,
            status=self.status(),
            candidates=tuple(candidates),
            search_ms=(time.perf_counter() - started) * 1000.0,
        )


class TextFieldChannel(_EntryChannel):
    """Frame-scoped retrieval over one optional text field (OCR, ASR, caption).

    Constructed for every optional field so its availability can be reported honestly. If
    the field is empty across the dataset — which it is for OCR, ASR and frame captions on
    the real L21 development scope — the channel reports `available=false` with a reason
    and never returns a candidate. Nothing is substituted for it: object labels are not
    OCR, and media metadata is not a frame caption.
    """

    def __init__(
        self,
        entry,
        name: str,
        source: Mapping[str, str] | Callable[[Any], str],
        *,
        enabled: bool = True,
    ):
        super().__init__(entry, enabled=enabled)
        self.name = name
        self._texts: dict[str, tuple[str, ...]] = {}
        for keyframe_id, raw in self.entry.raws.items():
            value = (
                source(raw)
                if callable(source)
                else str(source.get(keyframe_id, "") or "")
            )
            tokens = label_tokens(value) if value else ()
            if tokens:
                self._texts[keyframe_id] = tokens

    def status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            enabled=self.enabled,
            available=bool(self._texts),
            record_count=len(self._texts),
            index_type="token_sets",
            reason=None if self._texts else REASON_NO_DATA,
        )

    def search(self, query: QueryRepresentation, *, top_k: int) -> ChannelResult:
        import time

        started = time.perf_counter()
        terms = set(query.lexical_terms) - set(query.negated_terms)
        if not terms or not self._texts:
            return ChannelResult(
                channel=self.name,
                status=self.status(),
                search_ms=(time.perf_counter() - started) * 1000.0,
            )
        scored: list[tuple[float, str, tuple[str, ...]]] = []
        for keyframe_id, tokens in self._texts.items():
            overlap = tuple(sorted(terms & set(tokens)))
            if overlap:
                scored.append((len(overlap) / len(terms), keyframe_id, overlap))
        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates: list[ChannelCandidate] = []
        for rank, (score, keyframe_id, overlap) in enumerate(
            scored[: max(1, int(top_k))], start=1
        ):
            candidate = self._candidate(keyframe_id, rank, score, evidence=overlap)
            if candidate is not None:
                candidates.append(candidate)
        return ChannelResult(
            channel=self.name,
            status=self.status(),
            candidates=tuple(candidates),
            search_ms=(time.perf_counter() - started) * 1000.0,
        )


def build_channel_registry(
    entry,
    encode_query: Callable[[str], Any],
    config,
) -> RetrievalChannelRegistry:
    """Assemble every channel the configuration asks for.

    A disabled channel is still constructed so `/health` can report it; it simply never
    searches. Availability is measured from the data, never assumed from configuration.
    """
    channels: list[RetrievalChannel] = [
        ClipChannel(entry, encode_query, enabled=bool(config.clip_enabled)),
        Bm25Channel(entry, enabled=bool(config.bm25_enabled)),
        ObjectChannel(
            entry,
            enabled=bool(config.objects_enabled),
            confidence_threshold=float(config.object_confidence_threshold),
        ),
        MetadataChannel(
            entry,
            enabled=bool(config.metadata_enabled),
            frames_per_video=int(config.metadata_frames_per_video),
        ),
        TextFieldChannel(
            entry,
            CHANNEL_OCR,
            getattr(entry, "ocr_by_id", None) or {},
            enabled=bool(config.ocr_enabled),
        ),
        TextFieldChannel(
            entry,
            CHANNEL_ASR,
            getattr(entry, "asr_by_id", None) or {},
            enabled=bool(config.asr_enabled),
        ),
        TextFieldChannel(
            entry,
            CHANNEL_CAPTION,
            lambda raw: str(getattr(raw, "frame_caption", "") or ""),
            enabled=bool(config.caption_enabled),
        ),
    ]
    return RetrievalChannelRegistry(channels, normalization=str(config.normalization))


def channel_depths(config, *, scale: float = 1.0) -> dict[str, int]:
    """Per-channel retrieval depth, optionally deepened by a common factor."""
    factor = max(1.0, float(scale))
    return {
        CHANNEL_CLIP: int(config.clip_top_k * factor),
        CHANNEL_BM25: int(config.bm25_top_k * factor),
        CHANNEL_OBJECTS: int(config.objects_top_k * factor),
        CHANNEL_METADATA: int(config.metadata_top_k * factor),
        CHANNEL_OCR: int(config.ocr_top_k * factor),
        CHANNEL_ASR: int(config.asr_top_k * factor),
        CHANNEL_CAPTION: int(config.caption_top_k * factor),
    }


__all__ = [
    "CHANNEL_ASR",
    "CHANNEL_BM25",
    "CHANNEL_CAPTION",
    "CHANNEL_CLIP",
    "CHANNEL_METADATA",
    "CHANNEL_NAMES",
    "CHANNEL_OBJECTS",
    "CHANNEL_OCR",
    "CHANNEL_SCHEMA_VERSION",
    "REASON_DISABLED",
    "REASON_NO_DATA",
    "SCOPE_FRAME",
    "SCOPE_VIDEO",
    "Bm25Channel",
    "ChannelCandidate",
    "ChannelError",
    "ChannelResult",
    "ChannelStatus",
    "ChannelUnion",
    "ClipChannel",
    "MetadataChannel",
    "ObjectChannel",
    "PooledCandidate",
    "RetrievalChannel",
    "RetrievalChannelRegistry",
    "TextFieldChannel",
    "build_channel_registry",
    "channel_depths",
    "channel_diagnostics",
    "normalize_channel_scores",
]
