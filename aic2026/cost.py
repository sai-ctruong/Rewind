"""Per-query cost accounting.

Everything here is a COUNT of work performed: encoder calls, channel searches, decoded
frames, image embeddings, VLM calls, milliseconds. None of it says anything about
whether an answer is right. Cost and quality are different axes, and this module only
measures the first one — which is exactly why it can be trusted without ground truth.

Counters are additive and monotonic within one query. A stage that does nothing records
zero rather than being omitted, so an absent number always means "not implemented" and
never "happened to be skipped this time".
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# What a cost proxy is allowed to be used for: comparing two configurations of THIS
# system on THIS machine. It is not a portable FLOP count and is not money.
COST_PROXY_NOTE = (
    "Structural work counters for one query. Comparable between runs of this system on "
    "one machine; not a portable compute measure and not a quality signal."
)


@dataclass
class StageCost:
    """Wall time and work attributed to one named stage of a query."""

    name: str
    calls: int = 0
    wall_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "wall_ms": round(self.wall_ms, 3),
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass
class QueryCost:
    """Everything one query spent. Structural counters only."""

    task: str = ""
    query: str = ""

    # --- text side ---------------------------------------------------------------
    text_encoder_calls: int = 0        # times the encoder was actually invoked
    text_vectors_computed: int = 0     # prompt-template variants embedded
    text_encoder_cache_hits: int = 0

    # --- channel side ------------------------------------------------------------
    channel_search_calls: dict[str, int] = field(default_factory=dict)
    channel_candidate_counts: dict[str, int] = field(default_factory=dict)
    channel_search_ms: dict[str, float] = field(default_factory=dict)

    # --- video side --------------------------------------------------------------
    video_frames_requested: int = 0
    video_frames_decoded: int = 0
    video_decode_ms: float = 0.0

    # --- image embedding side ----------------------------------------------------
    image_embeddings_computed: int = 0
    image_embedding_ms: float = 0.0

    # --- Q&A side ----------------------------------------------------------------
    qa_vlm_calls: int = 0
    qa_vlm_images: int = 0
    qa_vlm_ms: float = 0.0

    # --- totals ------------------------------------------------------------------
    total_wall_ms: float = 0.0
    stages: list[StageCost] = field(default_factory=list)
    peak_process_rss_mb: Optional[float] = None
    gpu_available: bool = False
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------------- mutation
    def add_text_encode(self, *, variants: int, cached: bool = False) -> None:
        if cached:
            self.text_encoder_cache_hits += 1
            return
        self.text_encoder_calls += 1
        self.text_vectors_computed += int(variants)

    def add_channel_search(self, channel: str, *, candidates: int = 0, ms: float = 0.0) -> None:
        self.channel_search_calls[channel] = self.channel_search_calls.get(channel, 0) + 1
        self.channel_candidate_counts[channel] = (
            self.channel_candidate_counts.get(channel, 0) + int(candidates)
        )
        self.channel_search_ms[channel] = round(
            self.channel_search_ms.get(channel, 0.0) + float(ms), 3
        )

    def add_decode(self, *, requested: int = 0, decoded: int = 0, ms: float = 0.0) -> None:
        self.video_frames_requested += int(requested)
        self.video_frames_decoded += int(decoded)
        self.video_decode_ms += float(ms)

    def add_image_embeddings(self, count: int, *, ms: float = 0.0) -> None:
        self.image_embeddings_computed += int(count)
        self.image_embedding_ms += float(ms)

    def add_vlm_call(self, *, images: int = 0, ms: float = 0.0) -> None:
        self.qa_vlm_calls += 1
        self.qa_vlm_images += int(images)
        self.qa_vlm_ms += float(ms)

    @contextmanager
    def stage(self, name: str, **detail: Any) -> Iterator[StageCost]:
        entry = StageCost(name=name, detail=dict(detail))
        started = time.perf_counter()
        try:
            yield entry
        finally:
            entry.calls += 1
            entry.wall_ms += (time.perf_counter() - started) * 1000.0
            self.stages.append(entry)

    def sample_memory(self) -> Optional[float]:
        """Resident memory, when psutil is available. No new dependency is added."""
        try:
            import psutil
        except ImportError:  # pragma: no cover - optional
            return None
        value = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
        self.peak_process_rss_mb = max(self.peak_process_rss_mb or 0.0, value)
        return value

    # --------------------------------------------------------------------- reading
    @property
    def total_channel_calls(self) -> int:
        return sum(self.channel_search_calls.values())

    def cost_proxy(self) -> float:
        """A single comparable magnitude for ranking actions inside one machine.

        Weights are ORDER-OF-MAGNITUDE placeholders reflecting what each unit of work
        costs on this CPU-only setup — a decoded frame is far more expensive than a
        channel query, and a VLM call far more than a decode. They are not tuned, not
        calibrated against quality, and must never be reported as a score.
        """
        return round(
            1.0 * self.text_encoder_calls
            + 0.5 * self.total_channel_calls
            + 4.0 * self.video_frames_decoded
            + 4.0 * self.image_embeddings_computed
            + 200.0 * self.qa_vlm_calls,
            3,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "query": self.query,
            "text": {
                "encoder_calls": self.text_encoder_calls,
                "vectors_computed": self.text_vectors_computed,
                "cache_hits": self.text_encoder_cache_hits,
            },
            "channels": {
                "search_calls": dict(sorted(self.channel_search_calls.items())),
                "candidate_counts": dict(sorted(self.channel_candidate_counts.items())),
                "search_ms": dict(sorted(self.channel_search_ms.items())),
                "total_calls": self.total_channel_calls,
            },
            "video": {
                "frames_requested": self.video_frames_requested,
                "frames_decoded": self.video_frames_decoded,
                "decode_ms": round(self.video_decode_ms, 3),
            },
            "image_embedding": {
                "computed": self.image_embeddings_computed,
                "ms": round(self.image_embedding_ms, 3),
            },
            "qa": {
                "vlm_calls": self.qa_vlm_calls,
                "vlm_images": self.qa_vlm_images,
                "vlm_ms": round(self.qa_vlm_ms, 3),
            },
            "total_wall_ms": round(self.total_wall_ms, 3),
            "cost_proxy": self.cost_proxy(),
            "stages": [item.to_dict() for item in self.stages],
            "memory": {
                "peak_process_rss_mb": self.peak_process_rss_mb,
                # Never invented: an absent GPU reports unavailable rather than 0.
                "gpu": {"available": self.gpu_available, "vram_mb": None}
                if not self.gpu_available
                else {"available": True, "vram_mb": None},
            },
            "note": COST_PROXY_NOTE,
            **({"notes": list(self.notes)} if self.notes else {}),
        }


@contextmanager
def measure(cost: QueryCost) -> Iterator[QueryCost]:
    """Time a whole query into `total_wall_ms`."""
    started = time.perf_counter()
    cost.sample_memory()
    try:
        yield cost
    finally:
        cost.total_wall_ms += (time.perf_counter() - started) * 1000.0
        cost.sample_memory()


def merge_costs(costs: list[QueryCost]) -> dict[str, Any]:
    """Aggregate several query costs. Sums for work, mean/max for time."""
    if not costs:
        return {"queries": 0, "note": COST_PROXY_NOTE}
    channel_calls: dict[str, int] = {}
    for item in costs:
        for name, value in item.channel_search_calls.items():
            channel_calls[name] = channel_calls.get(name, 0) + value
    wall = [item.total_wall_ms for item in costs]
    return {
        "queries": len(costs),
        "text_encoder_calls": sum(item.text_encoder_calls for item in costs),
        "text_encoder_cache_hits": sum(item.text_encoder_cache_hits for item in costs),
        "channel_search_calls": dict(sorted(channel_calls.items())),
        "total_channel_calls": sum(item.total_channel_calls for item in costs),
        "video_frames_decoded": sum(item.video_frames_decoded for item in costs),
        "image_embeddings_computed": sum(item.image_embeddings_computed for item in costs),
        "qa_vlm_calls": sum(item.qa_vlm_calls for item in costs),
        "qa_vlm_images": sum(item.qa_vlm_images for item in costs),
        "cost_proxy_total": round(sum(item.cost_proxy() for item in costs), 3),
        "wall_ms": {
            "total": round(sum(wall), 1),
            "mean": round(sum(wall) / len(wall), 1),
            "max": round(max(wall), 1),
        },
        "peak_process_rss_mb": max(
            (item.peak_process_rss_mb for item in costs if item.peak_process_rss_mb), default=None
        ),
        "note": COST_PROXY_NOTE,
    }


__all__ = ["COST_PROXY_NOTE", "QueryCost", "StageCost", "measure", "merge_costs"]
