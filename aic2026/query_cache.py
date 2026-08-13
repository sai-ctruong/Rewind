"""Bounded, in-process caches for deterministic per-query work.

Two facts make this safe rather than clever:

* the text encoder is deterministic for a fixed model, template set and dimension, so a
  cached vector is the *same* vector, not an approximation of it;
* nothing here is persisted. A cache that outlives the process could outlive the model
  that produced it, and a stale embedding is indistinguishable from a fresh one.

The identity of an entry therefore includes everything that can change the answer:
model name, feature dimension, template identity, and the query text. Change any of
them and you get a different key rather than a wrong hit.

Sizes are small, fixed defaults. They are NOT tuned: no ground truth exists here, and a
cache size cannot improve retrieval quality anyway — it can only avoid recomputing an
identical result.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Hashable, Iterable, Optional, TypeVar

import numpy as np

# Deliberately modest: a competition session asks a few dozen distinct queries, and a
# TRAKE query re-asks the same event text at several depths within one request.
DEFAULT_QUERY_CACHE_SIZE = 256

T = TypeVar("T")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if not self.lookups else self.hits / self.lookups

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
        }


class BoundedCache(Generic[T]):
    """A plain LRU with a hard capacity and hit/miss accounting.

    `functools.lru_cache` is not used because the capacity, the statistics and the
    ability to clear it on a dataset change all have to be observable from the outside.
    """

    def __init__(self, max_entries: int = DEFAULT_QUERY_CACHE_SIZE) -> None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be > 0; an unbounded cache is not allowed")
        self.max_entries = int(max_entries)
        self._data: "OrderedDict[Hashable, T]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._data)

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.stats.hits += 1
                return self._data[key]
            self.stats.misses += 1
        # Computed outside the lock: encoding is slow and must not serialise every
        # caller. A duplicate computation under a race is wasteful, never incorrect,
        # because the value is a pure function of the key.
        value = compute()
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1
        return value

    def peek(self, key: Hashable) -> Optional[T]:
        """Read without counting a hit; for assertions and diagnostics."""
        with self._lock:
            return self._data.get(key)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def reset_stats(self) -> None:
        with self._lock:
            self.stats = CacheStats()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "entries": len(self._data),
            **self.stats.to_dict(),
        }


def template_signature(templates: Iterable[str]) -> str:
    """Stable identity of the prompt-template set, order included.

    The templates are averaged into one vector, so re-ordering them cannot change the
    result — but adding, removing or editing one does, and that must miss the cache.
    """
    payload = "\n".join(str(item) for item in templates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class QueryEmbeddingKey:
    """Everything that can change a query embedding."""

    query: str
    model_name: str
    feature_dim: int
    template_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "template_signature": self.template_signature,
        }


class QueryEmbeddingCache:
    """Bounded cache of query -> embedding, keyed on the full embedding identity."""

    def __init__(self, max_entries: int = DEFAULT_QUERY_CACHE_SIZE) -> None:
        self._cache: BoundedCache[np.ndarray] = BoundedCache(max_entries)

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats

    @property
    def max_entries(self) -> int:
        return self._cache.max_entries

    def __len__(self) -> int:
        return len(self._cache)

    def get_or_compute(
        self, key: QueryEmbeddingKey, compute: Callable[[], np.ndarray]
    ) -> np.ndarray:
        vector = self._cache.get_or_compute(key, compute)
        # Handed out read-only: one caller normalising or scaling a cached vector in
        # place would silently corrupt every later hit.
        view = vector.view()
        view.flags.writeable = False
        return view

    def peek(self, key: QueryEmbeddingKey) -> Optional[np.ndarray]:
        """Read without counting a hit; used for per-query cost attribution."""
        return self._cache.peek(key)

    def clear(self) -> None:
        self._cache.clear()

    def reset_stats(self) -> None:
        self._cache.reset_stats()

    def to_dict(self) -> dict[str, Any]:
        return self._cache.to_dict()


@dataclass
class QueryExecutionContext:
    """Per-request scratch space for work that is identical within one query.

    A TRAKE query retrieves the same event text at several candidate depths. Without a
    context each depth re-normalises the query and re-encodes it; with one, a deeper
    request either reuses the cached result outright or extends it, and a shallower
    request slices the deeper result it already has.

    It is request-local on purpose: an index or a dataset change between requests must
    not be able to serve a stale candidate list.
    """

    label: str = ""
    representations: dict[str, Any] = field(default_factory=dict)
    channel_results: dict[tuple[str, ...], Any] = field(default_factory=dict)
    reused_representations: int = 0
    reused_channel_results: int = 0
    extended_channel_results: int = 0

    def representation(self, query: str, build: Callable[[], Any]) -> Any:
        cached = self.representations.get(query)
        if cached is not None:
            self.reused_representations += 1
            return cached
        value = build()
        self.representations[query] = value
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "distinct_queries": len(self.representations),
            "reused_representations": self.reused_representations,
            "reused_channel_results": self.reused_channel_results,
            "extended_channel_results": self.extended_channel_results,
        }


__all__ = [
    "DEFAULT_QUERY_CACHE_SIZE",
    "BoundedCache",
    "CacheStats",
    "QueryEmbeddingCache",
    "QueryEmbeddingKey",
    "QueryExecutionContext",
    "template_signature",
]
