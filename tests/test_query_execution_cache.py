"""R0: caching may only remove work, never change an answer.

A cache that changes a ranking is not an optimization, it is a bug wearing one. These
tests pin bit-equality of the cached vector, order-equality of every task's output, the
hard capacity bound, and the fact that nothing is written to disk.
"""
from __future__ import annotations

import numpy as np
import pytest

from aic2026.query_cache import (
    DEFAULT_QUERY_CACHE_SIZE,
    BoundedCache,
    QueryEmbeddingCache,
    QueryEmbeddingKey,
    QueryExecutionContext,
    template_signature,
)
from tests.release_support import build_engine


# ------------------------------------------------------------------ bounded cache


def test_cache_refuses_to_be_unbounded() -> None:
    with pytest.raises(ValueError, match="unbounded"):
        BoundedCache(0)
    with pytest.raises(ValueError):
        BoundedCache(-1)


def test_cache_evicts_least_recently_used_and_stays_at_capacity() -> None:
    cache: BoundedCache[int] = BoundedCache(3)
    for key in range(3):
        cache.get_or_compute(key, lambda k=key: k)
    cache.get_or_compute(0, lambda: 999)          # refresh 0 -> 1 is now oldest
    cache.get_or_compute(3, lambda: 3)            # evicts 1
    assert len(cache) == 3
    assert cache.peek(1) is None
    assert cache.peek(0) == 0
    assert cache.stats.evictions == 1


def test_cache_counts_hits_and_misses() -> None:
    cache: BoundedCache[str] = BoundedCache(8)
    cache.get_or_compute("a", lambda: "A")
    cache.get_or_compute("a", lambda: "B")
    assert cache.stats.misses == 1 and cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5
    # A hit returns the FIRST computed value; the second factory never ran.
    assert cache.peek("a") == "A"


def test_peek_does_not_count_as_a_lookup() -> None:
    cache: BoundedCache[int] = BoundedCache(4)
    cache.get_or_compute(1, lambda: 1)
    before = cache.stats.hits
    cache.peek(1)
    assert cache.stats.hits == before


def test_default_size_is_small_and_declared() -> None:
    assert 64 <= DEFAULT_QUERY_CACHE_SIZE <= 1024


# ------------------------------------------------------------- embedding identity


def key_for(query: str, **overrides) -> QueryEmbeddingKey:
    payload = {
        "query": query,
        "model_name": "openai/clip-vit-base-patch32",
        "feature_dim": 512,
        "template_signature": "abc123",
    }
    payload.update(overrides)
    return QueryEmbeddingKey(**payload)


def test_cached_vector_is_bit_identical() -> None:
    cache = QueryEmbeddingCache(8)
    original = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    first = cache.get_or_compute(key_for("a"), lambda: original)
    second = cache.get_or_compute(key_for("a"), lambda: np.zeros(3, dtype=np.float32))
    assert np.array_equal(first, second)
    assert second.dtype == original.dtype


def test_cached_vector_is_handed_out_read_only() -> None:
    """One caller normalising in place would corrupt every later hit."""
    cache = QueryEmbeddingCache(4)
    vector = cache.get_or_compute(key_for("a"), lambda: np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        vector[0] = 5.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "other/model"},
        {"feature_dim": 256},
        {"template_signature": "different"},
    ],
)
def test_anything_that_changes_the_embedding_changes_the_key(overrides) -> None:
    assert key_for("a") != key_for("a", **overrides)


def test_different_queries_do_not_share_an_entry() -> None:
    cache = QueryEmbeddingCache(8)
    a = cache.get_or_compute(key_for("a"), lambda: np.array([1.0], dtype=np.float32))
    b = cache.get_or_compute(key_for("b"), lambda: np.array([2.0], dtype=np.float32))
    assert a[0] == 1.0 and b[0] == 2.0


def test_template_signature_tracks_the_template_set() -> None:
    assert template_signature(["a {q}"]) == template_signature(["a {q}"])
    assert template_signature(["a {q}"]) != template_signature(["a {q}", "b {q}"])
    assert template_signature(["a {q}"]) != template_signature(["A {q}"])


def test_cache_is_never_persisted(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    engine.search_kis("a", top_k=5)
    status = engine.query_cache_status()
    assert status["persisted"] is False
    # Nothing cache-shaped was written next to the index.
    assert not list(tmp_path.rglob("*query_cache*"))


# ------------------------------------------------------------- engine equivalence


def test_repeated_kis_query_is_identical_and_uses_the_cache(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    first = [(p.video_id, p.frame_id, round(p.score, 9)) for p in engine.search_kis("a", top_k=10)]
    hits_before = engine.query_cache_status()["query_embeddings"]["hits"]
    second = [(p.video_id, p.frame_id, round(p.score, 9)) for p in engine.search_kis("a", top_k=10)]
    assert first == second
    assert engine.query_cache_status()["query_embeddings"]["hits"] > hits_before


def test_repeated_trake_query_keeps_its_sequence_order(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    first = engine.search_trake_detailed(["a", "b"], max_results=10)
    second = engine.search_trake_detailed(["a", "b"], max_results=10)
    assert [p.row() for p in first.predictions] == [p.row() for p in second.predictions]


def test_repeated_qa_query_keeps_its_hypotheses(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    first, _ = engine.answer_qa("a", "what colour?", top_k=5)
    second, _ = engine.answer_qa("a", "what colour?", top_k=5)
    assert [(p.video_id, p.frame_id, p.answer) for p in first] == [
        (p.video_id, p.frame_id, p.answer) for p in second
    ]


def test_disabling_the_cache_by_size_one_still_produces_the_same_order(tmp_path) -> None:
    """Capacity affects only how often work is repeated, never the outcome."""
    big, _, _ = build_engine(tmp_path / "big", runtime={"query_embedding_cache_size": 256})
    small, _, _ = build_engine(tmp_path / "small", runtime={"query_embedding_cache_size": 1})
    assert [p.row() for p in big.search_kis("a", top_k=10)] == [
        p.row() for p in small.search_kis("a", top_k=10)
    ]


def test_engine_cache_is_bounded_by_configuration(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path, runtime={"query_embedding_cache_size": 2})
    for query in ("a", "b", "c", "d"):
        engine.search_kis(query, top_k=3)
    status = engine.query_cache_status()["query_embeddings"]
    assert status["max_entries"] == 2
    assert status["entries"] <= 2
    assert status["evictions"] >= 1


# --------------------------------------------------------- TRAKE work reduction


def test_trake_reuses_one_embedding_per_distinct_event_text(tmp_path) -> None:
    """The same event text is retrieved at several depths; it is encoded once."""
    engine, _, _ = build_engine(tmp_path)
    engine._query_embeddings.clear()
    before = engine._encode_calls
    outcome = engine.search_trake_detailed(["a", "b", "c"], max_results=10)
    distinct_events = 3
    assert engine._encode_calls - before <= distinct_events
    cache = outcome.diagnostics["query_embedding_cache"]
    assert cache["max_entries"] >= 1


def test_repeated_event_text_is_retrieved_once(tmp_path) -> None:
    """Two identical events at one depth are one retrieval, reused verbatim."""
    engine, _, _ = build_engine(tmp_path)
    outcome = engine.search_trake_detailed(["a", "a"], max_results=10)
    assert outcome.diagnostics["query_execution"]["reused_channel_results"] >= 1


def test_execution_context_is_request_local(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    first = engine.search_trake_detailed(["a", "b"], max_results=5)
    second = engine.search_trake_detailed(["a", "b"], max_results=5)
    # Each request reports its OWN reuse counters, not an accumulating total.
    assert first.diagnostics["query_execution"]["distinct_queries"] == (
        second.diagnostics["query_execution"]["distinct_queries"]
    )


def test_context_reuse_returns_candidates_for_the_right_event() -> None:
    """A reused candidate list must be re-labelled for the event that asked for it."""
    context = QueryExecutionContext(label="test")
    assert context.representation("a", lambda: "REP") == "REP"
    assert context.representation("a", lambda: "OTHER") == "REP"
    assert context.reused_representations == 1


def test_shallower_request_is_not_served_by_slicing_a_deeper_one(tmp_path) -> None:
    """Rank normalization depends on how many candidates a channel returned, so a
    top-40 slice of a depth-300 retrieval is NOT a depth-40 retrieval."""
    engine, _, _ = build_engine(tmp_path)
    context = QueryExecutionContext(label="test")
    deep = engine._trake_candidates(0, "a", 30, context=context)
    shallow = engine._trake_candidates(0, "a", 5, context=context)
    assert len(shallow) <= len(deep)
    # A separate key was computed rather than reused.
    assert ("a", 5) in context.channel_results and ("a", 30) in context.channel_results


# ------------------------------------------------------------------------ prewarm


def test_prewarm_is_off_unless_requested(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    state = engine.prewarm()
    assert state["prewarm_enabled"] is False
    assert state["requested"] is False and state["performed"] is False
    assert state["prewarm_ms"] == 0.0


def test_prewarm_changes_no_result(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    before = [p.row() for p in engine.search_kis("a", top_k=10)]
    state = engine.prewarm(force=True)
    after = [p.row() for p in engine.search_kis("a", top_k=10)]
    assert state["requested"] is True
    assert before == after


def test_prewarm_reports_its_own_failure_instead_of_raising(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)

    def boom(**kwargs):
        raise RuntimeError("no weights here")

    engine.encoder_status = boom
    state = engine.prewarm(force=True)
    assert state["performed"] is False
    assert state["model_state"] == "failed"
    assert "no weights here" in state["error"]
