"""R0: per-query cost counters measure work, and only work.

A counter that quietly reports zero when a stage did not run is indistinguishable from
a stage that ran for free, so absent capabilities are reported as unavailable rather
than as zero. Nothing here is a quality signal; `cost_proxy` is explicitly a within-
machine comparison aid, not a score.
"""
from __future__ import annotations

import json

import pytest

from aic2026.cost import COST_PROXY_NOTE, QueryCost, StageCost, measure, merge_costs
from tests.release_support import build_engine


# ---------------------------------------------------------------------- counters


def test_counters_start_at_zero_and_only_grow() -> None:
    cost = QueryCost(task="kis", query="a")
    assert cost.text_encoder_calls == 0 and cost.video_frames_decoded == 0
    cost.add_text_encode(variants=3)
    cost.add_text_encode(variants=3, cached=True)
    assert cost.text_encoder_calls == 1
    assert cost.text_vectors_computed == 3
    assert cost.text_encoder_cache_hits == 1


def test_channel_costs_are_per_channel() -> None:
    cost = QueryCost()
    cost.add_channel_search("clip", candidates=1200, ms=12.0)
    cost.add_channel_search("clip", candidates=800, ms=8.0)
    cost.add_channel_search("bm25", candidates=400, ms=3.0)
    assert cost.channel_search_calls == {"clip": 2, "bm25": 1}
    assert cost.channel_candidate_counts["clip"] == 2000
    assert cost.total_channel_calls == 3


def test_decode_and_embedding_counters_are_distinct() -> None:
    """Frames requested, frames actually decoded, and embeddings are three numbers."""
    cost = QueryCost()
    cost.add_decode(requested=32, decoded=28, ms=400.0)
    cost.add_image_embeddings(28, ms=120.0)
    payload = cost.to_dict()
    assert payload["video"]["frames_requested"] == 32
    assert payload["video"]["frames_decoded"] == 28
    assert payload["image_embedding"]["computed"] == 28


def test_vlm_counters_track_calls_and_images_separately() -> None:
    cost = QueryCost()
    cost.add_vlm_call(images=4, ms=900.0)
    cost.add_vlm_call(images=2, ms=700.0)
    assert cost.qa_vlm_calls == 2 and cost.qa_vlm_images == 6


def test_stage_timing_is_recorded() -> None:
    cost = QueryCost()
    with cost.stage("coarse", depth=100):
        pass
    assert len(cost.stages) == 1
    stage = cost.stages[0]
    assert stage.name == "coarse" and stage.calls == 1
    assert stage.detail == {"depth": 100}
    assert stage.wall_ms >= 0.0


def test_measure_records_total_wall_time() -> None:
    cost = QueryCost()
    with measure(cost):
        pass
    assert cost.total_wall_ms >= 0.0


# ------------------------------------------------------------------- honesty rules


def test_gpu_metrics_are_reported_unavailable_not_zero() -> None:
    payload = QueryCost().to_dict()
    gpu = payload["memory"]["gpu"]
    assert gpu["available"] is False
    assert gpu["vram_mb"] is None


def test_cost_dict_declares_what_it_is_not() -> None:
    payload = QueryCost().to_dict()
    assert payload["note"] == COST_PROXY_NOTE
    assert "not a quality signal" in payload["note"]
    text = json.dumps(payload).lower()
    for forbidden in ("accuracy", "recall@", "final score", "correct"):
        assert forbidden not in text


def test_cost_proxy_is_monotonic_in_expensive_work() -> None:
    cheap = QueryCost()
    cheap.add_channel_search("clip", candidates=10)
    expensive = QueryCost()
    expensive.add_channel_search("clip", candidates=10)
    expensive.add_vlm_call(images=4)
    assert expensive.cost_proxy() > cheap.cost_proxy()


def test_merge_reports_nothing_for_no_queries() -> None:
    assert merge_costs([])["queries"] == 0


def test_merge_sums_work_and_summarises_time() -> None:
    a, b = QueryCost(), QueryCost()
    a.add_decode(decoded=10)
    a.total_wall_ms = 100.0
    b.add_decode(decoded=6)
    b.total_wall_ms = 300.0
    merged = merge_costs([a, b])
    assert merged["queries"] == 2
    assert merged["video_frames_decoded"] == 16
    assert merged["wall_ms"]["max"] == 300.0
    assert merged["wall_ms"]["mean"] == 200.0


# ------------------------------------------------------------------ engine wiring


def test_kis_search_reports_its_cost(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    outcome = engine.search_kis_detailed("a", top_k=10)
    cost = outcome.diagnostics()["cost"]
    assert cost["task"] == "kis"
    assert cost["channels"]["total_calls"] >= 1
    assert cost["total_wall_ms"] >= 0.0


def test_second_identical_query_costs_no_encoder_call(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    first = engine.search_kis_detailed("a", top_k=10).diagnostics()["cost"]
    second = engine.search_kis_detailed("a", top_k=10).diagnostics()["cost"]
    assert first["text"]["encoder_calls"] >= 1
    assert second["text"]["encoder_calls"] == 0
    assert second["text"]["cache_hits"] >= 1


def test_trake_cost_counts_every_event_retrieval(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    cost = engine.search_trake_detailed(["a", "b"], max_results=5).diagnostics["cost"]
    assert cost["task"] == "trake"
    assert cost["channels"]["total_calls"] >= 2


def test_refinement_off_means_zero_decoded_frames(tmp_path) -> None:
    engine, _, _ = build_engine(tmp_path)
    cost = engine.search_kis_detailed("a", top_k=10).diagnostics()["cost"]
    assert cost["video"]["frames_decoded"] == 0
    assert cost["image_embedding"]["computed"] == 0


def test_non_visual_backend_reports_zero_vlm_calls(tmp_path) -> None:
    """A mock that reasons over text is not a VLM call and must not be counted as one."""
    engine, _, _ = build_engine(tmp_path)
    _, info = engine.answer_qa("a", "what colour?", top_k=5)
    cost = info["diagnostics"]["cost"]
    assert cost["qa"]["vlm_calls"] == 0
    assert cost["qa"]["vlm_images"] == 0
    assert info["diagnostics"]["vlm_budget"]["backend_visual_capable"] is False


# ------------------------------------------------------------- three-axis reporting


def variant(name, **kwargs):
    from evaluation.pareto import VariantReport

    return VariantReport(name=name, cache_fingerprint="same", queries=6, **kwargs)


def test_dominance_needs_no_worse_everywhere_and_better_somewhere() -> None:
    from evaluation.pareto import dominates

    fast = variant("fast", efficiency={"p50_latency_ms": 100.0}, cost={"vlm_calls_per_query": 1.0})
    slow = variant("slow", efficiency={"p50_latency_ms": 200.0}, cost={"vlm_calls_per_query": 1.0})
    assert dominates(fast, slow, ("p50_latency_ms", "vlm_calls_per_query"))
    assert not dominates(slow, fast, ("p50_latency_ms", "vlm_calls_per_query"))


def test_a_trade_off_dominates_nothing() -> None:
    from evaluation.pareto import dominates, pareto_front

    cheap = variant("cheap", quality={"Final Score": 0.4}, efficiency={"p50_latency_ms": 50.0})
    good = variant("good", quality={"Final Score": 0.6}, efficiency={"p50_latency_ms": 500.0})
    metrics = ("Final Score", "p50_latency_ms")
    assert not dominates(cheap, good, metrics)
    assert not dominates(good, cheap, metrics)
    assert set(pareto_front([cheap, good], metrics)) == {"cheap", "good"}


def test_quality_axis_is_absent_without_ground_truth() -> None:
    from evaluation.pareto import NO_GT_NOTE, build_report

    report = build_report([variant("a", efficiency={"p50_latency_ms": 10.0})])
    assert report["has_quality_axis"] is False
    assert report["note"] == NO_GT_NOTE
    assert report["variants"][0]["quality"] is None


def test_variants_from_different_caches_are_not_comparable() -> None:
    from evaluation.pareto import VariantReport, build_report

    a = VariantReport(name="a", cache_fingerprint="one", queries=3)
    b = VariantReport(name="b", cache_fingerprint="two", queries=3)
    report = build_report([a, b])
    assert report["comparable"] is False
    assert "NOT comparable" in report["comparability_warning"]


def test_report_writes_json_and_csv(tmp_path) -> None:
    from evaluation.pareto import build_report, write_report

    report = build_report([variant("a", efficiency={"p50_latency_ms": 10.0})])
    paths = write_report(report, tmp_path)
    assert paths["json"].is_file() and paths["csv"].is_file()
    assert "p50_latency_ms" in paths["csv"].read_text(encoding="utf-8")


def test_variant_from_costs_averages_per_query() -> None:
    from evaluation.pareto import variant_from_costs

    a, b = QueryCost(), QueryCost()
    a.add_decode(decoded=10)
    b.add_decode(decoded=20)
    row = variant_from_costs("v", costs=[a, b], latencies_ms=[100.0, 300.0])
    assert row.cost["decoded_frames_per_query"] == 15.0
    assert row.efficiency["warm_mean_ms"] == 200.0
    assert row.has_quality is False
