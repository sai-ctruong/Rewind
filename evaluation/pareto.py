"""Three-axis reporting: quality, efficiency, cost — kept separate on purpose.

The organizer's Final Score is one axis. Latency is another. Compute is a third. A single
blended "goodness" number hides exactly the trade-off this research is about, so nothing
here collapses the axes: a configuration is reported as dominating another only when it
is no worse on every compared axis and strictly better on at least one.

Quality columns exist ONLY when ground truth was supplied. Without it the report carries
efficiency and cost columns and says so — an empty quality column is a missing
measurement, never a zero.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# Higher is better.
QUALITY_METRICS = ("R@1", "R@5", "R@20", "R@50", "R@100", "Final Score")
# Lower is better.
EFFICIENCY_METRICS = ("p50_latency_ms", "p95_latency_ms", "first_query_ms", "warm_mean_ms")
COST_METRICS = (
    "decoded_frames_per_query",
    "image_embeddings_per_query",
    "vlm_calls_per_query",
    "vlm_images_per_query",
    "text_encoder_calls_per_query",
    "channel_calls_per_query",
    "cost_proxy_per_query",
    "peak_rss_mb",
    "api_cost_per_query_usd",
)

NO_GT_NOTE = (
    "No ground truth was supplied, so no quality column exists. Latency and cost are "
    "measured; nothing here says a configuration retrieves better."
)


@dataclass
class VariantReport:
    """One configuration's measurements across the three axes."""

    name: str
    quality: dict[str, float] = field(default_factory=dict)
    efficiency: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    structural: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    cache_fingerprint: str = ""
    queries: int = 0
    ground_truth: Optional[dict[str, Any]] = None

    @property
    def has_quality(self) -> bool:
        return bool(self.quality)

    def value(self, metric: str) -> Optional[float]:
        for bucket in (self.quality, self.efficiency, self.cost):
            if metric in bucket:
                return float(bucket[metric])
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "queries": self.queries,
            "config_hash": self.config_hash,
            "cache_fingerprint": self.cache_fingerprint,
            "quality": dict(self.quality) or None,
            "efficiency": dict(self.efficiency),
            "cost": dict(self.cost),
            "structural": dict(self.structural),
            "ground_truth": self.ground_truth,
        }


def _better(metric: str, left: float, right: float) -> bool:
    """Is `left` strictly better than `right` on this metric?"""
    if metric in QUALITY_METRICS:
        return left > right
    return left < right  # efficiency and cost: lower is better


def dominates(left: VariantReport, right: VariantReport, metrics: Sequence[str]) -> bool:
    """Pareto dominance over the given metrics, and only over those.

    `left` dominates `right` when it is no worse on every compared metric and strictly
    better on at least one. A metric missing from either side is skipped rather than
    guessed, because comparing a measured value against an absent one is not a comparison.
    """
    strictly_better = False
    compared = 0
    for metric in metrics:
        a, b = left.value(metric), right.value(metric)
        if a is None or b is None:
            continue
        compared += 1
        if _better(metric, b, a):
            return False
        if _better(metric, a, b):
            strictly_better = True
    return bool(compared) and strictly_better


def pareto_front(variants: Sequence[VariantReport], metrics: Sequence[str]) -> list[str]:
    """Names of the variants no other variant dominates."""
    return [
        item.name
        for item in variants
        if not any(other is not item and dominates(other, item, metrics) for other in variants)
    ]


def comparable(variants: Sequence[VariantReport]) -> tuple[bool, str]:
    """Two variants are comparable only if they searched the same index."""
    fingerprints = {item.cache_fingerprint for item in variants if item.cache_fingerprint}
    if len(fingerprints) > 1:
        return False, (
            "Variants ran against different caches; their measurements are NOT comparable."
        )
    counts = {item.queries for item in variants}
    if len(counts) > 1:
        return False, (
            f"Variants answered different query counts ({sorted(counts)}); per-query "
            "averages are not comparable."
        )
    return True, ""


def build_report(
    variants: Sequence[VariantReport],
    *,
    ground_truth: Optional[dict[str, Any]] = None,
    axes: Optional[dict[str, Sequence[str]]] = None,
) -> dict[str, Any]:
    """Assemble the machine-readable three-axis report."""
    ok, warning = comparable(variants)
    has_quality = any(item.has_quality for item in variants)
    axes = axes or {
        "quality_vs_latency": ("Final Score", "p50_latency_ms"),
        "quality_vs_decoded_frames": ("Final Score", "decoded_frames_per_query"),
        "quality_vs_vlm_calls": ("Final Score", "vlm_calls_per_query"),
        "quality_vs_memory": ("Final Score", "peak_rss_mb"),
    }
    fronts: dict[str, list[str]] = {}
    for label, metrics in axes.items():
        usable = [m for m in metrics if not (m in QUALITY_METRICS and not has_quality)]
        if not usable:
            continue
        fronts[label] = pareto_front(variants, usable)
    return {
        "variants": [item.to_dict() for item in variants],
        "comparable": ok,
        **({"comparability_warning": warning} if warning else {}),
        "has_quality_axis": has_quality,
        "ground_truth": ground_truth,
        "pareto_fronts": fronts,
        "note": (
            "Pareto dominance only: a variant is listed as dominating another when it is "
            "no worse on every compared axis and strictly better on at least one. No "
            "single best configuration is declared."
        )
        if has_quality
        else NO_GT_NOTE,
    }


def write_report(report: dict[str, Any], directory: str | Path, *, stem: str = "three_axis") -> dict[str, Path]:
    """Write the report as JSON plus a flat CSV suitable for plotting."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    columns: list[str] = ["variant", "queries", "config_hash", "cache_fingerprint"]
    for item in report["variants"]:
        for bucket in ("quality", "efficiency", "cost"):
            for key in (item.get(bucket) or {}):
                if key not in columns:
                    columns.append(key)
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in report["variants"]:
            row: dict[str, Any] = {
                "variant": item["name"],
                "queries": item["queries"],
                "config_hash": item["config_hash"],
                "cache_fingerprint": item["cache_fingerprint"],
            }
            for bucket in ("quality", "efficiency", "cost"):
                row.update(item.get(bucket) or {})
            writer.writerow(row)
    return {"json": json_path, "csv": csv_path}


def variant_from_costs(
    name: str,
    *,
    costs: Iterable[Any],
    latencies_ms: Sequence[float],
    quality: Optional[dict[str, float]] = None,
    structural: Optional[dict[str, Any]] = None,
    config_hash: str = "",
    cache_fingerprint: str = "",
    ground_truth: Optional[dict[str, Any]] = None,
) -> VariantReport:
    """Build one variant row from per-query `QueryCost` objects and wall times."""
    items = list(costs)
    count = max(1, len(items))
    ordered = sorted(latencies_ms)
    efficiency: dict[str, float] = {}
    if ordered:
        efficiency = {
            "p50_latency_ms": round(ordered[len(ordered) // 2], 1),
            "p95_latency_ms": round(ordered[min(len(ordered) - 1, int(-(-95 * len(ordered) // 100)) - 1)], 1),
            "warm_mean_ms": round(sum(ordered) / len(ordered), 1),
            "max_latency_ms": round(ordered[-1], 1),
        }
    cost = {
        "decoded_frames_per_query": round(sum(c.video_frames_decoded for c in items) / count, 3),
        "image_embeddings_per_query": round(
            sum(c.image_embeddings_computed for c in items) / count, 3
        ),
        "vlm_calls_per_query": round(sum(c.qa_vlm_calls for c in items) / count, 3),
        "vlm_images_per_query": round(sum(c.qa_vlm_images for c in items) / count, 3),
        "text_encoder_calls_per_query": round(
            sum(c.text_encoder_calls for c in items) / count, 3
        ),
        "channel_calls_per_query": round(sum(c.total_channel_calls for c in items) / count, 3),
        "cost_proxy_per_query": round(sum(c.cost_proxy() for c in items) / count, 3),
    }
    peaks = [c.peak_process_rss_mb for c in items if c.peak_process_rss_mb]
    if peaks:
        cost["peak_rss_mb"] = max(peaks)
    return VariantReport(
        name=name,
        quality=dict(quality or {}),
        efficiency=efficiency,
        cost=cost,
        structural=dict(structural or {}),
        config_hash=config_hash,
        cache_fingerprint=cache_fingerprint,
        queries=len(items),
        ground_truth=ground_truth,
    )


__all__ = [
    "COST_METRICS",
    "EFFICIENCY_METRICS",
    "NO_GT_NOTE",
    "QUALITY_METRICS",
    "VariantReport",
    "build_report",
    "comparable",
    "dominates",
    "pareto_front",
    "variant_from_costs",
    "write_report",
]
