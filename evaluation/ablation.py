"""Ablation registry; comparisons are valid only on the same labels and split."""
ABLATIONS = {
    "retrieval": ("clip_only", "clip_objects", "clip_metadata", "clip_sparse", "rrf_full", "adaptive_full"),
    "ranking": ("raw_top100", "diversified_top100"),
    "trake": ("independent_search", "hard_order_filter", "greedy_alignment", "dp_alignment", "dp_alignment_refine", "full_event_coverage_dp_refine"),
    "qa": ("single_frame", "temporal_window", "diverse_multi_frame", "multi_frame_plus_text_signals", "full_grounded_qa"),
}


def comparable(left: dict, right: dict) -> bool:
    return left.get("dataset") == right.get("dataset") and left.get("split") == right.get("split")
