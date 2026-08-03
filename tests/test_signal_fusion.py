from aic2026.fusion import ABLATIONS, FusionConfig, RankedCandidate, fuse_candidates, object_match_score


def candidate(frame, dense=0, obj=0, meta=0, sparse=0):
    return RankedCandidate("V", str(frame), float(frame), None, dense, obj, meta, sparse)


def test_confidence_aware_object_match_and_plural_normalization() -> None:
    score, labels = object_match_score("two cars", [
        {"label": "Car", "confidence": 0.9},
        {"label": "cars", "confidence": 0.1},
    ], confidence_threshold=0.2)
    assert score > 0 and labels == ("car",)


def test_weighted_rrf_and_adaptive_fusion_are_available() -> None:
    items = [candidate(1, dense=1), candidate(2, obj=1, sparse=1)]
    weighted = fuse_candidates("car", items, FusionConfig(method="weighted", adaptive=False))
    assert len(weighted.candidates) == 2
    rrf = fuse_candidates("CAR 123", items, FusionConfig(method="rrf", adaptive=True))
    assert rrf.weights["sparse"] > FusionConfig().sparse_weight
    assert {"clip_only", "rrf_full", "adaptive_full"} <= set(ABLATIONS)
