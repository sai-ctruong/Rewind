"""Unit test cho evaluation/bench_scale — benchmark quy mô ANN.

Chạy ở cỡ TÍ HON (vài nghìn vector, ít chiều) để test nhanh: kiểm ground-truth đúng
(so với numpy exact), đường cong efSearch có ý nghĩa (ef lớn -> recall không giảm), và
phần ngoại suy RAM tính đúng.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("psutil")

from evaluation.bench_scale import (  # noqa: E402
    SizeResult, bench_size, clustered_vectors, exact_topk, extrapolate, main,
    make_vectors, unit_vectors)


def _nn_sim(v: np.ndarray) -> float:
    """Độ giống trung bình với LÁNG GIỀNG GẦN NHẤT — thước đo 'có cụm hay không'."""
    s = v @ v.T
    np.fill_diagonal(s, -1.0)
    return float(s.max(axis=1).mean())


def test_unit_vectors_are_normalized() -> None:
    v = unit_vectors(50, 16, seed=1)
    assert v.shape == (50, 16) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_clustered_vectors_actually_cluster_at_high_dim() -> None:
    """HỒI QUY (bug thật): nhiễu σ·N(0,I) ở dim chiều có độ dài ≈ σ·√dim. Quên chia cho
    √dim thì ở 768 chiều nhiễu (~9.7) át tâm (1) -> dữ liệu 'clustered' thực chất NGẪU
    NHIÊN ĐỀU, và cả benchmark recall trở nên vô nghĩa mà không ai biết."""
    c = clustered_vectors(600, 768, seed=1)
    r = unit_vectors(600, 768, seed=1)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-5)
    assert _nn_sim(c) > 0.6          # có cụm rõ (giống embedding thật)
    assert _nn_sim(r) < 0.3          # ngẫu nhiên đều: gần như trực giao
    assert _nn_sim(c) > _nn_sim(r) + 0.4


def test_make_vectors_dispatch_and_guard() -> None:
    assert _nn_sim(make_vectors(400, 256, dist="clustered")) > \
           _nn_sim(make_vectors(400, 256, dist="random"))
    with pytest.raises(ValueError):
        make_vectors(10, 8, dist="khong-ton-tai")


def test_exact_topk_matches_bruteforce() -> None:
    """Ground-truth phải ĐÚNG — nếu sai thì mọi số recall đều vô nghĩa."""
    mat, q = unit_vectors(200, 8, seed=2), unit_vectors(5, 8, seed=3)
    got = exact_topk(mat, q, k=4)
    want = np.argsort(-(q @ mat.T), axis=1)[:, :4]
    assert np.array_equal(got, want)


def test_exact_topk_batches_do_not_change_result() -> None:
    """Chia lô (để tiết kiệm RAM) không được làm đổi kết quả."""
    mat, q = unit_vectors(500, 8, seed=4), unit_vectors(40, 8, seed=5)
    assert np.array_equal(exact_topk(mat, q, k=3),
                          np.argsort(-(q @ mat.T), axis=1)[:, :3])


def test_bench_size_reports_ram_and_ef_curve() -> None:
    q = unit_vectors(20, 32, seed=6)
    r = bench_size(2000, 32, q, k=5, m=16, ef_construction=40, ef_values=(8, 64))
    # build_seconds làm tròn 2 số lẻ -> ở cỡ tí hon có thể ra 0.0; chỉ cần không âm.
    assert r.n_vectors == 2000 and r.build_seconds >= 0
    # báo cáo làm tròn tới 0.1 MB (đủ cho cỡ thật hàng trăm MB) -> so cùng độ chính xác
    assert r.matrix_ram_mb == round(2000 * 32 * 4 / 2**20, 1)
    assert [e.ef_search for e in r.ef_curve] == [8, 64]
    for e in r.ef_curve:
        assert 0.0 <= e.recall_at_k <= 1.0 and e.latency_ms_p50 >= 0
        assert 0.0 <= e.score_ratio <= 1.001      # không thể tốt hơn exact


def test_score_ratio_forgives_near_ties() -> None:
    """recall khớp-ID phạt cả khi ANN trả hàng xóm 'khác ID nhưng tốt ngang'. score_ratio
    phải cao hơn recall trong tình huống đó -> nhìn được CHẤT LƯỢNG thật."""
    q = make_vectors(20, 64, seed=9, dist="clustered")
    r = bench_size(3000, 64, q, k=10, m=16, ef_construction=40, ef_values=(8,),
                   dist="clustered")
    e = r.ef_curve[0]
    assert e.score_ratio > e.recall_at_k


def test_higher_efsearch_does_not_hurt_recall() -> None:
    """Tính chất cốt lõi của HNSW: efSearch lớn hơn -> quét rộng hơn -> recall >= .
    Nếu vi phạm thì benchmark (hoặc cách đo) đang sai."""
    q = unit_vectors(30, 32, seed=7)
    r = bench_size(3000, 32, q, k=5, m=16, ef_construction=40, ef_values=(4, 128))
    lo, hi = r.ef_curve[0].recall_at_k, r.ef_curve[1].recall_at_k
    assert hi >= lo


def test_extrapolate_scales_linearly() -> None:
    r = SizeResult(n_vectors=100_000, dim=768, build_seconds=100.0,
                   index_ram_mb=350.0, matrix_ram_mb=293.0, bytes_per_vector=3670.0)
    e = extrapolate([r], 1_000_000)
    assert e["target_vectors"] == 1_000_000
    assert e["index_ram_gb"] == pytest.approx(350.0 * 10 / 1024, rel=0.01)
    assert e["matrix_ram_gb"] == pytest.approx(1_000_000 * 768 * 4 / 2**30, rel=0.01)
    # total = index + ma trận gốc (KeyframeIndex giữ CẢ HAI)
    assert e["total_ram_gb"] == pytest.approx(e["index_ram_gb"] + e["matrix_ram_gb"], rel=0.01)
    assert e["build_hours_lower_bound"] == pytest.approx(100.0 * 10 / 3600, rel=0.01)


def test_extrapolate_empty_is_safe() -> None:
    assert extrapolate([], 1_000_000) == {}


def test_main_writes_report(tmp_path) -> None:
    out = tmp_path / "scale.json"
    payload = main(sizes=(1000,), dim=16, n_queries=10, k=3, m=8,
                   ef_construction=20, ef_values=(8, 32), out=out)
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["config"]["dim"] == 16
    assert len(saved["sizes"]) == 1 and saved["sizes"][0]["n_vectors"] == 1000
    assert saved["extrapolation_1m"]["target_vectors"] == 1_000_000
    assert payload["caveat"]                      # có ghi rõ giới hạn phép đo
