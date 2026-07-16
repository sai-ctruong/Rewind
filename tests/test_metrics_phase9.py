"""Unit test Phase 9 — evaluation/metrics + run_eval (CLAUDE.md Mục 8, Mục 9).

DoD Phase 9: "Có báo cáo số liệu trên bộ test end-to-end". Ngoài smoke test cho
run_eval, ta kiểm TÍNH ĐÚNG của từng metric bằng các ví dụ có giá trị đã tính tay
(không dùng dữ liệu 'sạch' để tránh mọi metric = 1.0 che lỗi).
"""
from __future__ import annotations

import math

import pytest

from evaluation import metrics
from evaluation.metrics import mcnemar_paired, wilson_interval
from evaluation.run_eval import run_eval


# ------------------------------- IR metrics ----------------------------------
def test_recall_at_k_partial() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "e"}          # 3 đúng, top-3 chứa a & c
    assert metrics.recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_hit_at_k() -> None:
    assert metrics.hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert metrics.hit_at_k(["x", "a"], {"a"}, 1) == 0.0   # a nằm ngoài top-1


def test_reciprocal_rank() -> None:
    assert metrics.reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)
    assert metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0  # không có item đúng


def test_average_precision_known_value() -> None:
    # retrieved a(hit),b,c(hit),d ; relevant {a,c} -> AP = (1/1 + 2/3)/2 = 0.8333
    ap = metrics.average_precision(["a", "b", "c", "d"], {"a", "c"})
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)


def test_ndcg_at_k_known_value() -> None:
    # relevant {b}; b ở rank 2 -> DCG=1/log2(3); IDCG=1/log2(2)=1
    ndcg = metrics.ndcg_at_k(["a", "b"], {"b"}, 2)
    assert ndcg == pytest.approx(1.0 / math.log2(3))


def test_empty_relevant_is_zero() -> None:
    assert metrics.recall_at_k(["a"], set(), 1) == 0.0
    assert metrics.average_precision(["a"], set()) == 0.0
    assert metrics.ndcg_at_k(["a"], set(), 1) == 0.0


# ------------------------------- QA metrics ----------------------------------
def test_normalize_and_exact_match() -> None:
    assert metrics.exact_match("5.", "5") == 1.0            # bỏ dấu câu
    assert metrics.exact_match("Người ĐÀN ông", "người đàn ông") == 1.0  # hoa/thường
    assert metrics.exact_match("6", "5") == 0.0


def test_token_f1_partial() -> None:
    # pred 5 token, gold 3 token, chung 3 -> P=3/5, R=1 -> F1=0.75
    f1 = metrics.token_f1("người đàn ông áo xanh", "người đàn ông")
    assert f1 == pytest.approx(0.75)


def test_token_f1_no_overlap() -> None:
    assert metrics.token_f1("mèo", "chó") == 0.0


# ------------------------------- aggregates ----------------------------------
def test_aggregates() -> None:
    samples = [
        (["a", "b"], {"a"}),   # RR=1, hit@1=1
        (["x", "c"], {"c"}),   # RR=0.5, hit@1=0
    ]
    assert metrics.mean_reciprocal_rank(samples) == pytest.approx(0.75)
    assert metrics.top_k_accuracy(samples, 1) == pytest.approx(0.5)
    assert metrics.mean_recall_at_k(samples, 2) == pytest.approx(1.0)


# ------------------------------- run_eval ------------------------------------
def test_run_eval_report_structure(tmp_path, capsys) -> None:
    report = run_eval(save_dir=tmp_path)
    # Đủ 4 nhóm bài toán.
    assert set(report.keys()) == {"kis_coarse", "kis_reranked", "avs", "vqa"}
    # Mọi số liệu nằm trong [0, 1].
    for section in report.values():
        for key, val in section.items():
            if key == "n":
                continue
            assert 0.0 <= val <= 1.0, f"{key}={val} ngoài [0,1]"
    # Đã ghi file báo cáo (DoD: "ghi log" số liệu).
    assert (tmp_path / "phase9_report.json").is_file()
    # Có in bảng báo cáo ra màn hình.
    assert "BÁO CÁO ĐÁNH GIÁ" in capsys.readouterr().out


def test_run_eval_metrics_reasonable(tmp_path) -> None:
    report = run_eval(save_dir=tmp_path)
    # Trên dữ liệu mock có token độc nhất, retrieval phải bắt được đáp án.
    assert report["kis_coarse"]["top1"] >= 0.8
    assert report["avs"]["recall@10"] >= 0.8
    assert report["vqa"]["exact_match"] >= 0.6


# ===================== So sánh cấu hình (thống kê) ===========================
def test_wilson_interval_stays_inside_0_1_at_extremes() -> None:
    """Lý do dùng Wilson thay công thức normal: p=1.0, n nhỏ thì normal cho
    p ± 0 => (1.0, 1.0) — khẳng định 'chắc chắn 100%' từ 5 mẫu. Wilson thì không."""
    lo, hi = wilson_interval(5, 5)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 1.0, "5/5 mẫu KHÔNG được phép cho cận dưới = 1.0"


def test_mcnemar_ignores_agreeing_queries() -> None:
    """Query mà cả 2 cấu hình cùng đúng/cùng sai không mang thông tin so sánh.
    Thêm bao nhiêu query đồng thuận cũng không được làm p-value đổi."""
    a = [1, 0, 1, 0]
    b = [0, 1, 1, 0]
    p_small = mcnemar_paired(a, b)["p_value"]
    p_padded = mcnemar_paired(a + [1] * 50, b + [1] * 50)["p_value"]
    assert p_small == p_padded


def test_mcnemar_detects_consistent_winner() -> None:
    """B thắng ở mọi cặp bất đồng -> phải phát hiện được là có ý nghĩa."""
    a = [0] * 10 + [1] * 10
    b = [1] * 10 + [1] * 10          # B đúng hết; 10 cặp bất đồng, B thắng cả 10
    r = mcnemar_paired(a, b)
    assert r["b01_only_b_correct"] == 10 and r["b10_only_a_correct"] == 0
    assert r["significant_at_05"]


def test_rerank_pool_conclusion_was_not_supported() -> None:
    """KHOÁ LẠI MỘT SAI LẦM ĐÃ MẮC: TASKS từng ghi 'pool 8 (0.510) tốt hơn pool 32
    (0.471)'. Chênh 0.039 trên 51 nhãn = 2 query. Kể cả ở kịch bản THUẬN LỢI NHẤT cho
    kết luận đó (mọi cặp bất đồng đều do pool 8 thắng), nó vẫn KHÔNG đạt mức ý nghĩa.
    """
    n = 51
    # Kịch bản tốt nhất cho giả thuyết: chỉ 2 cặp bất đồng, pool 8 thắng cả 2.
    pool8 = [1] * 26 + [0] * 25
    pool32 = [1] * 24 + [0] * 2 + [0] * 25
    assert sum(pool8) - sum(pool32) == 2
    r = mcnemar_paired(pool32, pool8)
    assert not r["significant_at_05"], "2 query chênh lệch KHÔNG được coi là bằng chứng"
    assert len(pool8) == len(pool32) == n
