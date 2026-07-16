"""Độ đo đánh giá (CLAUDE.md Mục 9, Phase 9).

Cung cấp các metric chuẩn của Information Retrieval + QA để chấm hệ thống:
  - KIS: Top-1 / Top-5 accuracy, MRR (đúng ground-truth ở vị trí đầu là quan trọng nhất).
  - AVS: Recall@K, mAP (mean Average Precision) — đánh giá cả danh sách, không chỉ Top-1.
  - VQA: Exact Match + token F1 (câu trả lời dạng số/tên).

Thiết kế theo cặp "per-query" + "aggregate" để dễ ghép vào run_eval và tái dùng. Thuần
numpy/Python, không phụ thuộc model/API.

Quy ước: `retrieved` = danh sách id đã xếp hạng (giảm dần theo điểm); `relevant` = tập
id đúng (ground-truth). Với KIS thường |relevant| = 1; với AVS |relevant| >= 1.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence


# =============================== IR metrics ===================================
def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Tỉ lệ item ĐÚNG nằm trong top-k: |relevant ∩ retrieved[:k]| / |relevant|.

    Với KIS (|relevant|=1) đây chính là hit@k (1.0 nếu đáp án lọt top-k)."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(relevant & topk) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 nếu CÓ ít nhất 1 item đúng trong top-k, ngược lại 0.0 (dùng cho Top-1/5 acc)."""
    relevant = set(relevant)
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1/rank của item đúng ĐẦU TIÊN (rank từ 1). 0.0 nếu không có item đúng nào."""
    relevant = set(relevant)
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Average Precision cho 1 query (nền tảng của mAP cho AVS).

    AP = trung bình precision tại mỗi vị trí có item đúng, chuẩn hoá theo |relevant|.
    Thưởng cho việc xếp các item đúng lên CAO trong danh sách."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


def dcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Discounted Cumulative Gain (gain nhị phân 0/1) tới vị trí k."""
    relevant = set(relevant)
    dcg = 0.0
    for i, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k (IDCG = DCG lý tưởng khi mọi item đúng xếp trên đầu)."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    # Clamp về [0,1]: nDCG theo định nghĩa luôn <= 1, nhưng phép chia float có thể
    # trả 1.0000000000000002 khi DCG == IDCG -> chặn để không vượt biên.
    return min(1.0, dcg_at_k(retrieved, relevant, k) / idcg)


# =============================== QA metrics ===================================
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(text: str) -> str:
    """Chuẩn hoá câu trả lời trước khi so khớp: lowercase, bỏ dấu câu, gộp khoảng trắng.

    Giúp Exact Match/F1 không bị lệch bởi khác biệt vô nghĩa ('5.' vs '5', hoa/thường)."""
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(prediction: str, gold: str) -> float:
    """1.0 nếu câu trả lời (đã chuẩn hoá) trùng khớp hoàn toàn, ngược lại 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """F1 trên tập token (đo khớp một phần — hữu ích khi câu trả lời là cụm từ)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    gold_count = {t: gold_tokens.count(t) for t in set(gold_tokens)}
    pred_count = {t: pred_tokens.count(t) for t in set(pred_tokens)}
    num_same = sum(min(pred_count[t], gold_count.get(t, 0)) for t in pred_count)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ============================ Aggregate (nhiều query) =========================
# Mỗi "sample" IR = (retrieved_ids, relevant_ids).
IRSample = tuple[Sequence[str], Iterable[str]]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mean_recall_at_k(samples: list[IRSample], k: int) -> float:
    return _mean([recall_at_k(r, rel, k) for r, rel in samples])


def top_k_accuracy(samples: list[IRSample], k: int) -> float:
    """Tỉ lệ query có item đúng trong top-k (Top-1/Top-5 accuracy cho KIS)."""
    return _mean([hit_at_k(r, rel, k) for r, rel in samples])


def mean_reciprocal_rank(samples: list[IRSample]) -> float:
    return _mean([reciprocal_rank(r, rel) for r, rel in samples])


def mean_average_precision(samples: list[IRSample]) -> float:
    return _mean([average_precision(r, rel) for r, rel in samples])


def mean_ndcg_at_k(samples: list[IRSample], k: int) -> float:
    return _mean([ndcg_at_k(r, rel, k) for r, rel in samples])


# ========================= So sánh cấu hình (thống kê) ========================
# VÌ SAO CẦN MỤC NÀY: đã có kết luận SAI vì đọc nhiễu thành tín hiệu. Bảng rerank_pool
# ghi "pool 8 (0.510) tốt hơn pool 32 (0.471)" — chênh 0.039 trên 51 nhãn = ĐÚNG 2
# QUERY, trong khi sai số chuẩn của một tỉ lệ ở n=51 là ~0.07 (±7 query). Không kết
# luận được gì. Từ nay mọi so sánh cấu hình phải đi qua đây.


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy 95% cho MỘT tỉ lệ (hit@k), theo Wilson.

    Dùng Wilson thay vì công thức normal (p ± z·√(p(1-p)/n)): ở n nhỏ và p gần 0/1,
    normal cho khoảng vượt ra ngoài [0,1] và phủ sai. Wilson luôn nằm trong [0,1].
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_paired(a_hits: Sequence[float], b_hits: Sequence[float]) -> dict:
    """So 2 cấu hình trên CÙNG bộ query (theo cặp) — mạnh hơn so 2 tỉ lệ độc lập.

    Ý tưởng: các query mà CẢ HAI cùng đúng (hoặc cùng sai) không mang thông tin về
    việc cái nào hơn — chúng chỉ làm loãng phép đo. Chỉ các cặp BẤT ĐỒNG mới đáng kể:
      b01 = A sai, B đúng      b10 = A đúng, B sai
    Nếu 2 cấu hình thực sự như nhau thì b01 ~ Binomial(b01+b10, 0.5). p-value 2 phía
    tính CHÍNH XÁC bằng binomial (không xấp xỉ chi-square — số bất đồng thường < 25,
    chỗ xấp xỉ đó sai).

    Trả p_value + số cặp bất đồng. p_value >= 0.05 nghĩa là "CHƯA phân biệt được",
    KHÔNG phải "hai cái như nhau" — với n nhỏ, thiếu bằng chứng ≠ bằng chứng phủ định.
    """
    if len(a_hits) != len(b_hits):
        raise ValueError("So theo cặp thì 2 danh sách phải cùng bộ query, cùng thứ tự.")
    b01 = sum(1 for a, b in zip(a_hits, b_hits) if a < b)   # A sai, B đúng
    b10 = sum(1 for a, b in zip(a_hits, b_hits) if a > b)   # A đúng, B sai
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        # P(X <= min) * 2, X ~ Binomial(n_disc, 0.5); chặn tại 1.0.
        m = min(b01, b10)
        tail = sum(math.comb(n_disc, i) for i in range(m + 1)) / (2 ** n_disc)
        p = min(1.0, 2 * tail)
    return {
        "b01_only_b_correct": b01,
        "b10_only_a_correct": b10,
        "n_discordant": n_disc,
        "p_value": round(p, 4),
        "significant_at_05": p < 0.05,
    }


def compare_configs(a_hits: Sequence[float], b_hits: Sequence[float]) -> dict:
    """Báo cáo so sánh đầy đủ: 2 tỉ lệ + KTC Wilson + kiểm định cặp McNemar."""
    n = len(a_hits)
    sa, sb = int(sum(a_hits)), int(sum(b_hits))
    return {
        "n_queries": n,
        "a_rate": round(sa / n, 4) if n else 0.0,
        "b_rate": round(sb / n, 4) if n else 0.0,
        "a_ci95": tuple(round(x, 4) for x in wilson_interval(sa, n)),
        "b_ci95": tuple(round(x, 4) for x in wilson_interval(sb, n)),
        "delta": round((sb - sa) / n, 4) if n else 0.0,
        **mcnemar_paired(a_hits, b_hits),
    }
