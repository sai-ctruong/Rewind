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
