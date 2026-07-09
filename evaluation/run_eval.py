"""Đánh giá end-to-end (CLAUDE.md Mục 9, Phase 9).

Tạo một bộ test CÓ NHÃN (offline), chạy qua pipeline retrieval thật (coarse ± fine
rerank) cho KIS/AVS và VQA module cho hỏi-đáp, rồi tính các metric ở evaluation/metrics
và IN BÁO CÁO. Đây là hiện thực DoD Phase 9 ("có báo cáo số liệu trên bộ test end-to-end").

Bộ test dùng dữ liệu mock (Mục 1.5) — khi có ground-truth thật của BTC, chỉ cần thay
hàm build_* bằng loader dữ liệu thật, phần đo và báo cáo giữ nguyên.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation import metrics
from ingestion.build_index import KeyframeIndex
from ingestion.build_records import searchable_text
from ingestion.schemas import KeyframeRecord, RawKeyframe
from ingestion.build_records import IngestionPipeline
from retrieval.coarse_retriever import CoarseRetriever
from retrieval.fine_rerank import FineReranker, MockReranker, RerankConfig
from retrieval.vqa_module import VqaModule

CATEGORIES = ["mèo", "chó", "xe hơi", "bông hoa", "bánh kem"]


# ------------------------------------------------------------- build datasets
def build_ir_dataset(n_per_category: int = 6):
    """Sinh records + query KIS (1 đáp án) + query AVS (nhiều đáp án cùng category)."""
    raws: list[RawKeyframe] = []
    kis_queries: list[tuple[str, str]] = []          # (query, target_id)
    avs_relevant: dict[str, set[str]] = {c: set() for c in CATEGORIES}
    idx = 0
    for cat in CATEGORIES:
        for _ in range(n_per_category):
            uid = f"item{idx}"
            kf_id = f"gt/{idx}"
            raws.append(
                RawKeyframe(id=kf_id, video_id=f"vid{idx % 5}", timestamp=float(idx),
                            objects=[cat, uid])
            )
            kis_queries.append((f"tìm ảnh có vật thể {uid}", kf_id))
            avs_relevant[cat].add(kf_id)
            idx += 1
    records = IngestionPipeline().build(raws)
    avs_queries = [(f"tìm tất cả ảnh có {cat}", avs_relevant[cat]) for cat in CATEGORIES]
    return records, kis_queries, avs_queries


def build_vqa_dataset() -> list[tuple[str, list[KeyframeRecord], str]]:
    """Vài bộ (câu hỏi, records, đáp án chuẩn) cho VQA."""
    def rec(kf_id, ts, cap, objs):
        return KeyframeRecord(id=kf_id, video_id="V", timestamp=ts,
                              clip_embedding=np.zeros(4, dtype=np.float32),
                              objects=objs, llm_caption=cap)
    return [
        ("Có bao nhiêu ngọn nến trên bánh?",
         [rec("a", 5.0, "Chiếc bánh có 5 ngọn nến.", ["nến"])], "5"),
        ("Có bao nhiêu ngọn nến?",
         [rec("b", 5.0, "Bàn có ba ngọn nến nhỏ.", ["nến"])], "3"),
        ("Ai là người tặng quà?",
         [rec("c", 5.0, "Người đàn ông áo xanh đang tặng quà cho cô gái.", ["quà"])],
         "người đàn ông áo xanh"),
    ]


# --------------------------------------------------------------- evaluations
def eval_kis(coarse, kis_queries, reranker=None, contexts=None, k_list=(1, 5)) -> dict:
    samples: list[metrics.IRSample] = []
    for query, target in kis_queries:
        cands = coarse.search(query_text=query, top_k=100)
        if reranker is not None:
            reranked = reranker.rerank(query, cands, contexts)
            retrieved = [c.keyframe_id for c in reranked]
        else:
            retrieved = [c.keyframe_id for c in cands]
        samples.append((retrieved, {target}))
    return {
        "top1": metrics.top_k_accuracy(samples, 1),
        "top5": metrics.top_k_accuracy(samples, 5),
        "mrr": metrics.mean_reciprocal_rank(samples),
        "n": len(samples),
    }


def eval_avs(coarse, avs_queries, k: int = 10) -> dict:
    samples: list[metrics.IRSample] = []
    for query, relevant in avs_queries:
        cands = coarse.search(query_text=query, top_k=100)
        retrieved = [c.keyframe_id for c in cands]
        samples.append((retrieved, relevant))
    return {
        f"recall@{k}": metrics.mean_recall_at_k(samples, k),
        "map": metrics.mean_average_precision(samples),
        f"ndcg@{k}": metrics.mean_ndcg_at_k(samples, k),
        "n": len(samples),
    }


def eval_vqa(vqa_dataset) -> dict:
    module = VqaModule()
    ems, f1s = [], []
    for question, records, gold in vqa_dataset:
        ans = module.answer(question, records)
        ems.append(metrics.exact_match(ans.answer, gold))
        f1s.append(metrics.token_f1(ans.answer, gold))
    return {
        "exact_match": sum(ems) / len(ems),
        "f1": sum(f1s) / len(f1s),
        "n": len(vqa_dataset),
    }


def run_eval(save_dir: str | Path | None = None) -> dict:
    """Chạy toàn bộ đánh giá, trả report dict + in ra màn hình."""
    records, kis_queries, avs_queries = build_ir_dataset()
    index = KeyframeIndex.build(records)
    coarse = CoarseRetriever(index)
    contexts = {r.id: searchable_text(r) for r in records}
    reranker = FineReranker(MockReranker(),
                            RerankConfig(max_candidates=100, final_top_k=20,
                                         early_stop=False))

    report = {
        "kis_coarse": eval_kis(coarse, kis_queries),
        "kis_reranked": eval_kis(coarse, kis_queries, reranker, contexts),
        "avs": eval_avs(coarse, avs_queries),
        "vqa": eval_vqa(build_vqa_dataset()),
    }
    _print_report(report)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "phase9_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[Đã lưu báo cáo] {out}")
    return report


def _print_report(report: dict) -> None:
    print("=" * 60)
    print("BÁO CÁO ĐÁNH GIÁ END-TO-END (Phase 9)")
    print("=" * 60)
    k = report["kis_coarse"]
    kr = report["kis_reranked"]
    print(f"\n[KIS] (n={k['n']})")
    print(f"  Coarse    : Top-1={k['top1']:.3f}  Top-5={k['top5']:.3f}  MRR={k['mrr']:.3f}")
    print(f"  +Rerank   : Top-1={kr['top1']:.3f}  Top-5={kr['top5']:.3f}  MRR={kr['mrr']:.3f}")
    a = report["avs"]
    print(f"\n[AVS] (n={a['n']})")
    print(f"  Recall@10={a['recall@10']:.3f}  mAP={a['map']:.3f}  nDCG@10={a['ndcg@10']:.3f}")
    v = report["vqa"]
    print(f"\n[VQA] (n={v['n']})")
    print(f"  Exact Match={v['exact_match']:.3f}  F1={v['f1']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_eval(save_dir="evaluation/benchmarks")
