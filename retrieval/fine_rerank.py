"""Fine Rerank bằng LVLM (CLAUDE.md Mục 4.4, Phase 5).

VAI TRÒ: sau khi coarse + fusion cho top-K ứng viên (recall cao), tầng này dùng LVLM
"soi" từng ứng viên để tối ưu PRECISION — chấm độ khớp giữa ảnh keyframe và mô tả
query, rồi xếp lại hạng. Đây là tầng đắt (chậm) nên CHỈ chạy trên top-K nhỏ (Mục 3).

RÀNG BUỘC LÀ ĐỘ TRỄ, KHÔNG PHẢI CHI PHÍ (Mục 2.3, 4.4): BTC cấp API miễn phí, nên
giới hạn thực sự là TIME BUDGET của vòng thi. Vì vậy tầng này có:
  - `time_budget_s`: tổng ngân sách thời gian; dừng chấm khi sắp vượt (Mục 4.4).
  - Early stopping (Mục 11.1.4): dùng helper confidence-gap nội bộ
    — khi Top-1 đã vượt trội rõ rệt so với phần còn lại thì dừng sớm, khỏi tốn thêm
    lượt LVLM.
  - `max_candidates`: trần số ứng viên rerank (từ settings.yaml -> fine_rerank.top_k).

THIẾT KẾ (ABC + Mock + Claude lazy, Mục 1.5): MockReranker chấm bằng độ trùng token
giữa query và context text (offline, không API) — đủ để chạy/test toàn pipeline và đo
Top-1. ClaudeReranker (bản thật) chấm bằng vision qua Claude API.

Điểm rerank chuẩn hoá về [0, 1] để dùng chung ngưỡng early-stop với is_confident_enough.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from retrieval.coarse_retriever import Candidate


@dataclass
class RerankedCandidate:
    """Ứng viên sau rerank. `score` = điểm LVLM chuẩn hoá [0,1] (đặt tên `score` để
    tương thích duck-typing với helper confidence-gap)."""

    keyframe_id: str
    row: int
    score: float                 # điểm rerank [0,1]
    coarse_score: float          # điểm RRF từ coarse (giữ để truy vết)
    video_id: str
    timestamp: float
    explanation: Optional[str] = None


@dataclass
class RerankConfig:
    """Tham số rerank (khớp configs/settings.yaml -> fine_rerank / runtime).

    [PROVISIONAL] (Mục 11.3): time_budget_s và max_candidates phải suy từ giây/candidate
    ĐO THỰC NGHIỆM ở Phase 5, không đoán trước.
    """

    max_candidates: int = 100          # trần số ứng viên đưa vào LVLM (fine_rerank.top_k)
    final_top_k: int = 20              # số kết quả trả sau rerank
    time_budget_s: float = 20.0        # ngân sách độ trễ online (runtime.time_budget_seconds)
    early_stop: bool = True
    confidence_margin: float = 0.15    # gap Top1-Top2 để dừng sớm (early_stopping.confidence_margin)
    min_evaluated_before_stop: int = 5  # chấm tối thiểu bấy nhiêu trước khi cho phép dừng sớm


class Reranker(ABC):
    """Interface: chấm 1 ứng viên với query, trả (score∈[0,1], giải thích)."""

    @abstractmethod
    def score(self, query_text: str, keyframe_id: str, context: object) -> tuple[float, str]:
        ...


class FineReranker:
    """Điều phối rerank: áp time budget + early stop quanh một Reranker cụ thể."""

    def __init__(self, reranker: Reranker, config: Optional[RerankConfig] = None):
        self.reranker = reranker
        self.config = config or RerankConfig()

    def rerank(
        self,
        query_text: str,
        candidates: list[Candidate],
        context_by_id: dict[str, object],
        clock: Callable[[], float] = time.perf_counter,
    ) -> list[RerankedCandidate]:
        """Rerank top-K candidate. `context_by_id` ánh xạ keyframe_id -> ngữ cảnh cho
        reranker (text với Mock, đường dẫn ảnh với Claude). `clock` cho phép test bơm
        đồng hồ giả để kiểm time budget mà không phụ thuộc thời gian thực."""
        cfg = self.config
        # Coarse đã xếp hạng theo recall; chỉ rerank tối đa max_candidates đầu (Mục 4.4).
        pool = candidates[: cfg.max_candidates]

        scored: list[RerankedCandidate] = []
        start = clock()
        per_call_times: list[float] = []

        for i, cand in enumerate(pool):
            # --- Time budget guard (Mục 4.4): dừng nếu lượt kế tiếp có nguy cơ vượt.
            elapsed = clock() - start
            if per_call_times:
                est_next = sum(per_call_times) / len(per_call_times)
                if elapsed + est_next > cfg.time_budget_s:
                    break
            elif elapsed > cfg.time_budget_s:
                break

            t0 = clock()
            s, explain = self.reranker.score(
                query_text, cand.keyframe_id, context_by_id.get(cand.keyframe_id)
            )
            per_call_times.append(clock() - t0)

            scored.append(
                RerankedCandidate(
                    keyframe_id=cand.keyframe_id,
                    row=cand.row,
                    score=max(0.0, min(1.0, float(s))),
                    coarse_score=cand.score,
                    video_id=cand.video_id,
                    timestamp=cand.timestamp,
                    explanation=explain,
                )
            )

            # --- Early stopping (Mục 11.1.4): Top-1 vượt trội rõ -> dừng sớm.
            if (
                cfg.early_stop
                and len(scored) >= cfg.min_evaluated_before_stop
                and _dominant_top1(scored, cfg.confidence_margin)
            ):
                break

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: cfg.final_top_k]


def _dominant_top1(scored: list[RerankedCandidate], margin: float) -> bool:
    """Top-1 co vuot Top-2 toi thieu `margin` khong?

    Logic cuc bo cua rerank; khong phu thuoc module hoi thoai KISC cu.
    """
    if len(scored) < 2:
        return True
    top = sorted(scored, key=lambda c: float(c.score), reverse=True)
    return (float(top[0].score) - float(top[1].score)) >= margin


# ---------------------------------------------------------------------- Mock
class MockReranker(Reranker):
    """Chấm bằng độ trùng token giữa query và context text (objects + caption + OCR/ASR).

    Không phải LVLM thật, nhưng cho tín hiệu precision hợp lý để chạy/đo pipeline
    offline. `per_call_delay` mô phỏng độ trễ LVLM để test time budget."""

    def __init__(self, per_call_delay: float = 0.0):
        self.per_call_delay = per_call_delay

    def score(self, query_text: str, keyframe_id: str, context: object) -> tuple[float, str]:
        if self.per_call_delay:
            time.sleep(self.per_call_delay)
        from ingestion.build_index import tokenize

        q = set(tokenize(query_text))
        ctx = set(tokenize(str(context or "")))
        if not q:
            return 0.0, "query rỗng"
        overlap = len(q & ctx)
        s = overlap / len(q)  # tỉ lệ token query được context đáp ứng -> [0,1]
        return s, f"khớp {overlap}/{len(q)} token query"


# --------------------------------------------------------------------- Claude
RERANK_PROMPT = (
    "Cho ảnh keyframe này và mô tả: '{query}'. Đánh giá độ khớp giữa ảnh và mô tả "
    "trên thang 0-10 (10 = khớp hoàn hảo). Trả về DUY NHẤT một dòng dạng: "
    "<điểm>|<giải thích ngắn>."
)


class ClaudeReranker(Reranker):
    """Bản THẬT: chấm độ khớp bằng Claude vision (anthropic lazy, key từ env).

    context ở đây là ĐƯỜNG DẪN ẢNH keyframe (khác Mock dùng text). Trả điểm chuẩn hoá
    0-1 (chia 10). Batch/concurrency (Mục 4.4/11.1.5) sẽ thêm khi đo được rate limit
    thực tế — ở đây giữ 1 ảnh/lượt cho rõ ràng."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 100,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ClaudeReranker cần SDK 'anthropic'. Cài: pip install anthropic. "
                "(Đang mock-first — dùng MockReranker để test offline.)"
            ) from e
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Thiếu API key: đặt biến môi trường {api_key_env}.")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def score(self, query_text: str, keyframe_id: str, context: object) -> tuple[float, str]:  # pragma: no cover
        import base64
        from pathlib import Path

        image_path = str(context)
        img_b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
        ext = Path(image_path).suffix.lower()
        media = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media, "data": img_b64},
                        },
                        {"type": "text", "text": RERANK_PROMPT.format(query=query_text)},
                    ],
                }
            ],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        raw, _, explain = text.partition("|")
        try:
            score10 = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            score10 = 0.0
        return max(0.0, min(1.0, score10 / 10.0)), explain.strip()
