"""Fine-rerank bằng VLM LOCAL — hiểu từng chữ + ngữ cảnh (CLAUDE.md Mục 4.4).

VÌ SAO CẦN VLM (không chỉ SigLIP): SigLIP là dual-encoder — nén CẢ câu thành 1 vector,
so cosine với ảnh. Nó yếu ở TỔ HỢP TỪ và NGỮ CẢNH ("người đàn ông áo xanh ĐANG tặng
quà cho cô gái" vs "cô gái tặng quà cho người đàn ông" ra vector gần giống nhau). VLM
(Qwen2-VL) dùng CROSS-ATTENTION giữa ảnh và TỪNG token câu -> đọc hiểu ngôn ngữ thật,
suy luận đúng quan hệ/thứ tự/số lượng, và ĐA NGÔN NGỮ (tiếng Việt token-by-token).

VỊ TRÍ TRONG PIPELINE: coarse (SigLIP ensemble, nhanh, recall cao) -> lọc top-K nhỏ ->
VLM rerank top-K này (chậm nhưng chính xác). Đúng nguyên tắc Mục 3/4.4: chỉ chạy tầng
đắt trên số ít ứng viên.

KHÔNG CẦN API KEY — model chạy local. LƯU Ý: trên CPU rerank chậm (vài giây–chục giây/
ảnh) nên chỉ rerank top-K nhỏ. Model nạp LƯỜI. Là bản THẬT của Reranker (Mục 1.5): test
dùng MockReranker (fine_rerank.py), production dùng class này.
"""
from __future__ import annotations

import re
from pathlib import Path

from ingestion import model_cache  # noqa: F401 -- đặt HF_HOME (ổ D) trước transformers
from retrieval.fine_rerank import Reranker

# Prompt chấm điểm: ép VLM đọc mô tả + nhìn ảnh rồi cho 1 số 0-10. Song ngữ để ổn định.
RERANK_PROMPT = (
    "Nhìn kỹ ảnh này. Mô tả cần tìm: \"{query}\".\n"
    "Ảnh khớp với mô tả ở mức nào? Chú ý ĐÚNG các chi tiết: đối tượng, hành động, "
    "màu sắc, số lượng, quan hệ giữa các đối tượng. Chấm một điểm số nguyên từ 0 đến 10 "
    "(10 = khớp hoàn hảo, 0 = không liên quan). CHỈ trả về con số, không giải thích."
)


class Qwen2VLReranker(Reranker):
    """Reranker dùng Qwen2-VL local. context = ĐƯỜNG DẪN ẢNH keyframe.

    Trả (score∈[0,1], giải thích ngắn). score = điểm VLM chấm /10.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_new_tokens: int = 8,
        max_pixels: int = 512 * 512,   # hạ độ phân giải ảnh -> nhanh hơn trên CPU
    ):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Qwen2VLReranker cần 'torch' + 'transformers'. "
                "Cài: pip install torch transformers qwen-vl-utils accelerate."
            ) from e
        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        # Dùng GPU (fp16) nếu có CUDA -> nhanh hơn CPU 10-30x; CPU dùng float32.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()
        # Giới hạn max_pixels để tokenizer ảnh sinh ít token thị giác -> nhẹ VRAM/nhanh.
        self._proc = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)

    def score(self, query_text: str, keyframe_id: str, context: object) -> tuple[float, str]:
        from PIL import Image

        image_path = str(context)
        if not image_path or not Path(image_path).exists():
            return 0.0, "thiếu ảnh keyframe"
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": RERANK_PROMPT.format(query=query_text)},
            ],
        }]
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._proc(text=[text], images=[image], return_tensors="pt").to(self.device)
        torch = self._torch
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        answer = self._proc.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        return _parse_score(answer), f"VLM: {answer}"


def _parse_score(answer: str) -> float:
    """Rút số 0-10 đầu tiên trong câu trả lời VLM -> chuẩn hoá [0,1]. Không có -> 0."""
    m = re.search(r"\d+(?:\.\d+)?", answer)
    if not m:
        return 0.0
    val = float(m.group())
    return max(0.0, min(1.0, val / 10.0))
