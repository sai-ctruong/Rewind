"""Trích xuất SigLIP embedding cho keyframe (CLAUDE.md Mục 2.1, Phase 2).

VÌ SAO CÓ SigLIP BÊN CẠNH CLIP: Mục 2.1 quyết định dùng ENSEMBLE 2 encoder độc lập
(CLIP có sẵn + SigLIP tự trích xuất). SigLIP zero-shot tốt hơn CLIP ở benchmark
retrieval; dùng 2 encoder giảm rủi ro "cùng sai" (correlated errors) -> tăng accuracy.

KHÁC embed_clip.py: CLIP feature BTC cấp sẵn nên chỉ NẠP; còn SigLIP phải TỰ TRÍCH
XUẤT từ ảnh keyframe (cần model + GPU). Vì mock-first (chưa có GPU/data), bản thật
`SiglipEncoder` LAZY-IMPORT torch/transformers — import module này KHÔNG kéo theo
torch, chỉ khi gọi mới nạp -> pipeline mock chạy được ngay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .embed_clip import deterministic_unit_vector
from .schemas import RawKeyframe

# Chiều embedding SigLIP-large (khớp configs/settings.yaml -> embedding.siglip.dim).
SIGLIP_DIM = 1024


class SiglipEmbeddingProvider(ABC):
    """Interface: trả SigLIP embedding cho một RawKeyframe."""

    @abstractmethod
    def embed(self, raw: RawKeyframe) -> np.ndarray:
        ...


class MockSiglipEmbeddingProvider(SiglipEmbeddingProvider):
    """Mock: embedding tất định theo id, KHÁC không gian với CLIP (salt='siglip').

    Dùng salt khác để mock CLIP và mock SigLIP không trùng nhau — phản ánh việc 2
    encoder thật cho ra 2 biểu diễn độc lập (đúng tinh thần ensemble Mục 2.1).
    """

    def __init__(self, dim: int = SIGLIP_DIM):
        self.dim = dim

    def embed(self, raw: RawKeyframe) -> np.ndarray:
        return deterministic_unit_vector(raw.id, self.dim, salt="siglip")


class SiglipEncoder(SiglipEmbeddingProvider):
    """Bản THẬT: trích xuất SigLIP embedding từ ảnh keyframe.

    torch/transformers được LAZY-IMPORT trong __init__ để module này import được
    kể cả khi chưa cài (mock-first). Khi có GPU + data thật, tạo instance này thay
    cho MockSiglipEmbeddingProvider là chạy — không đổi phần còn lại của pipeline.

    Tối ưu theo Mục 5.2 (áp dụng khi chạy batch thật ở build_records): fp16 +
    batch lớn để tăng throughput ~40%. Ở đây để đơn giản, embed từng ảnh; phần
    batch hoá sẽ thêm khi cắm dữ liệu thật và đo được throughput.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-large-patch16-384",
        device: str | None = None,
        precision: str = "fp16",
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:  # pragma: no cover - chỉ chạy khi dùng bản thật
            raise ImportError(
                "SiglipEncoder cần 'torch' và 'transformers'. Cài: "
                "pip install torch transformers. (Đang mock-first nên chưa cài — "
                "dùng MockSiglipEmbeddingProvider để test offline.)"
            ) from e
        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device)
        if precision == "fp16" and self.device == "cuda":
            self._model = self._model.half()
        self._model.eval()

    def embed(self, raw: RawKeyframe) -> np.ndarray:  # pragma: no cover - bản thật
        if raw.image_path is None:
            raise ValueError(
                f"Keyframe {raw.id!r} thiếu image_path — SigLIP cần ảnh gốc."
            )
        from PIL import Image

        torch = self._torch
        image = Image.open(raw.image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        vec = feats[0].float().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
