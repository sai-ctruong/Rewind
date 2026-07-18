"""Cung cấp CLIP embedding cho mỗi keyframe (CLAUDE.md Mục 2.1, Phase 2).

VAI TRÒ: CLIP feature do BTC CẤP SẴN (ViT-L/14) — ta KHÔNG trích xuất lại (tiết
kiệm compute, Mục 2.1). Nhiệm vụ module này chỉ là NẠP đúng feature cho từng
keyframe theo id.

THIẾT KẾ (pattern ABC + Mock, giống dialogue/retriever.py::MockRetriever, Mục 1.5):
  - `ClipEmbeddingProvider` (ABC): interface `embed(raw) -> np.ndarray`.
  - `NpyClipEmbeddingProvider`: bản THẬT nạp từ file .npy BTC cấp. Định dạng thật
    của BTC chưa biết chắc -> code này giả định một layout phổ biến và ĐÁNH DẤU RÕ
    chỗ cần chỉnh khi thấy dữ liệu thật.
  - `MockClipEmbeddingProvider`: sinh vector TẤT ĐỊNH theo id để test offline mà
    không cần data thật/GPU.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .schemas import RawKeyframe

# Chiều embedding CLIP ViT-L/14 (khớp configs/settings.yaml -> embedding.clip.dim).
CLIP_DIM = 768


def deterministic_unit_vector(key: str, dim: int, salt: str = "") -> np.ndarray:
    """Sinh 1 vector đơn vị TẤT ĐỊNH từ một chuỗi khoá.

    Vì sao cần: mock phải cho ra cùng embedding mỗi lần chạy với cùng id (để test
    ổn định và để 2 frame "giống nhau" thực sự giống nhau). Ta seed một RNG bằng
    hash của (salt + key) rồi chuẩn hoá về norm 1 (khớp giả định cosine của Faiss).
    `salt` cho phép cùng một id sinh ra 2 không gian khác nhau (vd CLIP vs SigLIP).
    """
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class ClipEmbeddingProvider(ABC):
    """Interface: trả CLIP embedding cho một RawKeyframe."""

    @abstractmethod
    def embed(self, raw: RawKeyframe) -> np.ndarray:
        ...


class MockClipEmbeddingProvider(ClipEmbeddingProvider):
    """Mock: embedding tất định theo keyframe id (offline, không cần data thật).

    Tham số `noise` (>0): cộng thêm nhiễu ngẫu nhiên nhỏ rồi chuẩn hoá lại — dùng khi
    muốn tạo các vector "gần nhau nhưng không trùng" cho test. Mặc định noise=0 để
    embedding hoàn toàn tất định (cùng id -> cùng vector mỗi lần chạy).
    """

    def __init__(self, dim: int = CLIP_DIM, noise: float = 0.0, seed: int = 0):
        self.dim = dim
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    def embed(self, raw: RawKeyframe) -> np.ndarray:
        base = deterministic_unit_vector(raw.id, self.dim, salt="clip")
        if self.noise > 0:
            base = base + self.noise * self._rng.standard_normal(self.dim).astype(
                np.float32
            )
            n = np.linalg.norm(base)
            if n > 0:
                base = base / n
        return base.astype(np.float32)


class NpyClipEmbeddingProvider(ClipEmbeddingProvider):
    """Bản THẬT: nạp CLIP feature BTC cấp từ đĩa.

    GIẢ ĐỊNH LAYOUT (chỉnh lại khi biết định dạng BTC thật):
        {features_dir}/{video_id}.npy   -> mảng float32 shape (N_frames, CLIP_DIM)
        {features_dir}/{video_id}.txt    -> N dòng, mỗi dòng là keyframe id tương ứng
    Nếu không có file id, ta suy id theo quy ước "{video_id}/{index}" — nhưng khi đó
    RawKeyframe.id phải trùng quy ước này.

    Nạp lười theo video và cache lại để không đọc đĩa lặp.
    """

    def __init__(self, features_dir: str | Path):
        self.features_dir = Path(features_dir)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_video(self, video_id: str) -> dict[str, np.ndarray]:
        if video_id in self._cache:
            return self._cache[video_id]
        npy_path = self.features_dir / f"{video_id}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Không thấy CLIP feature cho video {video_id!r} tại {npy_path}. "
                "Kiểm tra lại configs/settings.yaml -> paths.clip_features_dir và "
                "định dạng BTC cấp (điều chỉnh NpyClipEmbeddingProvider nếu cần)."
            )
        matrix = np.load(npy_path).astype(np.float32)
        ids_path = self.features_dir / f"{video_id}.txt"
        if ids_path.exists():
            ids = ids_path.read_text(encoding="utf-8").splitlines()
        else:
            ids = [f"{video_id}/{i}" for i in range(len(matrix))]
        if len(ids) != len(matrix):
            raise ValueError(
                f"Số id ({len(ids)}) không khớp số vector ({len(matrix)}) cho "
                f"video {video_id!r}."
            )
        mapping = {kf_id: matrix[i] for i, kf_id in enumerate(ids)}
        self._cache[video_id] = mapping
        return mapping

    def embed(self, raw: RawKeyframe) -> np.ndarray:
        mapping = self._load_video(raw.video_id)
        if raw.id not in mapping:
            raise KeyError(
                f"Keyframe id {raw.id!r} không có trong feature của video "
                f"{raw.video_id!r}."
            )
        return mapping[raw.id]
