"""Dựng index cho tầng coarse retrieval (CLAUDE.md Phase 3, Mục 2.2 / 3 / 5).

Index gồm 3 thành phần song song, phục vụ tầng coarse "recall-first" (Mục 3 bước [2]):
  1. DENSE — Faiss HNSW cho embedding thị giác. Dựng RIÊNG 1 index cho CLIP và 1 cho
     SigLIP (ensemble 2 encoder độc lập — Mục 2.1) để giảm rủi ro "cùng sai".
  2. SPARSE — BM25 trên `searchable_text` (objects + OCR + ASR + llm_caption). Bắt
     các tín hiệu chữ/ngữ nghĩa mà embedding thuần thị giác bỏ sót (Mục 2.4).
  3. METADATA — mảng video_id / timestamp / objects để PRE-FILTER trước vector search
     (Mục 11.1.1: lọc cứng thu hẹp không gian mà không mất ứng viên đúng nào).

VÌ SAO HNSW (Mục 2.2): cân bằng tốc độ/độ chính xác, KHÔNG nén (giữ float) -> ưu tiên
accuracy theo Mục 1.1. Dùng METRIC_INNER_PRODUCT trên vector đã chuẩn hoá L2 = cosine.

VÌ SAO KHÔNG GIỮ MA TRẬN FLOAT RIÊNG (A9, [ĐO 2026-07-17]): tầng coarse cần EXACT
search trên tập con sau metadata pre-filter (Mục 11.1: pre-filter + exact trên subset
thay vì ANN post-filter, tránh mất recall). Trước đây ta giữ thêm `_clip_matrix` /
`_siglip_matrix` cho việc đó — nhưng `IndexHNSWFlat` VỐN ĐÃ lưu nguyên vector float32
bên trong (Flat = không nén), nên ta đang giữ CÙNG một dữ liệu hai lần.

Đo trên 20k×768: ma trận 58.6 MB + HNSW 67.2 MB = 125.8 MB -> bỏ ma trận tiết kiệm
**46.6% RAM** (và cả dung lượng file, vì ma trận còn được pickle vào meta.pkl).

KHÔNG mất gì: `reconstruct_batch` trả vector khớp TỪNG BIT, và còn NHANH HƠN fancy-
indexing của numpy (subset 10k: 5.9ms vs 10.5ms) vì `matrix[rows]` phải gather tạo bản
sao, còn Faiss làm trong C++. Lưu ý: gọi `reconstruct` từng dòng trong vòng lặp Python
thì CHẬM GẤP 5 (52ms) — luôn dùng bản batch.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .schemas import KeyframeRecord

EncoderName = Literal["clip", "siglip"]

# Tokenizer BM25: tách theo ký tự chữ-số Unicode (giữ được tiếng Việt có dấu).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho BM25: lowercase + tách \\w+ (hỗ trợ Unicode/tiếng Việt).

    Cố ý giữ đơn giản, KHÔNG stemming: tên riêng/màu sắc/đồ vật cần khớp nguyên dạng;
    stemming tiếng Việt phức tạp và dễ làm hại precision ở giai đoạn này.
    """
    return _TOKEN_RE.findall(text.lower())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 theo hàng, để inner product = cosine. Bảo vệ hàng norm 0."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class IndexConfig:
    """Tham số dựng index. Giá trị mặc định khớp configs/settings.yaml (index.hnsw).

    `ef_search` [ĐO 2026-07-15 — evaluation/bench_scale.py]: 2048, KHÔNG phải 128.
    Đo trên 768 chiều/dữ liệu mô phỏng embedding thật: ef=128 chỉ lấy được ~46% top-100
    đúng (0.6ms), ef=2048 lấy 98% (11ms). 11ms là vô nghĩa so với time_budget 20s và
    VLM rerank ~1s/ứng viên, trong khi ứng viên rớt ở coarse thì mất luôn (Mục 1.2) —
    nên mua recall bằng vài ms. Quy tắc: ef_search >= ~2x coarse top_k.

    `hnsw_m`/`ef_construction` vẫn là [PROVISIONAL] (chưa quét riêng).
    """

    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 2048


@dataclass
class KeyframeIndex:
    """Index đã dựng + metadata, cùng các phép search nguyên thuỷ (dense/sparse).

    Không tự làm fusion hay filtering logic — đó là việc của coarse_retriever.py.
    Ở đây chỉ cung cấp: dense_search, exact_dense_search (trên subset), sparse_search.
    """

    ids: list[str]
    video_ids: list[str]
    timestamps: list[float]
    objects: list[list[str]]
    config: IndexConfig = field(default_factory=IndexConfig)

    def __post_init__(self) -> None:
        self._clip_index = None
        self._siglip_index = None
        self._bm25 = None
        self._id_to_row = {kid: i for i, kid in enumerate(self.ids)}

    # ---------------------------------------------------------------- build
    @classmethod
    def build(
        cls, records: list[KeyframeRecord], config: Optional[IndexConfig] = None
    ) -> "KeyframeIndex":
        from .build_records import searchable_text  # tránh vòng import ở top-level

        config = config or IndexConfig()
        idx = cls(
            ids=[r.id for r in records],
            video_ids=[r.video_id for r in records],
            timestamps=[r.timestamp for r in records],
            objects=[list(r.objects) for r in records],
            config=config,
        )
        # DENSE: dựng HNSW cho từng encoder có mặt. Ma trận float chỉ là biến TẠM để
        # nạp vào Faiss — không giữ lại (A9): HNSWFlat đã lưu nguyên vector bên trong.
        clip_mat = np.stack([r.clip_embedding for r in records]).astype(np.float32)
        idx._clip_index = idx._build_hnsw(l2_normalize(clip_mat))
        del clip_mat

        if all(r.siglip_embedding is not None for r in records):
            sig_mat = np.stack([r.siglip_embedding for r in records]).astype(np.float32)
            idx._siglip_index = idx._build_hnsw(l2_normalize(sig_mat))
            del sig_mat

        # SPARSE: BM25 trên text gộp.
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(searchable_text(r)) for r in records]
        # BM25Okapi không chịu được doc rỗng hoàn toàn -> đảm bảo mỗi doc ≥1 token.
        corpus = [toks if toks else ["∅"] for toks in corpus]
        idx._bm25 = BM25Okapi(corpus)
        return idx

    def _build_hnsw(self, matrix: np.ndarray):
        import faiss

        dim = matrix.shape[1]
        index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.config.ef_construction
        index.hnsw.efSearch = self.config.ef_search
        index.add(matrix)
        return index

    # --------------------------------------------------------------- search
    def _vectors(self, encoder: EncoderName, rows: "Sequence[int]") -> np.ndarray:
        """Lấy embedding (đã L2-norm) của các `rows` — đọc THẲNG từ trong Faiss.

        Thay cho `matrix[rows]` trước đây (A9): HNSWFlat đã giữ nguyên vector float32,
        nên ma trận riêng chỉ là bản sao thừa (-46.6% RAM khi bỏ). `reconstruct_batch`
        khớp từng bit và nhanh hơn fancy-indexing numpy — KHÔNG dùng vòng lặp
        `reconstruct` từng dòng (chậm gấp 5).
        """
        index = self._index(encoder)
        if index is None:
            raise ValueError(f"Index cho encoder {encoder!r} chưa được dựng.")
        return index.reconstruct_batch(np.asarray(rows, dtype=np.int64))

    def _index(self, encoder: EncoderName):
        return self._clip_index if encoder == "clip" else self._siglip_index

    def has_encoder(self, encoder: EncoderName) -> bool:
        return self._index(encoder) is not None

    def mean_embedding(
        self, ids: "Sequence[str]", encoder: EncoderName
    ) -> Optional[np.ndarray]:
        """Trung bình embedding (đã L2-norm) của các keyframe `ids` cho 1 encoder.

        Dùng cho relevance feedback (Rocchio): dịch vector truy vấn về phía các ảnh
        người dùng đánh dấu. Trả None nếu encoder chưa dựng hoặc không id nào hợp lệ."""
        if not self.has_encoder(encoder):
            return None
        rows = [self._id_to_row[i] for i in ids if i in self._id_to_row]
        if not rows:
            return None
        return self._vectors(encoder, rows).mean(axis=0)

    def dense_search(
        self, query_vec: np.ndarray, encoder: EncoderName, top_k: int
    ) -> list[tuple[int, float]]:
        """ANN search HNSW trên TOÀN dataset. Trả [(row, cosine_score)] giảm dần."""
        index = self._index(encoder)
        if index is None:
            raise ValueError(f"Index cho encoder {encoder!r} chưa được dựng.")
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        scores, rows = index.search(q, min(top_k, len(self.ids)))
        return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]

    def exact_dense_search(
        self,
        query_vec: np.ndarray,
        encoder: EncoderName,
        candidate_rows: list[int],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """EXACT cosine trên TẬP CON đã lọc (Mục 11.1: pre-filter + exact, không mất
        recall). Dùng khi metadata filter đã thu hẹp đủ nhỏ."""
        if not candidate_rows:
            return []
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        rows = np.asarray(candidate_rows, dtype=np.int64)
        sims = self._vectors(encoder, rows) @ q
        order = np.argsort(-sims)[:top_k]
        return [(int(rows[i]), float(sims[i])) for i in order]

    def sparse_search(
        self, query_text: str, top_k: int, candidate_rows: Optional[list[int]] = None
    ) -> list[tuple[int, float]]:
        """BM25 search. Nếu truyền candidate_rows -> chỉ xếp hạng trong tập con đó."""
        if self._bm25 is None:
            raise ValueError("BM25 chưa được dựng.")
        tokens = tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        rows = (
            np.asarray(candidate_rows, dtype=np.int64)
            if candidate_rows is not None
            else np.arange(len(self.ids))
        )
        row_scores = [(int(r), float(scores[r])) for r in rows if scores[r] > 0]
        row_scores.sort(key=lambda x: x[1], reverse=True)
        return row_scores[:top_k]

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Lưu index ra đĩa (Mục 5.3 sharding/rebuild). Faiss index lưu riêng."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._clip_index is not None:
            faiss.write_index(self._clip_index, str(directory / "clip.hnsw"))
        if self._siglip_index is not None:
            faiss.write_index(self._siglip_index, str(directory / "siglip.hnsw"))
        payload = {
            "ids": self.ids,
            "video_ids": self.video_ids,
            "timestamps": self.timestamps,
            "objects": self.objects,
            "config": self.config,
            # A9: KHÔNG lưu ma trận float nữa — vector đã nằm trong *.hnsw. Trước đây
            # pickle cả hai nên meta.pkl phình gấp đôi một cách vô ích.
            "bm25": self._bm25,
        }
        with (directory / "meta.pkl").open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, directory: str | Path) -> "KeyframeIndex":
        import faiss

        directory = Path(directory)
        with (directory / "meta.pkl").open("rb") as fh:
            payload = pickle.load(fh)
        idx = cls(
            ids=payload["ids"],
            video_ids=payload["video_ids"],
            timestamps=payload["timestamps"],
            objects=payload["objects"],
            config=payload["config"],
        )
        # A9: index CŨ trên đĩa còn field clip_matrix/siglip_matrix — cố ý BỎ QUA
        # (vector lấy từ *.hnsw). Dùng .get() để file cũ vẫn nạp được, không phải
        # build lại từ đầu.
        idx._bm25 = payload["bm25"]
        clip_path = directory / "clip.hnsw"
        sig_path = directory / "siglip.hnsw"
        if clip_path.exists():
            idx._clip_index = faiss.read_index(str(clip_path))
        if sig_path.exists():
            idx._siglip_index = faiss.read_index(str(sig_path))
        return idx
