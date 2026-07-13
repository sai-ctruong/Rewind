"""Engine tìm kiếm video độ chính xác cao (không cần API key).

Nâng cấp so với bản demo 1-encoder ban đầu, áp dụng đúng các kỹ thuật blueprint:

1. ENSEMBLE 2 ENCODER (Mục 2.1): SigLIP2 (thế hệ mới, chính xác hơn) + SigLIP
   multilingual (đời trước, phân phối lỗi khác). Mỗi encoder cho 1 ranked list,
   fuse bằng RRF qua CoarseRetriever — giảm rủi ro "cùng sai" (correlated errors).
2. QUERY PROMPT ENSEMBLE (Mục 4.2 — Multi-Query Expansion, bản không cần LLM):
   encode nhiều biến thể prompt của cùng câu truy vấn ("{q}", "a photo of {q}",
   "một bức ảnh về {q}") rồi lấy TRUNG BÌNH embedding (chuẩn hoá lại). Một cách
   diễn đạt duy nhất có thể lệch khỏi phân phối caption model đã học — trung bình
   nhiều biến thể "trúng" biểu diễn đúng thường xuyên hơn.
3. LẤY MẪU DÀY + DEDUP NGỮ NGHĨA (Mục 5.1): sample mỗi 0.5s (dày gấp đôi) để không
   bỏ lỡ khoảnh khắc ngắn, rồi gộp frame gần trùng bằng ingestion/dedup (anchor-based,
   trên embedding — tinh hơn histogram) để index không phình và không nhiễu.
4. Điểm số minh bạch: giữ source_ranks của RRF để truy vết encoder nào ủng hộ.

Mọi tham số accuracy-critical đều [PROVISIONAL] theo Mục 11.3 — chỉnh qua benchmark,
không coi là số cuối.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ingestion.build_index import KeyframeIndex, l2_normalize
from ingestion.dedup import deduplicate_keyframes
from ingestion.schemas import KeyframeRecord, RawKeyframe
from ingestion.video_ingest import extract_keyframes
from retrieval.coarse_retriever import Candidate, CoarseRetriever
from retrieval.fine_rerank import FineReranker, RerankConfig, Reranker
from retrieval.temporal_check import TemporalMatch, temporal_consistency_filter

# Cặp encoder mặc định cho ensemble (Mục 2.1). Cả hai đều đa ngôn ngữ:
#   - SigLIP2: bản kế nhiệm, retrieval tốt hơn SigLIP1 trên benchmark.
#   - SigLIP1 multilingual: phân phối lỗi khác đời -> ensemble có ích.
DEFAULT_ENCODERS = (
    "google/siglip2-base-patch16-256",
    "google/siglip-base-patch16-256-multilingual",
)

# Biến thể prompt cho query ensemble (Mục 4.2). Trộn Anh + Việt vì cả 2 encoder đều
# đa ngôn ngữ; template "a photo of" khớp phân phối caption model được huấn luyện.
QUERY_TEMPLATES = (
    "{q}",
    "a photo of {q}.",
    "một bức ảnh về {q}.",
)


def adaptive_bm25_weight(query: str, low: float, high: float) -> float:
    """Chọn trọng số BM25 theo LOẠI query — dựa trên benchmark thật (2026-07-12):
    OCR/BM25 nặng GIÚP query CHỮ (tên biển hiệu) nhưng HẠI query THỊ GIÁC (kéo recall
    0.72→0.52). Vì một trọng số cố định không thắng cả hai, ta suy ra ý định từ câu:

    Nếu query chứa token IN HOA ≥2 chữ ASCII (kiểu tên biển hiệu: SAMSUNG, LANEIGE,
    OPEN, STREET FOOD, ATHOME) -> đây là query CHỮ -> dùng BM25 CAO để tín hiệu OCR nổi
    lên. Ngược lại (mô tả cảnh/vật bằng lời thường) -> BM25 THẤP để không nhiễu recall."""
    import re

    for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", query):
        if len(tok) >= 2 and tok.isupper():
            return high
    return low


@dataclass
class VideoIndexEntry:
    """Kết quả index 1 video: Faiss index + map id -> RawKeyframe (đường dẫn ảnh)."""

    video_id: str
    index: KeyframeIndex
    raws: dict[str, RawKeyframe]
    num_sampled: int          # số keyframe sau bước cắt thô (histogram)
    num_indexed: int          # số keyframe sau dedup ngữ nghĩa (vào index)
    ocr_by_id: dict[str, str] = field(default_factory=dict)  # chữ OCR đọc được / keyframe
    asr_by_id: dict[str, str] = field(default_factory=dict)  # lời nói (ASR) quanh keyframe
    caption_by_id: dict[str, str] = field(default_factory=dict)  # mô tả ngữ cảnh (VLM) / keyframe

    # ------------------------------------------------------------ persistence (A2)
    def save(self, directory: str | Path) -> None:
        """Lưu index ra đĩa để KHÔNG phải embed lại (dataset lớn: nạp 1 lần, mở tức thì).

        CỐ Ý KHÔNG lưu image_bytes (ảnh trong RAM ~80KB/frame → hàng TB với triệu
        frame). Chỉ lưu: Faiss index (embedding) + metadata + source_video/frame_idx
        để DỰNG LẠI ảnh từ video gốc khi cần hiển thị/rerank. Đây là điều kiện tiên
        quyết để thử nghiệm scale (blueprint Mục 5.3)."""
        import pickle

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.index.save(directory / "index")
        # raws BỎ image_bytes (nặng) — giữ đủ để decode lại từ source_video.
        raws_light = {
            rid: RawKeyframe(
                id=r.id, video_id=r.video_id, timestamp=r.timestamp,
                image_path=r.image_path, source_video=r.source_video,
                frame_idx=r.frame_idx, objects=list(r.objects),
            )
            for rid, r in self.raws.items()
        }
        payload = {
            "video_id": self.video_id, "num_sampled": self.num_sampled,
            "num_indexed": self.num_indexed, "ocr_by_id": self.ocr_by_id,
            "asr_by_id": self.asr_by_id, "caption_by_id": self.caption_by_id,
            "raws": raws_light,
        }
        with (directory / "entry.pkl").open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, directory: str | Path) -> "VideoIndexEntry":
        """Nạp lại entry đã lưu. Ảnh dựng lại lười từ source_video khi truy cập."""
        import pickle

        directory = Path(directory)
        index = KeyframeIndex.load(directory / "index")
        with (directory / "entry.pkl").open("rb") as fh:
            payload = pickle.load(fh)
        return cls(
            video_id=payload["video_id"], index=index, raws=payload["raws"],
            num_sampled=payload["num_sampled"], num_indexed=payload["num_indexed"],
            ocr_by_id=payload["ocr_by_id"], asr_by_id=payload.get("asr_by_id", {}),
            caption_by_id=payload.get("caption_by_id", {}),
        )


class VideoSearchEngine:
    """Điều phối: cắt keyframe -> embed ensemble -> dedup -> index -> search RRF.

    Model nạp LƯỜI ở lần dùng đầu (tải/khởi tạo chậm — không chặn app startup).
    """

    def __init__(
        self,
        encoder_names: Sequence[str] = DEFAULT_ENCODERS,
        sample_every_s: float = 1.0,           # [PROVISIONAL] 1 frame/giây
        max_frames: Optional[int] = None,       # None = KHÔNG giới hạn (index cả video)
        dedup_threshold: float = 0.97,          # [PROVISIONAL] khớp settings.yaml
        query_templates: Sequence[str] = QUERY_TEMPLATES,
        rerank_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        rerank_pool: int = 8,                   # [PROVISIONAL] số ứng viên coarse đưa vào VLM
        enable_ocr: bool = True,                # đọc chữ trên keyframe -> tìm biển hiệu/chữ
        ocr_langs: tuple[str, ...] = ("vi", "en"),
        enable_asr: bool = False,               # B3: chép lời nói cả video -> BM25 (nặng, opt-in)
        asr_model: str = "small",               # cỡ Whisper (tiny/base/small/medium/large)
        asr_language: Optional[str] = "vi",
        enable_caption: bool = False,           # Mục 2.4: VLM sinh mô tả ngữ cảnh -> BM25 (nặng, opt-in)
        caption_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        bm25_weight: float = 1.0,               # [ĐO 2026-07-12] 1.0 tối ưu: hit@5=0.72;
                                                # 3.0 làm hại recall (0.52), 0.0 dense-only (0.68)
        adaptive_bm25: bool = True,             # trọng số BM25 theo loại query (chữ vs thị giác)
        bm25_weight_high: float = 3.0,          # trọng số BM25 khi query là CHỮ (biển hiệu)
        embed_batch_size: int = 256,            # [PROVISIONAL] lô embed GPU (giảm nếu tràn VRAM)
        decode_backend: str = "auto",           # A3: "auto"|"cv2"|"decord" (decord=NVDEC nếu có)
        use_gpu_decode: bool = True,            # dùng NVDEC khi backend=decord + có CUDA
        parallel_index: bool = True,            # A4: song song decode ‖ embed (tắt để debug)
    ):
        self.encoder_names = list(encoder_names)[:2]  # KeyframeIndex có đúng 2 slot dense
        self.sample_every_s = sample_every_s
        self.max_frames = max_frames
        self.dedup_threshold = dedup_threshold
        self.query_templates = list(query_templates)
        self.rerank_model = rerank_model
        self.rerank_pool = rerank_pool
        self.enable_ocr = enable_ocr
        self.ocr_langs = ocr_langs
        self.enable_asr = enable_asr
        self.asr_model = asr_model
        self.asr_language = asr_language
        self.enable_caption = enable_caption
        self.caption_model = caption_model
        self.bm25_weight = bm25_weight
        self.adaptive_bm25 = adaptive_bm25
        self.bm25_weight_high = bm25_weight_high
        self.embed_batch_size = embed_batch_size
        self.decode_backend = decode_backend
        self.use_gpu_decode = use_gpu_decode
        self.parallel_index = parallel_index
        self._encoders: Optional[list] = None   # nạp lười
        self._reranker: Optional[FineReranker] = None  # VLM rerank, nạp lười
        self._ocr = None                        # EasyOCR, nạp lười
        self._asr = None                        # Whisper (ASR cấp-video), nạp lười
        self._captioner = None                  # Qwen2-VL captioner, nạp lười

    # ------------------------------------------------------------- encoders
    def _load_encoders(self) -> list:
        if self._encoders is None:
            from ingestion.embed_siglip import SiglipEncoder

            self._encoders = [SiglipEncoder(model_name=n) for n in self.encoder_names]
            # In DEVICE để phát hiện ngay nếu chạy nhầm CPU (chậm 10-30× -> "index lâu").
            devs = ", ".join(str(getattr(e, "device", "?")) for e in self._encoders)
            print(f"[video_engine] SigLIP encoders trên device: {devs} "
                  f"(cuda = GPU nhanh; cpu = CHẬM, kiểm tra torch CUDA!)", flush=True)
        return self._encoders

    def set_encoders(self, encoders: list) -> None:
        """Bơm encoder ngoài (mock trong test / model tuỳ chọn). Mỗi encoder cần
        `.embed(raw) -> vec` và `.encode_text(str) -> vec`."""
        self._encoders = list(encoders)[:2]

    def _free_encoders(self) -> None:
        """Xả SigLIP khỏi VRAM/RAM (để nạp VLM caption trên máy hạn bộ nhớ). Encoder
        tự nạp lại LƯỜI ở lần search kế. An toàn: chỉ gọi khi embedding đã tính xong."""
        if not self._encoders:
            return
        try:
            import gc
            for e in self._encoders:
                if hasattr(e, "_model"):
                    del e._model
            self._encoders = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - dọn bộ nhớ 'best-effort'
            self._encoders = None

    # ------------------------------------------------------------- reranker
    def _get_reranker(self) -> FineReranker:
        if self._reranker is None:
            from retrieval.vlm_rerank import Qwen2VLReranker

            self._reranker = FineReranker(
                Qwen2VLReranker(model_name=self.rerank_model),
                # Pool nhỏ + budget rộng (VLM CPU chậm); không early-stop để chấm hết pool.
                RerankConfig(max_candidates=self.rerank_pool, early_stop=False,
                             time_budget_s=3600.0),
            )
        return self._reranker

    def set_reranker(self, reranker: Reranker) -> None:
        """Bơm reranker ngoài (mock trong test). Bọc trong FineReranker."""
        self._reranker = FineReranker(
            reranker, RerankConfig(max_candidates=self.rerank_pool,
                                   early_stop=False, time_budget_s=3600.0))

    # ------------------------------------------------------------------- ocr
    def _get_ocr(self):
        if self._ocr is None:
            from ingestion.ocr_asr_extract import EasyOcrEngine
            self._ocr = EasyOcrEngine(langs=self.ocr_langs)
        return self._ocr

    def set_ocr(self, ocr) -> None:
        """Bơm OCR engine ngoài (mock trong test). Bật enable_ocr."""
        self._ocr = ocr
        self.enable_ocr = True

    # ------------------------------------------------------------------- asr
    def _get_asr(self):
        if self._asr is None:
            from ingestion.ocr_asr_extract import WhisperVideoAsrEngine
            self._asr = WhisperVideoAsrEngine(model_size=self.asr_model,
                                              language=self.asr_language)
        return self._asr

    def set_asr(self, asr) -> None:
        """Bơm ASR cấp-video ngoài (mock trong test). Bật enable_asr."""
        self._asr = asr
        self.enable_asr = True

    # ------------------------------------------------------------- captioner
    def _get_captioner(self):
        if self._captioner is None:
            from ingestion.llm_captioning import QwenVLCaptioner
            self._captioner = QwenVLCaptioner(model_name=self.caption_model)
        return self._captioner

    def set_captioner(self, captioner) -> None:
        """Bơm captioner ngoài (mock trong test / ClaudeCaptioner khi có API). Bật enable_caption."""
        self._captioner = captioner
        self.enable_caption = True

    def _apply_captions(self, raws_by_id: dict, kept: list[KeyframeRecord]) -> None:
        """Sinh caption ngữ cảnh (VLM) cho các keyframe ĐẠI DIỆN (sau dedup) -> llm_caption.

        VÌ SAO CHỈ SURVIVORS: caption VLM chậm (~vài giây/ảnh) nên chỉ chạy trên số ít
        đại diện sau dedup, KHÔNG phải mọi frame lấy mẫu. Caption (quan hệ + hoàn cảnh)
        vào searchable_text -> BM25 tìm được theo MÔ TẢ NGỮ NGHĨA, không chỉ hình ảnh
        thuần (Mục 2.4). Lỗi caption 1 frame không làm vỡ cả mẻ."""
        # GIẢI PHÓNG SigLIP trước khi nạp VLM caption LOCAL thật (Qwen): máy 6GB không
        # chứa nổi cả 2 (tràn paging file Windows -> crash). Chỉ xả khi captioner CHƯA
        # được bơm (tức sắp nạp Qwen local); captioner bơm sẵn (mock/Claude) thì không
        # cần — tránh làm hỏng test mock và tránh xả vô ích với Claude (không nạp local).
        if self._captioner is None:
            self._free_encoders()
        try:
            captioner = self._get_captioner()
        except Exception as e:  # pragma: no cover - nạp model lỗi (thiếu RAM/paging file)
            # KHÔNG làm sập cả lần index chỉ vì captioner nạp lỗi: bỏ caption, giữ index
            # (vẫn có SigLIP + OCR/ASR). Gợi ý khắc phục rõ ràng.
            print(f"[video_engine] Không nạp được captioner ({e!r}). BỎ QUA caption — "
                  "index vẫn chạy. Khắc phục: tăng paging file Windows, hoặc dùng "
                  "ClaudeCaptioner (API) khi có key.", flush=True)
            return
        for rec in kept:
            raw = raws_by_id.get(rec.id)
            if raw is None:
                continue
            try:
                cap = captioner.caption(raw)
            except Exception as e:  # pragma: no cover
                print(f"[video_engine] caption lỗi {rec.id} ({e!r}); bỏ qua.")
                cap = None
            if cap:
                rec.llm_caption = cap

    def _apply_asr(self, raws: list[RawKeyframe],
                   records: list[KeyframeRecord]) -> None:
        """Điền `asr_text` cho từng record bằng lời nói quanh timestamp của nó.

        Transcribe MỖI video 1 lần (cache theo source_video) rồi gán đoạn transcript
        phủ thời điểm keyframe (segment_text_at). Lỗi ASR 1 video KHÔNG làm vỡ cả mẻ —
        video đó chỉ thiếu asr_text (vẫn còn hình ảnh + OCR). Bổ sung tín hiệu 'lời nói'
        vào searchable_text -> BM25 tìm được cảnh theo điều người ta nói."""
        from ingestion.ocr_asr_extract import segment_text_at

        asr = self._get_asr()
        seg_cache: dict[str, list] = {}
        for raw, rec in zip(raws, records):
            src = raw.source_video
            if not src:
                continue
            if src not in seg_cache:
                try:
                    seg_cache[src] = asr.transcribe_segments(src)
                except Exception as e:  # pragma: no cover - lỗi Whisper/ffmpeg
                    print(f"[video_engine] ASR lỗi cho {src!r} ({e!r}); bỏ qua.")
                    seg_cache[src] = []
            text = segment_text_at(seg_cache[src], rec.timestamp)
            if text:
                rec.asr_text = text

    # -------------------------------------------------------------- indexing
    def _embed_all(self, encoder, raws: list[RawKeyframe]) -> list[np.ndarray]:
        """Embed toàn bộ `raws` bằng 1 encoder — ưu tiên THEO LÔ (GPU chạy hết công suất).

        Nếu encoder có `embed_batch` (bản SigLIP thật) thì gom lô -> nhanh gấp hàng
        chục lần; nếu không (mock trong test) thì lặp `embed()` từng ảnh. Nhờ vậy cùng
        một code chạy được cả bản thật lẫn mock (mock-first, Mục 5)."""
        batch_fn = getattr(encoder, "embed_batch", None)
        if callable(batch_fn):
            return list(batch_fn(raws, self.embed_batch_size))
        return [encoder.embed(r) for r in raws]

    def _embed_batch_shared(self, encoders: list,
                            raws: list[RawKeyframe]) -> list[list[np.ndarray]]:
        """Embed `raws` bằng MỌI encoder, GIẢI MÃ ẢNH 1 LẦN dùng chung (nếu bản thật).

        Trước đây mỗi encoder tự load ảnh -> ensemble 2 encoder giải mã JPEG 2 lần
        (lãng phí đúng phần tiền xử lý CPU vốn là nút thắt). Ở đây decode 1 lần/lô rồi
        đưa cùng ảnh cho cả 2 encoder. Mock không có embed_pil_batch -> fallback lối cũ."""
        if not all(hasattr(e, "embed_pil_batch") for e in encoders):
            return [self._embed_all(e, raws) for e in encoders]
        from concurrent.futures import ThreadPoolExecutor

        from ingestion.schemas import load_pil_image

        bs = max(1, self.embed_batch_size)
        outs: list[list[np.ndarray]] = [[] for _ in encoders]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(raws), bs):
                chunk = raws[start:start + bs]
                images = list(pool.map(load_pil_image, chunk))   # decode MỘT lần
                for ei, e in enumerate(encoders):
                    outs[ei].extend(e.embed_pil_batch(images, bs))
        return outs

    def _embed_raws(self, raws: list[RawKeyframe],
                    enable_ocr: Optional[bool] = None) -> list[KeyframeRecord]:
        """Embed mỗi keyframe bằng cả 2 encoder (slot clip + siglip) + OCR (nếu bật).

        Embed THEO LÔ cho từng encoder (A1) thay vì lẻ từng ảnh: gom hết raws -> một
        loạt tensor trên GPU. OCR đọc chữ trên khung hình (biển hiệu, phụ đề) -> lưu
        ocr_text để đưa vào BM25 -> tìm được bằng CHỮ chứ không chỉ hình ảnh."""
        encoders = self._load_encoders()
        use_ocr = self.enable_ocr if enable_ocr is None else enable_ocr
        ocr = self._get_ocr() if use_ocr else None
        embs = self._embed_batch_shared(encoders, raws)
        emb0 = embs[0]
        emb1 = embs[1] if len(embs) > 1 else None
        records: list[KeyframeRecord] = []
        for i, r in enumerate(raws):
            rec = KeyframeRecord(
                id=r.id, video_id=r.video_id, timestamp=r.timestamp,
                clip_embedding=emb0[i],
            )
            if emb1 is not None:
                rec.siglip_embedding = emb1[i]
            if ocr is not None:
                rec.ocr_text = ocr.extract(r)
            records.append(rec)
        return records

    def _build_entry(
        self, raws: list[RawKeyframe], records: list[KeyframeRecord], video_id: str,
        enable_caption: Optional[bool] = None,
    ) -> VideoIndexEntry:
        # Dedup NGỮ NGHĨA (Mục 5.1): gộp frame gần trùng theo embedding chính (slot 1).
        # deduplicate_keyframes gom theo video_id nên với DATASET nhiều video vẫn dedup
        # đúng TRONG TỪNG video, không gộp nhầm xuyên video.
        result = deduplicate_keyframes(
            records, similarity_threshold=self.dedup_threshold,
            embedding_attr="clip_embedding",
        )
        kept = result.representatives
        # Giải phóng RAM: chỉ giữ ảnh (image_bytes) của frame SỐNG SÓT sau dedup — số
        # còn lại (đã bị gộp) không bao giờ hiển thị/rerank nên bỏ bytes đi. Vẫn giữ
        # record raw (timestamp...) để tra cứu, chỉ None hoá phần ảnh nặng.
        kept_ids = {r.id for r in kept}
        # Caption ngữ cảnh (VLM) chỉ trên ĐẠI DIỆN — trước khi None-hoá ảnh của frame bị
        # loại, và trước khi build index (caption vào searchable_text -> BM25).
        if self.enable_caption if enable_caption is None else enable_caption:
            self._apply_captions({r.id: r for r in raws}, kept)
        for r in raws:
            if r.id not in kept_ids:
                r.image_bytes = None
        return VideoIndexEntry(
            video_id=video_id, index=KeyframeIndex.build(kept),
            raws={r.id: r for r in raws},
            num_sampled=len(records), num_indexed=len(kept),
            ocr_by_id={r.id: r.ocr_text for r in records if r.ocr_text},
            asr_by_id={r.id: r.asr_text for r in records if r.asr_text},
            caption_by_id={r.id: r.llm_caption for r in kept if r.llm_caption},
        )

    def _pipeline_records(
        self, video_specs: list[tuple], out_dir: str | Path, *,
        sample_every_s: Optional[float], max_frames: Optional[int],
        enable_ocr: Optional[bool],
    ) -> tuple[list[RawKeyframe], list[KeyframeRecord]]:
        """SONG SONG HOÁ decode ‖ embed (A4): 1 luồng PRODUCER stream keyframe (decode
        CPU/NVDEC), luồng chính CONSUMER embed theo lô trên GPU + OCR.

        VÌ SAO NHANH: decode (CPU/IO) và embed (GPU) chạy CHỒNG nhau — trong lúc GPU
        embed lô hiện tại, producer đã decode lô kế. cv2/torch đều nhả GIL khi chạy
        C++/CUDA nên thread cho song song thật. Queue GIỚI HẠN (maxsize) để producer
        không decode vượt quá xa consumer -> chặn phình RAM với video dài. Thứ tự
        keyframe được BẢO TOÀN (FIFO) nên dedup/index giống hệt đường tuần tự."""
        import queue
        import threading

        from ingestion.video_ingest import iter_keyframes

        encoders = self._load_encoders()
        use_ocr = self.enable_ocr if enable_ocr is None else enable_ocr
        ocr = self._get_ocr() if use_ocr else None
        bs = max(1, self.embed_batch_size)

        q: "queue.Queue" = queue.Queue(maxsize=4)   # tối đa ~4 lô chờ -> chặn RAM
        DONE = object()
        err: list[Exception] = []

        def producer() -> None:
            try:
                batch: list[RawKeyframe] = []
                for path, vid in video_specs:
                    for kf in iter_keyframes(
                        path, out_dir, video_id=vid,
                        sample_every_s=sample_every_s or self.sample_every_s,
                        max_frames=max_frames, decode_backend=self.decode_backend,
                        use_gpu=self.use_gpu_decode,
                    ):
                        batch.append(kf)
                        if len(batch) >= bs:
                            q.put(batch)
                            batch = []
                if batch:
                    q.put(batch)
            except Exception as e:  # chuyển lỗi decode về luồng chính
                err.append(e)
            finally:
                q.put(DONE)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        all_raws: list[RawKeyframe] = []
        all_records: list[KeyframeRecord] = []
        import time as _time
        t0 = _time.perf_counter()
        while True:
            item = q.get()
            if item is DONE:
                break
            embs = self._embed_batch_shared(encoders, item)
            emb0 = embs[0]
            emb1 = embs[1] if len(embs) > 1 else None
            for i, r in enumerate(item):
                rec = KeyframeRecord(id=r.id, video_id=r.video_id,
                                     timestamp=r.timestamp, clip_embedding=emb0[i])
                if emb1 is not None:
                    rec.siglip_embedding = emb1[i]
                if ocr is not None:
                    rec.ocr_text = ocr.extract(r)
                all_records.append(rec)
                all_raws.append(r)
            # Tiến độ: cho thấy đang chạy + tốc độ thực (frame/giây) để không tưởng treo.
            n = len(all_records)
            fps = n / max(1e-6, _time.perf_counter() - t0)
            print(f"[index] đã xử lý {n} keyframe ({fps:.1f} frame/giây"
                  f"{', có OCR' if ocr is not None else ''})", flush=True)
        t.join()
        if err:
            raise err[0]
        return all_raws, all_records

    def _collect_records(
        self, video_specs: list[tuple], out_dir: str | Path, *,
        sample_every_s: Optional[float], max_frames: Optional[int],
        enable_ocr: Optional[bool],
    ) -> tuple[list[RawKeyframe], list[KeyframeRecord]]:
        """Cắt + embed keyframe cho các video. Dùng pipeline song song (A4) nếu
        `parallel_index`, ngược lại chạy tuần tự (dễ debug/tất định). Hai đường cho
        KẾT QUẢ GIỐNG NHAU (cùng thứ tự)."""
        if self.parallel_index:
            return self._pipeline_records(
                video_specs, out_dir, sample_every_s=sample_every_s,
                max_frames=max_frames, enable_ocr=enable_ocr)
        all_raws: list[RawKeyframe] = []
        for path, vid in video_specs:
            all_raws.extend(extract_keyframes(
                path, out_dir, video_id=vid,
                sample_every_s=sample_every_s or self.sample_every_s,
                max_frames=max_frames, decode_backend=self.decode_backend,
                use_gpu=self.use_gpu_decode))
        return all_raws, self._embed_raws(all_raws, enable_ocr)

    def index_video(
        self, video_path: str | Path, out_dir: str | Path,
        video_id: Optional[str] = None, *,
        sample_every_s: Optional[float] = None, max_frames: Optional[int] = -1,
        enable_ocr: Optional[bool] = None, enable_asr: Optional[bool] = None,
        enable_caption: Optional[bool] = None,
    ) -> VideoIndexEntry:
        # max_frames=-1 (sentinel) -> dùng mặc định engine; None -> không giới hạn.
        mf = self.max_frames if max_frames == -1 else max_frames
        raws, records = self._collect_records(
            [(video_path, video_id)], out_dir,
            sample_every_s=sample_every_s, max_frames=mf, enable_ocr=enable_ocr)
        if not raws:
            raise RuntimeError("Không trích được keyframe nào từ video.")
        if self.enable_asr if enable_asr is None else enable_asr:
            self._apply_asr(raws, records)
        return self._build_entry(raws, records, video_id=raws[0].video_id,
                                 enable_caption=enable_caption)

    def index_dataset(
        self, video_paths: Sequence[str | Path], out_dir: str | Path,
        dataset_id: str = "__dataset__", *,
        sample_every_s: Optional[float] = None, max_frames: Optional[int] = -1,
        enable_ocr: Optional[bool] = None, enable_asr: Optional[bool] = None,
        enable_caption: Optional[bool] = None,
    ) -> VideoIndexEntry:
        """Index NHIỀU video vào MỘT index chung -> tìm xuyên suốt cả dataset.

        Mỗi keyframe giữ video_id thật nên kết quả biết rõ 'ở video nào, giây mấy'.
        Đây là hướng scale của blueprint (Mục 5): một index cho cả kho, có thể shard
        về sau. Keyframe id dạng '{video_id}/{n}' đã toàn cục duy nhất."""
        mf = self.max_frames if max_frames == -1 else max_frames
        raws, records = self._collect_records(
            [(vp, None) for vp in video_paths], out_dir,
            sample_every_s=sample_every_s, max_frames=mf, enable_ocr=enable_ocr)
        if not raws:
            raise RuntimeError("Không trích được keyframe nào từ dataset.")
        if self.enable_asr if enable_asr is None else enable_asr:
            self._apply_asr(raws, records)
        return self._build_entry(raws, records, video_id=dataset_id,
                                 enable_caption=enable_caption)

    # ---------------------------------------------------------------- query
    def encode_query(self, query: str) -> list[np.ndarray]:
        """Encode query cho TỪNG encoder, mỗi encoder = trung bình các biến thể prompt.

        Trung bình vector đơn vị rồi chuẩn hoá lại (Mục 4.2 — weighted average của
        embedding, trọng số đều)."""
        encoders = self._load_encoders()
        out: list[np.ndarray] = []
        for enc in encoders:
            variants = [
                enc.encode_text(tpl.format(q=query)) for tpl in self.query_templates
            ]
            mean = np.mean(np.stack(variants), axis=0).astype(np.float32)
            out.append(l2_normalize(mean.reshape(1, -1))[0])
        return out

    def encode_image_query(self, image_bytes: bytes) -> list[np.ndarray]:
        """Encode 1 ẢNH truy vấn vào CÙNG không gian với keyframe (mỗi encoder 1 vector).

        VÌ SAO: SigLIP là mô hình image-text -> vector ảnh và vector keyframe cùng không
        gian, so cosine trực tiếp. Bọc ảnh vào RawKeyframe rồi gọi encoder.embed(raw) —
        đúng interface chung nên chạy cả bản thật lẫn mock. Đây là 'đổi phương thức truy
        vấn' (slide Buổi 2): đưa ẢNH MẪU thay vì mô tả chữ, liên kết chặt hơn."""
        query_raw = RawKeyframe(id="__imgquery__", video_id="__query__",
                                timestamp=0.0, image_bytes=image_bytes)
        out: list[np.ndarray] = []
        for enc in self._load_encoders():
            v = np.asarray(enc.embed(query_raw), dtype=np.float32).reshape(1, -1)
            out.append(l2_normalize(v)[0])
        return out

    def neighbors(
        self, entry: VideoIndexEntry, frame_id: str, before: int = 4, after: int = 4,
    ) -> list[RawKeyframe]:
        """Trả các keyframe LÂN CẬN (cùng video, quanh thời điểm) của 1 frame — phục vụ
        'Video Browser' (slide Buổi 2): từ 1 kết quả, xem frame trước/sau để định vị bối
        cảnh, thay vì lưới phẳng rời rạc. Dùng MỌI frame đã lấy mẫu (kể cả bị dedup gộp)
        để timeline dày; ảnh dựng lại từ video gốc khi cần."""
        ref = entry.raws.get(frame_id)
        if ref is None:
            return []
        same = sorted((r for r in entry.raws.values() if r.video_id == ref.video_id),
                      key=lambda r: r.timestamp)
        idx = next((i for i, r in enumerate(same) if r.id == frame_id), None)
        if idx is None:
            return []
        return same[max(0, idx - before): idx + after + 1]

    def search_by_image(
        self, entry: VideoIndexEntry, image_bytes: bytes, top_k: int = 8,
    ) -> list:
        """Tìm keyframe GIỐNG một ảnh mẫu (image-to-video). Chỉ dùng tín hiệu THỊ GIÁC
        (dense ensemble), KHÔNG BM25 (ảnh không có 'chữ' để so). Trả top_k coarse."""
        qvecs = self.encode_image_query(image_bytes)
        retriever = CoarseRetriever(entry.index)
        coarse = retriever.search(
            query_clip_vec=qvecs[0],
            query_siglip_vec=qvecs[1] if len(qvecs) > 1 else None,
            query_text=None,          # query ẢNH: không có nhánh BM25
            top_k=top_k,
        )
        return coarse[:top_k]

    def search(
        self, entry: VideoIndexEntry, query: str, top_k: int = 8, rerank: bool = False,
    ) -> list:
        """Search 2 tầng.

        Tầng COARSE (luôn chạy): ensemble SigLIP -> RRF fuse -> ranked list nhanh.
        Tầng RERANK (nếu rerank=True): đưa top `rerank_pool` ứng viên coarse cho VLM
        chấm lại theo hiểu-ngôn-ngữ-thật -> trả top_k đã rerank. Chậm hơn nhiều (CPU)
        nhưng chính xác về tổ hợp từ/ngữ cảnh.
        """
        qvecs = self.encode_query(query)
        retriever = CoarseRetriever(entry.index)
        pool = max(top_k, self.rerank_pool) if rerank else top_k
        # Trọng số BM25 theo LOẠI query (benchmark: chữ cần cao, thị giác cần thấp).
        bm25_w = (adaptive_bm25_weight(query, self.bm25_weight, self.bm25_weight_high)
                  if self.adaptive_bm25 else self.bm25_weight)
        coarse = retriever.search(
            query_clip_vec=qvecs[0],
            query_siglip_vec=qvecs[1] if len(qvecs) > 1 else None,
            # query text -> BM25 trên OCR/objects/caption: khớp CHỮ trên biển hiệu.
            # RRF gộp dense (hình ảnh) + sparse (chữ) -> tìm được cả cảnh lẫn text.
            query_text=query,
            top_k=pool,
            weights={"bm25": bm25_w},
        )
        if not rerank or not coarse:
            return coarse[:top_k]

        # VLM rerank: context = RawKeyframe (VLM cần "nhìn" ảnh — lấy từ RAM/đĩa).
        context = {c.keyframe_id: entry.raws[c.keyframe_id]
                   for c in coarse if c.keyframe_id in entry.raws}
        try:
            reranked = self._get_reranker().rerank(query, coarse, context)
            return reranked[:top_k]
        except Exception as e:  # pragma: no cover - fallback khi VLM lỗi/thiếu model
            # VLM lỗi (chưa tải xong model, thiếu RAM...) -> KHÔNG làm vỡ tìm kiếm,
            # trả kết quả coarse SigLIP (vẫn tốt). Log để người dùng biết.
            print(f"[video_engine] VLM rerank lỗi ({e!r}); dùng kết quả coarse.")
            return coarse[:top_k]

    def search_temporal(
        self, entry: VideoIndexEntry, events: Sequence[str], *,
        per_event_k: int = 20, max_results: int = 50,
    ) -> list[TemporalMatch]:
        """Tìm chuỗi sự kiện ĐÚNG THỨ TỰ thời gian: "cảnh A TRƯỚC cảnh B (trước C…)".

        Giải thách thức #3 của đề thi (temporal logic, Mục 4.5): mỗi phần tử `events`
        là 1 câu mô tả một cảnh, theo THỨ TỰ mong muốn. Ta search coarse từng cảnh để
        lấy ứng viên keyframe, rồi LỌC CỨNG (hard-constraint) giữ lại các tổ hợp CÙNG
        video có timestamp tăng dần đúng thứ tự. Đây KHÔNG phải similarity — một tổ
        hợp sai thứ tự bị loại dứt khoát dù điểm cao (Mục 4.5, không gộp vào fusion).

        Args:
            events: >=2 câu mô tả cảnh, theo đúng thứ tự thời gian mong muốn.
            per_event_k: số ứng viên coarse lấy cho MỖI cảnh (rộng hơn -> nhiều tổ hợp).
            max_results: trần số chuỗi trả về (tránh bùng nổ tổ hợp).

        Returns:
            list[TemporalMatch] — mỗi cái là chuỗi keyframe cùng video, timestamp tăng
            dần đúng thứ tự; rỗng nếu không có tổ hợp hợp lệ (đúng cảnh nhưng sai thứ tự).
        """
        if len(events) < 2:
            raise ValueError("search_temporal cần >= 2 cảnh để kiểm thứ tự thời gian.")
        # Khoá theo CHỈ SỐ để không đụng nhau khi 2 cảnh mô tả trùng chữ.
        temporal_order = [{"event": f"{i}:{e}", "order": i} for i, e in enumerate(events)]
        candidates_by_event = {
            f"{i}:{e}": self.search(entry, e, top_k=per_event_k, rerank=False)
            for i, e in enumerate(events)
        }
        return temporal_consistency_filter(
            temporal_order, candidates_by_event, max_results=max_results)
