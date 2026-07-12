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


@dataclass
class VideoIndexEntry:
    """Kết quả index 1 video: Faiss index + map id -> RawKeyframe (đường dẫn ảnh)."""

    video_id: str
    index: KeyframeIndex
    raws: dict[str, RawKeyframe]
    num_sampled: int          # số keyframe sau bước cắt thô (histogram)
    num_indexed: int          # số keyframe sau dedup ngữ nghĩa (vào index)
    ocr_by_id: dict[str, str] = field(default_factory=dict)  # chữ OCR đọc được / keyframe

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
            ocr_by_id=payload["ocr_by_id"],
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
        bm25_weight: float = 3.0,               # [PROVISIONAL] trọng số OCR/text trong RRF
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
        self.bm25_weight = bm25_weight
        self.embed_batch_size = embed_batch_size
        self.decode_backend = decode_backend
        self.use_gpu_decode = use_gpu_decode
        self.parallel_index = parallel_index
        self._encoders: Optional[list] = None   # nạp lười
        self._reranker: Optional[FineReranker] = None  # VLM rerank, nạp lười
        self._ocr = None                        # EasyOCR, nạp lười

    # ------------------------------------------------------------- encoders
    def _load_encoders(self) -> list:
        if self._encoders is None:
            from ingestion.embed_siglip import SiglipEncoder

            self._encoders = [SiglipEncoder(model_name=n) for n in self.encoder_names]
        return self._encoders

    def set_encoders(self, encoders: list) -> None:
        """Bơm encoder ngoài (mock trong test / model tuỳ chọn). Mỗi encoder cần
        `.embed(raw) -> vec` và `.encode_text(str) -> vec`."""
        self._encoders = list(encoders)[:2]

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

    def _embed_raws(self, raws: list[RawKeyframe],
                    enable_ocr: Optional[bool] = None) -> list[KeyframeRecord]:
        """Embed mỗi keyframe bằng cả 2 encoder (slot clip + siglip) + OCR (nếu bật).

        Embed THEO LÔ cho từng encoder (A1) thay vì lẻ từng ảnh: gom hết raws -> một
        loạt tensor trên GPU. OCR đọc chữ trên khung hình (biển hiệu, phụ đề) -> lưu
        ocr_text để đưa vào BM25 -> tìm được bằng CHỮ chứ không chỉ hình ảnh."""
        encoders = self._load_encoders()
        use_ocr = self.enable_ocr if enable_ocr is None else enable_ocr
        ocr = self._get_ocr() if use_ocr else None
        emb0 = self._embed_all(encoders[0], raws)
        emb1 = self._embed_all(encoders[1], raws) if len(encoders) > 1 else None
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
        for r in raws:
            if r.id not in kept_ids:
                r.image_bytes = None
        return VideoIndexEntry(
            video_id=video_id, index=KeyframeIndex.build(kept),
            raws={r.id: r for r in raws},
            num_sampled=len(records), num_indexed=len(kept),
            ocr_by_id={r.id: r.ocr_text for r in records if r.ocr_text},
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
        while True:
            item = q.get()
            if item is DONE:
                break
            emb0 = self._embed_all(encoders[0], item)
            emb1 = self._embed_all(encoders[1], item) if len(encoders) > 1 else None
            for i, r in enumerate(item):
                rec = KeyframeRecord(id=r.id, video_id=r.video_id,
                                     timestamp=r.timestamp, clip_embedding=emb0[i])
                if emb1 is not None:
                    rec.siglip_embedding = emb1[i]
                if ocr is not None:
                    rec.ocr_text = ocr.extract(r)
                all_records.append(rec)
                all_raws.append(r)
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
        enable_ocr: Optional[bool] = None,
    ) -> VideoIndexEntry:
        # max_frames=-1 (sentinel) -> dùng mặc định engine; None -> không giới hạn.
        mf = self.max_frames if max_frames == -1 else max_frames
        raws, records = self._collect_records(
            [(video_path, video_id)], out_dir,
            sample_every_s=sample_every_s, max_frames=mf, enable_ocr=enable_ocr)
        if not raws:
            raise RuntimeError("Không trích được keyframe nào từ video.")
        return self._build_entry(raws, records, video_id=raws[0].video_id)

    def index_dataset(
        self, video_paths: Sequence[str | Path], out_dir: str | Path,
        dataset_id: str = "__dataset__", *,
        sample_every_s: Optional[float] = None, max_frames: Optional[int] = -1,
        enable_ocr: Optional[bool] = None,
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
        return self._build_entry(raws, records, video_id=dataset_id)

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
        coarse = retriever.search(
            query_clip_vec=qvecs[0],
            query_siglip_vec=qvecs[1] if len(qvecs) > 1 else None,
            # query text -> BM25 trên OCR/objects/caption: khớp CHỮ trên biển hiệu.
            # RRF gộp dense (hình ảnh) + sparse (chữ) -> tìm được cả cảnh lẫn text.
            query_text=query,
            top_k=pool,
            # BM25 (OCR/chữ) nặng hơn để khớp biển hiệu nổi lên (dense có 2 encoder).
            weights={"bm25": self.bm25_weight},
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
