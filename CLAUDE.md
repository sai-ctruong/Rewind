# CLAUDE.md — Blueprint xây dựng Trợ lý ảo Truy xuất Đa phương tiện (AIC 2026)

> **Cách dùng file này**: Đây là bản đặc tả kỹ thuật đầy đủ để Claude Code đọc và
> triển khai. Đọc TOÀN BỘ file này trước khi viết bất kỳ dòng code nào. Thực hiện
> **tuần tự theo từng Phase** ở Mục 8. Sau mỗi Phase, dừng lại, chạy test tương ứng,
> báo cáo kết quả và xin xác nhận trước khi sang Phase kế tiếp. Không bỏ qua Phase,
> không viết code cho Phase sau khi Phase trước chưa có Definition of Done đạt.

---

## 0. Bối cảnh & Mục tiêu (Context & Goal)

Xây dựng hệ thống **Trợ lý ảo truy xuất đa phương tiện** cho Hội thi AI Challenge
(AIC) 2026 TP.HCM, mô phỏng thể thức **Lifelog Search Challenge (LSC)** và
**Video Browser Showdown (VBS)**. Hệ thống phải xử lý được 4 dạng bài toán:

| Dạng | Input | Output |
|---|---|---|
| **KIS** (Known-Item Search — Video/Textual) | 1 mô tả hoặc 1 đoạn video mẫu | Top-1/Top-5 keyframe chính xác tuyệt đối |
| **AVS** (Ad-hoc Video Search) | Mô tả tổng quát | Danh sách TẤT CẢ đoạn khớp, xếp hạng theo độ tương đồng |
| **VQA** (Video Question Answering) | Video dài + câu hỏi | Câu trả lời text có suy luận (đếm, thứ tự thời gian) |
| **KISC** (Conversational KIS — mới 2026) | Hội thoại nhiều lượt | Trợ lý chủ động hỏi lại để thu hẹp phạm vi (**đã xây ở `kisc_module/`**) |

**Quy mô dữ liệu (Scale)**: video từ vài giờ đến **vài trăm giờ** → ước tính hàng
triệu keyframe. Đây là ràng buộc kiến trúc quan trọng nhất — mọi thiết kế phải
chịu được quy mô này, KHÔNG được thiết kế kiểu "demo nhỏ rồi tính sau".

---

## 1. Ràng buộc không thương lượng (Non-negotiable constraints)

1. **Độ chính xác là ưu tiên số 1** (accuracy over speed). Chấp nhận pipeline
   nhiều tầng (multi-stage), có tầng rerank tốn kém (LVLM), miễn tổng thời gian
   phản hồi nằm trong giới hạn cho phép của vòng thi (xem Mục 4.4 — time budget).
2. **Không được đánh mất recall ở tầng lọc thô** (coarse filtering). Bất kỳ
   keyframe nào bị loại ở tầng đầu sẽ KHÔNG BAO GIỜ được xét lại ở các tầng sau
   → tầng coarse phải ưu tiên **recall cao** hơn precision; tầng fine rerank mới
   tối ưu precision.
3. **Kiến trúc phải scale-ready ngay từ đầu**: không dùng brute-force linear scan
   cho vector search một khi số lượng keyframe vượt quá ~50,000. Xem Mục 5.
4. Toàn bộ code viết bằng **Python 3.10+**, có type hints, có docstring giải
   thích cả lý thuyết lẫn lý do lựa chọn kỹ thuật (không chỉ code suông).
5. Mọi module phải có unit test tối thiểu, và có thể chạy **offline bằng mock
   data** (không phụ thuộc GPU/API thật) để test logic trước khi cắm dữ liệu thật
   — theo đúng pattern đã dùng ở `kisc_module/retriever.py::MockRetriever`.

---

## 2. So sánh & lựa chọn công nghệ (Tech stack decisions)

### 2.1 Embedding model (multimodal encoder)

| Lựa chọn | Ưu điểm | Nhược điểm | Khuyến nghị |
|---|---|---|---|
| CLIP (OpenAI ViT-L/14) | Baseline chuẩn, dữ liệu BTC đã cấp sẵn feature này | Độ phân giải ngữ nghĩa hạn chế với văn bản dài | Dùng làm baseline bắt buộc (đã có sẵn) |
| **SigLIP / SigLIP2** | Zero-shot tốt hơn CLIP ở benchmark retrieval, sigmoid loss ổn định hơn | Cần tự trích xuất lại toàn bộ keyframe | **Khuyến nghị dùng làm encoder chính**, ensemble với CLIP có sẵn |
| EVA-CLIP / OpenCLIP (ViT-bigG) | Độ chính xác cao nhất trong họ CLIP mở | Nặng, chậm, cần GPU mạnh để trích xuất hàng triệu frame | Dùng nếu có đủ compute; nếu không, SigLIP-large là điểm cân bằng tốt |
| InternVideo2 / Video-LLaVA embedding | Hiểu được motion/action tốt hơn (video-level, không chỉ frame-level) | Phức tạp, chậm hơn nhiều | Dùng bổ sung cho VQA/action reasoning, không dùng cho coarse retrieval toàn bộ dataset |

**Quyết định**: Dùng **ensemble 2 encoder** — CLIP features có sẵn (không tốn
compute) + SigLIP tự trích xuất (độ chính xác cao hơn) — kết hợp điểm bằng
**weighted fusion** ở tầng coarse. Việc dùng 2 encoder độc lập giúp giảm rủi ro
"cùng sai" (correlated errors) của 1 model đơn lẻ — đây là kỹ thuật ensemble kinh
điển để tăng accuracy mà không cần model lớn hơn.

### 2.2 Vector Database / ANN search

| Lựa chọn | Ưu điểm | Nhược điểm | Khuyến nghị |
|---|---|---|---|
| Faiss (Flat, brute-force) | Chính xác 100% (exact search) | Không scale — hàng triệu vector sẽ quá chậm | Chỉ dùng cho tầng fine rerank (top 500-1000 candidate, không phải toàn bộ dataset) |
| Faiss IVF-PQ | Nén vector (product quantization) → tiết kiệm RAM đáng kể, tốc độ cao | Mất độ chính xác do nén lossy | Dùng cho tầng coarse trên toàn bộ dataset khi > 1 triệu vector |
| Faiss HNSW | Cân bằng tốt giữa tốc độ và độ chính xác, không nén (giữ float) | Tốn RAM hơn IVF-PQ (đổi lại chính xác hơn) | **Khuyến nghị dùng nếu tổng vector < ~5 triệu và đủ RAM** — ưu tiên vì đây là bài toán ưu tiên accuracy |
| Milvus / Qdrant | Hỗ trợ hybrid filter (metadata + vector) built-in, dễ scale phân tán | Thêm overhead vận hành (cần chạy service riêng) | Dùng nếu team quen vận hành service; nếu ưu tiên đơn giản, Faiss HNSW + filter thủ công là đủ |

**Quyết định**: **Faiss HNSW** cho tầng coarse (ưu tiên accuracy theo ràng buộc
Mục 1.1), kết hợp **metadata pre-filter** (time/location/objects) trước khi vào
HNSW nếu filter đã đủ hẹp — giảm không gian tìm kiếm mà không cần đánh đổi bằng
PQ compression. Chỉ chuyển sang IVF-PQ nếu profiling cho thấy RAM là nút thắt
thực sự (đo đạc ở Phase 3, không giả định trước).

### 2.3 LVLM cho tầng fine rerank & VQA

| Lựa chọn | Ưu điểm | Nhược điểm |
|---|---|---|
| Claude (Sonnet/Opus với vision) qua API | Suy luận tốt, dễ tích hợp qua `anthropic` SDK | Chi phí + độ trễ mỗi lần gọi — chỉ áp dụng cho top-K nhỏ |
| Gemini / GPT-4V | Tương tự, có thể ensemble thêm | Tương tự |
| Model mở (Qwen2-VL, InternVL) tự host | Không giới hạn API, chạy local | Cần GPU mạnh, setup phức tạp hơn |

**Quyết định**: Dùng Claude API cho tầng fine rerank (chỉ áp lên **top 50-100
candidate sau coarse retrieval**, không bao giờ áp lên toàn bộ dataset — đúng
tinh thần "coarse filter cực nhanh, fine rerank chính xác" từ thách thức #2
trong slide tập huấn BTC).

---

## 3. Kiến trúc pipeline tổng thể (Overall architecture)

```
                            ┌─────────────────────────┐
                            │   INDEXING (offline)     │
                            │  Video → Keyframe/Object  │
                            │  → CLIP + SigLIP embed    │
                            │  → Dedup gần trùng        │
                            │  → Faiss HNSW + metadata  │
                            └─────────────────────────┘
                                        │
  ┌─────────────────────────────────────┼─────────────────────────────────────┐
  │                          RETRIEVAL (online, mỗi query)                     │
  │                                                                             │
  │  Query text/hội thoại                                                      │
  │        │                                                                   │
  │        ▼                                                                   │
  │  [1] Query Understanding (LLM)  → JSON: {objects, actions, attrs,           │
  │       tách temporal order          time, location, temporal_order}         │
  │        │                                                                   │
  │        ▼                                                                   │
  │  [2] Coarse Retrieval (recall-first)                                       │
  │       Dense (CLIP+SigLIP ensemble) + Sparse (BM25 trên OCR/ASR/Objects)     │
  │       + Metadata pre-filter  →  top 500-1000 candidate                     │
  │        │                                                                   │
  │        ▼                                                                   │
  │  [3] Score Fusion (Reciprocal Rank Fusion — RRF)  → top 100                │
  │        │                                                                   │
  │        ▼                                                                   │
  │  [4] Fine Rerank (LVLM verifier, chỉ top 50-100)  → top 20                 │
  │        │                                                                   │
  │        ▼                                                                   │
  │  [5] Temporal consistency check (nếu query có ràng buộc thứ tự)            │
  │        │                                                                   │
  │        ▼                                                                   │
  │  [6] Route theo loại bài toán:                                             │
  │       KIS  → trả Top-1/Top-5                                               │
  │       AVS  → trả toàn bộ danh sách đã rerank, không cắt Top-K nhỏ           │
  │       VQA  → đưa temporal window liên quan cho LVLM trả lời câu hỏi         │
  │       KISC → nếu độ tự tin thấp, gọi kisc_module để hỏi lại (đã xây)        │
  └─────────────────────────────────────────────────────────────────────────┘
```

**Lý do bắt buộc phải có tầng [2] Coarse trước khi vào [4] Fine LVLM rerank**:
với vài trăm giờ video (~hàng triệu frame), chạy LVLM (vốn chậm, ~1-3s/frame)
trên toàn bộ dataset là bất khả thi trong thời gian thi. Tầng coarse (vector
search, mili-giây/query) phải làm nhiệm vụ "lọc thô cực nhanh" trước — đúng
nguyên tắc từ thách thức #2 (Data sparsity & Quy mô lớn) trong slide BTC.

---

## 4. Chi tiết thuật toán từng tầng

### 4.1 Query Understanding (LLM decomposition)

Bắt buộc parse câu query tự nhiên thành JSON có cấu trúc trước khi retrieval,
KHÔNG được search trực tiếp bằng câu thô (giảm accuracy do semantic gap):

```json
{
  "objects": ["chìa khóa", "móc khóa hình gấu bông màu hồng"],
  "actions": ["đánh rơi"],
  "location": "quầy bán hoa quả",
  "attributes": {"color": ["hồng"]},
  "time_constraint": null,
  "temporal_order": null,
  "query_type": "KIS_textual"
}
```

Với query có nhiều mệnh đề thời gian (vd "cởi mũ TRƯỚC KHI vào phòng"), bắt buộc
điền `temporal_order: [{"event": "cởi mũ", "order": 1}, {"event": "vào phòng",
"order": 2}]` — đây là input cho bước [5] Temporal consistency check.

### 4.2 Multi-Query Expansion (tăng recall — kỹ thuật bắt buộc)

Sinh **3-5 biến thể diễn đạt khác nhau** của cùng 1 query bằng LLM (đồng nghĩa,
góc nhìn khác — xem "Generative Query Expansion" trong slide BTC), encode từng
biến thể, lấy **trung bình có trọng số (weighted average) của các embedding**
hoặc union kết quả search của từng biến thể trước khi fusion. Lý do: một câu mô
tả duy nhất có thể không khớp chính xác cách model đã học biểu diễn khái niệm đó
— nhiều biến thể tăng khả năng "trúng" cách biểu diễn đúng.

### 4.3 Score Fusion — Reciprocal Rank Fusion (RRF)

Không dùng trung bình cộng điểm số thô (các thang điểm CLIP similarity, BM25,
metadata match không cùng scale) — dùng **RRF**:

```
RRF_score(d) = Σ (1 / (k + rank_i(d)))   với mỗi nguồn tín hiệu i, k=60 (mặc định chuẩn)
```

RRF không cần chuẩn hóa thang điểm giữa các nguồn khác nhau, là kỹ thuật fusion
chuẩn trong Information Retrieval khi kết hợp nhiều ranked list không đồng nhất
về đơn vị đo.

### 4.4 Fine Rerank bằng LVLM + Time budget

Với mỗi candidate trong top 50-100, gọi LVLM với prompt dạng:
`"Cho ảnh keyframe này và mô tả: '{query}', đánh giá độ khớp từ 0-10 và giải
thích ngắn gọn lý do."` — Batch hóa nhiều candidate/request khi API hỗ trợ để
giảm số lần gọi. **Time budget**: đặt giới hạn cứng (vd 15-20s/query) — nếu vượt
ngưỡng, cắt bớt top-K rerank (giảm từ 100 xuống 30) thay vì bỏ qua tầng rerank
hoàn toàn (giữ nguyên tắc accuracy-first trong phạm vi cho phép).

### 4.5 Temporal Consistency Check (giải quyết thách thức #3 — temporal logic)

Với các cặp sự kiện có ràng buộc thứ tự (`temporal_order`), sau khi có candidate
keyframe cho mỗi sự kiện, chỉ giữ lại các cặp (video, timestamp1, timestamp2)
thỏa `timestamp1 < timestamp2` (đúng thứ tự yêu cầu). Đây là bước **loại trừ
logic cứng (hard constraint filtering)**, không phải similarity — không được
gộp chung vào bước fusion điểm số ở 4.3.

---

## 5. Chiến lược xử lý quy mô lớn (Scale strategy — vài trăm giờ video)

1. **Deduplication trước khi index**: video lifelog có rất nhiều frame liên tiếp
   gần giống hệt nhau (người đứng yên, cảnh tĩnh). Trước khi đưa vào Faiss, tính
   cosine similarity giữa các keyframe liên tiếp cùng video; nếu > ngưỡng (vd
   0.97), gộp thành 1 cụm (giữ 1 đại diện + lưu khoảng thời gian cụm che phủ).
   Điều này có thể giảm 30-60% số vector cần index mà **không mất recall** (dữ
   liệu bị gộp gần như giống hệt nhau về ngữ nghĩa).
2. **Batch GPU embedding extraction**: xử lý theo batch lớn (256-512
   keyframe/batch), dùng `fp16`/`bfloat16` để tăng throughput, có thể giảm ~40%
   thời gian trích xuất so với fp32 mà độ chính xác gần như không đổi.
3. **Sharding theo video/thời gian**: chia index thành nhiều shard (vd theo
   từng batch video được nạp), search song song trên các shard rồi merge kết
   quả — tránh phải rebuild toàn bộ index khi có dữ liệu mới.
4. **Lazy-load rerank**: chỉ tải ảnh gốc/frame chất lượng cao cho top-K sau
   fusion (bước 4.4), KHÔNG tải toàn bộ ảnh vào RAM cùng lúc.
5. **Caching**: cache embedding của các query đã hỏi trong phiên KISC (nhiều
   lượt hội thoại dùng lại 1 phần filter cũ) — tránh encode lại từ đầu mỗi lượt.

---

## 6. Cấu trúc thư mục dự án (Project structure)

```
aic2026_system/
├── CLAUDE.md                    <- chính là file này
├── kisc_module/                 <- ĐÃ CÓ, không viết lại, chỉ mở rộng CANDIDATE_ATTRIBUTES
├── ingestion/
│   ├── dedup.py                 <- Phase 2: gộp keyframe gần trùng
│   ├── embed_clip.py            <- Phase 2: load CLIP feature có sẵn từ BTC
│   ├── embed_siglip.py          <- Phase 2: tự trích xuất SigLIP
│   ├── ocr_asr_extract.py       <- Phase 2: OCR (PaddleOCR) + ASR (Whisper)
│   └── build_index.py           <- Phase 3: build Faiss HNSW + BM25 index
├── retrieval/
│   ├── query_understanding.py   <- Phase 4: LLM decompose query -> JSON
│   ├── query_expansion.py       <- Phase 4: multi-query expansion
│   ├── coarse_retriever.py      <- Phase 3: implement HybridRetriever thật
│   ├── fusion.py                <- Phase 5: RRF
│   ├── fine_rerank.py           <- Phase 5: LVLM verifier
│   ├── temporal_check.py        <- Phase 6: temporal consistency filter
│   └── vqa_module.py            <- Phase 7: video question answering
├── evaluation/
│   ├── metrics.py                <- Phase 8: Recall@K, MRR, nDCG
│   └── run_eval.py
├── ui/                            <- Phase 9: giao diện (tùy chọn framework)
├── tests/                         <- unit test cho từng module trên
└── configs/
    └── settings.yaml              <- ngưỡng dedup, top-K từng tầng, time budget...
```

## 7. Data Schemas (bắt buộc tuân theo, không tự ý đổi field name)

```python
# ingestion/schemas.py
@dataclass
class KeyframeRecord:
    id: str
    video_id: str
    timestamp: float
    clip_embedding: np.ndarray       # từ BTC cấp sẵn
    siglip_embedding: np.ndarray     # tự trích xuất Phase 2
    objects: list[str]               # từ BTC cấp sẵn (Open Images 600 categories)
    ocr_text: str | None
    asr_text: str | None             # transcript đoạn audio quanh timestamp này
    is_cluster_representative: bool  # True nếu là đại diện sau dedup
    cluster_span: tuple[float, float] | None  # (t_start, t_end) nếu là cụm

@dataclass
class StructuredQuery:
    raw_text: str
    objects: list[str]
    actions: list[str]
    location: str | None
    attributes: dict
    time_constraint: str | None
    temporal_order: list[dict] | None
    query_type: str  # "KIS_video" | "KIS_textual" | "AVS" | "VQA" | "KISC"
```

---

## 8. Roadmap thực thi theo Phase (thực hiện tuần tự, KHÔNG nhảy cóc)

| Phase | Nội dung | Definition of Done |
|---|---|---|
| **0** | Setup project structure theo Mục 6, viết `configs/settings.yaml` với các ngưỡng mặc định (dedup threshold=0.97, coarse top-K=1000, rerank top-K=100, time_budget=20s) | Chạy `pytest` trống không lỗi, cấu trúc thư mục đúng Mục 6 |
| **1** | Viết `ingestion/dedup.py` + unit test với dữ liệu keyframe giả (synthetic) mô phỏng chuỗi frame gần giống nhau | Test chứng minh giảm >30% số record trên dữ liệu giả mà không mất representative nào |
| **2** | Viết `embed_clip.py` (load feature BTC cấp) + `embed_siglip.py` (trích xuất mới) + `ocr_asr_extract.py`, test trên 1 video mẫu nhỏ | Ra được `KeyframeRecord` đầy đủ field cho ≥1 video mẫu thật |
| **3** | Viết `build_index.py` (Faiss HNSW) + `coarse_retriever.py` implement `HybridRetriever` thật (thay `MockRetriever`) | Query mẫu trả về candidate hợp lý trong <200ms trên tập test |
| **4** | Viết `query_understanding.py` + `query_expansion.py` (gọi Claude API), test với 5 câu query mẫu từ case study trong slide BTC | JSON output đúng schema Mục 7 cho cả 5 câu |
| **5** | Viết `fusion.py` (RRF) + `fine_rerank.py` (LVLM verifier có time budget) | Trên tập test có ground-truth nhỏ (tự tạo ~20 cặp query-answer), Top-1 accuracy được đo và ghi log |
| **6** | Viết `temporal_check.py`, test với case "cởi mũ trước khi vào phòng" kiểu | Phân biệt đúng 2 trường hợp thứ tự khác nhau trong bộ test tự tạo |
| **7** | Viết `vqa_module.py` (retrieve temporal window + gọi LVLM trả lời, có xử lý đếm/counting) | Trả lời đúng ví dụ case study VQA trong slide BTC (đếm số nến, xác định người tặng quà) |
| **8** | Tích hợp `kisc_module/` đã có vào `retrieval/coarse_retriever.py` làm `HybridRetriever` thật (thay `MockRetriever` trong `kisc_module`) | `kisc_module/demo.py` chạy đúng trên dữ liệu thật thay vì mock |
| **9** | Viết `evaluation/metrics.py` (Recall@K, MRR cho KIS/AVS; Exact Match cho VQA) + `run_eval.py` | Có báo cáo số liệu trên bộ test end-to-end |
| **10** | UI tối giản (CLI hoặc web đơn giản) hiển thị kết quả + hỗ trợ hội thoại KISC trực tiếp | Demo được toàn bộ luồng end-to-end cho người ngoài xem |

**Sau mỗi Phase**: chạy test tương ứng, in báo cáo ngắn (số liệu đo được, giới
hạn phát hiện), rồi DỪNG lại chờ xác nhận — không tự động chạy tiếp Phase kế.

---

## 9. Tiêu chí đánh giá (Evaluation)

| Bài toán | Metric | Ghi chú |
|---|---|---|
| KIS | Top-1 accuracy, Top-5 accuracy | Đúng ground-truth ở vị trí đầu tiên là quan trọng nhất |
| AVS | Recall@K, mAP (mean Average Precision) | Đánh giá cả danh sách, không chỉ Top-1 |
| VQA | Exact Match, F1 (đối với câu trả lời dạng số/tên) | Cần bộ câu hỏi-đáp tự tạo để test trước khi có đề thật |
| KISC | Số lượt hội thoại trung bình đến khi hội tụ | Càng ít lượt mà vẫn đúng đáp án càng tốt (đã đo được ở `kisc_module/demo.py`: hội tụ 2 lượt) |

---

## 10. Hướng dẫn thực thi cho Claude Code

1. Đọc toàn bộ file này 1 lần trước khi bắt đầu.
2. Không cài đặt thư viện không cần thiết — dùng đúng stack đã quyết định ở Mục 2.
3. Nếu không chắc chắn về 1 quyết định kỹ thuật không có trong file này, DỪNG LẠI
   và hỏi người dùng, không tự suy đoán.
4. Viết code có docstring giải thích LÝ DO chọn cách làm đó (không chỉ mô tả code
   làm gì) — để người review (Trường) hiểu được kiến trúc, không chỉ chạy được.
5. Mỗi Phase phải có ít nhất 1 file test trong `tests/` chạy được bằng `pytest`.
6. Không refactor code của `kisc_module/` đã có sẵn trừ khi được yêu cầu rõ ràng.

