# 📋 TASKS.md — Kế hoạch đầy đủ (Hệ thống tìm kiếm video AIC 2026)

> **Nguồn yêu cầu:** `CLAUDE.md` (blueprint kỹ thuật) + **Slide Tập huấn Buổi 2**
> (khung "Hệ thống tìm kiếm video" — ThS. Nguyễn Quang Thức) + **benchmark thật**
> (RTX 3060, 37 nhãn). Trạng thái tổng quan xem `TEAM.md`.
>
> Cập nhật: 2026-07-14 · Nhánh: `main` · 185 test xanh.

---

## 0. Khung tham chiếu — hệ thống tìm kiếm video cần 3 TRỤ CỘT (theo slide)

```
   [1] MÔ HÌNH TRUY VẤN  →  [2] CƠ CHẾ HIỂN THỊ (Video Browser)  →  [3] PHẢN HỒI NGƯỜI DÙNG
       (query → keyframe)        (trình bày kết quả)                    (relevance feedback)
              └───────────────────────── vòng lặp ──────────────────────────────┘
```

Slide nhấn 3 đánh đổi cốt lõi: **tốc độ ↔ sức mạnh ↔ chi phí**; và kiến trúc VLM
**late-fusion** (nhanh, dữ liệu lớn — CLIP/SigLIP) cho tầng thô + **early-fusion**
(grounding, chạy lại mỗi query — Qwen2-VL) cho bước cuối. Cả hai đã có trong project.

**Tình trạng theo trụ cột:** [1] mạnh (thiếu truy vấn ảnh/sketch, multi-modal) ·
[2] cơ bản (thiếu browser nâng cao) · [3] có KISC nhưng chưa nối vào tìm video thật.

---

## 1. ✅ ĐÃ XONG (bản đồ nhanh)

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| **Late-fusion coarse** (SigLIP2 + SigLIP-ml ensemble, RRF) | ✅ | encoder chính, feature precompute |
| **Early-fusion rerank** (Qwen2-VL-2B, cross-attention) | ✅ | bước cuối top-K, +hit@1 0.35→0.60 |
| **Prompt ensemble** (nhiều biến thể câu) | ✅ | đúng "Prompt Engineering & Ensembling" |
| **Temporal** ("cảnh A trước B") | ✅ B4 | tab ⏱️ Chuỗi + `search_temporal` |
| **Dedup** frame gần giống (histogram + ngữ nghĩa) | ✅ | giảm "quá nhiều frame giống nhau" |
| **BM25 đa tín hiệu** (OCR + ASR + caption) + **adaptive weight** | ✅ | trọng số theo loại query |
| **ASR/Whisper** (tìm theo lời nói) | ✅ B3 | opt-in |
| **Caption ngữ cảnh** (quan hệ + hoàn cảnh) | 🟡 B5 | logic xong; local Qwen kẹt VRAM 6GB → chờ API |
| **KISC** (hội thoại, khám phá/khai phá) | ✅ | trên data mẫu — CHƯA nối video thật |
| **Scale**: batch embed ×2.15 · lưu/nạp index · decode backend · song song | ✅ A1–A4 | benchmark thật |
| **Benchmark harness** + 37 nhãn thật + tuning | ✅ B1 | xem Mục 5 |

---

## 2. TRỤ CỘT [1] — MÔ HÌNH TRUY VẤN

### ✅ Đã có
Late-fusion ensemble · early-fusion rerank · prompt ensemble · temporal · ASR · caption · adaptive BM25.

### 🔲 Q1. Truy vấn bằng ẢNH (image → video)  ⭐ ƯU TIÊN CAO (mới từ slide)
Slide: *"thay đổi phương thức truy vấn để liên kết chặt hơn"*. Người dùng đưa 1 ẢNH mẫu
→ SigLIP encode ảnh (ĐÃ có `embed`) → search như query text. **Dễ nhất** vì tận dụng
encoder sẵn có; giá trị cao (nhiều bài KIS cho sẵn ảnh/đoạn video mẫu).
**Làm:** endpoint `/api/video/search_image` (nhận ảnh upload) + ô upload trên UI.
**File:** `retrieval/video_engine.py` (search_by_image), `ui/app.py`, `ui/index.html`

### 🔲 Q2. Truy vấn bằng SKETCH / ảnh sinh từ mô tả (mới từ slide)
Slide nêu "truy vấn bằng ảnh sinh ra từ sketch". Cho phép vẽ phác/tạo ảnh từ text rồi
dùng làm image-query (dựa trên Q1). Bước sau Q1.
**File:** `ui/` (canvas sketch), tái dùng Q1.

### 🔲 Q3. Kết hợp NHIỀU KIỂU truy vấn (multi-modal fusion) (mới từ slide)
Slide: *"cơ chế kết hợp nhiều kiểu truy vấn"*. Gộp text + ảnh (+ temporal) trong MỘT
truy vấn → RRF nhiều nguồn. Dựa trên Q1.
**File:** `retrieval/video_engine.py`

### 🔲 Q4. Query understanding → pre-filter (B2 cũ)
Parse câu → `StructuredQuery {objects, actions, location, time, temporal_order}` trước
search → thu hẹp, tăng precision + tự động route sang temporal khi câu có "trước/sau".
`retrieval/query_understanding.py` có sẵn (mock/Claude) nhưng **chưa nối** vào video_engine.
**File:** `retrieval/video_engine.py`, `retrieval/query_understanding.py`

### 🔲 Q5. Nâng TRẦN recall (hit@5 ~0.68 bị chặn bởi encoder)
Benchmark cho thấy tinh chỉnh BM25/sample KHÔNG phá được trần. Hai đường:
- **LLM captioning qua API** (hoàn tất B5 khi có `ANTHROPIC_API_KEY`) — bắt quan hệ ngữ nghĩa.
- **Encoder mạnh hơn** (siglip2-large-384) hoặc **ensemble CLIP** (B6, blueprint Mục 2.1) —
  lưu ý CLIP yếu tiếng Việt, cân nhắc.
**File:** `ingestion/llm_captioning.py`, `retrieval/video_engine.py`, `ingestion/embed_clip.py`

---

## 3. TRỤ CỘT [2] — CƠ CHẾ HIỂN THỊ (Video Browser)

### ✅ Đã có
Lưới keyframe (ảnh + timestamp + video_id) · dedup giảm trùng · tab Chuỗi (A→B→C).

### 🔲 D1. Video Browser nâng cao (mới từ slide)
Slide: *"hiển thị gì khi người dùng không biết bắt đầu từ đâu"*, *"frame gần giống nhau"*.
- **Xem lân cận**: từ 1 keyframe, xem frame trước/sau trong CÙNG video (timeline nhỏ).
- **Nhảy tới giây** trong video gốc (mở video ở timestamp).
- **Gom cụm kết quả** theo video để không tràn frame giống nhau.
**File:** `ui/index.html`, `ui/app.py` (endpoint lân cận theo video_id + khoảng thời gian)

### 🔲 D2. Trang "khám phá" khi chưa biết bắt đầu (mới từ slide)
Hiển thị mẫu đại diện đa dạng của dataset (mỗi video/cụm 1 ảnh) để người dùng lướt chọn.
**File:** `ui/`

---

## 4. TRỤ CỘT [3] — PHẢN HỒI NGƯỜI DÙNG (Relevance Feedback)

### ✅ Đã có
**KISC** (`kisc_module/`): hội thoại hỏi lại, entropy/information-gain — đúng tinh thần
"khám phá (exploration) ↔ khai phá (exploitation)". NHƯNG chạy trên **data mẫu**.

### 🔲 F1. Nối KISC vào TÌM VIDEO THẬT  ⭐ (khoảng trống lớn)
Hiện KISC dùng `build_dataset()` giả. Cho KISC chạy trên keyframe video thật (khi độ tự
tin thấp → hỏi lại thu hẹp). Cần cầu nối attribute ↔ keyframe video.
**File:** `retrieval/kisc_adapter.py`, `retrieval/video_engine.py`, `ui/app.py`

### 🔲 F2. Relevance feedback trên tab tìm chính (mới từ slide)
Bấm "giống/không giống" trên kết quả → tinh chỉnh truy vấn (đẩy embedding về phía ảnh
được thích — Rocchio/PRF). Cân bằng khám phá (mở rộng) ↔ khai phá (tách video giống nhau).
**File:** `retrieval/video_engine.py` (feedback vector), `ui/`

### 🔲 F3. Gợi ý concept liên quan (mới từ slide)
Từ truy vấn, gợi ý concept: (a) model nghĩ người dùng muốn (khám phá), (b) giảm bất định
kết quả (khai phá). Có thể dùng object/caption thường gặp trong top-K.
**File:** `retrieval/video_engine.py`, `ui/`

---

## 5. NHÓM A — SCALE (dataset lớn: hàng trăm giờ → triệu keyframe)

**Phép thử:** 10.000 video × 5h ≈ 50.000h → ~80–120 triệu keyframe. HNSW float ≈ 300GB RAM.

| | Việc | Trạng thái |
|---|---|---|
| A1 | Batch embedding GPU | ✅ **×2.15** (đo thật; nút thắt = tiền xử lý CPU, không phải GPU) |
| A2 | Lưu/nạp index ra đĩa (bỏ image_bytes, dựng lại từ video gốc) | ✅ |
| A3 | Decode backend (decord/NVDEC nếu cài, else cv2 chỉ decode frame mẫu) | ✅ (cần `pip install decord` bản CUDA để có NVDEC thật) |
| A4 | Song song decode ‖ embed (queue giới hạn) | ✅ |
| **A5** | **IVF-PQ + sharding** khi > vài triệu vector | 🔲 đo RAM thật trước (blueprint Mục 2.2) |
| **A6** | **Progress bar** khi index dataset lớn (SSE/polling + ETA) | 🔲 |
| A7 | Load ảnh 1 lần cho cả 2 encoder + log device/tiến độ | ✅ (giảm decode 2×; phát hiện chạy nhầm CPU) |

---

## 6. NHÓM C — ĐÁNH GIÁ & VẬN HÀNH

| | Việc | Trạng thái |
|---|---|---|
| B1 | Harness benchmark (throughput + Recall/MRR) + **37 nhãn thật** + sweep | ✅ (xem Mục 7) |
| C1 | **Tăng nhãn 37 → ~100+** (giảm nhiễu hit@1 ±0.1) | 🔲 |
| C2 | **VQA trên video thật** (đếm/thứ tự) — cần object-detection hoặc API | 🔲 |
| C3 | **CI GitHub Actions** chạy pytest mỗi push | 🔲 |
| C4 | Quy trình **PR** khi 2 người làm song song | 🔲 |

---

## 7. 📊 KẾT QUẢ BENCHMARK ĐÃ CHỐT (RTX 3060, 37 nhãn — chi tiết `evaluation/benchmarks/`)

**Cấu hình đã chốt (ghi ở `configs/settings.yaml` mục 11 + default `video_engine.py`):**
- `bm25_weight = 1.0` + **adaptive** (query IN HOA → 3.0). Sweep: 1.0 cho hit@5 **0.72** (cao nhất);
  3.0 hại recall (0.52); 0 dense-only (0.68).
- `sample_every_s = 1.0` (hit@5 phẳng 0.68 mọi mức; 0.5 tăng hit@1). Recall bị chặn bởi **encoder**.
- **VLM rerank cho KIS** (không cho AVS): ~24s/query.

**Cấu hình tốt nhất cho KIS — dense-coarse + rerank:** hit@1 **0.595**, hit@5 0.676, MRR 0.623.

**Adaptive BM25 thắng cố định (coarse):** adaptive hit@1 0.351 / hit@5 0.649 > fixed-1.0
(0.297/0.649) và fixed-3.0 (0.324/0.514 — recall tụt).

**⚠️ Độ nhiễu:** hit@1 dao động ±0.1 giữa các lần (37 nhãn còn ít); **hit@5 ổn định**.
Đừng chốt trên chênh hit@1 <0.1 → cần thêm nhãn (C1).

---

## 8. 🎯 THỨ TỰ ĐỀ XUẤT LÀM TIẾP

Ưu tiên theo **tác động × đúng-khung-slide × công sức**:

1. **Q1 — Truy vấn bằng ẢNH** ⭐ (mới từ slide, dễ, tận dụng SigLIP, giá trị cao cho KIS).
2. **F1 — Nối KISC vào tìm video thật** ⭐ (trụ cột [3] đang trống với video thật).
3. **D1 — Video Browser nâng cao** (xem lân cận + gom cụm — đúng slide "hiển thị").
4. **Q4 — Query understanding + auto-route temporal** (tăng precision, nối module sẵn có).
5. **C1 — Tăng nhãn 100+** (mọi kết luận accuracy mới đáng tin).
6. **Q3/Q2 — Multi-modal + sketch** (sau Q1).
7. **A5/A6, C3/C4** — khi cần scale thật / làm nhóm.
8. **Q5 (LLM caption API / encoder mạnh)** — khi có `ANTHROPIC_API_KEY` để phá trần recall.

> Nguyên tắc xuyên suốt (blueprint Mục 11.3): **đo trước, tối ưu sau**. Mọi thay đổi
> accuracy phải chứng minh bằng `evaluation/bench_retrieval.py` trên nhãn thật.
