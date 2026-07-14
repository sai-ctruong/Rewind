# 📋 TASKS.md — Kế hoạch đầy đủ (Hệ thống tìm kiếm video AIC 2026)

> **Nguồn yêu cầu:** `CLAUDE.md` (blueprint kỹ thuật) + **Slide Tập huấn Buổi 2**
> (khung "Hệ thống tìm kiếm video" — ThS. Nguyễn Quang Thức) + **benchmark thật**
> (RTX 3060, 51 nhãn). Trạng thái tổng quan xem `TEAM.md`.
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
| **Benchmark harness** + 51 nhãn thật + tuning | ✅ B1 | xem Mục 5 |

---

## 2. TRỤ CỘT [1] — MÔ HÌNH TRUY VẤN

### ✅ Đã có
Late-fusion ensemble · early-fusion rerank · prompt ensemble · temporal · ASR · caption · adaptive BM25.

### ✅ Q1. Truy vấn bằng ẢNH (image → video)  ĐÃ XONG (2026-07-14)
Slide: *"thay đổi phương thức truy vấn"*. **Đã làm:** `encode_image_query` (bọc ảnh vào
RawKeyframe → `encoder.embed`, chung interface thật/mock) + `search_by_image` (dense
ensemble thuần, không BM25). Endpoint `/api/video/search_image` (multipart) + ô upload +
nút "Tìm theo ảnh" tab Video. Test: ảnh đỏ→cảnh đỏ, ảnh xanh→cảnh xanh.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

### ✅ Q2. Truy vấn bằng SKETCH (phác hoạ)  ĐÃ XONG (2026-07-14)
Slide: "truy vấn bằng ảnh sinh ra từ sketch". **Đã làm:** canvas vẽ phác (màu/xoá) trong
tab Video → `canvas.toBlob` → dùng lại endpoint ảnh (Q1) / multimodal (Q3, kết hợp câu ở
ô tìm). Thuần frontend, không đổi engine. `runImageSearch(blob|file)` dùng chung upload+sketch.
**LƯU Ý:** SigLIP hợp ẢNH THẬT hơn nét vẽ → sketch thô chất lượng hạn chế; "ảnh sinh từ
sketch" chuẩn cần model generative (sketch→ảnh thật) — để sau nếu có.
**File:** `ui/index.html`

### ✅ Q3. Kết hợp NHIỀU KIỂU truy vấn (multi-modal) ĐÃ XONG (2026-07-14)
Slide: *"cơ chế kết hợp nhiều kiểu truy vấn"*. **Đã làm:** `search_multimodal(query_text,
image_bytes, text_weight)` — trộn Ở MỨC VECTOR `q=norm(w·vec_chữ+(1−w)·vec_ảnh)` (SigLIP
cùng không gian chữ+ảnh); chữ vẫn cho BM25/rerank. Endpoint `/api/video/search_image`
nhận thêm `query` → có chữ thì multimodal, không thì ảnh thuần. UI: nút "Tìm theo ảnh
(+chữ)" tự lấy câu ở ô tìm. Test: lệch chữ→theo chữ, lệch ảnh→theo ảnh.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

### 🟡 Q4. Query understanding + auto-route temporal  PHẦN LỚN XONG (2026-07-14)
**Đã làm:** nối `query_understanding.py` (mock heuristic, không cần API) vào engine:
`understand(query)` → StructuredQuery; `temporal_events(query)` → tách sự kiện khi câu có
"A TRƯỚC/SAU KHI B". Endpoint search trả `parsed` (hiện cấu trúc đã hiểu) + **TỰ ĐỊNH
TUYẾN** sang `search_temporal` khi phát hiện thứ tự → UI hiện "🧠 Hiểu câu" + banner + chuỗi.
`set_query_understander` để cắm Claude khi có key. Test understand + routing.
**CÒN LẠI:** pre-filter theo object/location/time CHƯA hữu ích vì keyframe video chưa có
metadata đó (cần object-detection hoặc caption) — chỉ temporal là dùng được ngay.
**File:** `retrieval/video_engine.py`, `retrieval/query_understanding.py`, `ui/app.py`, `ui/index.html`

### 🟡 Q5. Nâng TRẦN recall — TURNKEY sẵn, chờ tài nguyên (2026-07-14)
Trần hit@5 ~0.68 bị chặn bởi encoder. **Đã nối turnkey (chỉ cần bật):**
- **Caption Claude tự bật:** `_get_captioner()` tự chọn `ClaudeCaptioner` khi có
  `ANTHROPIC_API_KEY` (chất lượng cao, KHÔNG tốn VRAM → hết kẹt như Qwen local), else Qwen.
  → Có key + bật Caption là caption 'xịn' vào BM25 → nâng recall query quan hệ. Test auto-select.
- **Preset encoder mạnh:** hằng `HQ_ENCODERS` (siglip2-large-384). Dùng
  `VideoSearchEngine(encoder_names=HQ_ENCODERS)` — nặng VRAM, đo lại bằng bench_retrieval.
**CÒN LẠI (cần tài nguyên user):** đặt API key để dùng caption Claude; hoặc chạy encoder
large trên GPU đủ khoẻ rồi benchmark để xác nhận lợi.
**File:** `retrieval/video_engine.py`, `ingestion/llm_captioning.py`

---

## 3. TRỤ CỘT [2] — CƠ CHẾ HIỂN THỊ (Video Browser)

### ✅ Đã có
Lưới keyframe (ảnh + timestamp + video_id) · dedup giảm trùng · tab Chuỗi (A→B→C).

### 🟡 D1. Video Browser nâng cao  PHẦN LỚN XONG (2026-07-14)
Slide: *"hiển thị gì khi người dùng không biết bắt đầu từ đâu"*, *"frame gần giống nhau"*.
**Đã làm:** `engine.neighbors()` + endpoint `/api/video/neighbors/<id>` (keyframe cùng
video quanh thời điểm); UI nút "🎞 Lân cận" mở dải trước/sau (highlight frame gốc); **gom
cụm kết quả theo video** khi tìm xuyên dataset. Test neighbors + 404.
**CÒN LẠI:** nhảy/mở video gốc ở timestamp (cần phục vụ file video); trang khám phá D2.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

### ✅ D2. Trang "khám phá" khi chưa biết bắt đầu  ĐÃ XONG (2026-07-14)
**Đã làm:** `engine.explore()` lấy mẫu keyframe ĐA DẠNG khắp dataset (mỗi video rải đều
theo thời gian) + `search_similar()` dùng THẲNG embedding đã lưu (không encode lại). Endpoint
`/api/video/explore` + `/api/video/similar/<id>`. UI nút "🧭 Khám phá" → grid; bấm 1 ảnh →
tìm cảnh tương tự. Test explore phủ nhiều video + similar cùng màu.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

---

## 4. TRỤ CỘT [3] — PHẢN HỒI NGƯỜI DÙNG (Relevance Feedback)

### ✅ Đã có
**KISC** (`kisc_module/`): hội thoại hỏi lại, entropy/information-gain — đúng tinh thần
"khám phá (exploration) ↔ khai phá (exploitation)". NHƯNG chạy trên **data mẫu**.

### ✅ F1. KISC cho video thật (hỏi lại bằng hình ảnh)  ĐÃ XONG (2026-07-14)
KISC gốc hỏi theo THUỘC TÍNH — keyframe video chưa có metadata đó (bị chặn). **Giải bằng
cách khác, không cần thuộc tính:** `disambiguation(entry, candidates)` — khi kết quả CÒN
MƠ HỒ (nhiều ứng viên, top-1 chưa nổi trội — tái dùng ý `is_confident_enough` của KISC),
hệ CHỦ ĐỘNG chọn `k` ảnh ĐA DẠNG (greedy farthest-point trên embedding) hỏi "cái nào
giống ý bạn nhất?"; bấm 1 ảnh → `search_similar` thu hẹp. Vòng khám phá↔khai phá chạy
THẲNG trên embedding. Endpoint search trả `disambiguation`; UI panel "🗣️ Thu hẹp". Test
đa dạng + đủ-tự-tin-thì-thôi.
**CÒN LẠI:** KISC theo THUỘC TÍNH (áo màu gì/ở đâu) cần object-detection hoặc caption.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

### ✅ F2. Relevance feedback trên tab tìm chính  ĐÃ XONG (2026-07-14)
**Đã làm:** `search_with_feedback` (Rocchio: q' = α·q + β·mean(pos) − γ·mean(neg), chuẩn
hoá) + `KeyframeIndex.mean_embedding`. Chạy THẲNG trên embedding (không cần thuộc tính).
Endpoint search nhận `positive`/`negative`. UI: nút 👍/👎 mỗi card + thanh "🔁 Lọc lại theo
phản hồi"; reset khi đổi câu. Test: mark xanh lá → top-1 về xanh lá.
**File:** `retrieval/video_engine.py`, `ingestion/build_index.py`, `ui/app.py`, `ui/index.html`

### ✅ F3. Gợi ý concept liên quan  ĐÃ XONG (2026-07-14)
**Đã làm:** `engine.suggest_concepts()` đếm từ khoá hay gặp trong caption/OCR/ASR của
top-K (trừ từ trong câu + từ dừng) → endpoint search trả `suggestions`; UI chip gợi ý,
bấm → thêm vào câu + tìm lại (thu hẹp/mở rộng). Rỗng nếu chưa bật caption/OCR/ASR.
Test gợi ý + loại từ query.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

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
| **A6** | Progress bar khi index (poll /api/video/progress, threaded server) | ✅ (count+fps+elapsed; UI poll 900ms) |
| A7 | Load ảnh 1 lần cho cả 2 encoder + log device/tiến độ | ✅ (giảm decode 2×; phát hiện chạy nhầm CPU) |

---

## 6. NHÓM C — ĐÁNH GIÁ & VẬN HÀNH

| | Việc | Trạng thái |
|---|---|---|
| B1 | Harness benchmark (throughput + Recall/MRR) + **51 nhãn thật** + sweep | ✅ (xem Mục 7) |
| C1 | **Tăng nhãn** (37 → **51** ✅; tiến tới ~100+) — giảm nhiễu hit@1 | 🟡 51 xong, thêm được |
| C2 | **VQA trên video thật** (đếm/thứ tự) — cần object-detection hoặc API | 🔲 |
| C3 | **CI GitHub Actions** chạy pytest mỗi push | 🔲 |
| C4 | Quy trình **PR** khi 2 người làm song song | 🔲 |

---

## 7. 📊 KẾT QUẢ BENCHMARK ĐÃ CHỐT (RTX 3060, đo trên 37 nhãn — chi tiết `evaluation/benchmarks/`)

> LƯU Ý: các số dưới đo trên bộ **37 nhãn**; bộ nhãn đã mở rộng lên **51** (C1) nhưng
> CHƯA chạy lại benchmark trên 51. Chạy lại để cập nhật: `python -m evaluation.bench_retrieval --labels evaluation/labels.json`.

**Cấu hình đã chốt (ghi ở `configs/settings.yaml` mục 11 + default `video_engine.py`):**
- `bm25_weight = 1.0` + **adaptive** (query IN HOA → 3.0). Sweep: 1.0 cho hit@5 **0.72** (cao nhất);
  3.0 hại recall (0.52); 0 dense-only (0.68).
- `sample_every_s = 1.0` (hit@5 phẳng 0.68 mọi mức; 0.5 tăng hit@1). Recall bị chặn bởi **encoder**.
- **VLM rerank cho KIS** (không cho AVS): ~24s/query.

**Cấu hình tốt nhất cho KIS — dense-coarse + rerank:** hit@1 **0.595**, hit@5 0.676, MRR 0.623.

**Adaptive BM25 thắng cố định (coarse):** adaptive hit@1 0.351 / hit@5 0.649 > fixed-1.0
(0.297/0.649) và fixed-3.0 (0.324/0.514 — recall tụt).

**⚠️ Độ nhiễu:** hit@1 dao động ±0.1 giữa các lần (bộ 37 nhãn còn ít); **hit@5 ổn định**.
Đừng chốt trên chênh hit@1 <0.1. Đã tăng lên **51 nhãn** (C1) để giảm nhiễu — chạy lại benchmark để thấy.

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
