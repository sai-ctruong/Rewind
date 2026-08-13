# AIC 2026 Current Work

**Status: CURRENT.** Everything above the horizontal rule describes the system that
exists today. Everything below it is HISTORICAL and describes a different, superseded
system — see the banner there before quoting any of it.

## What the competition runtime supports

Exactly three official tasks, and nothing else:

1. **Textual KIS**
2. **Q&A**
3. **TRAKE**

There is no agent, no KISC dialogue, no sketch search, no image-query search, no
user-feedback search, no generic AVS mode and no unrelated dashboard in the competition
runtime. Where the historical notes below describe those, they describe code that is not
part of this system.

## Frozen release

`0.11.0-aic2026`, commit `7dfe06e`, tag `aic2026-competition-ready`. That release is the
immutable baseline **B0**; the research branch does not modify it.

- [x] Repository/data audit
- [x] Production CLIP safeguards and encoder health
- [x] Multi-signal fusion and independent retrieval channels
- [x] Video-aware Top-100 ranking
- [x] Bounded local refinement and missing-MP4 fallback
- [x] Joint monotonic beam-pruned TRAKE alignment with k-best sequences
- [x] Grounded per-video-hypothesis Q&A and answer normalization
- [x] Shared submission validator and result-edit safety
- [x] Readiness preflight, system profile, release smoke

## Research branch (`research/aic2026-metric-budget`)

R0 — engineering cleanup and instrumentation, complete:

- [x] Remove the dead KIS rerank control and its engine shim
- [x] Remove six config knobs no code read (each now rejected with an explanation)
- [x] Disable OCR/ASR/caption in the competition config; report them as INFO, not WARN
- [x] Separate UI display count from the competition result pool
- [x] Add scope mode `retrieval_ready` (map + CLIP), separate from MP4 availability
- [x] Bounded query-embedding cache, TRAKE work reuse, optional startup prewarm
- [x] Per-query cost accounting and three-axis (quality/efficiency/cost) reporting
- [x] Private-development ground-truth schema with a provenance guard

R1 — metric-aware budgeted retrieval, experimental and disabled by default:

- [ ] Measure anything semantic. Blocked: no AIC ground truth exists.
- [ ] Compare variants on quality. Blocked by the same thing.

Open, and blocked on data rather than code:

- [ ] Supply AIC-format labels and run the first fixed-split baseline
- [ ] Run the full novelty/related-work review only after that baseline

---

> **HISTORICAL — NOT THE CURRENT SYSTEM.**
> Everything below is retained as engineering notes from the pre-AIC refactor. It
> describes a SigLIP-based multi-tab product with agent, dialogue, sketch and image-query
> features that the competition runtime does not contain. It is kept for provenance, not
> as a description of current capability, and must not be quoted as one.

---
# 📋 TASKS.md — Kế hoạch đầy đủ (Hệ thống tìm kiếm video AIC 2026)

> **Nguồn yêu cầu:** `CLAUDE.md` (blueprint kỹ thuật) + **Slide Buổi 2** (khung "Hệ
> thống tìm kiếm video" — 3 trụ cột) + **Slide Buổi 3** (khung "Agentic AI & LLM trong
> tìm kiếm" → Mục 9, Nhóm G) + **benchmark thật** (RTX 3060, 51 nhãn). Tổng quan: `TEAM.md`.
>
> Cập nhật: 2026-07-15 · Nhánh: `main` · ~200 test xanh.

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

### 🔴 Q5. Nâng TRẦN recall — **giả thuyết "encoder to hơn" ĐÃ BỊ BÁC BỎ** (2026-07-16)

Trần hit@5 ~0.65 giữ nguyên qua mọi cấu hình fusion/rerank → nghi là trần của ENCODER.
**Đã kiểm bằng thí nghiệm, không suy diễn** (`bench_retrieval.py --encoders`, 51 nhãn thật,
RTX 3060, truy vấn tiếng Anh để không thiệt cho model chỉ-biết-tiếng-Anh):

| Cấu hình | hit@1 | hit@5 | MRR |
|---|---|---|---|
| **base_ensemble** (2 encoder base — mặc định) | 0.353 | **0.647** ✅ | 0.482 |
| hq_ensemble (large-384 + base-ml) | 0.255 | 0.588 ↓ | 0.399 |
| base_single (1 encoder base) | 0.333 | 0.471 | 0.388 |
| large_single (1 encoder large-384) | 0.196 | **0.333** ↓↓ | 0.256 |

**KẾT LUẬN: encoder to hơn KHÔNG phá trần — nó làm KÉM đi.** Cái tạo ra độ chính xác là
**độ đa dạng của ensemble** (2 base = 0.647 **>>** 1 base = 0.471), không phải kích thước
model — đúng lý do Mục 2.1 chọn ensemble ("giảm rủi ro *cùng sai*").

**Đã loại trừ mọi khả năng "dùng sai" trước khi kết luận:**
- fp16 cho kết quả **y hệt** fp32, không NaN, norm = 1.0 → không phải lỗi số học.
- Text dùng đúng `padding='max_length', max_length=64` (bẫy kinh điển của SigLIP).
- Dedup **không** xoá mất ground-truth: cả hai index đều **0/51** nhãn mất đáp án, cùng
  **10.49** keyframe đúng/nhãn → chênh lệch 234 keyframe không phải nguyên nhân.
- So **song phẳng 1-đối-1** (base_single vs large_single) để không lẫn với lợi thế ensemble.
- VRAM **không** phải rào cản: ensemble large chỉ tốn **2.35/6 GB** (nhờ fp16 sẵn có).

**Phụ:** truy vấn Việt vs Anh trên base_ensemble gần như bằng nhau (hit@5 0.608 vs 0.647,
trong sai số của 51 nhãn) → encoder đa ngôn ngữ xử lý tiếng Việt tốt, **dịch truy vấn sang
Anh cũng không phải đòn bẩy**. Nhãn tiếng Anh lưu ở `evaluation/labels_en.json`.

**CÒN LẠI (đòn bẩy thật, cần tài nguyên):** caption LLM lúc index (`ANTHROPIC_API_KEY`) —
`_get_captioner()` tự bật `ClaudeCaptioner` khi có key. Đây là hướng duy nhất chưa thử.
**File:** `retrieval/video_engine.py` (HQ_ENCODERS đã ghi cảnh báo), `evaluation/bench_retrieval.py::encoder_sweep`

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
**KISC** (`dialogue/`): hội thoại hỏi lại, entropy/information-gain — đúng tinh thần
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

### ✅ C2. VQA trên VIDEO THẬT  ĐÃ XONG (2026-07-15)
Trước: tab VQA chỉ chạy trên 4 keyframe "sinh nhật" cứng trong `ui/app.py` (chữ bịa sẵn).
**Đã làm:** `answer_on_video(engine, entry, question)` — dùng **chính câu hỏi** để định vị
cửa sổ keyframe liên quan (blueprint bước [6]: retrieve window rồi mới hỏi), `entry_records()`
gộp caption/OCR/ASR của entry thành `KeyframeRecord`, `default_answerer()` tự chọn Claude
vision khi có key / Mock khi không. Endpoint `/api/video/vqa` + tab VQA có chọn-nạp video,
hiện **dải ảnh thật** đã dùng để suy luận (highlight frame được dùng).
**BUG THẬT phát hiện:** `ClaudeVqaAnswerer` đọc `f.image_path` — mà `KeyframeRecord` KHÔNG
có field đó (Mục 7) → **chưa bao giờ gửi được ảnh nào** cho Claude. Sửa: ảnh đi qua map
`images` (id → JPEG bytes); thêm `frame_jpeg_bytes()` dùng chung cho cả endpoint ảnh.
**Trung thực về giới hạn:** không có API key thì mock suy luận trên CHỮ → video chưa bật
Caption/OCR/ASR gần như không trả lời được; UI **nói thẳng** điều này thay vì trả lời rỗng.
**Gỡ kèm:** `/api/vqa` + `build_vqa_records()` (không còn UI gọi).
**File:** `retrieval/vqa_module.py`, `ingestion/schemas.py`, `ui/app.py`, `ui/index.html`
**Test:** `tests/test_vqa_on_video.py` (7) + guard HTTP.

### ✅ F5. Tab 🧠 Agent — đưa lớp Agentic (G1–G4) lên UI  ĐÃ XONG (2026-07-15)
Trước: G1–G4 xong ở tầng code + test nhưng `ui/app.py` KHÔNG import `search_agent`/
`session_memory`/`Reader` — người dùng giao diện không chạm tới được. **Đã làm:**
endpoints `/api/agent/ask|reset` (`SearchAgent` + `MockReader`; tự nâng lên
`ClaudePlanner`+`ClaudeReader` khi có `ANTHROPIC_API_KEY`) + tab `🧠 Agent`: ô hỏi →
lưới ảnh / chuỗi thời gian, panel **"Agent đã làm gì"** (tool + lý do + số kết quả),
**trí nhớ phiên** (lượt · 👍/👎 · câu gần đây), đáp án Reader, panel Agent chủ động hỏi lại.
**2 BUG THẬT do smoke test bắt được** (test không thấy): `MockReader.read` và
`SearchAgent.chat` đều giả định mọi kết quả có `keyframe_id` → truy vấn "A trước khi B"
(trả CHUỖI `{video_id, steps[]}`) làm **sập cả vòng Agent** với `KeyError`. Đã sửa cả hai
+ thêm test hồi quy; gom việc rút id vào `_result_ids()`.
**Đo thật** (walking.mp4): "người đi bộ" → `search` 8 ảnh · "đi bộ **trước khi** xe chạy"
→ tự chuyển `search_temporal`, 50 chuỗi (UI vẽ 8, vẫn báo tổng) · 👍 → `search_with_feedback`,
ảnh 👍 lên top, nhớ 3 lượt.
**File:** `ui/app.py`, `ui/index.html`, `retrieval/vqa_module.py`, `retrieval/search_agent.py`

### ✅ F4. Bộ lọc ẢNH hội thoại — thay tab KISC  ĐÃ XONG (2026-07-15)
**Vấn đề:** tab KISC chạy `build_dataset()` — 200 bản ghi lifelog TỔNG HỢP (`embedding=0`,
KHÔNG có ảnh) → chỉ hiện được text `kf_0102 (video_0020, t=500s)`. **Giải:** cho vòng thu
hẹp chạy trên **video THẬT** (đã có ảnh qua `/api/video/frame/<id>`).
- `ImageFilterSession` (`retrieval/image_filter.py`): giữ **pool** ứng viên, mỗi lượt xếp
  hạng lại *trong pool* rồi cắt theo `shrink=0.5` (sàn `min_k`) → số ảnh **đảm bảo giảm**.
  Thu hẹp bằng **3 tín hiệu dùng đồng thời**: thêm mô tả (truy vấn cộng dồn) · 👍/👎
  (Rocchio) · chọn ảnh đại diện (`pick` → 👍, `others` → 👎). Tái dùng `SessionMemory` (G3).
- Endpoints `/api/filter/start|refine|reset`; UI tab KISC = lưới ảnh + panel "cái nào gần
  ý nhất?" + thanh 🔁 lọc theo phản hồi + ô "20 → 8". Gỡ sạch cụm mô phỏng KISC offline.
- **Đo thật** (walking.mp4, 11 keyframe): start 11 ảnh → +"xe cộ" 6 → 👍 3, ảnh 👍 lên top.
**File:** `retrieval/image_filter.py`, `ui/app.py`, `ui/index.html` · **Test:**
`tests/test_image_filter.py` (11) + 4 test UI.

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
| **A5** | **IVF-PQ + sharding** khi > vài triệu vector | 🟡 **ĐÃ ĐO** — xem Mục 5b; ngưỡng kích hoạt giờ có bằng chứng |
| **A8** | **Benchmark quy mô + tune efSearch** | ✅ **ĐÃ XONG 2026-07-15** — `evaluation/bench_scale.py`; phát hiện lỗi cấu hình thật (Mục 5b) |
| **A6** | Progress bar khi index (poll /api/video/progress, threaded server) | ✅ (count+fps+elapsed; UI poll 900ms) |
| A7 | Load ảnh 1 lần cho cả 2 encoder + log device/tiến độ | ✅ (giảm decode 2×; phát hiện chạy nhầm CPU) |

---

## 5b. 📏 BENCHMARK QUY MÔ (A8) — "scale-ready" giờ đã có BẰNG CHỨNG

Trước đây mọi số đo đều trên video ngắn (~50 keyframe) — mà **HNSW ở 50 vector là vô
nghĩa** (hành xử như brute-force). `evaluation/bench_scale.py` đo trực tiếp ở **768 chiều**
(số chiều thật), 10k→200k vector, dữ liệu mô phỏng embedding thật (có cụm ngữ nghĩa).

**Tái lập** (JSON kết quả bị `.gitignore` như mọi artifact sinh ra — chạy để có lại):
```bash
python -m evaluation.bench_scale --sizes 10000,50000,100000,200000 --dim 768   # tổng quan
python -m evaluation.bench_scale --sizes 200000 --k 10  --ef 128,256,512,1024,2048  # điểm khuỷu tay
python -m evaluation.bench_scale --sizes 100000 --k 100 --ef 128,256,1024,2048      # sát coarse thật
```
Số dưới đây đo trên **RTX 3060 / 16 GB RAM**, `M=32`, `efConstruction=200`.

### 🔴 Phát hiện 1 — `efSearch=128` đang LÀM MẤT ~54% ứng viên đúng

| efSearch | recall@100 (100k vector) | latency p50 |
|---|---|---|
| **128** (giá trị cũ) | **0.465** ❌ | 0.6 ms |
| 256 | 0.604 | 1.1 ms |
| 1024 | 0.911 | 4.6 ms |
| **2048** (đã chốt) | **0.981** ✅ | **11 ms** |

Đây là vi phạm trực tiếp **ràng buộc không thương lượng #2** ("ứng viên bị loại ở coarse
KHÔNG BAO GIỜ được xét lại"). **Vì sao chọn 2048 chứ không phải "điểm khuỷu tay" 1024:**
khuỷu tay chỉ đáng theo khi latency khan hiếm — ở đây 11 ms là **0,05% của time_budget
20s** và nhỏ hơn VLM rerank (~1s/ứng viên) **hai bậc độ lớn**. Mua recall bằng vài ms là
món hời. **Quy tắc chốt:** `efSearch ≥ ~2× coarse.top_k`.
→ Đã sửa `configs/settings.yaml` + `IndexConfig` (không còn `[PROVISIONAL]`).
*Ở quy mô hiện tại (vài nghìn keyframe) thay đổi này là no-op — nó chặn thiệt hại khi
dữ liệu vượt ~50k.*

### 🔴 Phát hiện 2 — trần RAM thật: **~12 GB cho 1 triệu keyframe**

Đo được **3.343 B/vector** (ổn định mọi cỡ → tuyến tính). Nhưng phải nhân **×2** vì hệ
dùng **ensemble 2 encoder**, và `KeyframeIndex` giữ **cả index lẫn ma trận gốc**:

| Quy mô | 1 encoder | **×2 encoder (thực tế)** | Trên máy 16 GB |
|---|---|---|---|
| 200k keyframe (~80–140 giờ video sau dedup) | 1.2 GB | **~2.4 GB** | ✅ thoải mái |
| 1M keyframe (~400–700 giờ) | 6.0 GB | **~12 GB** | ❌ **không vừa** |

→ **Ngưỡng kích hoạt A5 (IVF-PQ/sharding) giờ có con số:** khoảng **300–400 giờ video**.
Dưới mức đó HNSW float là lựa chọn đúng (đúng như blueprint Mục 2.2 dự đoán).

### 🟢 Phát hiện 3 — build index & latency KHÔNG phải nút thắt
Build 200k vector: **43s** → 1M ước ~4–6 phút. Nhỏ xíu so với **trích embedding**
(~4–8 frame/s → 200k frame ≈ **7–14 giờ**). Latency coarse **<1 ms** (ef=128) đến **11 ms**
(ef=2048) ở 200k — tầng coarse không hề là nút thắt; VLM rerank mới là.

### ⚠️ Bài học phương pháp (2 lỗi ĐO ĐẠC tự bắt được)
1. **Bẫy nhiều chiều:** nhiễu `σ·N(0,I)` ở 768 chiều dài `σ·√768`. Quên chia `√dim` →
   dữ liệu "có cụm" thực chất **ngẫu nhiên đều** → cả benchmark recall vô nghĩa mà không
   ai biết. Đã khoá bằng test `test_clustered_vectors_actually_cluster_at_high_dim`.
2. **recall khớp-ID gây hiểu lầm khi nhiều điểm gần bằng nhau** → thêm `score_ratio`
   (điểm thu được / điểm exact). Ở ef=2048: recall 0.98 **và** score_ratio 0.999.
   Vector ngẫu nhiên đều (`--dist random`) là ca **bệnh lý**, giữ lại làm sàn tuyệt đối
   (`scale_bench_random.json`) nhưng **không dùng để chọn tham số**.

---

## 6. NHÓM C — ĐÁNH GIÁ & VẬN HÀNH

| | Việc | Trạng thái |
|---|---|---|
| B1 | Harness benchmark (throughput + Recall/MRR) + **51 nhãn thật** + sweep | ✅ (xem Mục 7) |
| C1 | **Tăng nhãn** (37 → **51** ✅; tiến tới ~100+) — giảm nhiễu hit@1 | 🟡 51 xong, thêm được |
| C2 | **VQA trên video thật** (đếm/thứ tự) — cần object-detection hoặc API | 🔲 |
| C3 | **CI GitHub Actions** + sửa 3 lỗi onboarding | ✅ **XONG 2026-07-16** — xem Mục 6b |
| C4 | Quy trình **PR** khi 2 người làm song song | 🔲 |

---

## 6b. 🔧 C3 — CI + 3 LỖI ONBOARDING (đã sửa 2026-07-16)

Máy dev đã cài sẵn đủ thứ nên `pytest` xanh **kể cả khi khai báo phụ thuộc sai**. Ba lỗi
dưới đây chỉ lộ ra trên **máy sạch** — tức là đúng lúc người mới clone repo.

| # | Lỗi | Hậu quả |
|---|---|---|
| 1 | `psutil` **thiếu** trong `requirements.txt` nhưng `bench_scale.py` dùng | benchmark quy mô **nổ** trên máy mới; test lặng lẽ bị skip |
| 2 | Số test **sai ở 2 nơi**: README ghi `262`, HUONG_DAN ghi `243` — thực tế **273** | tài liệu tự mâu thuẫn ngay dòng đầu |
| 3 | `requirements.txt` ép cài **torch + easyocr + transformers (~5 GB)** ngay từ đầu | trái chính lời hứa *"chạy test offline, không cần GPU"* |

### Đã sửa
- **Tách phụ thuộc:** `requirements.txt` (**lõi nhẹ**) ↔ `requirements-full.txt`
  (torch/OCR/VLM, có `-r requirements.txt`). **ĐO THẬT trên venv sạch Python 3.14:**
  `pip install -r requirements.txt && pytest` → **273 passed, ~7 giây, ~280 MB** —
  **không cần torch**. (Trước: ~5 GB.)
- Thêm `psutil` vào lõi; sửa số test ở README + HUONG_DAN; trỏ hướng dẫn cài sang
  `requirements-full.txt` thay vì liệt kê gói thủ công.

### CI (`.github/workflows/ci.yml`) — ngăn tái diễn, không chỉ vá một lần
Runner GitHub bắt đầu từ số 0 nên **bắt buộc** khai báo phụ thuộc phải đúng. 2 job:

| Job | Kiểm |
|---|---|
| **test** (Python 3.10 + 3.13) | cài **chỉ lõi nhẹ** → `pytest`; **khẳng định torch KHÔNG lọt vào lõi**; **chặn test bị skip vì thiếu thư viện** (bẫy: CI "xanh" mà thực chất không chạy gì) |
| **docs** | **số test trong tài liệu phải khớp thực tế** (lỗi #2 sẽ không tái diễn); mọi **ảnh + link nội bộ** phải tồn tại |

### 2 lỗi trong chính CI, tự bắt được khi chạy thử
1. `pyproject.toml` đã đặt `addopts = "-ra -q"` → thêm `-q` nữa thành **double-quiet**,
   pytest **ngừng in** dòng `N passed` → bước đếm test sập dù code không sao. Sửa: bỏ `-q`
   và đếm qua **hook `pytest_terminal_summary`** thay vì bới chữ trong stdout.
2. Hook pytest đòi **đúng tên tham số** (`terminalreporter`) — đổi tên là `PluginValidationError`.

**Đã chạy thật cả 4 bước trên venv sạch** (trích đúng code từ `ci.yml`, không gõ lại):
lõi-nhẹ PASS · 273 passed · không skip · tài liệu khớp · ảnh/link OK. Guard cũng được
kiểm **cả hai chiều**: PASS trên venv sạch, **FAIL** trên venv có torch → chứng minh nó
thật sự có tác dụng.


## 6c. ⚡ NÂNG CẤP: độ chính xác · tốc độ · chi phí (2026-07-16)

Ba đặc tính chỉ đánh đổi nhau khi áp MỘT phương pháp đắt cho MỌI thứ. Đo để tìm chỗ
được cả ba (hoặc ít nhất không mất gì).

### ✅ TỐC ĐỘ + CHI PHÍ — nút thắt là **VRAM**, không phải model (×5.25)
| | VRAM | s/ứng viên | s/truy vấn (pool=8) |
|---|---|---|---|
| SigLIP + Qwen2-VL cùng trong VRAM | 5.56/6.0 · **đỉnh 6.06 = tràn** | 2.77 s | **21.16 s** ❌ vượt budget |
| **Đẩy SigLIP sang RAM khi rerank** | 4.12 · đỉnh **5.11** | **0.48 s** | **4.03 s** ✅ |

`_offload_encoders(to_cpu)` — chỉ **chuyển thiết bị**, khác `_free_encoders` (xoá hẳn →
lượt sau nạp lại **từ đĩa**). Query encode xong **trước** rerank nên encoder chắc chắn
nhàn rỗi. Best-effort: CPU-only → no-op; encoder lạ không có `.to()` → nuốt lỗi.
3 test khoá hành vi. **File:** `retrieval/video_engine.py`

### ❌ 2 GIẢ THUYẾT BỊ BÁC BỎ (đo rồi gỡ, không giữ code chết)
1. **Gộp batch nhiều ứng viên vào 1 lần `generate()`** → **chậm hơn ×0.17** (điểm vẫn
   khớp 16/16 nên không phải lỗi). VRAM đã nghẹt thì không còn chỗ song song hoá.
2. **Tăng `rerank_pool` để VLM nhìn sâu hơn** → **không giúp, còn hơi hại**:

| `rerank_pool` | hit@1 | hit@5 | s/truy vấn |
|---|---|---|---|
| **8** (giữ nguyên) | **0.510** | 0.608 | **4.71 s** |
| 16 | 0.471 | 0.608 | 6.91 s |
| 24 | 0.471 | 0.608 | 23.64 s |
| 32 | 0.471 | 0.608 | 13.73 s |

hit@5 **đứng im** ở mọi pool → 24 ứng viên thêm **chưa bao giờ** góp đáp án đúng; pool
sâu chỉ thêm nhiễu. **Kết quả lưu:** `evaluation/benchmarks/rerank_pool_bench.json`

> #### 🔴 SỬA: cột hit@1 ở bảng trên KHÔNG kết luận được gì (2026-07-16)
> Tôi từng viết "pool 8 (0.510) tốt hơn pool 32 (0.471)". **Sai về phương pháp.** Chênh
> 0.039 trên 51 nhãn = **đúng 2 query**. KTC95% của 0.510 là **[0.377, 0.641]** — rộng
> 0.265, nuốt trọn mọi giá trị trong bảng. Kiểm định McNemar theo cặp ở **kịch bản
> thuận lợi nhất** cho giả thuyết (2 cặp bất đồng, pool 8 thắng cả 2): **p = 0.50**.
> Ngang tung đồng xu.
>
> Phần **hit@5 bằng nhau tuyệt đối** ở mọi pool thì VẪN VỮNG — bằng nhau y hệt không
> phải chuyện may rủi. Kết luận đúng phải là: *pool sâu không giúp, và pool 8 rẻ hơn
> nhiều* (4.71s vs 13.73s — chênh lệch tốc độ mới là thứ có thật), **không phải** "pool
> 8 chính xác hơn".
>
> **Không cần 2400 nhãn** để sửa: các cấu hình chạy trên CÙNG 51 query nên phải so
> **theo cặp**, không so 2 tỉ lệ độc lập. So cặp chỉ nhìn query có kết quả khác nhau →
> chỉ cần **6 cặp bất đồng thắng sạch** là p < 0.05. Đã thêm
> `evaluation/metrics.py::compare_configs` (Wilson + McNemar) và test khoá chính sai
> lầm này lại. **Từ nay mọi so sánh cấu hình phải đi qua đó.**

### 🔴 SỬA KẾT LUẬN CŨ: 0.65 KHÔNG phải trần của encoder
| Đáp án nằm trong | top-1 | top-5 | top-10 | top-30 | **top-100** |
|---|---|---|---|---|---|
| % số lần | 0.372 | 0.627 | 0.608 | 0.824 | **0.902** |

Coarse **tìm được đáp án 90% số lần** — nó chỉ không đẩy lên đỉnh. Cộng với 2 kết quả
bác bỏ ở trên (encoder to hơn: không; pool sâu hơn: không) ⇒ nút thắt thật là **khả năng
PHÂN BIỆT của bộ chấm** (SigLIP ở coarse, Qwen2-VL-2B ở rerank). Đòn bẩy còn lại: **bộ
rerank mạnh hơn (Claude vision)** hoặc **caption LLM** — đều cần API key.

### ✅ ĐÃ SỬA — fusion phụ thuộc kích thước pool (2026-07-16)

Đã tách `fusion_depth` (cố định) khỏi `top_k` (chỉ cắt kết quả) trong `CoarseRetriever`.
**Kiểm chứng test bắt được bug thật** (chạy lại code cũ qua `git stash`):

```
CODE CŨ  top5(xin 5)  : k42 k512 k546 k305 k424
CODE CŨ  top5(xin 50) : k42 k512 k157 k414 k546   ← 3/5 khác nhau
```

Cơ chế tinh vi hơn tưởng ban đầu: điểm RRF của một ảnh = tổng `1/(k+rank)` **trên các
list nó CÓ MẶT**. List sâu hơn không đổi rank của ảnh, nhưng làm nó **xuất hiện thêm ở
nguồn khác** → được cộng thêm điểm. Ảnh ở dense rank 2 + BM25 rank 12 vô hình khi
depth=5, nhưng ở depth=20 được cộng điểm BM25 và vọt lên trên ảnh dense rank 1.

#### 📊 Quét độ sâu (51 nhãn · 2679 kf · top_k=5 · coarse-only)
| depth | hit@1 | hit@5 | s/query |
|---|---|---|---|
| 5 | 0.431 | **0.667** | 0.066 |
| 10 | 0.451 | **0.667** | 0.063 |
| 20 | 0.451 | 0.647 | 0.066 |
| 50 | **0.490** | 0.628 | 0.068 |
| 100 · 300 · **1000** | 0.471 | 0.608 | 0.065 |

**Kết luận: GIỮ 1000 vì KHÔNG CÓ BẰNG CHỨNG để đổi** — không phải vì nó tối ưu.
Trông như một đánh đổi đẹp (sâu → hit@1 tăng, hit@5 giảm) nhưng McNemar theo cặp cho
hit@1: **p ≥ 0.625** ở mọi độ sâu. Với hit@5, chênh lệch **lớn nhất toàn bảng = 3
query** → kể cả khi cả 3 cùng nghiêng một phía thì **p = 0.25**. Thí nghiệm **không đủ
lực** để phát hiện hiệu ứng cỡ này (cần ≥ 6 cặp bất đồng) → phải làm **C1 (tăng nhãn)**
trước, không chỉnh số dựa trên bảng này. Quá depth 100 thì **bão hoà tuyệt đối** (100 ≡
300 ≡ 1000, 0 cặp bất đồng) — ứng viên ở rank sâu chỉ được cộng `1/(60+rank)`, quá nhỏ
để lật thứ hạng.

> #### 🔴 LẦN CHẠY ĐẦU HỎNG — tôi tự phá thí nghiệm của mình
> Bench đầu cho **số liệu y hệt nhau ở 5 độ sâu** → suýt kết luận "độ sâu không ảnh
> hưởng gì". Dấu hiệu hỏng: `depth=5` mà `hit@100=0.902` là **bất khả thi** (fusion sâu
> 5 trên 3 nguồn cho tối đa 15 ứng viên). Thủ phạm là `max(top_k, depth)` tôi thêm cho
> "an toàn": bench gọi `top_k=100` nên mọi depth < 100 **bị kẹp lên 100**. Tệ hơn: cái
> kẹp đó **tái lập đúng bug vừa sửa** — để `top_k` chi phối độ sâu trở lại.
> **Đã gỡ kẹp** (`depth < top_k` → trả về ít hơn top_k, đó mới đúng) + test khoá:
> `depth=3, top_k=100` mà trả đủ 100 kết quả là hỏng.
>
> **Bài học:** số liệu quá "đẹp"/quá đều là dấu hiệu HỎNG, không phải dấu hiệu thành công.

### 🟡 (đã sửa — giữ lại để tra cứu) BUG: fusion phụ thuộc kích thước pool
`hit@5` **đổi theo `top_k` yêu cầu**: 0.627 (top_k=5) → 0.510 (top_k=100). Xin **nhiều**
ứng viên hơn lại làm **top-5 tệ đi** — vô lý. Nguyên nhân: `pool = max(top_k, rerank_pool)`
(`video_engine._run_search`) làm **độ sâu fusion phụ thuộc số kết quả người gọi muốn**;
pool lớn → BM25 bơm thêm ứng viên tầm thường, RRF cộng điểm từ nhiều nguồn đẩy chúng
vượt kết quả dense tốt. **Cách sửa:** tách "độ sâu fusion" (cố định, theo `coarse.top_k`)
khỏi "số kết quả trả về" → kết quả ổn định bất kể `top_k`.


### ✅ A9 — bỏ ma trận float trùng lặp: **−46.6% RAM**, và còn NHANH HƠN (2026-07-17)

`IndexHNSWFlat` **vốn đã lưu nguyên vector float32** bên trong (Flat = không nén), nhưng
ta còn giữ thêm `_clip_matrix`/`_siglip_matrix` chứa **đúng những vector đó** — cùng một
dữ liệu lưu hai lần, trong RAM **và** trong `meta.pkl`.

| đo trên 20k × 768 | |
|---|---|
| ma trận float (thừa) | 58.6 MB |
| HNSW index (đã gồm vector) | 67.2 MB |
| **bỏ ma trận** | **−46.6% RAM** |

**Không phải đánh đổi RAM ↔ tốc độ — `reconstruct_batch` còn NHANH HƠN:**

| subset | `matrix[rows]` | `reconstruct_batch` | vòng lặp `reconstruct` |
|---|---|---|---|
| 1 000 | 1.50 ms | **1.11 ms** | 6.20 ms |
| 10 000 | 10.46 ms | **5.92 ms** | 51.97 ms ⚠️ |

Vì `matrix[rows]` (fancy-indexing) phải gather tạo bản sao, còn Faiss làm trong C++.
⚠️ Gọi `reconstruct` **từng dòng** trong vòng lặp Python thì **chậm gấp 5** — luôn dùng
bản batch.

**Kiểm chứng, không chỉ tin test xanh:** chạy song song code cũ (`git stash`) và mới
trên cùng dữ liệu → `exact_dense_search`, `dense_search`, `mean_embedding` **giống hệt
từng ký tự**. Faiss trả vector khớp **từng bit** (HNSWFlat không nén).

**Ngoại suy quy mô thật** (1 triệu keyframe × 768 × 2 encoder): tiết kiệm **~5.7 GB RAM**.

3 test khoá: (1) index không được giữ ma trận trùng — *đã kiểm chứng test này bắt được
khi cố tình thêm lại*; (2) vector từ Faiss khớp bit-for-bit với embedding gốc — **test
này PHẢI đỏ nếu sau này đổi sang IVF-PQ** (A5), vì lúc đó exact rerank không được lấy
vector từ index nén nữa; (3) index CŨ trên đĩa (còn field `clip_matrix`) vẫn nạp được —
không bắt người dùng index lại vài trăm giờ video. **File:** `ingestion/build_index.py`


## 7. 📊 KẾT QUẢ BENCHMARK ĐÃ CHỐT (RTX 3060, **51 nhãn** — chi tiết `evaluation/benchmarks/`)

**Bảng chính (index có OCR, sample 1.0):**
| cấu hình | hit@1 | hit@5 | MRR |
|---|---|---|---|
| coarse fixed bm25=1.0 | 0.275 | 0.647 | 0.438 |
| coarse fixed bm25=3.0 | 0.314 | **0.529 ↓** | 0.408 |
| **coarse adaptive** ✅ | 0.333 | 0.647 | 0.477 |
| **adaptive + VLM rerank (KIS)** ✅ | **0.549** | **0.667** | **0.604** |

**Cấu hình đã chốt (ghi ở `configs/settings.yaml` mục 11 + default `video_engine.py`):**
- `bm25_weight = 1.0` + **adaptive** (query IN HOA → 3.0): adaptive giữ recall CAO NHẤT
  (hit@5 0.647) mà tăng hit@1; bm25=3.0 HẠI recall (0.529). (Sweep 37 nhãn: 1.0 cho hit@5 0.72.)
- `sample_every_s = 1.0` (recall phẳng qua mọi mức — bị chặn bởi **encoder**).
- **VLM rerank cho KIS** (~24s/query): kéo hit@1 **0.333 → 0.549** (+0.22, cú nhảy lớn nhất). KHÔNG cho AVS.

**Đối chiếu 37↔51 nhãn:** kết luận GIỐNG HỆT (adaptive thắng, bm25=3.0 hại recall, rerank
+hit@1 mạnh). Số 51 nhãn hơi thấp hơn 37 (adaptive+rerank hit@1 0.549 vs 0.595) vì 14 nhãn
mới KHÓ hơn (mục tiêu Seoul đặc thù/thoáng qua). **hit@5 ~0.65 cực ổn định** qua mọi lần chạy.
**⚠️** hit@1 vẫn nhiễu ±0.05 giữa các lần; hit@5 mới là thước đo recall đáng tin.

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

---

## 9. NHÓM G — LỚP AGENTIC (theo Slide Buổi 3: "Kiến trúc Agentic AI & LLM trong tìm kiếm")

> **Nguồn:** Slide Buổi 3 (Hồ Lê Minh Quân). Buổi 3 KHÔNG thêm dataset — nó là **khung
> tư duy kiến trúc**: LLM làm "bộ não" (reasoning core) **điều phối bộ công cụ** thay
> vì chạy pipeline cứng. Hai ứng dụng sát đề: **VideoQA STAR agent** (slide 30 — Planner
> luân phiên gọi Tool *Thời gian* + Tool *Không gian*) và **MemoriEase 2.0/3.0** (slide
> 31 — Conversational Lifelog + RAG: trích bộ lọc metadata → Rocchio vector → **Rerank →
> Reader sinh đáp án**). Project đã có **đủ mọi "công cụ" rời** (search, temporal, image,
> rerank, understand, feedback-Rocchio, VQA, KISC) nhưng **chưa có bộ não điều phối**.

**Nguyên tắc:** mọi task dưới đây theo pattern **ABC + Mock + Claude-lazy** (Mục 1.5
CLAUDE.md) — `Mock*` chạy offline (rule-based) để test/đo NGAY khi chưa có API key;
`Claude*` (function-calling) bật khi có `ANTHROPIC_API_KEY`. KHÔNG chặn tiến độ vì thiếu key.

### ✅ G1. Tool Registry — hình thức hoá "Action Space"  ĐÃ XONG (2026-07-15)
Slide: *Action Space = tập công cụ khai báo được*. **Đã làm:** `ToolRegistry` bind
(engine, entry) → 9 tool có schema JSON: `search`, `search_temporal`, `search_by_image`,
`search_multimodal`, `understand`, `neighbors`, `search_similar`, `suggest_concepts`,
`disambiguation`. Mỗi tool có `description` tiếng Việt (nói RÕ khi nào dùng) + `parameters`
JSON Schema. `specs("anthropic"|"openai")` xuất đúng định dạng function-calling 2 SDK.
`call(name, **kw)` dispatch + **nuốt lỗi thành `ToolResult(ok=False, error=…)`** (Agent
self-reflect, không sập). Ảnh truy vấn qua `image_ref` (không nhét bytes vào schema).
Output CHUẨN HOÁ (`norm_candidates/raws/temporal`) → JSON-friendly cho G2/Reader.
**File:** `retrieval/agent_tools.py` · **Test:** `tests/test_agent_tools.py` (12 test:
đăng ký đủ 9 tool · schema 2 SDK · call thành công từng tool · 3 đường lỗi self-reflect).

### ✅ G2. Search Agent (Orchestrator loop) — STAR / MemoriEase  ĐÃ XONG (2026-07-15)
Slide 30–31: vòng **observe → reason → act**. **Đã làm:** `Planner` (ABC) + `SearchAgent`
(dựng registry G1 rồi giao Planner tự lái, trả `AgentRun` truy vết đủ bước/tool).
- `MockPlanner` (offline, tất định): `understand` (định tuyến) → chọn nhánh
  `search_temporal` / `search_by_image` / `search_multimodal` / `search` theo cấu trúc
  câu + có ảnh không → nếu tìm-chữ và có kết quả thì `disambiguation` (cờ mơ hồ) → finish.
- `ClaudePlanner` (thật, lazy): Anthropic tool-use loop — `specs("anthropic")` làm tools,
  vòng tool_use→tool_result→text cuối; cho tiêm `client` để test offline.
**File:** `retrieval/search_agent.py` · **Test:** `tests/test_search_agent.py` (9 test:
4 nhánh định tuyến, bước finish, + ClaudePlanner loop bằng fake client, dừng max_steps,
đòi API key). Định vị: Agent là "smart path" — KHÔNG thay `engine.search` "fast path".

### ✅ G3. Session Memory (episodic + semantic) — trí nhớ xuyên lượt  ĐÃ XONG (2026-07-15)
Slide "Memory": **episodic** (append-only) + **semantic** (feedback tích luỹ + facts).
**Đã làm:** `SessionMemory` + `Turn`:
- **episodic** `list[Turn]` append-only, đọc theo độ mới (`recent`/`recent_queries`).
- **semantic** — feedback 👍/👎 TÍCH LUỸ (`note_feedback`, quy tắc *phản hồi mới thắng*)
  + `facts` dict (tri thức tự do). `feedback_context()` bơm vào `ToolRegistry.context`.
- **Nối G2:** `SearchAgent.chat()` (lượt CÓ trí nhớ, khác `run()` fast-path độc lập) —
  gấp phản hồi lượt trước → Planner định tuyến qua **`search_with_feedback` (Rocchio)** ở
  lượt sau ("lượt 2 nhớ lượt 1") → ghi Turn → đính `summary` vào `run.meta`. Thêm tool
  `search_with_feedback` vào registry (G1).
**File:** `retrieval/session_memory.py`, `search_agent.py`, `agent_tools.py` · **Test:**
`tests/test_session_memory.py` (6) + 3 test tích hợp (feedback blue→Rocchio kéo lên đầu).

### ✅ G4. RAG Reader — sinh đáp án có dẫn chứng (MemoriEase 3.0)  ĐÃ XONG (2026-07-15)
Slide 31: sau Rerank là **Reader** tổng hợp câu trả lời. **Đã làm:** thêm `Reader` (ABC),
`MockReader`, `ClaudeReader`, `ReaderAnswer` vào `vqa_module.py`. Reader nhận KẾT QUẢ TOOL
đã chuẩn hoá (list dict) + entry (tra caption/OCR/ASR/ảnh) → câu trả lời tiếng Việt
**trích dẫn keyframe_id** + vị trí video/thời gian.
- `MockReader` (offline): tổng hợp caption/OCR/ASR + timestamp của top-K, cite id.
- `ClaudeReader` (lazy): gửi top-K ảnh (từ `raws[kid].image_bytes`) + text cho Claude
  vision; tiêm `client` để test cấu trúc request offline.
- **Nối G2:** `SearchAgent(reader=…)` — điền `answer` khi Planner chưa tự trả lời (Mock),
  KHÔNG ghi đè câu của ClaudePlanner. Đúng mắt xích "Rerank → Reader".
**File:** `retrieval/vqa_module.py`, `retrieval/search_agent.py` · **Test:**
`tests/test_rag_reader.py` (6) + 2 test tích hợp trong `test_search_agent.py`.

### 🟢 G5. Reasoning trace CoT/ToT — nâng cấp Query Understanding
Slide "Reasoning": System-2, CoT (chuỗi đơn), ToT (nhiều nhánh + tự đánh giá + backtrack).
Với query nhiều ràng buộc/temporal: sinh **vài kế hoạch truy vấn ứng viên**, tự chấm,
chọn (hoặc chạy song song rồi RRF-merge). Nâng cấp Q4. Chỉ đáng làm sau khi có API
(ToT thật cần LLM); `Mock` = enumerate biến thể có sẵn.
**File:** mở rộng `retrieval/query_understanding.py` + `query_expansion.py`.

### Thứ tự đề xuất Nhóm G
**G1 → G2 → G4 → G3 → G5.** G1+G2 làm được NGAY (MockPlanner, không cần key) và cho giá
trị demo lớn nhất ("hệ có bộ não"). G3/G4/G5 phát huy tối đa khi có `ANTHROPIC_API_KEY`.

> ⚠️ **Định vị:** Nhóm G là lớp *điều phối trên nền đã có*, KHÔNG thay thế pipeline hiện
> tại — pipeline vẫn là "fast path" mặc định; Agent là "smart path" cho query khó/hội thoại.
> Đúng đánh đổi **tốc độ ↔ sức mạnh ↔ chi phí** mà cả Buổi 2 lẫn Buổi 3 đều nhấn.
