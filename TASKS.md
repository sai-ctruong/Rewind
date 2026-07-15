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
