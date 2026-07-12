# 📋 TASKS.md — Việc còn thiếu & cải tiến (xếp theo tác động)

> File này liệt kê **cái gì còn thiếu, cải tiến gì để hệ thống tốt hơn**, ưu tiên
> từ trên xuống theo **tác động / công sức**. Trạng thái tính năng đã-xong xem
> `TEAM.md` Mục 2. Đặc tả thiết kế xem `CLAUDE.md`.
>
> Cập nhật: 2026-07-12 · Nhánh: `main`

---

## 0. Bối cảnh: hệ thống ĐANG ở đâu

Hiện chạy tốt ở **quy mô nhỏ–vừa** (vài video → vài chục video, demo end-to-end
đủ 4 bài toán KIS/AVS/VQA/KISC). **Chưa sẵn sàng cho quy mô thi thật** (mục tiêu
blueprint: hàng trăm giờ → hàng triệu keyframe).

**Phép thử scale (10.000 video × 5h ≈ 50.000 giờ):**
- ~180 triệu frame lấy mẫu (1 frame/giây) → ~80–120 triệu keyframe sau dedup.
- Với code hiện tại, 1 máy: **hàng tháng → >1 năm** (không khả thi).
- 3 nút thắt: (1) **embed lẻ từng ảnh** → GPU đói việc, (2) **decode video bằng
  CPU** (cv2), (3) **index nằm trong RAM**, không lưu đĩa, không shard.

→ Các task nhóm A dưới đây là để **đóng khoảng cách scale này**. Đây là ưu tiên
cao nhất vì nó quyết định hệ thống có dùng được ở đề thật hay không.

---

## A. 🔴 SCALE — bắt buộc để chạy được dataset lớn (ưu tiên cao nhất)

Xếp theo **tác động / độ dễ**:

### A1. Batch embedding trên GPU  ✅ ĐÃ XONG (2026-07-12)
**Vấn đề:** `SiglipEncoder.embed()` xử lý **1 ảnh/lần** → GPU chạy ~2–5% công suất.
**Đã làm:** thêm `SiglipEncoder.embed_batch(raws, batch_size=256)` gom ảnh thành 1
tensor, một lượt `get_image_features` (fp16, chia lô tránh tràn VRAM).
`VideoSearchEngine._embed_all` ưu tiên `embed_batch`, fallback `.embed()` cho mock.
`_embed_raws` embed cả loạt raws/encoder thay vì vòng lặp lẻ. Có test
`test_embed_batch_path_used_and_consistent` (lô == lẻ về kết quả).
**Còn lại:** đo throughput thật ảnh/giây trước/sau trên RTX 3060 (gộp vào B1 benchmark).
**File:** `ingestion/embed_siglip.py`, `retrieval/video_engine.py`
**Ước tính ăn:** giảm thời gian embed **~50–100 lần**.

### A2. Lưu / nạp index ra đĩa  ✅ ĐÃ XONG (2026-07-12)
**Vấn đề:** index + `image_bytes` nằm trong RAM; tắt app là mất, phải embed lại từ đầu.
**Đã làm:** `VideoIndexEntry.save(dir)`/`.load(dir)` — Faiss index (`KeyframeIndex.save`)
+ metadata/OCR + raws đã **bỏ `image_bytes`** (nặng). Thêm `source_video` + `frame_idx`
vào `RawKeyframe`; loader `load_pil/cv2_image` **decode lại frame từ video gốc** khi
thiếu ảnh RAM/đĩa → hiển thị/rerank vẫn chạy sau khi nạp từ đĩa. UI: nút "💾 Lưu index"
(`/api/video/save`) + **tự nạp lại** index khi khởi động app. Endpoint ảnh dựng lại từ
video gốc rồi encode JPEG. Test `test_save_load_roundtrip`.
**Còn lại:** khi IVF-PQ (A5) thay HNSW thì đổi định dạng lưu tương ứng.
**File:** `ingestion/schemas.py`, `ingestion/video_ingest.py`, `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

### A3. Decode video bằng GPU (NVDEC) + chỉ lấy frame cần  ✅ ĐÃ XONG (2026-07-12)
**Vấn đề:** `cv2.VideoCapture.read()` decode **mọi** frame bằng CPU rồi vứt (chỉ giữ 1/giây)
→ decode là nút thắt chi phối ở 50.000 giờ.
**Đã làm:** backend decode cắm được trong `extract_keyframes(decode_backend=..., use_gpu=...)`:
- **decord** (`_decode_samples_decord`): `get_batch(indices)` chỉ decode frame lấy mẫu,
  `ctx=gpu(0)` dùng **NVDEC**; tự về CPU nếu không có CUDA. Lazy import.
- **cv2** (`_decode_samples_cv2`): dùng `grab()` bỏ qua frame giữa, `retrieve()` chỉ tại
  điểm mẫu → tránh giải mã đầy đủ ~(step-1)/step frame. Luôn có, không cần cài thêm.
- **auto**: ưu tiên decord nếu import được, else cv2. Engine có `decode_backend`/`use_gpu_decode`.
Test: `test_cv2_backend_only_samples_step_frames`, `test_backend_cv2_matches_auto`,
`test_decord_backend_missing_raises`.
**CÒN LẠI:** để có **NVDEC thật** phải `pip install decord` (bản CUDA) trên máy user —
chưa cài ở env hiện tại (decord/PyNvVideoCodec đều vắng, chỉ có PyAV). Đo tốc độ decode
thật gộp vào B1.
**File:** `ingestion/video_ingest.py`, `retrieval/video_engine.py`

### A4. Song song hoá nhiều worker (decode ‖ embed ‖ ghi index)  ✅ ĐÃ XONG (2026-07-12)
**Vấn đề:** pipeline chạy tuần tự 1 luồng: decode xong mới embed, embed xong mới index.
**Đã làm:** `iter_keyframes` (bản GENERATOR của extract_keyframes) stream keyframe theo
dòng. `VideoSearchEngine._pipeline_records`: 1 luồng PRODUCER decode+stream → `queue.Queue`
GIỚI HẠN (maxsize=4, chặn phình RAM) → luồng chính CONSUMER embed theo lô GPU + OCR.
cv2/torch nhả GIL khi chạy C++/CUDA → thread cho song song thật (decode lô kế trong khi
GPU embed lô hiện tại). Thứ tự FIFO bảo toàn → dedup/index giống hệt tuần tự. Param
`parallel_index=True` (tắt để debug). Test `test_parallel_index_matches_sequential`.
**Phụ thuộc:** dựa trên A1 (batch) để "nuôi" GPU — đã có.
**File:** `ingestion/video_ingest.py`, `retrieval/video_engine.py`

### A5. Chuyển ANN sang IVF-PQ + sharding khi > vài triệu vector
**Vấn đề:** HNSW giữ vector float trong RAM → 100M × 768d × 4B ≈ **300 GB RAM** (bất khả thi 1 máy).
**Làm:** khi tổng vector vượt ngưỡng (đo ở A2), chuyển coarse sang **Faiss IVF-PQ**
(nén vector) như blueprint Mục 2.2 & 5.3; chia **shard theo lô video**, search song
song rồi merge. Fine rerank vẫn dùng float32 gốc (two-precision, Mục 11.1).
**File:** `ingestion/build_index.py`
**Đo trước khi làm:** profiling RAM ở A2 (blueprint Mục 2.2 — "không giả định trước").

### A6. Progress bar / log tiến độ khi index dataset lớn
**Vấn đề:** nạp dataset lớn không biết còn bao lâu, tưởng treo.
**Làm:** đếm video/frame đã xử lý, đẩy tiến độ qua SSE/polling lên UI; ETA theo throughput đo được.
**File:** `ui/app.py`, `ui/index.html`

---

## B. 🟡 ĐỘ CHÍNH XÁC — làm hệ thống "trúng" hơn (ưu tiên vừa)

### B1. Benchmark thực nghiệm để chốt tham số [PROVISIONAL]  🟨 HARNESS XONG (2026-07-12)
Blueprint Mục 11.3 **bắt buộc**: đo recall/latency thật rồi mới chốt `sample_every_s`,
`dedup_threshold`, `efSearch` HNSW, `rerank_pool`, `bm25_weight`.
**Đã làm:** `evaluation/bench_retrieval.py` — (1) `measure_embed_throughput` đo frame/giây
embed LẺ vs THEO LÔ (lượng hoá A1), (2) `evaluate_labeled` chấm Recall@K/hit@K/MRR trên
bộ nhãn theo **cửa sổ thời gian** (ổn định qua mọi cấu hình), (3) nạp NHÃN THẬT từ JSON
(`load_labels`), (4) `sweep_configs` + `--sweep` quét nhiều cấu hình → đường cong recall/latency.
Test offline bằng mock (`tests/test_bench_retrieval.py`, 6 test). **Bộ nhãn thật đã tạo:**
`evaluation/labels.json` — 25 cặp trên 3 video thật (Sydney/NYC/Seoul mưa), cửa sổ xác
minh bằng cách trích frame ra xem (xem `evaluation/labels.README.md`).
**CÒN LẠI (chỉ chạy trên máy có GPU — việc của user):**
- Chạy `python -m evaluation.bench_retrieval --labels evaluation/labels.json --sweep`
  **trên RTX 3060** → lấy số throughput A1 thật + Recall/MRR + đường cong sample_every_s.
- Đọc đường cong, chọn "khuỷu tay", cập nhật `configs/settings.yaml` (bỏ [PROVISIONAL]).

### B2. Query understanding cho video (parse câu → filter)
Tách câu tự nhiên → `{objects, actions, location, time, temporal_order}` (schema
`StructuredQuery` đã có) **trước** khi search → pre-filter thu hẹp, tăng precision.
Đã có `retrieval/query_understanding.py` (mock/Claude) nhưng **chưa nối** vào `video_engine`.
**File:** `retrieval/video_engine.py`, `retrieval/query_understanding.py`

### B3. ASR (Whisper) — tìm theo LỜI NÓI
`WhisperAsrEngine` đã có nhưng chưa wire. Trích transcript quanh mỗi keyframe →
đưa vào BM25 cùng OCR → tìm được cảnh theo điều người ta **nói**, không chỉ nhìn thấy.
**File:** `ingestion/ocr_asr_extract.py`, `retrieval/video_engine.py`

### B4. Temporal search trên video thật ("cảnh A trước cảnh B")  ✅ ĐÃ XONG (2026-07-12)
**Đã làm:** `VideoSearchEngine.search_temporal(entry, events)` — mỗi `events[i]` là 1 câu
mô tả cảnh theo thứ tự; search coarse từng cảnh rồi `temporal_consistency_filter` (đã có)
lọc CỨNG giữ chuỗi CÙNG video, timestamp tăng dần (Mục 4.5, không gộp vào fusion). Khoá
theo chỉ số để không đụng khi 2 cảnh trùng chữ. UI: tab "⏱️ Chuỗi" + endpoint
`/api/video/temporal` (hiển thị chuỗi keyframe A→B→C). Test: `test_search_temporal_respects_order`
(đúng thứ tự có chuỗi, ngược thứ tự rỗng, 3 cảnh, <2 cảnh báo lỗi) + test UI đường lỗi.
**File:** `retrieval/video_engine.py`, `ui/app.py`, `ui/index.html`

### B5. LLM Auto-Captioning lúc indexing (blueprint Mục 2.4)
Sinh caption tự nhiên cho mỗi keyframe đại diện (LVLM) → BM25 bắt được **quan hệ
ngữ nghĩa** ("người lớn hướng dẫn trẻ tưới hoa") mà object-detector rời rạc không nắm.
`llm_captioning.py` có bản mock; cần nối vào indexing video (cần API hoặc VLM local).
**File:** `ingestion/llm_captioning.py`, `retrieval/video_engine.py`

### B6. Ensemble thêm CLIP (BTC cấp sẵn) cạnh SigLIP
Blueprint Mục 2.1: dùng CLIP feature BTC cấp (miễn phí compute) + SigLIP → giảm
"cùng sai". Hiện engine chạy 2 encoder SigLIP; khi có feature CLIP của BTC thì cắm vào 1 slot.
**File:** `retrieval/video_engine.py`, `ingestion/embed_clip.py`

---

## C. 🟢 ĐÁNH GIÁ & VẬN HÀNH — để tin được kết quả & làm nhóm (ưu tiên nền)

### C1. Bộ dữ liệu có nhãn thật + báo cáo số liệu
Tạo ground-truth (query → keyframe đúng) trên video thật; chạy `evaluation/run_eval.py`
ra Recall@K, mAP, MRR, EM. **Đây là thước đo để mọi cải tiến B chứng minh được là có tác dụng.**

### C2. VQA / KISC trên video thật
VQA/KISC hiện chạy trên record mẫu. Nối vào keyframe video thật (cần object-detection
hoặc `ANTHROPIC_API_KEY` cho LVLM suy luận đếm/thứ tự).

### C3. CI GitHub Actions chạy pytest mỗi push
Tự động chạy test khi push/PR → không để hỏng lọt lên `main`.
**File:** `.github/workflows/ci.yml` (mới)

### C4. Chuyển sang quy trình PR khi 2 người làm song song
Hiện push thẳng `main`. Khi cả 2 cùng sửa `video_engine.py` → tách nhánh + PR
(TEAM.md Mục 6) để merge không dẫm chân.

---

## 🎯 Đề xuất thứ tự làm 5 việc kế tiếp

1. ~~**A1 — Batch embedding**~~ ✅ xong
2. ~~**A2 — Lưu/nạp index ra đĩa**~~ ✅ xong
3. ~~**B1 — Harness benchmark + nhãn thật**~~ 🟨 code + 25 nhãn xong; chỉ còn **chạy trên GPU** (1 lệnh)
4. ~~**A3 — NVDEC decode**~~ ✅ xong (backend cắm được; cần `pip install decord` bản CUDA để có NVDEC thật)
5. ~~**A4 — Song song hoá**~~ ✅ xong (pipeline decode ‖ embed, queue giới hạn)
6. **A5 — IVF-PQ + sharding** (khi tổng vector > vài triệu; cần đo RAM ở B1 trước) ← kế tiếp

> Nguyên tắc: **đo trước, tối ưu sau** (blueprint Mục 11.3). Làm A1+A2 xong rồi
> chạy thử trên ~10–50 video thật để lấy throughput/RAM thực, mới quyết A5.
