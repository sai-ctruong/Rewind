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

### A2. Lưu / nạp index ra đĩa  ⭐ (bắt buộc để không embed lại)
**Vấn đề:** index + `image_bytes` nằm trong RAM; tắt app là mất, phải embed lại từ đầu.
**Làm:** serialize `VideoIndexEntry` (Faiss `write_index` + metadata/OCR ra parquet/
npz; **KHÔNG** lưu `image_bytes` vào index — chỉ lưu embedding + timestamp + đường
dẫn video + frame_idx để trích lại ảnh khi cần hiển thị). API `/api/video/save` &
tự nạp lại lúc khởi động.
**File:** `ingestion/build_index.py`, `retrieval/video_engine.py`, `ui/app.py`
**Chặn:** không có cái này thì mọi test scale đều phải re-embed → không làm nổi.

### A3. Decode video bằng GPU (NVDEC) + chỉ lấy frame cần
**Vấn đề:** `cv2.VideoCapture` decode **mọi** frame bằng CPU rồi vứt (chỉ giữ 1/giây)
→ decode là nút thắt chi phối ở 50.000 giờ.
**Làm:** thay bằng `decord`/`PyNvVideoCodec` (NVDEC) hoặc ffmpeg `-hwaccel cuda -vf fps=1`
để chỉ giải mã frame ở mốc lấy mẫu, trên GPU. Giữ `cv2` làm fallback khi không có CUDA.
**File:** `ingestion/video_ingest.py`
**Ước tính ăn:** giảm mạnh thời gian decode + giải phóng CPU cho việc khác.

### A4. Song song hoá nhiều worker (decode ‖ embed ‖ ghi index)
**Vấn đề:** pipeline chạy tuần tự 1 luồng: decode xong mới embed, embed xong mới index.
**Làm:** hàng đợi sản xuất–tiêu thụ: worker CPU decode → queue → GPU embed theo lô →
index. Nhiều video decode song song để luôn có ảnh sẵn cho GPU (GPU không chờ CPU).
**File:** `retrieval/video_engine.py` (thêm lớp điều phối indexing)
**Phụ thuộc:** nên làm sau A1 (batch) để có gì mà "nuôi" GPU.

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

### B1. Benchmark thực nghiệm để chốt tham số [PROVISIONAL]
Blueprint Mục 11.3 **bắt buộc**: đo recall/latency thật rồi mới chốt `sample_every_s`,
`dedup_threshold`, `efSearch` HNSW, `rerank_pool`, `bm25_weight`. Hiện đang là số đoán.
**Cần:** một bộ test có nhãn nhỏ (~20–50 cặp query–đáp trên video thật) →
`evaluation/bench_video_engine.py` mở rộng, vẽ đường cong, chọn điểm "khuỷu tay".
**Chặn nhiều thứ:** không có nhãn thì mọi tinh chỉnh đều mù.

### B2. Query understanding cho video (parse câu → filter)
Tách câu tự nhiên → `{objects, actions, location, time, temporal_order}` (schema
`StructuredQuery` đã có) **trước** khi search → pre-filter thu hẹp, tăng precision.
Đã có `retrieval/query_understanding.py` (mock/Claude) nhưng **chưa nối** vào `video_engine`.
**File:** `retrieval/video_engine.py`, `retrieval/query_understanding.py`

### B3. ASR (Whisper) — tìm theo LỜI NÓI
`WhisperAsrEngine` đã có nhưng chưa wire. Trích transcript quanh mỗi keyframe →
đưa vào BM25 cùng OCR → tìm được cảnh theo điều người ta **nói**, không chỉ nhìn thấy.
**File:** `ingestion/ocr_asr_extract.py`, `retrieval/video_engine.py`

### B4. Temporal search trên video thật ("cảnh A trước cảnh B")
`temporal_check.py` đã có cho pipeline record; nối vào luồng video: search 2 sự kiện,
lọc cặp `t_A < t_B` trong cùng video.
**File:** `retrieval/video_engine.py`, `retrieval/temporal_check.py`

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
2. **A2 — Lưu/nạp index ra đĩa** (mở khoá mọi thử nghiệm scale) ← kế tiếp
3. **B1 — Bộ nhãn nhỏ + benchmark** (để đo được, chốt được tham số; gồm cả đo throughput A1)
4. **A3 — NVDEC decode** (gỡ nút thắt CPU)
5. **A4/A5 — Song song hoá + IVF-PQ/shard** (khi A2–A3 đã đo được throughput thật)

> Nguyên tắc: **đo trước, tối ưu sau** (blueprint Mục 11.3). Làm A1+A2 xong rồi
> chạy thử trên ~10–50 video thật để lấy throughput/RAM thực, mới quyết A5.
