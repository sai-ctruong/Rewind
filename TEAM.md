# 👥 TEAM.md — Tài liệu phối hợp & tiến độ dự án AIC 2026

> Đọc file này trước khi bắt tay vào code. Mục tiêu: người mới nắm được **dự án làm
> gì · đang tới đâu · file nào làm gì · setup thế nào · làm gì tiếp theo**. Đặc tả kỹ
> thuật đầy đủ nằm ở `CLAUDE.md` (blueprint, chỉ có trên máy — không đẩy lên GitHub).

Cập nhật lần cuối: 2026-07-12 · Nhánh chính: `main` · 159 unit test xanh.

---

## 1. Dự án là gì

**Trợ lý ảo Truy xuất Đa phương tiện** cho Hội thi AI Challenge (AIC) 2026 TP.HCM.
Tìm kiếm trong video/lifelog theo 4 dạng bài toán:

| | Bài toán | Ý nghĩa |
|---|---|---|
| **KIS** | Known-Item Search | Tìm đúng 1 khoảnh khắc (Top-1/5) |
| **AVS** | Ad-hoc Video Search | Tìm tất cả đoạn khớp, xếp hạng |
| **VQA** | Video QA | Trả lời câu hỏi (đếm, ai làm gì) |
| **KISC** | Conversational KIS | Hội thoại hỏi lại để thu hẹp |

Nguyên tắc cốt lõi (blueprint Mục 1): **accuracy > speed**, pipeline nhiều tầng,
tầng lọc thô ưu tiên recall, tầng rerank tối ưu precision, **không mất recall**.

---

## 2. ✅ Tiến độ hiện tại (đã xong & chạy được)

### Lõi hệ thống (roadmap Phase 0–10)
- [x] **Ingestion**: dedup keyframe (anchor-based), schema `KeyframeRecord`
- [x] **Index**: Faiss HNSW (dense) + BM25 (sparse) + metadata — `ingestion/build_index.py`
- [x] **Coarse retrieval**: ensemble + RRF fusion, recall-first — `retrieval/coarse_retriever.py`
- [x] **Query understanding / expansion** (mock-first, Claude lazy)
- [x] **Fine rerank** (LVLM verifier, time budget, early-stop)
- [x] **Temporal check** (thứ tự "A trước B")
- [x] **VQA module** (đếm/nhận diện, mock-first)
- [x] **Tích hợp KISC** (retriever thật thay MockRetriever)
- [x] **Evaluation**: Recall@K, MRR, mAP, nDCG, EM, F1 — `evaluation/`

### Tìm kiếm VIDEO THẬT (mở rộng sau roadmap — không cần API key)
- [x] 🎞️ **Cắt keyframe từ .mp4** (OpenCV + bỏ frame trùng) — `ingestion/video_ingest.py`
- [x] 🧩 **Ensemble 2 encoder** SigLIP2 + SigLIP-multilingual (RRF) — `retrieval/video_engine.py`
- [x] 🌏 **Đa ngôn ngữ** (query tiếng Việt lẫn Anh)
- [x] 🔀 **Query prompt ensemble** (trung bình embedding nhiều biến thể câu)
- [x] 🗂️ **Tìm xuyên CẢ DATASET** (nhiều video, biết kết quả ở video nào)
- [x] 📁 **Nạp dataset từ THƯ MỤC bất kỳ** trên máy (quét đệ quy)
- [x] 📝 **Tìm bằng CHỮ / biển hiệu** (OCR EasyOCR → BM25 fusion) — `ingestion/ocr_asr_extract.py::EasyOcrEngine`
- [x] 🤖 **VLM rerank** hiểu từng chữ + ngữ cảnh (Qwen2-VL-2B, local) — `retrieval/vlm_rerank.py`
- [x] ⚡ **GPU (CUDA)** — SigLIP embedding ~8× nhanh hơn CPU
- [x] 🎨 **Web UI** 5 tab (KISC · KIS · AVS · VQA · Video) — `ui/`
- [x] 💾 **Cache model trên ổ D** (không phồng ổ C) — `ingestion/model_cache.py`

### Hạ tầng
- [x] 159 unit test (pytest, chạy offline bằng mock)
- [x] Chạy trên **Python 3.14** + venv `.venv`
- [x] GPU: **NVIDIA RTX 3060 6GB**, torch `2.13.0+cu126`

---

## 3. 🗺️ Bản đồ thư mục (file nào làm gì)

```
kisc_module/            💬 Hội thoại KISC (entropy/Information Gain) — CÓ SẴN, ít sửa
ingestion/
  ├─ schemas.py          KeyframeRecord, RawKeyframe, StructuredQuery
  ├─ video_ingest.py     .mp4 → keyframe (cv2)
  ├─ embed_siglip.py     SiglipEncoder (GPU) + encode_text (cross-modal)
  ├─ embed_clip.py       nạp CLIP feature BTC (mock)
  ├─ ocr_asr_extract.py  EasyOcrEngine (OCR chữ) + Whisper ASR (chưa wire)
  ├─ llm_captioning.py   caption bằng LVLM (mock/Claude)
  ├─ dedup.py            gộp keyframe gần trùng
  ├─ build_index.py      KeyframeIndex: Faiss HNSW + BM25 + metadata
  ├─ build_records.py    ráp RawKeyframe → KeyframeRecord; searchable_text
  └─ model_cache.py      trỏ HF_HOME sang ổ D
retrieval/
  ├─ video_engine.py     ⭐ TRUNG TÂM tìm video: ensemble + OCR + rerank + dataset
  ├─ coarse_retriever.py CoarseRetriever: pre-filter + dense + sparse → RRF
  ├─ fusion.py           Reciprocal Rank Fusion (có trọng số)
  ├─ fine_rerank.py      FineReranker (time budget, early-stop) + Reranker ABC
  ├─ vlm_rerank.py       Qwen2VLReranker (VLM local, GPU)
  ├─ query_understanding.py / query_expansion.py
  ├─ temporal_check.py   lọc thứ tự thời gian
  ├─ vqa_module.py       hỏi–đáp video
  ├─ kisc_adapter.py     cầu nối KISC ↔ retriever thật
  └─ video_search_demo.py  CLI tìm video
ui/
  ├─ app.py              backend Flask (API: kisc/search/vqa/video/index_folder…)
  └─ index.html          frontend 1 trang (5 tab, tự phát hiện backend)
evaluation/              metrics.py + run_eval.py + bench_video_engine.py
tests/                   159 test (mock, offline)
configs/settings.yaml    ngưỡng tập trung ([PROVISIONAL] cần benchmark)
```

**File hay đụng nhất:** `retrieval/video_engine.py` (mọi tính năng tìm video hội tụ ở đây)
và `ui/app.py` + `ui/index.html` (giao diện).

---

## 4. 🚀 Setup cho thành viên mới

```powershell
git clone https://github.com/sai-ctruong/KISC_module.git
cd KISC_module

# Python 3.14 (bắt buộc 3.10+). Tạo venv:
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# CÓ GPU NVIDIA? cài torch CUDA (thay bản CPU) — tăng tốc 8-30x:
pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126

# Chạy test cho chắc:
pytest                      # → 159 passed

# Chạy web demo:
python -m ui.app            # http://127.0.0.1:5000
```

**Lưu ý môi trường:**
- Console vỡ tiếng Việt → `set PYTHONUTF8=1` trước khi chạy.
- Model (SigLIP, Qwen2-VL, EasyOCR) **tự tải về ổ D** (`D:\hf_cache`) lần đầu dùng — venv chỉ chứa code.
- Bỏ video thật vào `data\videos\` (đã gitignore) hoặc trỏ thư mục bất kỳ trong UI.

---

## 5. 🎮 Cách chạy nhanh

| Việc | Lệnh |
|---|---|
| Web UI | `python -m ui.app` |
| Tìm 1 video (CLI) | `python -m retrieval.video_search_demo data\videos\clip.mp4 "câu tìm" --topk 5` |
| Đánh giá | `python -m evaluation.run_eval` |
| Benchmark encoder | `python -m evaluation.bench_video_engine` |
| Demo KISC | `python -m retrieval.kisc_real_demo` |
| Test | `pytest` |

---

## 6. 🔧 Quy ước làm việc (2 người)

- **Nhánh:** để tránh dẫm chân nhau, mỗi tính năng làm 1 nhánh riêng rồi mở PR:
  `git checkout -b feat/<ten>` → code → `git push -u origin feat/<ten>` → mở PR.
  (Hiện repo đang push thẳng `main`; khi 2 người làm song song NÊN chuyển sang PR.)
- **Test bắt buộc:** mỗi thay đổi logic phải có/không phá test → `pytest` xanh trước khi PR.
- **Mock-first:** phần cần GPU/API/model nặng phải có bản mock chạy offline (test không tải model).
- **Không commit:** `CLAUDE.md` (blueprint riêng, đã gitignore), `data/`, `artifacts/`, model, `.venv`.
- **Model:** để ở ổ D (`D:\hf_cache`) — không commit, không để phồng ổ C.
- **settings.yaml:** số `[PROVISIONAL]` chỉ chốt sau khi benchmark (Mục 11.3), không đoán.

---

## 7. 🧑‍💻 Phân vai đề xuất (điều chỉnh tuỳ 2 bạn)

Chia theo **mảng** để ít đụng file của nhau:

**Bạn A — Retrieval & Ranking** (`retrieval/`)
- Encoder/ensemble, fusion (RRF + trọng số), VLM rerank, temporal, VQA
- Tinh chỉnh độ chính xác, benchmark (`evaluation/bench_*`)

**Bạn B — Data & UI** (`ingestion/` + `ui/`)
- Video ingest, OCR/ASR, dedup, build_index, dataset/folder, cache model
- Web UI (tab, hiển thị, tương tác), trải nghiệm người dùng

**Chung:** `evaluation/`, `tests/`, tài liệu. File giao thoa lớn nhất là
`retrieval/video_engine.py` → khi cả hai cùng sửa, tách nhánh + PR sớm để merge dễ.

---

## 8. 📋 Backlog — việc tiếp theo (ưu tiên từ trên xuống)

Đều **không cần API key** trừ khi ghi rõ:

1. 💾 **Lưu/nạp index ra đĩa** — dataset lớn nạp 1 lần, mở lại tức thì (không embed lại). *(B)*
2. 📊 **Progress bar** khi nạp dataset lớn (biết còn bao lâu). *(B)*
3. 🔘 **Toggle tắt/bật OCR** trên UI (nạp nhanh khi không cần chữ). *(B)*
4. 🎙️ **ASR (Whisper) — tìm theo LỜI NÓI** trong video (đã có WhisperAsrEngine, chưa wire vào video_engine). *(B)*
5. 🧭 **Query understanding cho video** — parse câu → tách filter (thời gian/đối tượng) trước khi search. *(A)*
6. ⏱️ **Temporal search trên video thật** — "cảnh A trước cảnh B". *(A)*
7. 📈 **Đánh giá trên dữ liệu có nhãn thật** — tạo bộ test ground-truth, đo Recall@K/mAP. *(chung)*
8. 🧠 **VQA/KISC trên video thật** — cần object-detection hoặc `ANTHROPIC_API_KEY` (LVLM). *(A)*
9. 🗃️ **Sharding index** cho kho rất lớn (hàng trăm giờ). *(A/B)*
10. ⚙️ **CI GitHub Actions** chạy pytest mỗi push. *(chung)*

---

## 9. ⚠️ Ghi chú kỹ thuật quan trọng

- **Python 3.14**, venv `.venv`. torch `2.13.0+cu126` (GPU). transformers 5.13.
- **GPU RTX 3060 6GB**: SigLIP ensemble + Qwen2-VL-2B fp16 ≈ ~5GB VRAM → vừa. Nếu hết VRAM khi bật rerank: giảm `max_pixels` trong `Qwen2VLReranker` hoặc dùng model nhỏ hơn.
- **OCR** bật mặc định (`enable_ocr=True`) → nạp chậm hơn ~0.6s/frame; trọng số BM25 = 3.0 (`video_engine.bm25_weight`).
- **Qwen2-VL-2B (~8GB)** chỉ tải khi bật checkbox "Rerank VLM" lần đầu.
- Blueprint đầy đủ + lý do thiết kế: xem `CLAUDE.md` (trên máy).

---

*Có gì chưa rõ, hỏi trong nhóm hoặc xem `CLAUDE.md` / `README.md`. Chúc build vui! 🎬*
