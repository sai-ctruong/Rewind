# 📖 Hướng dẫn sử dụng Rewind

Tài liệu này đưa bạn đi từ *chạy được ngay* → *hiểu từng chức năng* → *bật bản thật*.
Đọc README trước để nắm bức tranh lớn; file này là **cẩm nang thao tác**.

> Toàn bộ ví dụ chạy trên **Windows (PowerShell)**. Trên Linux/macOS đổi `\` → `/` và
> `.venv\Scripts\activate` → `source .venv/bin/activate`.

---

## Mục lục

1. [Hiểu trong 1 phút: hai đường đi](#1-hiểu-trong-1-phút-hai-đường-đi)
2. [Cài đặt](#2-cài-đặt)
3. [Ba cách dùng](#3-ba-cách-dùng)
4. [Bảng tra nhanh: muốn làm X → dùng gì](#4-bảng-tra-nhanh-muốn-làm-x--dùng-gì)
5. [Đi qua từng chức năng (Python API)](#5-đi-qua-từng-chức-năng-python-api)
   - [5.1 Index một video](#51-index-một-video)
   - [5.2 Tìm bằng chữ](#52-tìm-bằng-chữ)
   - [5.3 Tìm bằng ảnh / đa phương thức](#53-tìm-bằng-ảnh--đa-phương-thức)
   - [5.4 Tìm theo thứ tự thời gian](#54-tìm-theo-thứ-tự-thời-gian)
   - [5.5 Rerank bằng VLM (chính xác hơn)](#55-rerank-bằng-vlm-chính-xác-hơn)
   - [5.6 OCR · ASR · Caption](#56-ocr--asr--caption)
   - [5.7 Duyệt video: lân cận, tương tự, explore](#57-duyệt-video-lân-cận-tương-tự-explore)
   - [5.8 Phản hồi liên quan (Rocchio) & gợi ý concept](#58-phản-hồi-liên-quan-rocchio--gợi-ý-concept)
   - [5.9 Hỏi–đáp trên video (VQA)](#59-hỏiđáp-trên-video-vqa)
   - [5.10 Hội thoại thu hẹp (KISC)](#510-hội-thoại-thu-hẹp-kisc)
6. [Lớp Agentic: để hệ thống tự điều phối](#6-lớp-agentic-để-hệ-thống-tự-điều-phối)
7. [Bật bản "thật" (API key / model local)](#7-bật-bản-thật-api-key--model-local)
8. [Đánh giá & benchmark](#8-đánh-giá--benchmark)
9. [Cấu hình (settings.yaml)](#9-cấu-hình-settingsyaml)
10. [Gỡ rối thường gặp](#10-gỡ-rối-thường-gặp)

---

## 1. Hiểu trong 1 phút: hai đường đi

Có **2 cách** dùng cùng một bộ máy truy xuất:

| | **Fast path** — `VideoSearchEngine` | **Smart path** — `SearchAgent` |
|---|---|---|
| Là gì | Gọi thẳng pipeline truy xuất | Bộ não LLM điều phối các công cụ |
| Khi nào | Đa số truy vấn, cần nhanh & rõ ràng | Query khó, hội thoại, cần suy luận nhiều bước |
| Bạn gọi | `engine.search(entry, "…")` | `agent.run("…")` / `agent.chat("…")` |
| Tự định tuyến? | Không — bạn chọn hàm | Có — tự chọn tìm-chữ / thời gian / ảnh / hỏi lại |

Mọi thứ bắt đầu bằng việc **index một video** để tạo `entry`, rồi truy vấn trên `entry` đó.

---

## 2. Cài đặt

### Bước tối thiểu (chạy test offline, không cần GPU/API)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest                       # → 243 passed  (xác nhận mọi thứ chạy)
```

### Thêm để xử lý VIDEO THẬT (SigLIP local, vẫn không cần API key)

```powershell
pip install opencv-python-headless torch transformers sentencepiece protobuf pillow
```

> Lần đầu chạy sẽ **tải model SigLIP** (vài trăm MB) về cache — chỉ tải 1 lần.

---

## 3. Ba cách dùng

### A. Web UI (dễ nhất — bấm chuột)

```powershell
python -m ui.app          # mở http://127.0.0.1:5000
```

Trang web có các tab:
- **Video** — index video, tìm bằng chữ / ảnh / sketch, bật OCR·ASR·Caption·Rerank,
  xem lân cận, gom cụm, explore, phản hồi 👍/👎, gợi ý concept.
- **Hội thoại (KISC)** — mô tả khoảnh khắc, trợ lý hỏi lại thu hẹp dần.
- **Hỏi–đáp (VQA)** — đặt câu hỏi trên cửa sổ keyframe.

### B. Dòng lệnh (CLI) — nhanh, không viết code

```powershell
# Cắt keyframe từ video ra thư mục
python -m ingestion.video_ingest phim.mp4 --out artifacts/frames --every 1.0

# Tìm bằng chữ trên video thật
python -m retrieval.video_search_demo phim.mp4 "người đang đi bộ trên phố" --topk 5

# Demo hội thoại KISC (dữ liệu mẫu)
python -m kisc_module.demo
```

### C. Python API — linh hoạt nhất (phần 5 & 6 bên dưới)

---

## 4. Bảng tra nhanh: muốn làm X → dùng gì

| Bạn muốn… | Hàm / cách dùng |
|---|---|
| Tạo index từ 1 video | `engine.index_video(video, out_dir)` |
| Tìm bằng câu mô tả | `engine.search(entry, "câu", top_k=5)` |
| Tìm chính xác hơn (Top-1 quan trọng) | `engine.search(entry, "câu", rerank=True)` |
| Tìm bằng 1 ảnh mẫu | `engine.search_by_image(entry, image_bytes)` |
| Tìm bằng chữ + ảnh cùng lúc | `engine.search_multimodal(entry, "câu", image_bytes)` |
| Tìm chuỗi "A trước B" | `engine.search_temporal(entry, ["A", "B"])` |
| Xem frame trước/sau 1 kết quả | `engine.neighbors(entry, frame_id)` |
| "Tìm cảnh giống ảnh này" | `engine.search_similar(entry, keyframe_id)` |
| Lướt mẫu khắp dataset | `engine.explore(entry)` |
| Tinh chỉnh bằng 👍/👎 | `engine.search_with_feedback(entry, "câu", positive_ids=[…])` |
| Gợi ý từ khoá thu hẹp | `engine.suggest_concepts(entry, ids, "câu")` |
| Hỏi–đáp trên video | `VqaModule().answer("câu hỏi?", records)` |
| Hội thoại thu hẹp | `python -m kisc_module.demo` / tab KISC |
| **Để hệ tự quyết mọi thứ** | `SearchAgent(engine, entry).run("câu")` |
| **Hội thoại có trí nhớ** | `agent.chat("câu", positive_ids=[…])` |

---

## 5. Đi qua từng chức năng (Python API)

Khởi tạo chung cho mọi ví dụ dưới đây:

```python
from retrieval.video_engine import VideoSearchEngine

engine = VideoSearchEngine()                       # cấu hình mặc định
entry  = engine.index_video("phim.mp4", "artifacts/frames")
```

`entry` (kiểu `VideoIndexEntry`) là "kho đã đánh chỉ mục" của video — mọi truy vấn đều
chạy trên nó.

### 5.1 Index một video

```python
engine = VideoSearchEngine(
    sample_every_s=1.0,    # lấy 1 keyframe mỗi giây (nhỏ hơn = dày hơn = chậm hơn)
    max_frames=None,       # None = index CẢ video; đặt số để giới hạn
    enable_ocr=True,       # đọc chữ trên khung hình (biển hiệu…) -> tìm được text
)
entry = engine.index_video("phim.mp4", "artifacts/frames")
print(entry.num_sampled, "->", entry.num_indexed)   # sau dedup còn bao nhiêu keyframe
```

Index nhiều video vào **một index chung** (tìm xuyên suốt cả kho):
`engine.index_dataset(["v1.mp4", "v2.mp4", …], out_dir)` — mỗi keyframe vẫn giữ đúng
`video_id`, nên kết quả biết rõ "ở video nào, giây mấy".

### 5.2 Tìm bằng chữ

```python
results = engine.search(entry, "người mặc áo đỏ đứng ở quầy hoa quả", top_k=5)
for c in results:
    print(f"{c.score:.3f}  {c.video_id} @ {c.timestamp:.1f}s  ({c.keyframe_id})")
```

Hoạt động cả **tiếng Việt lẫn tiếng Anh**. Bên trong: SigLIP (ảnh) + BM25 (chữ) trộn
bằng RRF, trọng số BM25 **tự điều chỉnh** theo loại câu (câu nhiều chữ in hoa / biển
hiệu → tăng BM25; câu mô tả thị giác → giảm).

### 5.3 Tìm bằng ảnh / đa phương thức

```python
img = open("mau.jpg", "rb").read()

# Chỉ ảnh (image-to-video)
engine.search_by_image(entry, img, top_k=5)

# Chữ + ảnh cùng lúc: "giống ảnh này nhưng nhấn mạnh ý câu"
engine.search_multimodal(entry, "vào ban đêm", img, text_weight=0.5, top_k=5)
```

`text_weight` trong [0,1]: gần 1 = nghiêng về chữ, gần 0 = nghiêng về ảnh.

### 5.4 Tìm theo thứ tự thời gian

Cho truy vấn dạng *"cảnh A xảy ra **trước** cảnh B"*:

```python
matches = engine.search_temporal(entry, ["người cởi mũ", "người bước vào phòng"])
for m in matches:
    print(m.video_id, [f"{s.timestamp:.1f}s" for s in m.steps])
```

Chỉ giữ các tổ hợp **cùng video, timestamp tăng dần đúng thứ tự** — một tổ hợp sai thứ
tự bị loại dứt khoát dù điểm cao (đây là ràng buộc logic, không phải similarity).

### 5.5 Rerank bằng VLM (chính xác hơn)

```python
engine.search(entry, "hai người bắt tay nhau", top_k=5, rerank=True)
```

Bật `rerank=True` khi **Top-1 quan trọng**. Coarse (SigLIP) nén cả câu thành 1 vector nên
yếu về tổ hợp từ; VLM (Qwen2-VL) đọc ảnh + *từng token* câu → hiểu đúng quan hệ/thứ
tự/số lượng. Đổi lại **chậm hơn** (chạy trên top-K nhỏ). Chạy local, không cần API key.

### 5.6 OCR · ASR · Caption

Ba tín hiệu chữ làm giàu cho BM25, bật khi khởi tạo engine:

```python
engine = VideoSearchEngine(
    enable_ocr=True,      # đọc CHỮ trên khung hình (biển hiệu, phụ đề cứng) — mặc định BẬT
    enable_asr=True,      # chép LỜI NÓI cả video (Whisper) — nặng, mặc định TẮT
    enable_caption=True,  # VLM sinh MÔ TẢ ngữ cảnh mỗi keyframe — nặng, mặc định TẮT
)
```

- **OCR**: tìm được cảnh theo text ("SEPHORA", "NEW YORK"). ⚠️ Lưu ý: benchmark cho thấy
  đặt trọng số OCR quá cao **làm hại recall** — cứ để mặc định (adaptive).
- **ASR**: tìm theo điều ai đó nói.
- **Caption**: hiểu *quan hệ tương tác* ("người lớn hướng dẫn trẻ tưới hoa") mà object
  detector rời rạc bỏ lỡ. Bản local (Qwen2-VL) tốn VRAM; nên dùng **Claude** khi có key.

### 5.7 Duyệt video: lân cận, tương tự, explore

```python
# Frame trước/sau một kết quả (định vị bối cảnh theo timeline)
engine.neighbors(entry, frame_id, before=4, after=4)

# "Tìm cảnh giống ảnh kết quả này" (dùng embedding đã lưu, không encode lại)
engine.search_similar(entry, keyframe_id, top_k=8)

# Lướt mẫu đa dạng khắp dataset khi chưa biết bắt đầu từ đâu
engine.explore(entry, per_video=3, limit=30)
```

### 5.8 Phản hồi liên quan (Rocchio) & gợi ý concept

```python
# Người dùng bấm 👍 vào vài kết quả -> kéo truy vấn về hướng đó, ra xa cái 👎
engine.search_with_feedback(
    entry, "cảnh trên phố",
    positive_ids=["kf_12", "kf_18"],   # thích
    negative_ids=["kf_03"],            # không thích
    top_k=5,
)

# Gợi ý từ khoá hay xuất hiện trong top-K để thu hẹp/mở rộng truy vấn
ids = [c.keyframe_id for c in engine.search(entry, "phố", top_k=10)]
engine.suggest_concepts(entry, ids, "phố")     # -> ['đêm', 'mưa', 'taxi', …]
```

### 5.9 Hỏi–đáp trên video (VQA)

```python
from retrieval.vqa_module import VqaModule

# records: danh sách KeyframeRecord (đã có caption/objects). Xem tests/test_vqa_phase7.py
vqa = VqaModule()                                  # MockVqaAnswerer (offline)
ans = vqa.answer("Trên bánh có mấy ngọn nến?", records, video_id="birthday")
print(ans.answer, ans.value, ans.used_frame_ids)   # "5", 5, [...]
```

Trả lời có suy luận: **đếm số lượng**, **xác định ai làm gì**, mô tả. Bản thật
(`ClaudeVqaAnswerer`) gửi ảnh keyframe cho Claude vision.

### 5.10 Hội thoại thu hẹp (KISC)

Trợ lý **chủ động hỏi lại** để khoanh vùng khi mô tả còn mơ hồ — chọn câu hỏi tối đa
hoá Information Gain (giảm entropy nhanh nhất).

```powershell
python -m kisc_module.demo          # bản mẫu, hội tụ trung bình ~2 lượt
```

Nối KISC vào retriever thật: xem `retrieval/kisc_adapter.py` và
`retrieval/kisc_real_demo.py`.

---

## 6. Lớp Agentic: để hệ thống tự điều phối

Thay vì bạn chọn hàm nào, **Search Agent** tự quyết. Nó dùng đúng các công cụ ở phần 5,
nhưng có một *bộ não* điều phối chúng theo truy vấn.

### 6.1 Chạy một lượt tự định tuyến

```python
from retrieval.search_agent import SearchAgent
from retrieval.vqa_module import MockReader

agent = SearchAgent(engine, entry, reader=MockReader())   # offline, không cần key

run = agent.run("người cởi mũ trước khi vào phòng")
print(run.tools_used())   # ['understand', 'search_temporal', ...] — TỰ chọn tìm-thời-gian
print(run.answer)         # câu trả lời có trích dẫn [keyframe_id]
print(run.results)        # danh sách kết quả chuẩn hoá
```

Agent tự nhận ra: câu có "trước khi" → gọi `search_temporal`; câu có ảnh → `search_by_image`;
câu thường → `search`; kết quả mơ hồ → `disambiguation` (chọn vài ứng viên để hỏi lại).

### 6.2 Hội thoại CÓ TRÍ NHỚ (nhớ lượt trước)

```python
agent.chat("cảnh trên phố")                         # lượt 1
agent.chat("cảnh trên phố", positive_ids=["kf_42"]) # lượt 2: 👍 kf_42
# -> lượt 2 tự dùng Rocchio với phản hồi tích luỹ, kéo kết quả về hướng kf_42

print(agent.memory.summary())    # {turns, recent_queries, positive_ids, facts, ...}
```

`SessionMemory` giữ **episodic** (chuỗi lượt) + **semantic** (phản hồi tích luỹ, facts).
Phản hồi 👎 ở lượt sau ghi đè 👍 lượt trước (người dùng đổi ý được tôn trọng).

### 6.3 Xem "Agent đã nghĩ gì"

```python
for step in run.steps:
    a = step.action
    print(a.kind, a.tool, a.rationale)    # từng bước quyết định + lý do
```

### 6.4 Dùng bộ não Claude thật

```python
from retrieval.search_agent import ClaudePlanner
from retrieval.vqa_module import ClaudeReader
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-…"

agent = SearchAgent(engine, entry,
                    planner=ClaudePlanner(),      # function-calling thật
                    reader=ClaudeReader())        # đáp án vision có dẫn chứng
run = agent.run("tìm khoảnh khắc thổi nến sinh nhật")
```

Mặc định (không key) là `MockPlanner` + `MockReader` — vẫn tự định tuyến, chạy offline.

---

## 7. Bật bản "thật" (API key / model local)

Mọi thành phần nặng theo mẫu **Mock (offline) ↔ bản thật (lazy)**. Đổi khi có tài nguyên,
**không cần đổi code gọi**:

| Thành phần | Bản offline | Bản thật | Cần gì |
|---|---|---|---|
| Encoder | (ColorMock trong test) | `SiglipEncoder` (mặc định khi index thật) | `torch transformers` |
| VLM rerank | coarse-only | `Qwen2-VL` local | `torch transformers` (VRAM) |
| Caption | tắt | `QwenVLCaptioner` / `ClaudeCaptioner` | VRAM / `ANTHROPIC_API_KEY` |
| ASR | tắt | `WhisperVideoAsrEngine` | `openai-whisper` |
| OCR | tắt được | `EasyOcrEngine` | `easyocr` |
| VQA | `MockVqaAnswerer` | `ClaudeVqaAnswerer` | `ANTHROPIC_API_KEY` |
| Agent Planner | `MockPlanner` | `ClaudePlanner` | `ANTHROPIC_API_KEY` |
| Reader | `MockReader` | `ClaudeReader` | `ANTHROPIC_API_KEY` |

Đặt key một lần cho cả phiên:

```powershell
$env:ANTHROPIC_API_KEY = "sk-…"       # PowerShell
```

Lưu ý mức "tự động":
- **Caption** — khi có `ANTHROPIC_API_KEY`, engine **tự** chọn `ClaudeCaptioner`.
- **Rerank** — mặc định luôn dùng **Qwen2-VL local** (không tự chuyển Claude); muốn đổi thì
  `engine.set_reranker(...)`.
- **Planner / Reader** — **không** tự động; phải truyền tường minh
  `SearchAgent(engine, entry, planner=ClaudePlanner(), reader=ClaudeReader())`.

---

## 8. Đánh giá & benchmark

```powershell
# Benchmark truy xuất trên nhãn thật: throughput embedding + Hit@K + so cấu hình
python -m evaluation.bench_retrieval --labels evaluation/labels.json

# Báo cáo đánh giá end-to-end
python -m evaluation.run_eval
```

Nguyên tắc: **đo trước, tối ưu sau**. Đừng hard-code tham số (efSearch, top-K, trọng số
BM25) — chạy benchmark trên dữ liệu thật rồi chọn theo đường cong recall/latency. Kết quả
tham chiếu lưu ở `evaluation/benchmarks/`.

---

## 9. Cấu hình (settings.yaml)

`configs/settings.yaml` gom các ngưỡng. Đánh dấu:
- `[FIXED]` — đã đo, dùng luôn (vd `bm25_weight=1.0`, `dedup_threshold=0.97`).
- `[PROVISIONAL]` — **phải benchmark** trước khi chốt cho dataset của bạn
  (vd `efSearch`, `top-K` mỗi tầng, `time_budget`).

Đa số tham số cũng truyền thẳng được vào `VideoSearchEngine(...)` như phần 5.1.

---

## 10. Gỡ rối thường gặp

| Hiện tượng | Nguyên nhân & cách xử lý |
|---|---|
| Console vỡ tiếng Việt (Windows) | `set PYTHONUTF8=1` (hoặc `$env:PYTHONUTF8=1`) trước khi chạy |
| `ModuleNotFoundError: retrieval` | Chạy từ **thư mục gốc dự án**, hoặc đặt `PYTHONPATH=.` |
| Lần đầu chạy rất lâu | Đang **tải model** SigLIP/Qwen về cache — chỉ lần đầu |
| Hết VRAM khi caption/rerank local (6GB) | Dùng `ANTHROPIC_API_KEY` (Claude) thay bản local; hoặc giảm `embed_batch_size` |
| Bật OCR mà recall tệ đi | Đừng tăng trọng số OCR thủ công — để `adaptive_bm25=True` (mặc định) |
| Rerank quá chậm | Giảm `rerank_pool`, hoặc chỉ bật `rerank=True` khi thật cần Top-1 |
| Muốn index cả video dài | `max_frames=None` (mặc định) + tăng `sample_every_s` để bớt frame |
| Agent không gọi Claude | Chưa đặt `ANTHROPIC_API_KEY`, hoặc quên truyền `planner=ClaudePlanner()` |

---

<div align="center">

Cần chi tiết một module? Mở file tương ứng trong `retrieval/` hoặc `ingestion/` —
mỗi file có docstring giải thích **lý do** thiết kế, không chỉ mô tả code.

</div>
