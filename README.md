<p align="center">
  <img src="docs/assets/banner.svg" alt="Rewind — Tua lại tìm khoảnh khắc" width="100%">
</p>

<p align="center"><b>Tua lại tìm khoảnh khắc.</b><br/>
Công cụ tìm kiếm ngữ nghĩa cho video quy mô lớn — tìm bất kỳ khoảnh khắc nào bằng <em>lời mô tả</em>, <em>ảnh mẫu</em>, hoặc <em>hội thoại</em>.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-262%20passing-2f9e6f" alt="Tests">
  <img src="https://img.shields.io/badge/ANN-Faiss%20HNSW-F5A623" alt="Faiss">
  <img src="https://img.shields.io/badge/encoder-SigLIP2-6E8BFF" alt="SigLIP">
  <img src="https://img.shields.io/badge/layer-Agentic-8B5CF6" alt="Agentic">
  <img src="https://img.shields.io/badge/offline-no%20API%20key-06b6d4" alt="Offline">
</p>

<p align="center">
  <a href="#-bắt-đầu-nhanh"><b>Bắt đầu</b></a> ·
  <a href="#-hai-cách-tìm">Cách tìm</a> ·
  <a href="#-kiến-trúc">Kiến trúc</a> ·
  <a href="#-tài-liệu">Tài liệu</a> ·
  <a href="#-số-đo-thật">Số đo</a>
</p>

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🎬 Rewind là gì?

Rewind biến một kho video khổng lồ — từ vài giờ tới **hàng trăm giờ** (hàng triệu
keyframe) — thành thứ bạn **tra cứu theo ý nghĩa**, không phải tua tay hay lục theo tên
file. Mô tả khoảnh khắc cần tìm, đưa một **ảnh mẫu**, hoặc **trò chuyện** để hệ hỏi lại
thu hẹp dần — nó trả về đúng khung hình.

<table>
<tr>
<td width="50%" valign="top">

**🗣️ Tìm bằng lời** — *"người mặc áo đỏ ở quầy hoa quả"*

**🖼️ Tìm bằng ảnh / phác hoạ** — đưa ảnh mẫu → ra cảnh giống

</td>
<td width="50%" valign="top">

**⏱️ Tìm theo thứ tự** — *"cởi mũ **trước khi** vào phòng"*

**💬 Tìm bằng hội thoại** — hệ chủ động hỏi lại để khoanh vùng

</td>
</tr>
</table>

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🎯 Bộ lọc ảnh — thu hẹp dần tới đúng khoảnh khắc

Khi bạn **khó mô tả đủ trong một câu**: cứ nói đại khái, rồi thu hẹp bằng **ảnh thật**.

<p align="center">
  <img src="docs/assets/filter.svg" alt="Bộ lọc ảnh: 20 → 8 → 3 → 1" width="100%">
</p>

Ba tín hiệu thu hẹp, **dùng lẫn nhau tuỳ ý** trong cùng một lượt:

| Thao tác | Cơ chế |
|---|---|
| gõ thêm chi tiết | truy vấn cộng dồn |
| 👍 / 👎 trên ảnh | **Rocchio** — kéo về ảnh thích, ra xa ảnh không thích |
| chọn ảnh "gần ý nhất" khi hệ hỏi | ảnh chọn → 👍, các ảnh còn lại → 👎 |

Mỗi lượt chỉ xếp hạng lại **trong tập ứng viên hiện tại**, nên số ảnh **đảm bảo giảm** —
đúng mô hình một *bộ lọc*, không phải tìm lại từ đầu.

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🧠 Hai cách tìm

<p align="center">
  <img src="docs/assets/agent.svg" alt="Agent tự chọn công cụ" width="100%">
</p>

Rewind có **hai đường đi** trên cùng một bộ máy truy xuất:

<table>
<tr><th width="50%">⚡ Fast path — <code>VideoSearchEngine</code></th><th width="50%">🧠 Smart path — <code>SearchAgent</code></th></tr>
<tr valign="top">
<td>

Bạn chọn công cụ, gọi thẳng pipeline.

```python
engine.search(entry, "người đi bộ trên phố")
```

Nhanh, rõ ràng — hợp đa số truy vấn.

</td>
<td>

Bạn gõ một câu, **Agent tự quyết** dùng công cụ nào.

```python
agent.run("cởi mũ trước khi vào phòng")
# → ['understand', 'search_temporal', …]
```

Tự định tuyến · nhớ phản hồi xuyên lượt · trả lời kèm **trích dẫn keyframe** · và **cho
bạn xem nó đã gọi tool nào, vì sao**.

</td>
</tr>
</table>

> Agent là lớp **điều phối** trên chính các công cụ đã có — không cài lại thuật toán
> truy xuất. Chạy được **không cần API key** (bộ não luật, tất định); có
> `ANTHROPIC_API_KEY` thì tự nâng lên Claude function-calling.

<img src="docs/assets/divider.svg" width="100%" alt="">

## ✨ Điểm nổi bật

|  |  |
|---|---|
| 🧩 **Truy xuất recall-first nhiều tầng** | coarse (nhanh, giữ recall) → RRF fusion → VLM rerank → kiểm nhất quán thời gian |
| 🎯 **Bộ lọc ảnh hội thoại** | lưới ảnh thật co dần: mô tả · 👍/👎 · chọn ảnh gần ý nhất |
| 🧠 **Lớp Agentic** | Tool Registry · Search Agent · Session Memory · RAG Reader — **đều có trên UI** |
| ❓ **VQA trên video thật** | dùng chính câu hỏi định vị cửa sổ keyframe rồi mới trả lời |
| ⚡ **Sẵn sàng scale** | Faiss HNSW + metadata pre-filter · batch embed · lưu/nạp index · decode ‖ embed |
| 🤝 **Ensemble 2 encoder** | SigLIP2 + SigLIP đa ngôn ngữ — giảm "cùng sai", **Việt lẫn Anh** |
| 🔎 **Hybrid dense + sparse** | SigLIP trộn BM25 (OCR/ASR/caption) qua RRF, trọng số **thích ứng** |
| 🧪 **Mock-first** | mọi thứ cần GPU/API đều có bản Mock chạy **offline**; bản thật *lazy-import* |
| ✅ **262 unit test** | `pytest` chạy hoàn toàn offline — kể cả vòng lặp Agent |

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🏗️ Kiến trúc

```mermaid
flowchart TD
    subgraph IDX["🗂️ INDEXING (offline)"]
        V[Video] --> KF[Cắt keyframe + objects]
        KF --> DD[Dedup gần trùng]
        DD --> EMB[SigLIP2 + SigLIP-ml embed]
        DD --> ENR[Caption LLM · OCR · ASR]
        EMB --> IX[(Faiss HNSW + BM25 + metadata)]
        ENR --> IX
    end

    subgraph RET["🔎 RETRIEVAL (online)"]
        Q[Truy vấn: chữ / ảnh / hội thoại] --> QU[Query understanding]
        QU --> CO[Coarse recall-first<br/>dense ensemble + BM25 + pre-filter]
        CO --> FU[RRF fusion]
        FU --> RR[VLM fine rerank]
        RR --> TC[Kiểm nhất quán thời gian]
    end

    subgraph AG["🧠 AGENTIC LAYER"]
        AGENT[Search Agent] --> TOOLS[Tool Registry]
        AGENT --> MEM[Session Memory<br/>episodic + semantic]
        AGENT --> READER[RAG Reader<br/>đáp án có dẫn chứng]
    end

    IX -.-> CO
    TOOLS -.điều phối.-> RET
    TC --> RES{Định tuyến}
    RES -->|known-item| K[Top-1/5]
    RES -->|ad-hoc| A[Danh sách xếp hạng]
    RES -->|VQA| VQ[Trả lời + suy luận]
    RES -->|hội thoại| KC[Lọc ảnh / hỏi lại]
```

> **Nguyên tắc:** *độ chính xác trên tốc độ*. Tầng lọc thô ưu tiên **recall cao** (không
> đánh mất ứng viên đúng), tầng rerank tối ưu precision. Mọi thứ **tính trước được** dồn
> vào lúc index; chỉ thứ phụ thuộc query mới tối ưu độ trễ.

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🚀 Bắt đầu nhanh

```bash
# 1. Môi trường (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate            # Windows · source .venv/bin/activate (Linux/macOS)
pip install -r requirements.txt

# 2. Chạy toàn bộ test — offline, không cần GPU/API
pytest                            # → 262 passed

# 3. Web demo (pipeline thật)
python -m ui.app                  # → http://127.0.0.1:5000
```

Bỏ video vào `data/videos/`, mở tab **🎥 Video** → **Nạp video** → tìm.

<details>
<summary><b>🎥 Tìm trên video thật bằng dòng lệnh (SigLIP local, không cần API key)</b></summary>

```bash
pip install opencv-python-headless torch transformers sentencepiece protobuf pillow

python -m ingestion.video_ingest phim.mp4 --out artifacts/frames --every 1.0
python -m retrieval.video_search_demo phim.mp4 "người đang đi bộ trên phố" --topk 5
```
</details>

<details>
<summary><b>🧠 Lớp Agentic & bộ lọc ảnh từ code</b></summary>

```python
from retrieval.video_engine import VideoSearchEngine
from retrieval.search_agent import SearchAgent
from retrieval.image_filter import ImageFilterSession
from retrieval.vqa_module import MockReader, answer_on_video

engine = VideoSearchEngine()
entry  = engine.index_video("phim.mp4", "artifacts/frames")

# Agent tự định tuyến (offline; đổi sang ClaudePlanner/ClaudeReader khi có API key)
agent = SearchAgent(engine, entry, reader=MockReader())
run = agent.run("cảnh người cởi mũ trước khi vào phòng")
print(run.tools_used(), run.answer)

# Hội thoại có trí nhớ: lượt sau nhớ phản hồi lượt trước (Rocchio)
agent.chat("cảnh trên phố")
agent.chat("cảnh trên phố", positive_ids=["walking/7"])

# Bộ lọc ảnh: thu hẹp dần
s = ImageFilterSession(engine, entry, start_k=20)
s.start("cảnh đường phố")            # → 20 ảnh
s.refine(text="áo trắng")            # → ~10
s.refine(positive=["walking/7"])     # → ~5

# VQA: câu hỏi tự định vị cửa sổ keyframe rồi mới trả lời
ans, info = answer_on_video(engine, entry, "Có bao nhiêu người đi bộ?")
```
</details>

<img src="docs/assets/divider.svg" width="100%" alt="">

## 📊 Số đo thật

Benchmark trên **51 nhãn thật** (RTX 3060) — *đo, không đoán*:

| Cấu hình | Hit@1 | Hit@5 | MRR |
|---|:---:|:---:|:---:|
| coarse · BM25 cố định 1.0 | 0.275 | 0.647 | 0.438 |
| coarse · BM25 cố định 3.0 | 0.314 | 0.529 ↓ | 0.408 |
| **coarse · BM25 thích ứng** | 0.333 | **0.647** | 0.477 |
| **+ VLM rerank** | **0.549** | **0.667** | **0.604** |

**Đọc số:** VLM rerank là cú nhảy lớn nhất (**Hit@1 +0.22**). Trọng số BM25 cao *làm hại*
recall — nên mới có cơ chế **thích ứng theo loại truy vấn**. Hit@5 ~0.65 rất ổn định qua
mọi lần chạy: đó là **trần của encoder**, chỉ phá được bằng caption LLM / encoder mạnh hơn.

### Còn "sẵn sàng scale" thì sao? — cũng đo, không nói suông

`evaluation/bench_scale.py` đo tầng ANN ở **768 chiều** (số chiều thật), tới 200k vector:

| Đo được | Kết quả | Nghĩa là |
|---|---|---|
| RAM | **3.343 B/vector** (tuyến tính) | 200k keyframe ≈ **2.4 GB** (×2 encoder) → thoải mái; 1M ≈ **12 GB** → **cần IVF-PQ** |
| Build index | 200k trong **43 s** | không phải nút thắt — **trích embedding** mới là (200k frame ≈ 7–14 h) |
| Latency coarse | **<1 ms → 11 ms** @200k | ~0,05% của time budget 20 s; VLM rerank mới chiếm phần lớn |
| recall@100 vs `efSearch` | 128 → **0.465** · 2048 → **0.981** | 👇 |

> 🔴 **Benchmark này bắt được một lỗi cấu hình thật:** `efSearch=128` **âm thầm đánh mất
> ~54% ứng viên đúng** ở tầng lọc thô — vi phạm chính ràng buộc "không được mất recall ở
> coarse". Đã chốt **`efSearch=2048`**: recall **0.98**, trả giá **11 ms** (bằng 0,05%
> ngân sách). Kết luận vận hành: **HNSW float đúng cho tới ~300–400 giờ video**, quá mức
> đó mới cần IVF-PQ/sharding.

**Nguyên tắc:** tham số `[PROVISIONAL]` trong `configs/settings.yaml` chỉ được chốt sau khi
qua `bench_retrieval.py` (accuracy) hoặc `bench_scale.py` (quy mô) — **đo, không đoán**.

<img src="docs/assets/divider.svg" width="100%" alt="">

## 📚 Tài liệu

| Tài liệu | Dành cho |
|---|---|
| 🖥️ [`HUONG_DAN_GIAO_DIEN.md`](HUONG_DAN_GIAO_DIEN.md) | Dùng **giao diện web** — từng tab, từng nút, vòng phản hồi |
| 📘 [`HUONG_DAN.md`](HUONG_DAN.md) | Thao tác bằng **code/CLI** — bảng tra "muốn làm X → dùng gì" |
| 💬 [`kisc_module/`](kisc_module/) | Hội thoại thu hẹp bằng Information Gain |

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🧠 Tech stack & lý do chọn

| Thành phần | Lựa chọn | Vì sao |
|-----------|----------|--------|
| Encoder | **SigLIP2** + **SigLIP đa ngôn ngữ** (ensemble) | Sigmoid loss ổn định, zero-shot tốt; ensemble giảm correlated errors |
| Vector search | **Faiss HNSW** (cosine) | Cân bằng tốc độ/chính xác, giữ float — ưu tiên accuracy |
| Sparse | **BM25** trên objects/OCR/ASR/caption | Bắt tín hiệu chữ (biển hiệu, lời nói) mà embedding bỏ sót |
| Fusion | **Reciprocal Rank Fusion** (k=60) | Gộp nhiều ranked list khác thang đo, không cần chuẩn hoá |
| Rerank & VQA | **Qwen2-VL** (local) / **Claude** vision | Cross-attention đọc *từng token* → hiểu quan hệ, thứ tự, số lượng |
| Phản hồi | **Rocchio** trên embedding | Vòng khám phá↔khai phá, không cần metadata thuộc tính |
| Bộ não Agent | **LLM function-calling** | Điều phối công cụ, suy luận nhiều bước, hội thoại |

<img src="docs/assets/divider.svg" width="100%" alt="">

## 🗺️ Trạng thái

| Nhóm | Nội dung | |
|---|---|:---:|
| Truy xuất | dedup · ingestion · Faiss HNSW · coarse · fusion · rerank · temporal | ✅ |
| Truy vấn | hiểu query · mở rộng · ảnh · sketch · đa phương thức · temporal | ✅ |
| Hiển thị | video browser · lân cận · gom cụm · explore · progress | ✅ |
| Phản hồi | Rocchio · gợi ý concept · **bộ lọc ảnh** · hỏi lại bằng ảnh | ✅ |
| Agentic | Tool Registry · Search Agent · Session Memory · RAG Reader | ✅ |
| VQA | trên video thật (cửa sổ theo câu hỏi) | ✅ |
| Scale/eval | batch embed · lưu/nạp index · benchmark harness · metrics | ✅ |

*Còn lại — đều chờ tài nguyên, không phải thiếu code:* caption LLM để phá trần recall ·
reasoning CoT/ToT (cần LLM) · IVF-PQ/sharding (khi profiling cho thấy RAM là nút thắt).

<img src="docs/assets/divider.svg" width="100%" alt="">

<p align="center">
  <img src="docs/assets/banner.svg" width="66%" alt="Rewind">
</p>
<p align="center"><i>Rewind — tua lại tìm khoảnh khắc.</i></p>
<p align="center">
  <sub>Chạy offline bằng mock · cắm model/API thật là dùng · 262 test xanh</sub>
</p>
