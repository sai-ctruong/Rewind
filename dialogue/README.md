# KISC Module — Conversational Known-Item Search (AIC 2026)

Module xử lý bài toán **Conversational KIS (KISC)** — điểm mới và khó nhất của AIC 2026
(slide "Bài toán Conversational KIS (KISC)"): thay vì người dùng nhập 1 câu truy vấn rồi
chờ kết quả, trợ lý ảo **chủ động hội thoại** để làm rõ ý định trước khi trả kết quả cuối.

## 1. Lý thuyết nền tảng

**Vấn đề**: mô tả ban đầu của người dùng thường mơ hồ → tập ứng viên khớp rất lớn.
Câu hỏi cốt lõi: hệ thống nên hỏi về **thuộc tính (attribute) nào tiếp theo** để thu hẹp
tập ứng viên nhanh nhất?

**Giải pháp — Information Gain / Shannon Entropy** (cùng nguyên lý với thuật toán
ID3/C4.5 xây Decision Tree, hoặc chiến lược tối ưu trong trò chơi Twenty Questions):

```
H(attribute) = -Σ p_i · log2(p_i)
```

trong đó `p_i` là tỷ lệ ứng viên hiện tại có giá trị `i` ở thuộc tính đang xét.

- Entropy **càng cao** → giá trị thuộc tính càng phân tán đều trong tập ứng viên
  → hỏi về thuộc tính này sẽ **chia tập ứng viên thành các nhóm cân bằng nhất**
  → giảm độ bất định (uncertainty) nhanh nhất cho mỗi câu hỏi.
- Entropy = 0 → mọi ứng viên còn lại giống nhau ở thuộc tính đó → hỏi thêm **vô ích**.

**Điều kiện dừng hội thoại** (`ambiguity.is_confident_enough`):
1. Số ứng viên còn lại đã đủ nhỏ (`<= max_candidates`), hoặc
2. Có khoảng cách điểm số rõ rệt giữa Top-1 và Top-2 (score gap) — một kết quả nổi
   trội hẳn, không cần hỏi thêm.

## 2. Kiến trúc module

```
dialogue/
├── schemas.py            # Keyframe, DialogueTurn, DialogueState
├── ambiguity.py           # entropy(), best_attribute_to_ask(), is_confident_enough()
├── retriever.py           # HybridRetriever (interface) + MockRetriever (demo)
├── slot_extractor.py      # trích filter từ câu trả lời tự nhiên (rule-based + LLM stub)
├── question_generator.py  # sinh câu hỏi làm rõ (template + LLM stub)
├── dialogue_manager.py    # vòng lặp chính, ghép tất cả module trên
└── demo.py                # demo CLI tái hiện case study trong slide AIC 2026
```

**Vòng lặp hội thoại** (`KISCDialogueManager`):

```
User input
   │
   ▼
SlotExtractor.extract()          -> filters mới
   │
   ▼
state.filters.update(...)        -> cộng dồn (Dialogue State Tracking)
   │
   ▼
HybridRetriever.search()         -> candidate set mới (áp toàn bộ filters)
   │
   ▼
is_confident_enough()? ──Yes──▶ trả kết quả cuối, kết thúc
   │No
   ▼
best_attribute_to_ask()          -> chọn attribute entropy cao nhất, chưa hỏi
   │
   ▼
QuestionGenerator.generate()     -> câu hỏi tự nhiên gửi lại người dùng
```

## 3. Chạy demo (offline, không cần API/GPU)

```bash
cd <thư mục cha chứa dialogue/>
python3 -m dialogue.demo
```

Demo tái hiện đúng kịch bản trong slide (Case Study 2 / KISC): "Tìm giúp tôi đoạn
video tôi gặp một người bạn cũ vào tuần trước" → hệ thống hỏi lại → hội tụ về kết quả.

## 4. Tích hợp vào hệ thống thật — checklist

Module này được thiết kế để cắm thẳng vào pipeline chính (xem kiến trúc Indexing/Retrieval
đã bàn ở phase trước) mà **không cần sửa dialogue_manager.py**:

| Bước | Việc cần làm | File cần sửa |
|---|---|---|
| 1 | Viết `FaissCLIPRetriever(HybridRetriever)`: encode query bằng CLIP text encoder, search trên Faiss/Milvus đã nạp Keyframes + CLIP features do BTC cấp, áp filter lên Objects/metadata | `retriever.py` |
| 2 | Thay `SlotExtractor.extract()` bằng `extract_with_llm()`: gọi Claude API với prompt yêu cầu trả JSON slot values — xử lý ngôn ngữ tự nhiên tốt hơn keyword matching nhiều | `slot_extractor.py` |
| 3 | Thay `QuestionGenerator.generate()` bằng `generate_with_llm()`: câu hỏi tự nhiên hơn, có ngữ cảnh hội thoại trước đó | `question_generator.py` |
| 4 | Mở rộng `CANDIDATE_ATTRIBUTES` trong `dialogue_manager.py` theo schema Objects/CLIP concepts thật (ví dụ: số người, loại vật thể, hành động chi tiết hơn) | `dialogue_manager.py` |
| 5 | Nối `KISCDialogueManager` vào UI thật (mỗi lượt hội thoại = 1 lần gọi `start()`/`respond()`, hiển thị `state.candidates` dạng lưới ảnh) | UI layer (ngoài module này) |

## 5. Hạn chế của bản demo & TODO

- `MockRetriever` dùng exact-match filter (rule cứng) — hệ thống thật cần **similarity
  search mềm** (CLIP cosine similarity) kết hợp filter cứng (post-filtering hoặc
  metadata pre-filtering trên Faiss/Milvus).
- `SlotExtractor` rule-based chỉ nhận diện được từ khóa đã định nghĩa trước — cần
  thay bằng LLM extraction để xử lý được diễn đạt tự do.
- Chưa xử lý **temporal logic constraints** (thứ tự trước/sau hành động — thách thức
  #3 trong slide "Ba thách thức kỹ thuật cốt lõi"). Có thể mở rộng bằng cách thêm
  attribute `sequence_order` và so khớp theo timestamp giữa các keyframe liên quan.
- Chưa có cơ chế "nới lỏng filter" khi hội thoại đi vào ngõ cụt (0 ứng viên) — hiện
  tại chỉ yêu cầu người dùng mô tả lại; có thể cải tiến bằng cách tự động bỏ filter
  gần nhất được thêm vào và thử lại.
