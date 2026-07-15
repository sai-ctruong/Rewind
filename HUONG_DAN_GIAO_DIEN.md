# 🖥️ Hướng dẫn sử dụng giao diện Rewind

Tài liệu này hướng dẫn dùng **trang web** của Rewind — không cần viết code, chỉ gõ và
bấm chuột. (Muốn dùng bằng lệnh/Python xem [`HUONG_DAN.md`](HUONG_DAN.md).)

---

## 1. Mở giao diện

```powershell
python -m ui.app
```

Mở trình duyệt vào **http://127.0.0.1:5000**.

Góc trên có:
- **Thanh tab**: `KISC` (bộ lọc ảnh) · `KIS` · `AVS` · `VQA` · `🎥 Video` · `⏱️ Chuỗi` · `🧠 Agent`.
- **◐** — nút đổi giao diện sáng/tối.
- **Đèn trạng thái** — báo hệ thống sẵn sàng / đang xử lý.

> 💡 **Nên bắt đầu ở tab `🎥 Video`** — đây là tab đầy đủ nhất (tìm chữ, ảnh, sketch,
> phản hồi, duyệt lân cận). Các tab khác là phiên bản rút gọn cho từng kiểu bài toán.
> Riêng tab **`KISC`** giờ là **bộ lọc ảnh** (mục 7) — hay nhất khi bạn khó mô tả đủ
> trong một câu: cứ nói đại khái rồi thu hẹp dần bằng ảnh.

---

## 2. Thanh video chung: **nạp một lần, mọi tab dùng**

Ngay dưới thanh tab là **thanh video chung** — chỗ **duy nhất** để chọn và nạp video.
Nạp xong, **cả 7 tab dùng được ngay**, không phải chọn/nạp lại ở từng tab.

| Điều khiển | Làm gì |
|---|---|
| **Ô chọn video** | Liệt kê video trong `data/videos/`; cái nào đã nạp có dấu **✓ đã nạp** |
| **Nạp video** | Nạp video đang chọn → cắt keyframe + tạo chỉ mục |
| **Ô đường dẫn + Nạp thư mục** | Hoặc dán đường dẫn thư mục bất kỳ (VD `D:\my_videos`) → nạp cả kho |
| **💾 Lưu index** | Lưu ra đĩa → lần sau mở app **tự nạp lại**, khỏi embed lại |

Đổi video ở thanh này → mọi tab tự cập nhật theo (phiên lọc ảnh / trí nhớ Agent của video
cũ được xoá, vì chúng không còn ý nghĩa với video mới).

> 📁 Muốn video hiện trong ô chọn: đặt file vào **`data/videos/`**. Mở app lần sau, video
> đã lưu index được **tự chọn sẵn** — dùng được ngay.

Khi nạp, thanh trạng thái hiện **tiến trình** (số keyframe · fps · thời gian). Video dài
lần đầu sẽ lâu (tải model + cắt frame).

> Hệ **luôn quét toàn bộ video** (tự lấy mẫu keyframe) — không cần chọn giây/số frame.

### Ba công tắc **lúc nạp** (ở thanh chung)

Bật **trước khi bấm Nạp video** để làm giàu dữ liệu:

| Công tắc | Ý nghĩa | Lưu ý tốc độ |
|---|---|---|
| **OCR** | Đọc **chữ trên khung hình** (biển hiệu, phụ đề cứng) → tìm được theo text | Chậm ~5× khi nạp |
| **Caption** | VLM **mô tả ngữ cảnh** mỗi cảnh → tìm theo **quan hệ/hoàn cảnh**; **VQA cần cái này** để trả lời | Rất chậm (~vài giây/cảnh) — chỉ dùng video ngắn |
| **ASR** | Chép **lời nói** (Whisper) → tìm theo điều ai đó nói | Chậm, cần `openai-whisper` |

> Mặc định cả ba **tắt** để nạp nhanh. Bật khi thật sự cần kiểu tìm tương ứng.

### Công tắc **lúc tìm** (ở từng tab)

**Rerank VLM** nằm trong tab KIS/AVS/Video — vì nó ảnh hưởng lúc *tìm*, không phải lúc
*nạp*: dùng VLM chấm lại top kết quả → chính xác hơn nhưng chậm hơn mỗi lần tìm.

---

## 3. Tab `🎥 Video` — tìm kiếm đầy đủ

Đây là "phòng làm việc" chính. Sau khi nạp video:

### 3.1 Tìm bằng chữ
1. Gõ mô tả vào ô (VD: *"a person walking on the street"* hoặc *"người đi bộ trên phố"*).
2. Bấm **Tìm trong video** → lưới kết quả hiện ra, xếp theo độ khớp, kèm **video + giây**.
3. Muốn chính xác hơn cho vị trí Top-1 → tick **Rerank VLM** rồi tìm lại.

### 3.2 Tìm bằng ảnh mẫu (và ảnh + chữ)
1. Ở nhãn **🖼️ Ảnh mẫu**, chọn một ảnh từ máy.
2. Bấm **Tìm theo ảnh (+chữ)**.
   - Nếu **ô tìm để trống** → tìm khung hình *giống ảnh*.
   - Nếu **ô tìm có chữ** → kết hợp **ảnh + chữ** (multimodal), VD ảnh một con phố + chữ
     "vào ban đêm".

### 3.3 Tìm bằng phác hoạ (sketch)
1. Bấm **✏️ Phác hoạ** → hiện khung vẽ.
2. Vẽ bố cục/màu bạn hình dung; **🧹 Xoá** để làm lại.
3. Bấm **Tìm theo phác hoạ** → hệ tìm khung hình giống nét vẽ (kết hợp câu ở ô tìm nếu có).

### 3.4 Khám phá khi chưa biết tìm gì
Bấm **🧭 Khám phá** → hệ hiện một loạt keyframe **đa dạng** khắp video. Bấm vào một ảnh
bất kỳ để **tìm các cảnh tương tự** — cách hay để "lần" ra thứ cần tìm mà không phải nghĩ
câu mô tả.

### 3.5 Tinh chỉnh bằng phản hồi 👍 / 👎 (rất mạnh)
Trên mỗi kết quả có nút **👍** và **👎**:
1. Bấm 👍 vào ảnh **đúng ý**, 👎 vào ảnh **lạc đề**.
2. Thanh **🔁 Lọc lại theo phản hồi (👍n 👎m)** hiện số đã đánh dấu — bấm nó.
3. Hệ **kéo kết quả** về hướng ảnh bạn thích, tránh xa ảnh không thích (thuật toán
   Rocchio). Lặp vài vòng là ra đúng thứ cần.

### 3.6 Gợi ý từ khoá (concept)
Dưới ô tìm có các **chip từ khoá** rút ra từ kết quả hiện tại (VD: *đêm, mưa, taxi*).
Bấm một chip để **thêm vào truy vấn**, thu hẹp hoặc đổi hướng nhanh.

### 3.7 Xem lân cận theo timeline
Trên mỗi kết quả có **🎞 Lân cận** → xem các khung hình **ngay trước/sau** cảnh đó trong
cùng video, để định vị đúng bối cảnh (đoạn đó diễn ra thế nào).

### 3.8 Khi hệ *chủ động hỏi lại*
Nếu kết quả còn **mơ hồ** (nhiều cảnh na ná, không cái nào nổi trội), hệ tự hiện một nhóm
**vài ảnh đại diện khác nhau** và hỏi *"cái nào giống ý bạn nhất?"*. Bấm chọn → hệ dùng
lựa chọn đó để lọc tiếp. Đây là KISC áp cho video thật.

---

## 4. Tab `KIS` và `AVS` — bản gọn theo bài toán

Giống tab Video nhưng tối giản, dùng khi bạn biết rõ mình cần kiểu nào:

| Tab | Dùng khi | Nút tìm | Kết quả |
|---|---|---|---|
| **KIS** (Known-Item) | Cần tìm **một** khoảnh khắc cụ thể | **Tìm Top-5** | 5 ứng viên khả dĩ nhất |
| **AVS** (Ad-hoc) | Cần **mọi** cảnh khớp một mô tả chung | **Tìm tất cả** | Danh sách đầy đủ, xếp hạng |

Cả hai dùng video từ **thanh chung** và có công tắc **Rerank VLM** riêng (tuỳ chọn lúc tìm).

---

## 5. Tab `⏱️ Chuỗi` — tìm theo thứ tự thời gian

Dùng khi truy vấn là **chuỗi sự kiện có thứ tự** ("A xảy ra **trước** B").

1. Nhập **mỗi dòng một cảnh, theo đúng thứ tự**. Ví dụ:
   ```
   người cầm ô đỏ
   xe buýt hai tầng màu vàng
   ```
2. Bấm **Tìm chuỗi**.
3. Hệ chỉ trả về các đoạn **cùng video** có timestamp **tăng dần đúng thứ tự** — cảnh
   đúng nhưng sai thứ tự bị loại.

---

## 6. Tab `VQA` — hỏi–đáp trên video thật

Dùng để **hỏi một câu và nhận câu trả lời có suy luận**, không phải tìm ảnh.

1. Video lấy từ **thanh video chung** (mục 2) — nạp một lần là xong.
2. Gõ câu hỏi hoặc bấm **câu hỏi gợi ý** (VD: *"Có bao nhiêu người đi bộ?"*) → **Hỏi**.
3. Hệ **dùng chính câu hỏi để tìm cửa sổ keyframe liên quan**, rồi trả lời trên cửa sổ đó
   (thay vì đọc cả video — vừa chậm vừa loãng). Bên phải hiện **dải ảnh thật** đã dùng để
   suy luận, ảnh được dùng trực tiếp có viền nổi bật.

> ⚠️ **Điều kiện để trả lời được:** hệ cần "nhìn" hoặc "đọc" được nội dung.
> - **Không có `ANTHROPIC_API_KEY`** → chỉ suy luận trên **chữ** (caption/OCR/ASR). Video
>   nạp mà **không bật Caption** thì gần như không trả lời được — tab sẽ hiện cảnh báo này.
>   Cách khắc phục: tick **Caption** ở thanh video chung rồi nạp lại.
> - **Có API key** → dùng **Claude vision**, nhìn thẳng ảnh, không cần caption.

---

## 7. Tab `KISC` — 🎯 Bộ lọc ảnh (thu hẹp dần)

Dùng khi bạn **khó mô tả một câu cho đủ**. Thay vì đọc danh sách chữ, bạn **nhìn ảnh thật
và thu hẹp dần** cho tới khi còn đúng khoảnh khắc cần tìm.

1. Chọn + nạp video ở **thanh video chung** (mục 2) — chỉ một lần cho mọi tab.
2. Gõ mô tả thô (VD: *"người đi bộ trên phố"*) → **Gửi**.
   → Hiện **lưới ảnh keyframe thật** (VD 11 ảnh). Ô **Còn lại** đếm số ảnh.
3. Thu hẹp bằng **3 cách, dùng lẫn nhau tuỳ ý**:

| Cách | Thao tác | Kết quả |
|---|---|---|
| **Thêm chi tiết** | gõ tiếp *"xe cộ"* → Gửi | truy vấn cộng dồn, lưới co lại (11 → 6) |
| **👍 / 👎** | bấm nút dưới mỗi ảnh, rồi **🔁 Lọc theo phản hồi** | kéo về ảnh thích, tránh xa ảnh không thích (6 → 3) |
| **Chọn ảnh gần ý nhất** | khi hệ hỏi, bấm 1 ảnh trong panel trên cùng | thu hẹp quanh ảnh đó; các ảnh còn lại thành 👎 |

4. Lặp cho tới khi còn 1 ảnh — ô **Còn lại** chuyển xanh: *"Đã còn 1 — đúng khoảnh khắc"*.

Ô **20 → 8** cạnh tiêu đề cho thấy lưới vừa co bao nhiêu; **chip Truy vấn** hiện câu đã
cộng dồn; **thanh bên phải** ghi lại diễn tiến hội thoại.

> ↺ **Lọc lại từ đầu** — xoá phiên và bắt đầu lại. Cần dùng khi mô tả ban đầu sai hướng:
> bộ lọc chỉ thu hẹp (ảnh đã bị loại **không** quay lại), giống mọi bộ lọc.

> ⚠️ Tab này **cần backend thật** (`python -m ui.app`) vì phải lấy ảnh keyframe.

---

## 8. Tab `🧠 Agent` — để hệ **tự chọn cách tìm**

Các tab trên: *bạn* chọn công cụ (tìm chữ / ảnh / chuỗi…). Tab này: **gõ một câu, Agent tự
quyết**.

1. Video lấy từ **thanh video chung** (mục 2).
2. Gõ câu bất kỳ → **Hỏi Agent**. Nó tự hiểu và tự định tuyến:

| Bạn gõ | Agent tự làm |
|---|---|
| *"người đi bộ trên phố"* | `understand` → `search` → lưới ảnh |
| *"đi bộ **trước khi** xe chạy qua"* | `understand` → **`search_temporal`** → các chuỗi cảnh đúng thứ tự |
| bấm 👍/👎 rồi **🔁 Hỏi lại theo phản hồi** | **`search_with_feedback`** (Rocchio) → kéo về ảnh bạn thích |
| kết quả mơ hồ | Agent **chủ động hỏi lại**: chọn 1 ảnh gần ý nhất |

3. Panel bên phải cho thấy **"Agent đã làm gì"** — từng công cụ đã gọi kèm **lý do** và số
   kết quả. Đây là điểm khác biệt: bạn *thấy được* nó suy nghĩ, không phải hộp đen.
4. **Trí nhớ phiên** hiển thị số lượt · 👍/👎 đã tích luỹ · các câu gần đây — Agent **nhớ
   xuyên lượt**, nên càng trao đổi càng sát ý. **↺ Phiên mới** để quên hết.

Ô **Trả lời** là phần tổng hợp có **trích dẫn keyframe** (`[walking/7]`).

> Không cần API key: mặc định Agent dùng bộ não luật (tất định, chạy offline) — vẫn tự
> định tuyến đủ 4 nhánh trên. Có `ANTHROPIC_API_KEY` thì tự nâng lên **Claude**
> (function-calling) để suy luận linh hoạt hơn.

---

## 9. Vòng làm việc khuyến nghị

```
Nạp video MỘT LẦN ở thanh chung (💾 lưu index để tái dùng)
      │
      ▼
Tab 🎥 Video: gõ mô tả → Tìm trong video
      │
      ├─ Kết quả gần đúng?  → bấm 👍/👎 → 🔁 Lọc lại theo phản hồi (lặp)
      ├─ Bí ý tưởng?        → 🧭 Khám phá, bấm ảnh giống để lần ra
      ├─ Cần chính xác Top-1? → tick Rerank VLM, tìm lại
      ├─ Có ảnh mẫu?        → 🖼️ Ảnh mẫu → Tìm theo ảnh (+chữ)
      └─ Cần đúng bối cảnh?  → 🎞 Lân cận trên kết quả
```

---

## 9. Mẹo & lưu ý

- **Nạp một lần dùng mọi tab**: thanh video chung là chỗ duy nhất cần nạp.
- **Lưu index**: sau khi nạp video lớn, bấm **💾 Lưu index** — lần sau mở lại tức thì.
- **OCR chỉ bật khi cần tìm theo chữ** (biển hiệu). Bật bừa sẽ nạp chậm hơn và đôi khi
  *giảm* độ chính xác cho truy vấn thị giác.
- **Caption rất chậm với video dài** — dùng cho clip ngắn, hoặc cấu hình Claude (xem
  [`HUONG_DAN.md`](HUONG_DAN.md) mục 7) để nhanh và mạnh hơn.
- **Rerank VLM** chỉ nên bật khi vị trí Top-1 quan trọng — nó chậm hơn vì đọc kỹ từng ảnh.
- **Tiếng Việt hay tiếng Anh đều được** ở ô tìm.
- Console vỡ tiếng Việt trên Windows: chạy `set PYTHONUTF8=1` trước `python -m ui.app`.

---

<div align="center">

Giao diện gói gọn toàn bộ chức năng của Rewind. Cần thao tác nâng cao (tự động điều phối
bằng Agent, benchmark, cắm API thật) → xem [`HUONG_DAN.md`](HUONG_DAN.md).

</div>
