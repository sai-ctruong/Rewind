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
- **Thanh tab**: `KISC` · `KIS` · `AVS` · `VQA` · `🎥 Video` · `⏱️ Chuỗi`.
- **◐** — nút đổi giao diện sáng/tối.
- **Đèn trạng thái** — báo hệ thống sẵn sàng / đang xử lý.

> 💡 **Nên bắt đầu ở tab `🎥 Video`** — đây là tab đầy đủ nhất (tìm chữ, ảnh, sketch,
> phản hồi, duyệt lân cận). Các tab khác là phiên bản rút gọn cho từng kiểu bài toán.
> Riêng tab **`KISC`** giờ là **bộ lọc ảnh** (mục 7) — hay nhất khi bạn khó mô tả đủ
> trong một câu: cứ nói đại khái rồi thu hẹp dần bằng ảnh.

---

## 2. Bước chung cho mọi tab: Nạp video

Trước khi tìm, phải **nạp (index)** video ít nhất một lần. Ở đầu mỗi tab tìm kiếm có 3 nút:

| Điều khiển | Làm gì |
|---|---|
| **Ô chọn video** (dropdown) | Liệt kê các video hệ tìm thấy trong `data/videos/` — chọn một cái |
| **Nạp video** | Nạp video đang chọn ở dropdown → cắt keyframe + tạo chỉ mục |
| **Ô đường dẫn + Nạp thư mục** | Hoặc dán đường dẫn 1 thư mục bất kỳ (VD `D:\my_videos`) → nạp cả kho nhiều video |
| **💾 Lưu index** | Lưu chỉ mục ra đĩa để lần sau mở lại **không phải nạp lại** |

> 📁 Muốn video của bạn hiện trong dropdown: đặt file vào thư mục **`data/videos/`** của
> dự án. Hoặc dùng ô đường dẫn để trỏ tới thư mục bất kỳ trên máy.

Khi nạp, một **thanh tiến trình** hiện số keyframe đã xử lý. Video dài lần đầu sẽ lâu
(đang tải model + cắt frame). Nạp xong là có thể tìm.

> Hệ **luôn quét toàn bộ video** (tự lấy mẫu keyframe) — bạn không cần chọn giây/số frame.

### Bốn công tắc khi nạp (checkbox cạnh ô tìm)

Bật **trước khi nạp** để làm giàu dữ liệu tìm kiếm:

| Công tắc | Ý nghĩa | Lưu ý tốc độ |
|---|---|---|
| **OCR** | Đọc **chữ trên khung hình** (biển hiệu, phụ đề cứng) → tìm được theo text | Chậm ~5× khi nạp |
| **ASR** | Chép **lời nói** trong video (Whisper) → tìm theo điều ai đó nói | Chậm, cần `openai-whisper` |
| **Caption** | VLM **mô tả ngữ cảnh** mỗi cảnh → tìm theo **quan hệ/hoàn cảnh** ("người lớn hướng dẫn trẻ tưới hoa") | Rất chậm (~vài giây/cảnh) — chỉ dùng video ngắn |
| **Rerank VLM** | Khi *tìm*, dùng VLM chấm lại top kết quả → **chính xác hơn** | Chậm hơn lúc tìm (không ảnh hưởng lúc nạp) |

> Mặc định **OCR/ASR/Caption tắt** để nạp nhanh. Bật khi thật sự cần kiểu tìm tương ứng.

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

Cả hai đều có công tắc **Rerank VLM / OCR / Caption / ASR** như trên.

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

## 6. Tab `VQA` — hỏi–đáp trên video

Dùng để **hỏi một câu và nhận câu trả lời có suy luận**, không phải tìm ảnh. Tab này chạy
trên một **video demo sẵn có** ("video sinh nhật") để minh hoạ — không cần nạp video.

1. Gõ câu hỏi, hoặc bấm một **câu hỏi gợi ý** (VD: *"Có bao nhiêu ngọn nến trên bánh?"*).
2. Bấm **Hỏi**.
3. Nhận câu trả lời (đếm số lượng, xác định "ai làm gì"…). Bên phải hiện **cửa sổ keyframe**
   mà trợ lý đã đọc để suy luận.

---

## 7. Tab `KISC` — 🎯 Bộ lọc ảnh (thu hẹp dần)

Dùng khi bạn **khó mô tả một câu cho đủ**. Thay vì đọc danh sách chữ, bạn **nhìn ảnh thật
và thu hẹp dần** cho tới khi còn đúng khoảnh khắc cần tìm.

1. Chọn video ở dropdown → bấm **Nạp video** (dùng chung cache với tab 🎥 Video, đã nạp
   rồi thì tức thì).
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

## 8. Vòng làm việc khuyến nghị

```
Nạp video (💾 lưu index để tái dùng)
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

- **Lưu index**: sau khi nạp video lớn, bấm **💾 Lưu index** — lần sau mở lại tức thì,
  khỏi nạp lại.
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
