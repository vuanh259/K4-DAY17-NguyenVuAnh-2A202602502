# Bốn hướng bonus và bằng chứng

## 1. Confidence threshold

Live extractor trả về key/value, evidence và confidence. Chỉ nhận fact khi evidence là chuỗi có thật trong lượt user và confidence >= `MEMORY_MIN_CONFIDENCE` (mặc định 0,85). Câu hỏi, giả định, câu đùa và tin tức tạm thời được yêu cầu loại bỏ trong prompt extraction.

Test `test_extractor_rejects_unsupported_evidence_and_low_confidence` chứng minh pipeline loại cập nhật thiếu bằng chứng và cập nhật confidence thấp. Đây là kiểm thử hợp đồng với transport giả, không phải khẳng định model thật được hiệu chỉnh tốt. Lợi ích dự kiến là giảm memory poisoning; chưa có đo A/B live về tỷ lệ recall của riêng bonus này.

Rủi ro: confidence do model tự báo có thể sai; bằng chứng khớp chuỗi không bảo đảm cách diễn giải đúng. Ngưỡng quá cao làm mất fact hợp lệ. Mỗi lượt trích xuất là một lời gọi API bổ sung, được tính riêng trong usage.

## 2. Entity extraction có cấu trúc

Model lựa chọn trường theo nội dung, không giới hạn vào danh sách tên/thành phố/nghề của dataset. Runtime kiểm tra schema; User.md lưu một dòng cho mỗi khóa. Test sử dụng trường `collection` và thông tin tiếng Anh về bản đồ cổ để xác nhận nhánh live không phụ thuộc regex tiếng Việt hay mẫu đáp án offline.

Lợi ích: facts có thể cập nhật riêng và profile không cần mang toàn bộ transcript. Rủi ro: model có thể tạo hai khóa khác nhau cho cùng một khái niệm; prompt yêu cầu tái sử dụng khóa nhưng chưa có ontology hay canonicalization ngữ nghĩa bảo đảm tuyệt đối.

## 3. Conflict handling

`upsert_fact()` thay giá trị cũ qua `edit_text()`, thay vì nối lịch sử mâu thuẫn. Bài test correction kiểm tra backend → MLOps và Huế → Đà Nẵng, đồng thời bỏ nhiễu nơi đi họp và câu đùa. Thí nghiệm synthetic trong `results/bonus_policies.json` xác nhận giá trị cũ biến mất sau correction và fact mới có hiệu lực.

Lợi ích: profile không tích lũy hai giá trị đối lập cho cùng khóa, giảm nguy cơ trả lời theo bản cũ. Rủi ro: “mới nhất thắng” sẽ sai nếu khai báo mới sai; cập nhật ở hai khóa đồng nghĩa khác nhau vẫn cần model nhận diện. Chưa có version history hoặc xác nhận thủ công cho fact quan trọng.

## 4. Memory decay

`DecayingProfileStore` giữ tuổi fact theo số lượt riêng của từng user trong `freshness.json`. Trọng số `2 ** (-age / half_life_turns)`. Fact dưới `MEMORY_MIN_WEIGHT` bị ẩn khỏi persistent context đưa vào câu trả lời, vẫn còn trên đĩa. Khi user khẳng định lại fact, tuổi trở về 0; correction cũng làm mới tuổi. Chỉ đọc/nhắc lại câu hỏi không làm mới fact.

Mặc định `MEMORY_HALF_LIFE_TURNS=0` tắt decay cho benchmark chính, để đo độc lập persistent memory và compact. Có thể bật, ví dụ:

```dotenv
MEMORY_HALF_LIFE_TURNS=64
MEMORY_MIN_WEIGHT=0.125
```

Thí nghiệm độc lập, không gọi API:

```powershell
python src/bonus_experiments.py
```

Kết quả với half-life 2 lượt, ngưỡng 0,5, sau 3 lượt và xác nhận lại một fact:

| Chỉ số | Kết quả |
|---|---:|
| Facts trên đĩa | 2 |
| Facts đưa vào prompt | 1 |
| Token profile đầy đủ (ước lượng) | 39 |
| Token profile sau lọc (ước lượng) | 18 |
| Giảm token trong context profile | 53,85% |
| Dung lượng hồ sơ + metadata | 223 byte |

Các test kiểm tra: fact cũ bị ẩn nhưng không xóa, xác nhận lại khôi phục fact, restart vẫn giữ tuổi và hai user được cách ly. Test tích hợp agent cho thấy tên không còn được recall trong thread mới khi đã qua ngưỡng decay.

Đánh đổi được quan sát: prompt nhỏ hơn nhưng mất khả năng recall fact cũ nếu user không nhắc lại; sidecar metadata tăng dung lượng đĩa. Vì vậy không bật decay mặc định chỉ để làm bảng benchmark “đẹp”. Tuổi tính theo lượt có thể phạt người chat nhiều; đây không phải thời gian thực. Trong thread hiện tại, recent messages/summary vẫn có thể chứa fact cũ. Live extractor vẫn nhận toàn bộ profile để nhận ra correction; giảm profile trong prompt trả lời không đồng nghĩa giảm toàn bộ API usage.

## Giới hạn bằng chứng live

Lần chạy Gemini thực đã bắt đầu và ghi trace, nhưng API dừng với HTTP 429 do quota free-tier 20 request/ngày. Chưa có hai bảng live hoàn chỉnh; không thay số đo thiếu bằng số offline. Các bonus đã được kiểm thử code và thí nghiệm synthetic, nhưng chưa có đánh giá chất lượng live đầy đủ cho tất cả cơ chế.
