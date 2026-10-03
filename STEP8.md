# Bước 8 — Phân tích Memory Systems for AI Agent

**Nguyễn Vũ Anh — K4 — 2A202602502**
Ngày thực hiện: 03/10/2026. Python 3.12.10, Windows, chế độ offline.

> **Phạm vi số liệu:** Tất cả bảng bên dưới là mô phỏng offline, chưa phải kết quả model thật. Đã thử chạy Gemini thực với judge, nhưng API dừng sau 16 lời gọi trả lời và 1 lời gọi judge vì quota 20 request/ngày (HTTP 429); xem `results/live_status.json`. Không dùng các tỷ lệ offline này làm kết luận về một model cụ thể. Đề gốc cho phép benchmark offline; phần live đầy đủ cần quota bổ sung.

## Thiết kế phép đo

Giữ nguyên hai file trong `data/`: Standard gồm 10 conversation/101 lượt chat và 14 câu recall; Stress gồm 1 conversation/16 lượt chat và 3 câu recall. Mỗi câu recall được hỏi trong một thread mới riêng. Advanced giữ cùng hồ sơ của user qua các conversation trong cùng suite; mỗi suite và mỗi lần chạy có trạng thái ban đầu sạch.

Hai agent dùng chung `extract_profile_updates()` và `respond()`: Baseline chỉ trích facts từ user messages của thread hiện tại; Advanced dùng thêm hồ sơ trên đĩa và summary. Không truyền `expected_contains` vào agent. Do đó khác biệt đến từ dữ liệu memory được truy cập, không phải cung cấp đáp án cho Advanced.

Cách tính sáu chỉ số:

- **Agent tokens only**: cộng `ceil(len(text.strip()) / 4)` của mọi câu trả lời assistant, gồm cả chat và recall.
- **Prompt tokens processed**: cộng ước lượng ngữ cảnh trước từng câu trả lời. Baseline tính toàn bộ history, gồm user message hiện tại. Advanced tính đủ User.md + summary + recent messages sau compact, trước khi sinh câu trả lời. Không tính trùng message hiện tại.
- **Cross-session recall**: mỗi câu được 1 khi đủ mọi chuỗi kỳ vọng, 0,5 khi đúng một phần, 0 khi không có; lấy trung bình mọi câu, không lấy trung bình conversation.
- **Response quality**: tỷ lệ chuỗi kỳ vọng tìm thấy, lấy trung bình. Đây chỉ là proxy độ đầy đủ câu recall; không đánh giá tự nhiên, đúng kiến thức, suy luận hoặc nội dung các đoạn tin tức.
- **Memory growth (bytes)**: tổng kích thước hồ sơ cuối trừ đầu của các user duy nhất. Không tính history/summary trong RAM, dung lượng directory hay tổng byte đã ghi lại nhiều lần.
- **Compactions**: tổng bộ đếm compact của các thread chat và recall.

Token là ước lượng ký tự, không phải tokenizer/billing của provider. Trích xuất và summary offline không gọi LLM, nên chi phí CPU/I/O không được biểu diễn bằng token. Số đo live cũng chưa phải hóa đơn API.

## Kết quả thực nghiệm

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 901 | 11850 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 915 | 19296 | 1.000 | 1.000 | 324 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 174 | 21444 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 205 | 13115 | 1.000 | 1.000 | 265 | 2 |

Nguồn: [benchmark.json](results/benchmark.json). Test chạy hai lần suite với trạng thái mới và xác nhận toàn bộ hàng kết quả bằng nhau.

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Advanced đạt **1,0**, Baseline **0,0** ở cả hai suite. Baseline không ghi file (0 byte) và khóa history theo thread; sang thread recall, không còn tên, nghề nghiệp hay sở thích cũ. Test còn kiểm tra Baseline vẫn nhớ tên trong cùng thread, nên điểm 0 xuyên phiên không phải do cố tình làm hỏng khả năng nhớ trong phiên.

Advanced đưa fact khai báo vào User.md; agent mới có thể đọc lại file ngay cả khi instance cũ đã mất. Các test xác nhận việc khởi tạo lại agent vẫn nhớ tên và user khác không đọc được hồ sơ đó. Cập nhật nơi ở và nghề nghiệp thay giá trị cũ trước khi trả lời, nên stress recall không bị nhiễu bởi Huế, Hà Nội hay product manager.

Giới hạn: recall 100% chỉ đúng trên 17 câu hỏi của hai dataset này. Regex phục vụ phạm vi tiếng Việt của lab và có thể bỏ sót cách nói khác. Không suy rộng thành trí nhớ tổng quát của LLM.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trong Standard, prompt load tăng **11.850 → 19.296** (tăng **62,84%**). Advanced nạp profile vào mỗi lượt dù các thread chưa vượt ngưỡng 1.200 token; Compactions bằng 0. Lợi ích nhớ xuyên phiên đổi lấy ngữ cảnh bổ sung và thao tác đọc/ghi file.

Agent output chỉ tăng **901 → 915** (14 token, khoảng 1,55%). Cột này chỉ đo câu trả lời, không gồm profile. Không thể giải thích mức tăng Agent tokens only bằng việc “mang profile vào prompt”; profile thuộc Prompt tokens processed. Hai agent dùng cùng mẫu trả lời; Advanced có thêm thông tin cho một số câu recall, còn độ dài câu “chưa có thông tin” cũng được tính ở Baseline.

## 3. Vì sao compact giúp ở hội thoại dài?

Dù Stress chỉ có 16 lượt chat, Baseline xử lý **21.444** prompt token, nhiều hơn cả Standard với 101 lượt ngắn. Mỗi turn dài bị nạp lại qua nhiều lượt. Với độ dài mỗi lượt gần cố định, prompt riêng một lượt tăng gần tuyến tính theo chiều dài history, còn tổng cộng dồn qua n lượt tăng gần bậc hai.

Advanced compact **2 lần**, thay message cũ bằng summary và giữ 4 message mới nhất; prompt load còn **13.115**, giảm **8.329 token, tương đương 38,84%** so với Baseline.

Đối chứng `--no-compact`:

| Stress | Prompt tokens processed | Agent tokens only | Recall | Compactions |
|---|---:|---:|---:|---:|
| Baseline | 21444 | 174 | 0.0 | 0 |
| Advanced tắt compact | 22493 | 205 | 1.0 | 0 |
| Advanced có compact | 13115 | 205 | 1.0 | 2 |

Nguồn: [ablation_no_compact.json](results/ablation_no_compact.json). Tắt compact khiến Advanced vượt Baseline về prompt load vì vẫn phải nạp thêm hồ sơ. Bật compact giảm **9.378 token (41,69%)** so với chính Advanced tắt compact; output vẫn **205 token** và recall vẫn **1,0**. Đây là bằng chứng trực tiếp rằng compact tối ưu **Prompt tokens processed**, không trực tiếp giảm **Agent tokens only**.

Summary heuristic giữ facts khai báo và tối đa 6 trích đoạn ngắn, được gộp với summary cũ thay vì nối mãi. Đây là nén mất mát: có thể mất chi tiết, quan hệ giữa các chủ đề hoặc số liệu trong tin tức. Bộ recall hiện chỉ kiểm tra profile xuyên phiên, chưa chứng minh giữ tốt abstraction của bốn chủ đề news. Test xác nhận summary giữ fact qua nhiều lần compact nhưng không thay thế đánh giá semantic bằng con người.

Ngưỡng compact là ngưỡng kích hoạt, không phải hard cap. Một message rất dài nằm trong 4 message mới nhất vẫn được giữ nguyên; summary và phần recent có thể vượt ngưỡng sau khi nén.

## 4. Memory tăng trưởng ra sao, có rủi ro gì?

Baseline tăng **0 byte**. Advanced tăng **324 byte** ở Standard và **265 byte** ở Stress. Hai suite độc lập và có tập trường được trích xuất khác nhau, nên Stress dài hơn không đồng nghĩa User.md lớn hơn. Toàn bộ đoạn news không được đổ vào profile.

Profile được upsert theo trường: cập nhật location/profession không tạo thêm dòng lịch sử. Trong triển khai này số trường được giới hạn bởi bộ extractor; file không tăng theo từng lượt như một bản transcript. Tuy vậy giá trị dài, mở rộng schema hoặc lưu thêm sở thích có thể tăng dung lượng. Chưa có cơ chế decay, TTL hay giới hạn tổng byte cho profile.

Rủi ro còn lại gồm trích sai một khai báo, bỏ sót correction không khớp regex, mất thông tin khi compact và race condition nếu nhiều tiến trình cùng sửa một hồ sơ. Ghi file hiện là đồng bộ đơn tiến trình, chưa có transaction/lock. ID được băm và đường dẫn được kiểm tra để ngăn traversal/va chạm trên Windows; đây chưa phải cơ chế xác thực user cho sản phẩm thật.

## Bonus — Structured extraction và conflict handling

Các fact được tách theo khóa name/location/profession/drink/food/pet/interests và các thành phần style. Khi có correction, `upsert_fact()` dùng `edit_text()` thay giá trị trước, không giữ đồng thời nghề cũ và nghề mới. Ví dụ backend engineer → MLOps engineer và Huế → Đà Nẵng được kiểm chứng trong test.

Các câu hỏi, yêu cầu nhắc lại, giả định và câu đùa bị loại khỏi đầu vào ghi profile. Điều này ngăn một câu như “Mình đang ở Hà Nội phải không?” bị lưu thành nơi ở mới. Test bao gồm câu hỏi không có dấu hỏi như “Nhắc lại giúp mình tên và style trả lời mình thích…”.

Lợi ích quan sát được là recall 1,0 trên dataset có correction/nhiễu và mỗi khóa chỉ giữ một giá trị; tránh tích lũy các phiên bản mâu thuẫn trong prompt. Không tuyên bố một tỷ lệ cải thiện riêng cho bonus vì chưa có full benchmark ablation tắt riêng conflict handling.

Đánh đổi: quy tắc thận trọng có thể bỏ qua thông tin thật trong câu pha trộn hỏi/khai báo hoặc lịch sử/correction; “giá trị mới nhất thắng” cũng có thể nhận một khai báo sai. Các regex và quy tắc về phạm vi câu làm code phức tạp hơn. Nhánh live bổ sung confidence/evidence filtering và schema validation; confidence chưa được hiệu chỉnh thống kê.

Phiên bản cuối bổ sung **memory decay tùy chọn**. Với thí nghiệm riêng, profile context giảm từ 39 xuống 18 token ước lượng (53,85%), giữ lại fact vừa xác nhận và ẩn fact cũ mà không xóa trên đĩa. Test chứng minh đánh đổi là mất recall fact đã quá tuổi. Decay tắt trong hai bảng chính để không trộn hiệu ứng với compact. Chi tiết bốn bonus, giới hạn và lệnh tái lập ở [BONUS.md](BONUS.md), số đo ở [bonus_policies.json](results/bonus_policies.json).

## Cấu hình và tái lập

Mặc định `COMPACT_THRESHOLD_TOKENS=1200`, `COMPACT_KEEP_MESSAGES=4`, estimator `ceil(len(strip(text))/4)`; nội dung User.md ghi UTF-8 với LF để byte count ổn định giữa Windows/Linux.

Tên biến model: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_API_KEY`, `LLM_BASE_URL`. Các fallback key/URL theo provider và nhóm `JUDGE_*` được liệt kê tại [README.md](README.md). Các số liệu trong tài liệu này được đo offline. Phiên bản hiện tại cho phép `--mode live` và `--judge`; `LLM_LIVE=1` đặt mặc định live nếu không truyền cờ `--mode`.

```powershell
python -m pip install -r requirements.txt
python src/benchmark.py --mode offline
python src/benchmark.py --mode offline --no-compact
pytest src/test_agents.py -v
```

Kết quả ban đầu: **23 passed** trong `test_agents.py`. Bộ kiểm thử cuối gồm **35 passed** khi chạy `pytest src -v`, bổ sung hợp đồng live, cách ly đáp án benchmark và memory decay. Adapter sáu provider đã khởi tạo được; Gemini đã có lời gọi thực nhưng full live benchmark bị chặn bởi quota. Không tuyên bố chất lượng live dựa trên các test dùng transport giả.
