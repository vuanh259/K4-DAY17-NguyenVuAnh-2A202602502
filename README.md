# Lab 17 — Memory Systems for AI Agent

**Nguyễn Vũ Anh — 2A202602502 — K4**

So sánh hai kiến trúc memory trên **cùng một model**: Baseline chỉ giữ lịch sử trong thread; Advanced thêm User.md và compact. Hỗ trợ sáu provider để lựa chọn: openai, custom, gemini, anthropic, ollama, openrouter. Không tự chạy sáu provider cùng lúc.

## Số liệu hiện tại đến từ đâu?

Các bảng đã lưu tại [results/benchmark.md](results/benchmark.md), [results/benchmark.json](results/benchmark.json) và bài [STEP8.md](STEP8.md) là **mô phỏng OFFLINE**, không phải kết quả gọi model thật. Chương trình tính chúng từ hai file JSON gốc trong data/, dùng câu trả lời theo quy tắc và token ước lượng ký tự. Các số này không được gán cứng trong code benchmark, nhưng logic offline bị giới hạn bởi regex và mẫu trả lời.

**Trạng thái live:** Đã gọi Gemini `gemini-2.5-flash` thực tế, hoàn thành 16 lời gọi trả lời và 1 lời gọi judge trước khi gặp HTTP 429 (quota 20 request/ngày). Chưa có hai bảng live hoàn chỉnh; xem [trạng thái](results/live_status.json) và trace. Các provider khác mới được smoke-test khởi tạo. Test bằng transport giả chỉ kiểm chứng code, không thay thế đánh giá model thật.

**Bài nộp:** Các yêu cầu offline của đề đã được hoàn thiện; xem [SUBMISSION.md](SUBMISSION.md). Có đủ bốn hướng [bonus và bằng chứng](BONUS.md). Bộ kiểm thử hiện có 35 test.

## Chạy với model thật — cần cấu hình .env

Tại root repo, dùng Python >= 3.11:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-live.txt
Copy-Item .env.example .env
~~~

Nếu .env đã tồn tại, sửa trực tiếp file đó, không chép đè. Điền:

~~~dotenv
LLM_LIVE=1
LLM_PROVIDER=gemini
LLM_MODEL=<ID model thực có trong tài khoản của bạn>
LLM_API_KEY=<key của bạn>
~~~

Provider trong ví dụ có thể đổi. Model ID không được gán mặc định sang một model khác; thiếu cấu hình thì báo lỗi. Với Ollama chọn LLM_PROVIDER=ollama, đặt model đã tải trên máy, LLM_BASE_URL nếu dùng server khác mặc định, không cần API key. Với custom cần thêm LLM_BASE_URL; key tùy server. Với provider còn lại có thể dùng key theo tên OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY thay LLM_API_KEY.

Chạy thử riêng stress suite trước:

~~~powershell
.\.venv\Scripts\python.exe src/benchmark.py --mode live --suite stress
~~~

Chạy cả hai suite:

~~~powershell
.\.venv\Scripts\python.exe src/benchmark.py --mode live
~~~

Bật model judge nếu muốn Response quality cũng được model đánh giá:

~~~powershell
.\.venv\Scripts\python.exe src/benchmark.py --mode live --judge
~~~

Mặc định judge dùng cùng model/provider/key. Có thể cấu hình riêng JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_TEMPERATURE. Không có --judge thì Response quality vẫn là heuristic đối chiếu chuỗi, kể cả khi câu trả lời do model thật sinh.

## Live khác offline thế nào?

| Thành phần | Offline | Live |
|---|---|---|
| Trả lời | Mẫu câu dùng chung | Model sinh câu trả lời |
| Trích xuất profile | Regex tiếng Việt | Model trả JSON facts, evidence và confidence |
| Khóa profile | Các trường định sẵn trong extractor | Model chọn khóa theo nội dung, tái sử dụng khóa khi correction |
| Compact | Heuristic facts + đoạn trích | Model tóm tắt ngữ cảnh cũ và summary trước |
| Token | Ước lượng ký tự | Usage provider; nếu thiếu thì ghi rõ character_estimate |
| Quality | Đối chiếu chuỗi | Đối chiếu chuỗi hoặc model judge với --judge |

Nhánh live **không gọi regex extractor, mẫu trả lời offline hoặc heuristic summarizer**. Không có tên, nơi ở, nghề nghiệp, sở thích từ dataset trong prompt live. Đáp án expected_contains chỉ đi vào evaluator/judge, không đi vào agent. Khi API lỗi hoặc JSON sai, benchmark dừng và giữ trace đã ghi, không âm thầm chuyển offline.

Bộ lọc memory chỉ chấp nhận evidence thực sự xuất hiện trong message, confidence đạt ngưỡng và schema hợp lệ. Confidence là tự đánh giá của model, chưa được hiệu chỉnh thống kê; evidence khớp chuỗi không bảo đảm model hiểu đúng câu hỏi/câu đùa. Cần đánh giá live thực tế.

## Kết quả, trace và chi phí

Mỗi lần chạy tạo file riêng, mặc định:
- results/benchmark_live_<run-id>.json hoặc benchmark_offline_<run-id>.json.
- File cùng tên đuôi .trace.jsonl: lời gọi model, prompt/response, usage và điểm recall; offline chỉ ghi đánh giá recall.

JSON gồm metadata (mode, provider, model, cấu hình, hash dataset), results (hai bảng), usage (chi phí theo mục đích). Không lưu key trong metadata/trace. Trace có chứa nội dung hội thoại.

Sáu cột chính giữ theo đề. Hai cột token chỉ tính **lời gọi trả lời của agent**. Muốn tính tổng sử dụng API, cộng các mục answer, profile_extract, compact và judge trong usage. Không chỉ nhìn mức giảm prompt trong bảng để kết luận tổng chi phí tiền giảm: Advanced còn gọi model để trích xuất memory ở mỗi lượt và tóm tắt khi compact.

Ngưỡng compact vẫn dùng estimator ký tự để quyết định lúc nén; số đo token live dùng usage trả về nếu provider có. Summary được giới hạn theo ký tự; có thể mất chi tiết. Temperature 0 không bảo đảm API cho kết quả giống tuyệt đối mỗi lần.

Đặt --output đường dẫn JSON mới để tự đặt tên. Chương trình từ chối ghi đè output/trace cũ. Dữ liệu memory benchmark dùng thư mục tạm độc lập, không cần xóa hồ sơ người dùng.

## Offline theo yêu cầu lab và test

Không cần .env/API key:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/benchmark.py --mode offline
.\.venv\Scripts\python.exe src/benchmark.py --mode offline --no-compact
.\.venv\Scripts\python.exe -m pytest src/test_agents.py src/test_live.py -v
~~~

Có thể kích hoạt .venv rồi dùng python/pytest trực tiếp. Lệnh python src/benchmark.py lấy LLM_LIVE từ .env; nếu không có, chạy offline. Cờ --mode luôn ưu tiên hơn .env. Bốn test bắt buộc và test offline vẫn ở src/test_agents.py; test_live.py kiểm tra hợp đồng tích hợp bằng model giả.

## Cấu hình memory

| Biến | Mặc định |
|---|---:|
| COMPACT_THRESHOLD_TOKENS | 1200 |
| COMPACT_KEEP_MESSAGES | 4 message |
| MEMORY_MIN_CONFIDENCE | 0.85 |
| SUMMARY_BUDGET_TOKENS | 300 token ước lượng |
| LLM_TEMPERATURE | 0 |
| MEMORY_HALF_LIFE_TURNS | 0 (tắt decay trong benchmark chính) |
| MEMORY_MIN_WEIGHT | 0.125 |

.env được gitignore; .env.example không chứa bí mật. requirements-lock.txt ghi môi trường Python 3.12.10/Windows đã cài.

## Source và bài nộp

- src/model_provider.py, config.py: adapter và cấu hình.
- src/live_runtime.py: gọi model, extraction, summary, usage, trace.
- src/memory_store.py: profile, cập nhật fact, quản lý compact; giữ heuristic cho offline.
- src/agent_baseline.py, agent_advanced.py: hai kiến trúc memory.
- src/benchmark.py: chạy suite, chấm điểm, lưu số đo.
- data/: hai dataset gốc, không sửa.
- STEP8.md: phân tích các số đo offline cũ; bổ sung kết quả live sau khi chạy thực tế.

Live dùng LangChain chat model cùng lớp memory thủ công, chưa có graph/tool-calling. GitHub Actions được cấu hình kiểm thử trên Windows và Ubuntu; trạng thái CI cần xem trực tiếp tại tab Actions. Nộp link repo trên VLearn sau khi kiểm tra bài.
