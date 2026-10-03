# Nguồn và phạm vi kết quả

- `benchmark.json`, `benchmark.md`, `ablation_no_compact.json`: số đo **offline** ban đầu, dùng heuristic và token ước lượng; không gọi model.
- `tests.txt`: log 23 test ban đầu.
- `offline_runtime_verified.json`: chạy lại offline sau khi thêm model-driven runtime; có metadata và usage phân biệt rõ.
- `offline_runtime_verified.trace.jsonl`: câu hỏi, câu trả lời và điểm recall của lần chạy lại.

`live_submission.trace.jsonl` là trace Gemini thực đã chạy một phần; `live_status.json` ghi lý do dừng HTTP 429, chưa đủ bảng so sánh. `bonus_policies.json` là thí nghiệm synthetic cho decay/correction, không gọi API. `tests_final.txt` ghi bộ test cuối.

Test trong `test_live.py` dùng transport giả, chỉ kiểm tra tích hợp code. Model API thực cần `.env`, quota đủ và lệnh `python src/benchmark.py --mode live`. Mỗi lần chạy có ID và file riêng; không ghi đè số đo cũ. Metadata không chứa API key, nhưng trace chứa nội dung hội thoại.
