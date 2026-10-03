# Hồ sơ bài nộp Day 17

**Nguyễn Vũ Anh — K4 — MSSV 2A202602502**

Repo: https://github.com/vuanh259/K4-DAY17-NguyenVuAnh-2A202602502

## Các yêu cầu đã thực hiện

- Baseline chỉ nhớ trong thread, không lưu profile.
- Advanced có User.md, short-term và compact; xử lý correction.
- Hai bộ dữ liệu gốc giữ nguyên; hai bảng đủ sáu chỉ số.
- Benchmark offline không cần key; số đo tái lập được.
- 35 test gồm bốn test bắt buộc, các kiểm tra live bằng transport giả và decay.
- [STEP8.md](STEP8.md) phân tích bốn câu hỏi với số liệu, đối chứng tắt compact và giới hạn.
- [BONUS.md](BONUS.md) có cả confidence threshold, structured extraction, conflict handling và memory decay, kèm kiểm chứng và đánh đổi.
- Sáu adapter provider; live dùng model để trả lời, trích profile và compact, không chuyển sang heuristic khi API lỗi.
- Requirements, hướng dẫn môi trường và workflow GitHub Actions cho Windows/Ubuntu.
- .env, API key, .venv và state không thuộc bài nộp.

## Chạy lại để chấm

```powershell
python -m pip install -r requirements.txt
python src/benchmark.py --mode offline
pytest src -v
python src/bonus_experiments.py
```

Trên máy sạch không có .env, `python src/benchmark.py` mặc định chạy offline như yêu cầu đề. Dùng `--mode offline` trên máy đang có .env live để chọn đúng chế độ.

## Trạng thái phần live mở rộng

Gemini đã được gọi thật nhưng dừng bởi quota free-tier 20 request/ngày. Trace và tình trạng lỗi có trong results; không có số liệu live hoàn chỉnh để so sánh hai agent. Đây là giới hạn tài khoản, không phải test pass hay bằng chứng đạt recall live. Benchmark offline là kết quả chính phù hợp yêu cầu bắt buộc của đề. Khi có quota, chạy `python src/benchmark.py --mode live --judge` rồi phân tích thêm các bảng mới.

Không bảo đảm điểm số; quyết định chấm thuộc giảng viên. Người học cần đọc code và phần phân tích để giải thích thiết kế, cách đo và đánh đổi.
