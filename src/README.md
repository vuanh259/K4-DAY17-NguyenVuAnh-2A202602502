# Source đã hoàn thiện

Chạy từ root repo:

```
python src/benchmark.py
pytest src/test_agents.py -v
```

Các module dùng import phẳng theo scaffold. Không chạy bằng `python -m src.benchmark`.

Thứ tự phụ thuộc: model_provider → config → memory_store → hai agent → benchmark.
Hai agent cùng dùng offline_response, khác nhau ở bộ nhớ được cung cấp.
Chi tiết cấu hình và kết quả tại README.md và STEP8.md ở root.
