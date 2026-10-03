"""Shared answer policy keeps the comparison about memory, not model skill."""
import re


def respond(message: str, facts: dict[str, str]) -> str:
    low = message.casefold()
    query = ("?" in message or any(s in low for s in
             ("nhắc lại", "nhớ lại xem", "tóm tắt ngắn về mình", "mô tả ngắn gọn mình")))
    if not query:
        return "Mình đã tiếp nhận thông tin."
    keys = []
    mappings = {
        "name": ("tên", "là ai"),
        "location": ("ở đâu", "nơi ở", "ở huế", "hiện đang ở"),
        "profession": ("nghề", "làm gì"),
        "drink": ("đồ uống",),
        "food": ("món ăn",),
        "pet": ("nuôi", "thú cưng"),
        "interests": ("quan tâm", "kỹ thuật chính"),
    }
    for key, words in mappings.items():
        if any(word in low for word in words):
            keys.append(key)
    if any(word in low for word in ("style", "kiểu trả lời", "thích kiểu", "thích trả lời")):
        keys += [key for key in facts if key.startswith("style_")]
    values = [facts[key] for key in keys if key in facts]
    if not values:
        return "Mình chưa có thông tin này trong bộ nhớ hiện có."
    if "3 bullet" == facts.get("style_format"):
        groups = ["; ".join(values[i::3]) for i in range(3)]
        return "\n".join("- " + (group or "Chỉ dùng thông tin đã được cung cấp.") for group in groups)
    return "Thông tin mình nhớ: " + "; ".join(values) + "."
