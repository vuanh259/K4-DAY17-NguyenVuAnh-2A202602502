"""Deterministic Vietnamese profile extraction and bounded conversation memory."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def estimate_tokens(text: str) -> int:
    text = (text or "").strip()
    return (len(text) + 3) // 4


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # Hash every ID: case-insensitive Windows paths cannot merge Alice/alice,
        # and punctuation/traversal or reserved device names cannot escape root.
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:40] or "user"
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        root = self.root_dir.resolve()
        path = (root / f"u_{slug}_{digest}" / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile path escapes root")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content or search_text == replacement:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_facts(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Invalid fact key")
        value = " ".join(value.split())
        previous = self.facts(user_id)
        if previous.get(key) == value:
            return
        if key in previous:
            self.edit_text(user_id, f"- {key}: {previous[key]}", f"- {key}: {value}")
        else:
            content = self.read_text(user_id) or "# User profile\n\n"
            self.write_text(user_id, content.rstrip() + f"\n- {key}: {value}\n")


def parse_facts(text: str) -> dict[str, str]:
    return dict(re.findall(r"^- ([a-z_]+): (.+)$", text, re.MULTILINE))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Only explicit declarations/preferences, never facts inferred from questions.

    Conservative rules are intentionally limited to this Vietnamese lab domain.
    Historical/hypothetical clauses are ignored; later declarations win.
    """
    updates: dict[str, str] = {}
    message = unicodedata.normalize("NFC", message)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", message):
        lower = sentence.casefold()
        if re.match(r"\s*(?:nhắc lại|tóm tắt|mô tả|sang thread mới)", lower):
            continue
        if "?" in sentence or re.search(r"\b(nếu|giả sử|đùa|hỏi lại|sẽ hỏi|lát nữa)\b", lower):
            continue
        for clause in re.split(r"\bnhưng\b|[;]", sentence, flags=re.I):
            low = clause.casefold()
            if any(x in low for x in ("lúc đầu", "trước đó", "ví dụ cũ", "không phải nơi ở")):
                continue
            patterns = {
                "name": r"(?:mình|tôi)\s+tên(?:\s+là)?\s+([^.,;:!?]+)",
                "location": r"(?:(?:mình|tôi)\s+(?:(?:vẫn|đang|hiện)\s+)*ở|hiện ở|nơi ở hiện tại là)\s+([^.,;:!?]+)",
                "profession": r"(?:(?:mình|tôi)\s+(?:vẫn\s+)?(?:đang\s+)?làm|đang làm|giờ chuyển sang|nghề nghiệp (?:hiện tại |thì )?vẫn (?:là )?)\s+([^.,;:!?]+)",
                "drink": r"đồ uống yêu thích(?: của mình)? là\s+([^.,;!?]+)",
                "food": r"món ăn yêu thích(?: của mình)? là\s+([^.,;!?]+)",
                "pet": r"(?:mình|tôi) nuôi\s+([^.,;!?]+)",
            }
            for key, pattern in patterns.items():
                for match in re.finditer(pattern, clause, re.I):
                    value = re.split(r"\s+(?:và|chứ|cho|dù|trong|mỗi ngày|vài tháng|để|nữa)\b",
                                     match.group(1), maxsplit=1, flags=re.I)[0].strip()
                    if key == "profession" and not re.search(r"engineer|developer|manager|giáo viên|bác sĩ|kỹ sư", value, re.I):
                        continue
                    if value and not re.search(r"\b(gì|đâu|không|việc)\b", value, re.I):
                        updates[key] = value
            # Working in another city is residence only with explicit relocation.
            if "correction" in low or "từ tuần này" in low:
                match = re.search(r"mình đang làm việc ở\s+(.+?)(?: vài tháng| để|[.,]|$)", clause, re.I)
                if match:
                    updates["location"] = match.group(1).strip()
            if re.search(r"mình (?:vẫn )?(?:thích|đang quan tâm|quan tâm)", low):
                interests = [term for term in ("Python", "AI", "MLOps", "RAG")
                             if re.search(rf"\b{term}\b", clause, re.I)]
                if interests:
                    updates["interests"] = ", ".join(interests)
            style_context = re.search(r"(?:mình|tôi).*(?:muốn|thích)|hãy trả lời|style trả lời|cách giải thích", low)
            if style_context and re.search(r"trả lời|giải thích|style|trình bày", low):
                if re.search(r"ngắn|gọn|bullet", low):
                    updates["style_length"] = "ngắn gọn"
                bullet = re.search(r"(\d+)\s+bullet", low)
                if bullet:
                    updates["style_format"] = f"{bullet.group(1)} bullet"
                elif "bullet" in low:
                    updates["style_format"] = "bullet"
                if "ví dụ" in low:
                    updates["style_example"] = "có ví dụ thực chiến" if "thực chiến" in low else "có ví dụ thực tế"
                if "trade-off" in low:
                    updates["style_focus"] = "ưu tiên so sánh trade-off"
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded, lossy summary: latest explicit facts + recent content excerpts.

    Previous summary facts are merged rather than recursively quoting the whole
    summary. News excerpts remain session memory, never persistent profile facts.
    """
    if max_items <= 0:
        return ""
    facts: dict[str, str] = {}
    excerpts: list[str] = []
    for message in messages:
        if message["role"] == "summary":
            facts.update(parse_facts(message["content"]))
            excerpts.extend(re.findall(r"^> (.+)$", message["content"], re.MULTILINE))
        elif message["role"] == "user":
            facts.update(extract_profile_updates(message["content"]))
            excerpts.append(" ".join(message["content"].split())[:140])
    facts_text = [f"- {key}: {value[:120]}" for key, value in facts.items()]
    unique = list(dict.fromkeys(excerpts))
    return "\n".join(facts_text + [f"> {item}" for item in unique[-max_items:]])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    summarizer: Callable[[list[dict], str], str] | None = None

    def __post_init__(self):
        if self.threshold_tokens < 1 or self.keep_messages < 1:
            raise ValueError("Compact threshold and keep_messages must be positive")

    def append(self, thread_id: str, role: str, content: str) -> None:
        ctx = self.context(thread_id)
        ctx["messages"].append({"role": role, "content": content})
        tokens = estimate_tokens(ctx["summary"]) + sum(estimate_tokens(m["content"]) for m in ctx["messages"])
        if tokens > self.threshold_tokens and len(ctx["messages"]) > self.keep_messages:
            older = ctx["messages"][:-self.keep_messages]
            source = [{"role": "summary", "content": ctx["summary"]}] + older
            ctx["summary"] = (self.summarizer(source, thread_id) if self.summarizer
                              else summarize_messages(source))
            ctx["messages"] = ctx["messages"][-self.keep_messages:]
            ctx["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
