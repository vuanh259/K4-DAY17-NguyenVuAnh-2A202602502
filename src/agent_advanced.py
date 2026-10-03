from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, parse_facts
from model_provider import build_chat_model
from offline_response import respond
from live_runtime import ANSWER_SYSTEM, ModelRuntime
from memory_decay import DecayingProfileStore


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Persistent profile plus bounded, per-thread conversation memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False,
                 *, live: bool | None = None, trace=None):
        self.config = config or load_config()
        self.force_offline = force_offline
        self.live = (os.getenv("LLM_LIVE", "0") == "1") if live is None else live
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        if self.config.memory_half_life_turns:
            self.profile_store = DecayingProfileStore(
                self.config.state_dir / "profiles", self.config.memory_half_life_turns,
                self.config.memory_min_weight)
        self.compact_memory = CompactMemoryManager(
            self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()
        self.runtime = ModelRuntime(self.langchain_agent, trace) if self.langchain_agent is not None else None
        if self.runtime:
            self.compact_memory.summarizer = lambda messages, thread: self.runtime.summarize(
                messages, thread, self.config.summary_budget_tokens)

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        owner = self.thread_users.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError("A thread cannot be shared by different users")
        return self._run(user_id, thread_id, message, live=self.langchain_agent is not None)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._run(user_id, thread_id, message, live=False)

    def _run(self, user_id: str, thread_id: str, message: str, live: bool) -> dict[str, Any]:
        if isinstance(self.profile_store, DecayingProfileStore):
            self.profile_store.advance_turn(user_id)
        updates = (self.runtime.profile_updates(self.profile_store.facts(user_id), message,
                   thread_id, self.config.memory_min_confidence) if live
                   else extract_profile_updates(message))
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if live:
            ctx = self.compact_memory.context(thread_id)
            # Profile and summary are untrusted context, not model instructions.
            messages = [{"role": "system", "content": ANSWER_SYSTEM},
                        {"role": "user", "content": "Memory data:\n" +
                         self._profile_context(user_id) + "\n" + ctx["summary"]}]
            messages += ctx["messages"]
            answer, prompt_tokens, output_tokens = self.runtime.invoke(messages, "answer", thread_id)
        else:
            answer = self._offline_response(user_id, thread_id, message)
            output_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {"response": answer, "token_usage": output_tokens,
                "prompt_tokens_processed": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        ctx = self.compact_memory.context(thread_id)
        return (estimate_tokens(self._profile_context(user_id)) +
                estimate_tokens(ctx["summary"]) +
                sum(estimate_tokens(m["content"]) for m in ctx["messages"]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        ctx = self.compact_memory.context(thread_id)
        facts = parse_facts(ctx["summary"])
        for item in ctx["messages"]:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        # Persisted, corrected values override stale values in older summaries.
        facts.update(parse_facts(self._profile_context(user_id)))
        return respond(message, facts)

    def _profile_context(self, user_id: str) -> str:
        if isinstance(self.profile_store, DecayingProfileStore):
            return self.profile_store.context_text(user_id)
        return self.profile_store.read_text(user_id)

    def _maybe_build_langchain_agent(self):
        if self.force_offline or not self.live:
            return None
        return build_chat_model(self.config.model)
