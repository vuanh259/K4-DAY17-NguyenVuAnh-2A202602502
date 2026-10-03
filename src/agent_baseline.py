from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any
from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from offline_response import respond
from live_runtime import ANSWER_SYSTEM, ModelRuntime


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Control agent: conversation history only, keyed by thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False,
                 *, live: bool | None = None, trace=None):
        self.config = config or load_config()
        self.force_offline = force_offline
        self.live = (os.getenv("LLM_LIVE", "0") == "1") if live is None else live
        self.sessions: dict[str, SessionState] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()
        self.runtime = ModelRuntime(self.langchain_agent, trace) if self.langchain_agent is not None else None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        owner = self.thread_users.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError("A thread cannot be shared by different users")
        return self._run(thread_id, message, live=self.langchain_agent is not None)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        return self._run(thread_id, message, live=False)

    def _run(self, thread_id: str, message: str, live: bool) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in state.messages)
        if live:
            answer, prompt_tokens, output_tokens = self.runtime.invoke(
                [{"role": "system", "content": ANSWER_SYSTEM}] + state.messages,
                "answer", thread_id)
        else:
            facts = {}
            for item in state.messages:
                if item["role"] == "user":
                    facts.update(extract_profile_updates(item["content"]))
            answer = respond(message, facts)
            output_tokens = estimate_tokens(answer)
        state.messages.append({"role": "assistant", "content": answer})
        state.token_usage += output_tokens
        state.prompt_tokens_processed += prompt_tokens
        return {"response": answer, "token_usage": output_tokens,
                "prompt_tokens_processed": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        # Live is explicit, never activated just because a key happens to exist.
        if self.force_offline or not self.live:
            return None
        return build_chat_model(self.config.model)
