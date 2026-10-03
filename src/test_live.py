"""Contract tests with fake transports, NOT evidence of live model accuracy."""
import json
from types import SimpleNamespace
from dataclasses import replace

import pytest

import agent_advanced
import agent_baseline
import memory_store
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from live_runtime import ModelRuntime
from memory_store import CompactMemoryManager
from model_provider import ProviderConfig, build_chat_model


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        response = next(self.responses)
        return SimpleNamespace(content=response, usage_metadata={"input_tokens": 37, "output_tokens": 9})


def forbidden(*args, **kwargs):
    raise AssertionError("Live called an offline heuristic")


def test_live_advanced_uses_model_for_unseen_fields_and_answers(tmp_path, monkeypatch):
    statement = "I collect antique maps and prefer a monthly digest."
    model = FakeModel([
        json.dumps({"updates": [{"key": "collection", "value": "antique maps",
                                "evidence": "collect antique maps", "confidence": 0.95}]}),
        "Noted your collection.",
        '{"updates": []}', "Your collection is antique maps.",
    ])
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: model)
    monkeypatch.setattr(agent_advanced, "extract_profile_updates", forbidden)
    monkeypatch.setattr(agent_advanced, "respond", forbidden)
    cfg = replace(LabConfig(), state_dir=tmp_path)
    agent = AdvancedAgent(cfg, live=True)
    agent.reply("u", "first", statement)
    assert agent.profile_store.facts("u") == {"collection": "antique maps"}
    answer = agent.reply("u", "fresh", "What do I collect?")
    assert answer["response"] == "Your collection is antique maps."
    assert answer["token_usage"] == 9
    assert answer["prompt_tokens_processed"] == 37
    assert "antique maps" in model.messages[-1][1]["content"]
    assert agent.runtime.totals()["profile_extract"]["calls"] == 2
    assert agent.runtime.totals()["answer"]["calls"] == 2


def test_live_baseline_only_sends_current_thread(tmp_path, monkeypatch):
    model = FakeModel(["Noted.", "I do not know."])
    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: model)
    monkeypatch.setattr(agent_baseline, "extract_profile_updates", forbidden)
    monkeypatch.setattr(agent_baseline, "respond", forbidden)
    agent = BaselineAgent(replace(LabConfig(), state_dir=tmp_path), live=True)
    agent.reply("u", "first", "My name is Mira.")
    agent.reply("u", "second", "What is my name?")
    assert "Mira" not in json.dumps(model.messages[-1])
    assert agent.prompt_token_usage("second") == 37
    assert not list(tmp_path.rglob("User.md"))


def test_live_compact_uses_model_and_accounts_for_usage(monkeypatch):
    monkeypatch.setattr(memory_store, "summarize_messages", forbidden)
    runtime = ModelRuntime(FakeModel(["An unresolved scheduling task."]))
    manager = CompactMemoryManager(10, 1, summarizer=lambda messages, thread: runtime.summarize(messages, thread, 50))
    manager.append("t", "user", "A long scheduling discussion. " * 10)
    manager.append("t", "assistant", "Which day?")
    assert manager.context("t")["summary"] == "An unresolved scheduling task."
    assert manager.compaction_count("t") == 1
    assert runtime.totals()["compact"]["input_tokens"] == 37


def test_extractor_rejects_unsupported_evidence_and_low_confidence():
    runtime = ModelRuntime(FakeModel([json.dumps({"updates": [
        {"key": "city", "value": "X", "evidence": "not in message", "confidence": 0.99},
        {"key": "city", "value": "Y", "evidence": "I moved", "confidence": 0.2},
    ]})]))
    assert runtime.profile_updates({}, "I moved yesterday.", "t") == {}


def test_model_failure_and_invalid_json_do_not_fallback():
    runtime = ModelRuntime(FakeModel(["This is not JSON."]))
    with pytest.raises(json.JSONDecodeError):
        runtime.profile_updates({}, "Hello", "t")
    class BrokenModel:
        def invoke(self, messages):
            raise RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        ModelRuntime(BrokenModel()).invoke([], "answer", "t")


def test_live_requires_explicit_model_and_credentials():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_chat_model(ProviderConfig("gemini", "stub"))
    with pytest.raises(ValueError, match="API_KEY"):
        build_chat_model(ProviderConfig("gemini", "configured-model"))


def test_missing_provider_usage_is_marked_estimated():
    class NoUsage:
        def invoke(self, messages):
            return SimpleNamespace(content="A reply", usage_metadata=None)
    runtime = ModelRuntime(NoUsage())
    runtime.invoke([{"role": "user", "content": "Hello"}], "answer", "t")
    assert runtime.calls[0]["token_source"] == "character_estimate"


def test_judge_inherits_selected_model_credentials(tmp_path, monkeypatch):
    for name in ("JUDGE_API_KEY", "GEMINI_API_KEY", "JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-test")
    cfg = load_config(tmp_path)
    assert cfg.judge_model.api_key == cfg.model.api_key
    assert cfg.judge_model.model_name == cfg.model.model_name


def test_benchmark_live_routing_and_usage_without_answer_leak(tmp_path, monkeypatch):
    from benchmark import run_suites
    class Transport:
        def invoke(self, messages):
            serialized = json.dumps(messages)
            assert "SECRET_EXPECTED" not in serialized
            response = '{"updates": []}' if "NEW_MESSAGE" in serialized else "Transport response"
            return SimpleNamespace(content=response, usage_metadata={"input_tokens": 10, "output_tokens": 5})
    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: Transport())
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: Transport())
    monkeypatch.setattr(agent_baseline, "respond", forbidden)
    monkeypatch.setattr(agent_advanced, "extract_profile_updates", forbidden)
    monkeypatch.setenv("LLM_LIVE", "0")
    data = tmp_path / "data"
    data.mkdir()
    (data / "conversations.json").write_text(json.dumps([
        {"id": "arbitrary", "user_id": "someone", "turns": ["A new topic"],
         "recall_questions": [{"question": "What topic?", "expected_contains": ["SECRET_EXPECTED"]}]}
    ]), encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    usage = {}
    rows = run_suites(replace(LabConfig(), data_dir=data, state_dir=state), mode="live",
                      suite="standard", usage_report=usage)["Standard Benchmark"]
    assert all(row.prompt_tokens_processed == 20 for row in rows)
    assert all(row.agent_tokens_only == 10 for row in rows)
    advanced = usage["Standard Benchmark"]["Advanced"]["model_calls"]
    assert advanced["profile_extract"]["calls"] == 2
    assert advanced["answer"]["calls"] == 2
