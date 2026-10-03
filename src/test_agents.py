from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import recall_points, run_suites
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import ProviderConfig, normalize_provider


def make_config(tmp_path: Path):
    return LabConfig(tmp_path, tmp_path / "data", tmp_path / "state", 80, 2,
                     ProviderConfig("openai", "stub"), ProviderConfig("openai", "stub"))


def test_user_markdown_read_write_edit(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text("u") == ""
    assert store.file_size("u") == 0
    path = store.write_text("u", "# User\n- location: Huế\n")
    assert path.name == "User.md"
    assert store.edit_text("u", "Huế", "Đà Nẵng")
    assert "Đà Nẵng" in store.read_text("u")
    assert not store.edit_text("u", "Huế", "Hà Nội")
    assert store.file_size("u") == len(store.read_text("u").encode("utf-8"))
    store.upsert_fact("u", "location", "Huế")
    assert store.facts("u") == {"location": "Huế"}
    assert "Đà Nẵng" not in store.read_text("u")


def test_compact_trigger(tmp_path):
    manager = CompactMemoryManager(80, 2)
    manager.append("t", "user", "Mình tên là Lan.")
    for i in range(10):
        manager.append("t", "user", f"Lượt {i}: " + "Nội dung hội thoại dài. " * 20)
    ctx = manager.context("t")
    assert manager.compaction_count("t") > 0
    assert len(ctx["messages"]) == 2
    assert "Lượt 9" in ctx["messages"][-1]["content"]
    assert "Lan" in ctx["summary"]
    assert estimate_tokens(ctx["summary"]) < 500
    assert manager.context("other")["messages"] == []


def test_cross_session_recall(tmp_path):
    cfg = make_config(tmp_path)
    advanced, baseline = AdvancedAgent(cfg, True), BaselineAgent(cfg, True)
    for agent in (advanced, baseline):
        agent.reply("u", "first", "Mình tên là Lan.")
        assert "Lan" in agent.reply("u", "first", "Mình tên gì?")["response"]
    # Recreate the Advanced instance: persistence is on disk, not an instance dict.
    advanced = AdvancedAgent(cfg, True)
    assert "Lan" in advanced.reply("u", "second", "Mình tên gì?")["response"]
    assert "Lan" not in baseline.reply("u", "second", "Mình tên gì?")["response"]
    assert "Lan" not in advanced.reply("other", "third", "Mình tên gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path):
    cfg = replace(make_config(tmp_path), compact_threshold_tokens=500)
    agents = [BaselineAgent(cfg, True), AdvancedAgent(cfg, True),
              AdvancedAgent(replace(cfg, state_dir=tmp_path / "ablation", compact_threshold_tokens=10**9), True)]
    for agent in agents:
        for i in range(20):
            agent.reply("u", "long", f"Tin {i}. " + "Nội dung tạm thời về thử nghiệm hệ thống. " * 30)
    baseline, advanced, no_compact = agents
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.6
    assert no_compact.prompt_token_usage("long") == baseline.prompt_token_usage("long")
    assert advanced.token_usage("long") == baseline.token_usage("long")


def test_corrections_and_noise_do_not_poison_memory(tmp_path):
    agent = AdvancedAgent(make_config(tmp_path), True)
    turns = [
        "Mình ở Huế và đang làm backend engineer.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
        "Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng để tiện gặp team.",
        "Mình đùa là chuyển sang product manager, nhưng chỉ là câu đùa.",
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.",
        "Mình đang ở Hà Nội phải không?",
    ]
    for turn in turns:
        agent.reply("u", "t", turn)
    facts = agent.profile_store.facts("u")
    assert facts["location"] == "Đà Nẵng"
    assert facts["profession"] == "MLOps engineer"
    text = agent.profile_store.read_text("u")
    assert "Huế" not in text and "backend engineer" not in text
    assert "Hà Nội" not in text and "product manager" not in text


@pytest.mark.parametrize("message", [
    "Mình tên gì?", "Bạn có biết DũngCT không?",
    "Nếu mình ở Hà Nội thì sao?", "Giả sử mình làm product manager.",
    "Mình đùa là mình tên Nam.", "Nhắc lại style trả lời mình thích.",
    "Nhắc lại giúp mình tên và style trả lời mình thích trong stress test này.",
])
def test_questions_and_hypotheticals_are_not_facts(message):
    assert extract_profile_updates(message) == {}


@pytest.mark.parametrize("user", ["../escape", "a/b", "a\\b", "CON", "", "Alice", "alice"])
def test_profile_paths_are_safe(tmp_path, user):
    store = UserProfileStore(tmp_path)
    assert store.path_for(user).is_relative_to(tmp_path.resolve())
    store.write_text(user, "Tiếng Việt")
    assert store.read_text(user) == "Tiếng Việt"
    assert store.path_for("Alice") != store.path_for("alice")
    assert store.path_for("a/b") != store.path_for("a\\b")


def test_token_accounting_and_no_profile_from_question(tmp_path):
    cfg = replace(make_config(tmp_path), compact_threshold_tokens=10000)
    for agent in (BaselineAgent(cfg, True), AdvancedAgent(cfg, True)):
        first = agent.reply("u", "t", "Mình tên là Lan.")
        second = agent.reply("u", "t", "Mình tên gì?")
        assert agent.token_usage("t") == first["token_usage"] + second["token_usage"]
        assert agent.prompt_token_usage("t") == first["prompt_tokens_processed"] + second["prompt_tokens_processed"]
        with pytest.raises(ValueError):
            agent.reply("another", "t", "Mình tên gì?")
    advanced = AdvancedAgent(cfg, True)
    advanced.reply("unknown", "q", "Mình tên gì?")
    assert advanced.memory_file_size("unknown") == 0


def test_estimator_and_scoring():
    assert estimate_tokens("") == estimate_tokens("   ") == 0
    assert estimate_tokens("Tiếng Việt") == estimate_tokens("Tiếng Việt")
    assert estimate_tokens("a" * 20) >= estimate_tokens("a" * 10)
    assert recall_points("DŨNG, Python", ["Dũng", "Python"]) == 1
    assert recall_points("Dũng", ["Dũng", "Python"]) == 0.5
    assert recall_points("Không biết", ["Dũng"]) == 0


def test_config_defaults_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthorpic")
    monkeypatch.setenv("LLM_MODEL", "example")
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "400")
    cfg = load_config(tmp_path)
    assert cfg.model.provider == "anthropic"
    assert cfg.model.model_name == "example"
    assert cfg.compact_threshold_tokens == 400
    assert cfg.state_dir.is_dir()
    assert normalize_provider(" Google ") == "gemini"
    with pytest.raises(ValueError):
        normalize_provider("unknown")


def test_benchmark_is_reproducible_and_data_independent(tmp_path):
    root = Path(__file__).resolve().parent.parent
    cfg = replace(make_config(tmp_path), data_dir=root / "data",
                  compact_threshold_tokens=1200, compact_keep_messages=4)
    cfg.state_dir.mkdir()
    first, second = run_suites(cfg), run_suites(cfg)
    assert first == second
    for rows in first.values():
        baseline, advanced = rows
        assert baseline.recall_score == 0
        assert advanced.recall_score > baseline.recall_score
        assert baseline.memory_growth_bytes == 0 < advanced.memory_growth_bytes
    baseline, advanced = first["Long-Context Stress Benchmark"]
    assert advanced.compactions > 0
    assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed
