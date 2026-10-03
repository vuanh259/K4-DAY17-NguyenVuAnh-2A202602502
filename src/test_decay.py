from dataclasses import replace

from agent_advanced import AdvancedAgent
from config import LabConfig
from memory_decay import DecayingProfileStore


def test_decay_hides_old_facts_without_deleting_them(tmp_path):
    store = DecayingProfileStore(tmp_path, half_life_turns=2, minimum_weight=0.5)
    store.upsert_fact("u", "project", "An old project")
    for _ in range(3):
        store.advance_turn("u")
    assert store.active_facts("u") == {}
    assert store.facts("u") == {"project": "An old project"}
    store.upsert_fact("u", "project", "An old project")
    assert store.active_facts("u") == {"project": "An old project"}


def test_decay_survives_restart_and_isolates_users(tmp_path):
    store = DecayingProfileStore(tmp_path, 1, 0.5)
    store.upsert_fact("u", "preference", "Detailed answers")
    store.upsert_fact("v", "preference", "Brief answers")
    store.advance_turn("u")
    store.advance_turn("u")
    restored = DecayingProfileStore(tmp_path, 1, 0.5)
    assert restored.active_facts("u") == {}
    assert restored.active_facts("v") == {"preference": "Brief answers"}


def test_decay_applies_to_agent_new_thread_context(tmp_path):
    cfg = replace(LabConfig(), state_dir=tmp_path, memory_half_life_turns=1, memory_min_weight=0.5)
    agent = AdvancedAgent(cfg, force_offline=True)
    agent.reply("u", "a", "Mình tên là Minh.")
    agent.reply("u", "b", "Xin chào.")
    answer = agent.reply("u", "c", "Mình tên gì?")["response"]
    assert "Minh" not in answer
    assert "Minh" in agent.profile_store.read_text("u")
