"""Reproducible synthetic ablations for memory policies; no API calls."""
import json
import tempfile
from pathlib import Path

from memory_decay import DecayingProfileStore
from memory_store import estimate_tokens


def run_experiment(folder: Path) -> dict:
    store = DecayingProfileStore(folder, half_life_turns=2, minimum_weight=0.5)
    store.upsert_fact("demo", "project", "Research archival maps and prepare a detailed catalog of historical collections.")
    store.upsert_fact("demo", "response_preference", "Use concise explanations.")
    before = estimate_tokens(store.read_text("demo"))
    for _ in range(3):
        store.advance_turn("demo")
    store.upsert_fact("demo", "response_preference", "Use concise explanations.")
    active = store.active_facts("demo")
    result = {
        "kind": "synthetic_policy_experiment_not_live_model",
        "half_life_turns": 2, "minimum_weight": 0.5, "elapsed_turns": 3,
        "facts_on_disk": len(store.facts("demo")), "facts_in_prompt": len(active),
        "full_profile_estimated_tokens": before,
        "active_profile_estimated_tokens": estimate_tokens(store.context_text("demo")),
        "stale_fact_hidden": "project" not in active,
        "reconfirmed_fact_kept": "response_preference" in active,
        "storage_bytes_including_metadata": store.file_size("demo"),
    }
    store.upsert_fact("demo", "project", "Maintain the new catalog.")
    result["correction_replaces_old_value"] = "Research archival maps" not in store.read_text("demo")
    result["correction_restores_active_fact"] = store.active_facts("demo")["project"] == "Maintain the new catalog."
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    (root / "state").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "state", prefix="bonus-") as folder:
        result = run_experiment(Path(folder))
    output = root / "results" / "bonus_policies.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
