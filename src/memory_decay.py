"""Optional turn-based decay: hide stale facts from prompts, retain them on disk."""
from __future__ import annotations

import json
from dataclasses import dataclass

from memory_store import UserProfileStore


@dataclass
class DecayingProfileStore(UserProfileStore):
    half_life_turns: int = 64
    minimum_weight: float = 0.125

    def __post_init__(self):
        if self.half_life_turns < 1 or not 0 < self.minimum_weight <= 1:
            raise ValueError("Decay half-life must be positive and weight in (0, 1]")

    def _metadata(self, user_id):
        path = self.path_for(user_id).with_name("freshness.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"turn": 0, "confirmed": {}}

    def _save_metadata(self, user_id, metadata):
        path = self.path_for(user_id).with_name("freshness.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    def advance_turn(self, user_id):
        metadata = self._metadata(user_id)
        metadata["turn"] += 1
        self._save_metadata(user_id, metadata)

    def upsert_fact(self, user_id, key, value):
        super().upsert_fact(user_id, key, value)
        metadata = self._metadata(user_id)
        metadata["confirmed"][key] = metadata["turn"]
        self._save_metadata(user_id, metadata)

    def active_facts(self, user_id):
        metadata = self._metadata(user_id)
        return {key: value for key, value in self.facts(user_id).items()
                if 2 ** (-max(0, metadata["turn"] - metadata["confirmed"].get(key, 0))
                         / self.half_life_turns) >= self.minimum_weight}

    def context_text(self, user_id):
        facts = self.active_facts(user_id)
        return "# Active user profile\n" + "".join(f"- {key}: {value}\n" for key, value in facts.items()) if facts else ""

    def file_size(self, user_id):
        metadata = self.path_for(user_id).with_name("freshness.json")
        return super().file_size(user_id) + (metadata.stat().st_size if metadata.exists() else 0)
