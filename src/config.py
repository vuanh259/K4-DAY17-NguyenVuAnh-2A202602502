from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LabConfig:
    base_dir: Path = ROOT
    data_dir: Path = ROOT / "data"
    state_dir: Path = ROOT / "state"
    compact_threshold_tokens: int = 1200
    compact_keep_messages: int = 4
    model: ProviderConfig = field(default_factory=lambda: ProviderConfig("openai", "stub"))
    judge_model: ProviderConfig = field(default_factory=lambda: ProviderConfig("openai", "stub"))
    memory_min_confidence: float = 0.85
    summary_budget_tokens: int = 300
    memory_half_life_turns: int = 0
    memory_min_weight: float = 0.125

    def __post_init__(self):
        if self.compact_threshold_tokens < 1 or self.compact_keep_messages < 1:
            raise ValueError("Compact threshold and keep_messages must be positive")
        if not 0 <= self.memory_min_confidence <= 1 or self.summary_budget_tokens < 1:
            raise ValueError("Invalid memory confidence or summary budget")
        if self.memory_half_life_turns < 0 or not 0 < self.memory_min_weight <= 1:
            raise ValueError("Invalid memory decay configuration")


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
    key_prefix = provider.upper()
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", fallback.model_name if fallback and fallback.provider == provider else "stub"),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=(os.getenv(f"{prefix}_API_KEY") or os.getenv(f"{key_prefix}_API_KEY")
                 or (fallback.api_key if fallback and fallback.provider == provider else None)),
        base_url=(os.getenv(f"{prefix}_BASE_URL") or os.getenv(f"{key_prefix}_BASE_URL")
                  or (fallback.base_url if fallback and fallback.provider == provider else None)),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or ROOT).resolve()
    load_dotenv(root / ".env", override=False)
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    model = _provider_config("LLM")
    return LabConfig(root, root / "data", state,
                     int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
                     int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
                     model, _provider_config("JUDGE", model),
                     float(os.getenv("MEMORY_MIN_CONFIDENCE", "0.85")),
                     int(os.getenv("SUMMARY_BUDGET_TOKENS", "300")),
                     int(os.getenv("MEMORY_HALF_LIFE_TURNS", "0")),
                     float(os.getenv("MEMORY_MIN_WEIGHT", "0.125")))
