"""Provider adapters are lazy: offline runs never need an SDK or API key."""
from __future__ import annotations
from dataclasses import dataclass, field
from importlib import import_module


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    value = value.strip().lower()
    value = {"anthorpic": "anthropic", "google": "gemini",
             "google-genai": "gemini", "openai-compatible": "custom"}.get(value, value)
    if value not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value!r}")
    return value


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    if not config.model_name.strip() or config.model_name == "stub":
        raise ValueError("Live mode requires LLM_MODEL set to an actual model ID in .env")
    if provider not in {"ollama", "custom"} and not config.api_key:
        raise ValueError(f"Live mode requires {provider.upper()}_API_KEY or LLM_API_KEY")
    adapters = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "custom": ("langchain_openai", "ChatOpenAI"),
        "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    }
    if provider == "custom" and not config.base_url:
        raise ValueError("custom provider requires CUSTOM_BASE_URL")
    module, name = adapters[provider]
    cls = getattr(import_module(module), name)
    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key and provider != "ollama":
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return cls(**kwargs)
