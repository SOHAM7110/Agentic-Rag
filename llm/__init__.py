from __future__ import annotations
from config import LLM_PROVIDER, LLM_MODEL
from .base import LLM, LLMError, LLMResult, OfflineLLM


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """provider: offline | groq | hf | ollama | openai. Falls back to offline if the provider cannot start
    (missing key, etc.) so a public demo never crashes."""
    provider = (provider or LLM_PROVIDER).lower()
    model = model if model is not None else LLM_MODEL
    if provider == "offline":
        return OfflineLLM()
    from .providers import OpenAICompatLLM
    try:
        return OpenAICompatLLM(provider, model)
    except LLMError as e:
        print(f"[llm] {provider} unavailable ({e}); falling back to offline rule-based backend")
        return OfflineLLM()
