"""One interface for every model. Nodes call `llm.generate(...)`; swap providers via env vars only."""
from __future__ import annotations
from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    """Rate/usage limit with a long wait (e.g. the free daily token allowance is used up). Not retried and NOT swallowed by
    the agent's fallback logic: callers should stop and report, instead of silently degrading or sleeping for hours."""


@dataclass
class LLMResult:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    cached: bool = False


class LLM:
    name = "base"
    offline = False          # True -> callers use the rule-based heuristics instead of calling a model

    def generate(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 600) -> LLMResult:
        raise NotImplementedError


class OfflineLLM(LLM):
    """Sentinel: 'no LLM'. Agent nodes detect `offline` and run deterministic rule-based logic instead."""
    name = "offline-rules"
    offline = True

    def generate(self, *a, **k):
        raise LLMError("offline backend has no generative model")
