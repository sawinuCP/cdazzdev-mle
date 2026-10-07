# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'shared offline fakes: scripted LLM and synthetic OHLCV frames', Date: 2026-10-06
"""Shared offline fakes for the agentic test suite (no network, no real LLM)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.llm_client import LLMSettings


class FakeLLM:
    """Scripted LLM: pops one item per complete_json call.

    Items can be dicts (validated into the requested model), exceptions, or
    raw strings (parsed through the client's fence stripping).
    """

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.settings = LLMSettings("http://fake", "test-key", "fake-model", 1.0)
        self.last_cache_hit = False

    def complete_json(self, messages, model_cls, max_tokens=None):
        self.calls.append({"messages": messages, "model": model_cls.__name__})
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            item = json.loads(item)
        return model_cls.model_validate(item)


def synthetic_frame(periods: int = 300, seed: int = 11) -> pd.DataFrame:
    """Deterministic OHLCV frame ending today (runtime dates, never literals)."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    rng = np.random.default_rng(seed)
    close = np.clip(100.0 + np.cumsum(rng.normal(0.05, 1.0, periods)), 20.0, None)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, periods).astype(float),
        },
        index=index,
    )


def ok_response(payload: dict) -> SimpleNamespace:
    """Shape an OpenAI-style response around one content string."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
