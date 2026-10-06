# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline LLM client verification: empty-content budget retry, cache hit, failure logging', Date: 2026-10-06
"""Offline verification for the LLM client (fake SDK, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import llm_client
from src.llm_client import (
    LLMClient,
    LLMError,
    LLMSettings,
    LLMValidationError,
    failure_records,
)
from src.schemas import HeadlineSentiment


class FakeStatusError(Exception):
    """Transport-style error carrying a status code, like the openai SDK."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("unexpected extra call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        content = item if isinstance(item, str) else ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _make_client(monkeypatch, tmp_path, script, use_cache=False):
    completions = FakeCompletions(script)
    fake_openai = lambda **kwargs: SimpleNamespace(  # noqa: E731
        chat=SimpleNamespace(completions=completions)
    )
    monkeypatch.setattr(llm_client, "OpenAI", fake_openai)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    llm_client._FAILURE_RECORDS.clear()
    settings = LLMSettings(base_url="http://fake", api_key="test-key", model="test-model", timeout_s=1.0)
    client = LLMClient(settings=settings, use_cache=use_cache, cache_path=tmp_path / "cache.json")
    return client, completions


_MESSAGES = [
    {"role": "system", "content": "system text"},
    {"role": "user", "content": "user text"},
]


def test_empty_content_retries_with_doubled_budget(monkeypatch, tmp_path):
    payload = json.dumps(
        {"headline": "h", "sentiment": "positive", "confidence": 0.5, "brief_reason": "r"}
    )
    client, completions = _make_client(monkeypatch, tmp_path, ["", payload])
    value = client.chat_json(_MESSAGES, HeadlineSentiment)
    assert value.sentiment == "positive"
    assert [c["max_tokens"] for c in completions.calls] == [
        llm_client.config.LLM_MIN_MAX_TOKENS,
        llm_client.config.LLM_MIN_MAX_TOKENS * 2,
    ]


def test_cache_hit_avoids_second_call(monkeypatch, tmp_path):
    client, completions = _make_client(monkeypatch, tmp_path, [], use_cache=True)
    payload = json.dumps(
        {"headline": "h", "sentiment": "neutral", "confidence": 0.2, "brief_reason": "r"}
    )
    key = client._cache_key("test-model", _MESSAGES)
    client.cache_path.parent.mkdir(parents=True, exist_ok=True)
    client.cache_path.write_text(json.dumps({key: payload}), encoding="utf-8")
    value = client.chat_json(_MESSAGES, HeadlineSentiment)
    assert value.sentiment == "neutral"
    assert completions.calls == []          # the network was never touched
    assert client.cache_hits == 1


def test_transport_errors_retry_with_backoff_then_succeed(monkeypatch, tmp_path):
    payload = json.dumps(
        {"headline": "h", "sentiment": "negative", "confidence": 0.4, "brief_reason": "r"}
    )
    script = [FakeStatusError(500), FakeStatusError(429), payload]
    client, completions = _make_client(monkeypatch, tmp_path, script)
    value = client.chat_json(_MESSAGES, HeadlineSentiment)
    assert value.sentiment == "negative"
    assert len(completions.calls) == 3


def test_persistent_failure_raises_and_is_logged(monkeypatch, tmp_path):
    script = [FakeStatusError(503)] * 6
    client, completions = _make_client(monkeypatch, tmp_path, script)
    with pytest.raises(LLMError):
        client.chat_json(_MESSAGES, HeadlineSentiment)
    assert len(completions.calls) == llm_client.config.LLM_MAX_ATTEMPTS
    assert any(rec["stage"] == "chat.completions" for rec in failure_records())


def test_repair_retry_carries_error_and_schema(monkeypatch, tmp_path):
    payload = json.dumps(
        {"headline": "h", "sentiment": "positive", "confidence": 0.5, "brief_reason": "r"}
    )
    client, completions = _make_client(monkeypatch, tmp_path, ["this is not json", payload])
    value = client.chat_json(_MESSAGES, HeadlineSentiment)
    assert value.sentiment == "positive"
    repair_call = completions.calls[1]
    repair_message = repair_call["messages"][-1]["content"]
    assert "Validation error" in repair_message
    assert '"headline"' in repair_message  # the schema is appended for the repair


def test_fenced_json_is_stripped(monkeypatch, tmp_path):
    fenced = "```json\n" + json.dumps(
        {"headline": "h", "sentiment": "neutral", "confidence": 0.1, "brief_reason": "r"}
    ) + "\n```"
    client, _ = _make_client(monkeypatch, tmp_path, [fenced])
    value = client.chat_json(_MESSAGES, HeadlineSentiment)
    assert value.sentiment == "neutral"


def test_missing_api_key_raises_clear_configuration_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "load_env", lambda: None)
    with pytest.raises(llm_client.LLMConfigurationError) as excinfo:
        llm_client.settings_from_env()
    assert "LLM_API_KEY" in str(excinfo.value)
