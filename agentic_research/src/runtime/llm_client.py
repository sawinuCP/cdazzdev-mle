"""OpenAI-compatible LLM client for the agents.

- Config only from environment (``LLM_BASE_URL/_API_KEY/_MODEL/_TIMEOUT_S``).
- Temperature 0; thinking-style models spend ``max_tokens`` on reasoning, so
  the budget has a generous floor and one doubled retry on empty content.
- JSON mode with fence stripping, Pydantic validation, ONE repair retry that
  appends the exact validation error + schema, then ``LLMValidationError``.
- Response cache keyed by sha256(model + normalized messages) with a
  ``use_cache`` flag and a ``last_cache_hit`` flag the caller traces.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .. import config

TModel = TypeVar("TModel", bound=BaseModel)

_LOGGER = logging.getLogger(__name__)

class LLMConfigurationError(RuntimeError):
    """Missing/invalid LLM environment configuration."""

class LLMError(RuntimeError):
    """LLM call failed after retries (transport, budget, or empty content)."""

class LLMValidationError(LLMError):
    """Response could not be parsed/validated even after the repair attempt."""

def load_env() -> None:
    """Load a ``.env`` walking upwards from the current directory."""
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:  # pragma: no cover
        pass

@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout_s: float

def settings_from_env() -> LLMSettings:
    api_key = os.environ.get(config.ENV_API_KEY, "").strip()
    if not api_key:
        raise LLMConfigurationError(
            "LLM_API_KEY is not set. Copy agentic_research/.env.example to '.env' "
            "at the repository root, or export LLM_BASE_URL / LLM_API_KEY / LLM_MODEL."
        )
    return LLMSettings(
        base_url=os.environ.get(config.ENV_BASE_URL, config.DEFAULT_BASE_URL).strip(),
        api_key=api_key,
        model=os.environ.get(config.ENV_MODEL, config.DEFAULT_MODEL).strip(),
        timeout_s=float(os.environ.get(config.ENV_TIMEOUT_S, 120.0) or 120.0),
    )

def client_from_env(use_cache: bool = True) -> "LLMClient":
    """Standard entry point."""
    return LLMClient(settings=settings_from_env(), use_cache=use_cache)

class LLMClient:
    def __init__(self, settings: LLMSettings, use_cache: bool = True,
                 cache_path: Path = config.LLM_CACHE_JSON) -> None:
        self.settings = settings
        self.use_cache = use_cache
        self.cache_path = Path(cache_path)
        self._client: Optional[OpenAI] = None
        self.calls_made = 0
        self.cache_hits = 0
        self.last_cache_hit = False  # read by the caller for the trace

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.api_key, base_url=self.settings.base_url,
                timeout=self.settings.timeout_s,
            )
        return self._client

    def _cache_key(self, messages: List[Dict[str, str]]) -> str:
        normalized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self.settings.model}|{normalized}".encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[str]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8")).get(key)
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, content: str) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache: Dict[str, str] = {}
            if self.cache_path.exists():
                try:
                    cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cache = {}
            cache[key] = content
            self.cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            _LOGGER.warning("cache write failed: %s", exc)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        return getattr(exc, "status_code", None) in config.RETRYABLE_STATUS_CODES

    def _create(self, messages: List[Dict[str, str]], max_tokens: int):
        kwargs: Dict[str, Any] = dict(
            model=self.settings.model, messages=messages,
            temperature=config.LLM_TEMPERATURE, max_tokens=max_tokens,
        )
        if config.LLM_USE_RESPONSE_FORMAT:
            try:
                return self.client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            except Exception as exc:
                if getattr(exc, "status_code", None) == 400:
                    # some gateways reject response_format; plain instructions
                    # still work
                    return self.client.chat.completions.create(**kwargs)
                raise
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return (getattr(choices[0].message, "content", None) or "").strip()

    def _complete(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        budget = max_tokens
        last_error: Optional[Exception] = None
        for attempt in range(config.LLM_MAX_ATTEMPTS * 2):
            try:
                self.calls_made += 1
                response = self._create(messages, budget)
            except Exception as exc:  # noqa: BLE001 - classified immediately below
                self.calls_made -= 1
                last_error = exc
                if self._is_retryable(exc) and attempt < config.LLM_MAX_ATTEMPTS - 1:
                    time.sleep(config.LLM_BACKOFF_BASE_S * (2 ** attempt))
                    continue
                raise LLMError(f"LLM call failed after retries: {exc}") from exc
            content = self._extract_content(response)
            if content:
                return content
            # thinking models burn reasoning tokens first: escalate the budget
            # on EVERY empty attempt instead of a single doubling
            if getattr(response, "choices", None) and \
                    getattr(response.choices[0], "finish_reason", None) == "length":
                budget *= config.LLM_TOKEN_BUDGET_MULTIPLIER
                continue
            raise LLMError("model returned empty content without hitting the "
                           "token budget")
        raise LLMError(f"LLM call failed after retries: {last_error or 'empty content'}")

    @staticmethod
    def _strip_fences(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.lstrip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        return text.strip()

    def _parse(self, content: str, model_cls: Type[TModel]) -> TModel:
        try:
            data = json.loads(self._strip_fences(content))
        except json.JSONDecodeError as exc:
            raise LLMValidationError(f"invalid JSON ({exc}); raw head: {content[:200]!r}") from exc
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc

    def complete_json(
        self, messages: List[Dict[str, str]], model_cls: Type[TModel],
        max_tokens: Optional[int] = None,
    ) -> TModel:
        """Chat and return a validated Pydantic instance (one repair retry)."""
        budget = max_tokens or config.LLM_MAX_TOKENS
        key = self._cache_key(messages)
        self.last_cache_hit = False
        if self.use_cache:
            cached = self._read_cache(key)
            if cached is not None:
                try:
                    self.cache_hits += 1
                    self.last_cache_hit = True
                    return self._parse(cached, model_cls)
                except LLMValidationError:
                    self.cache_hits -= 1
                    self.last_cache_hit = False
        content = self._complete(messages, budget)
        try:
            value = self._parse(content, model_cls)
        except LLMValidationError as first_error:
            repair_messages = list(messages) + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous response could not be validated.\n"
                        f"Validation error: {str(first_error)[:500]}\n"
                        "Return ONLY a corrected JSON object that validates against "
                        f"this schema:\n{json.dumps(model_cls.model_json_schema())}"
                    ),
                },
            ]
            content = self._complete(repair_messages, budget)
            try:
                value = self._parse(content, model_cls)
            except LLMValidationError as exc:
                _LOGGER.error("LLM validation failed after repair: %s", exc)
                raise
        if self.use_cache:
            # cache the FINAL successful content first-attempt or repaired
            self._write_cache(key, content)
        return value
