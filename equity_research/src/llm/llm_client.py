# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'OpenAI-compatible LLM client: retries, JSON repair, cache, failure log', Date: 2026-10-06
"""OpenAI-compatible LLM client.

Design notes:
- Provider-agnostic: configuration comes only from environment variables
  (``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``, ``LLM_TIMEOUT_S``), so
  switching gateway/model requires no code change.
- Temperature 0 everywhere: analyst output must be reproducible.
- Thinking-style models spend the token budget on reasoning tokens and can
  return empty content; the client therefore uses a generous floor and retries
  ONCE with a doubled budget on empty responses.
- Transport problems (timeouts, 429, 5xx) back off exponentially.
- JSON outputs are stripped of code fences, parsed, and validated against a
  Pydantic model; on failure the client performs ONE repair call that appends
  the exact validation error and the schema, then raises ``LLMValidationError``
  so callers can apply their own fallback.
- A committed response cache (``outputs/.llm_cache.json``) makes re-runs
  deterministic and free; failures are logged to ``logs/llm_failures.log`` and
  kept in memory for notebook display.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .. import config
from . import prompts

TModel = TypeVar("TModel", bound=BaseModel)


class LLMConfigurationError(RuntimeError):
    """Missing/invalid LLM environment configuration."""


class LLMError(RuntimeError):
    """LLM call failed after retries (transport, budget, or empty content)."""


class LLMValidationError(LLMError):
    """Response could not be parsed/validated even after the repair attempt."""


_LOGGER = logging.getLogger("equity_research.llm")
_FAILURE_HANDLER_ATTACHED = False
_FAILURE_RECORDS: List[Dict[str, str]] = []


def _attach_failure_handler() -> None:
    """Attach the failures file handler once (idempotent)."""
    global _FAILURE_HANDLER_ATTACHED
    if _FAILURE_HANDLER_ATTACHED:
        return
    config.ensure_dirs()
    handler = logging.FileHandler(config.LLM_FAILURE_LOG, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.WARNING)
    _LOGGER.propagate = False
    _FAILURE_HANDLER_ATTACHED = True


def record_failure(stage: str, error_type: str, detail: str) -> None:
    """Log a failure to the failure log and the in-memory notebook mirror."""
    _attach_failure_handler()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "error_type": error_type,
        "detail": detail[:400],
    }
    _FAILURE_RECORDS.append(entry)
    _LOGGER.warning("stage=%s type=%s detail=%s", stage, error_type, entry["detail"])


def failure_records() -> List[Dict[str, str]]:
    """In-memory failure mirror (displayed in the notebook)."""
    return list(_FAILURE_RECORDS)


def load_env() -> None:
    """Load a ``.env`` walking upwards from the current directory.

    Works whether the process starts in the project folder or the repository
    root; the notebook additionally uses Colab Secrets (handled there).
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM connection settings."""

    base_url: str
    api_key: str
    model: str
    timeout_s: float


def settings_from_env() -> LLMSettings:
    """Resolve settings from the environment; fail loudly but kindly."""
    load_env()
    api_key = os.environ.get(config.ENV_API_KEY, "").strip()
    if not api_key:
        raise LLMConfigurationError(
            "LLM_API_KEY is not set. Copy equity_research/.env.example to '.env' "
            "(repository root), fill in LLM_API_KEY, or export the variable: "
            "LLM_BASE_URL, LLM_API_KEY, LLM_MODEL."
        )
    base_url = os.environ.get(config.ENV_BASE_URL, config.DEFAULT_BASE_URL).strip()
    model = os.environ.get(config.ENV_MODEL, config.DEFAULT_MODEL).strip()
    try:
        timeout_s = float(os.environ.get(config.ENV_TIMEOUT_S, config.DEFAULT_TIMEOUT_S))
    except ValueError:
        timeout_s = config.DEFAULT_TIMEOUT_S
    return LLMSettings(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s)

class LLMClient:
    """OpenAI-compatible chat client with retry, repair, cache, and logging."""

    def __init__(
        self,
        settings: Optional[LLMSettings] = None,
        use_cache: bool = True,
        cache_path: Path = config.LLM_CACHE_JSON,
    ) -> None:
        self.settings = settings or settings_from_env()
        self.use_cache = use_cache
        self.cache_path = Path(cache_path)
        self._client: Optional[OpenAI] = None
        self.calls_made = 0
        self.cache_hits = 0

    # ── plumbing ──────────────────────────────────────────────────────
    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_s,
            )
        return self._client

    @staticmethod
    def _cache_key(model: str, messages: List[Dict[str, str]]) -> str:
        normalized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{model}|{normalized}".encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[str]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                cache = json.load(handle)
            return cache.get(key)
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
            self.cache_path.write_text(
                json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:  # cache is an optimisation; never fatal
            record_failure("cache_write", type(exc).__name__, str(exc))

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        return getattr(exc, "status_code", None) in config.RETRYABLE_STATUS_CODES

    def _create(self, messages: List[Dict[str, str]], max_tokens: int):
        """One chat completion; degrades gracefully if JSON mode is unsupported."""
        kwargs: Dict[str, Any] = dict(
            model=self.settings.model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens,
        )
        try:
            return self.client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 400:
                # Some gateways reject response_format; plain instructions still work.
                record_failure("response_format", type(exc).__name__, str(exc)[:200])
                return self.client.chat.completions.create(**kwargs)
            raise

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return (getattr(message, "content", None) or "").strip()

    def _complete(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Complete with transport backoff and the empty-content budget retry."""
        budget = max_tokens
        doubled = False
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
                record_failure("chat.completions", type(exc).__name__, str(exc))
                raise LLMError(f"LLM call failed after retries: {exc}") from exc
            content = self._extract_content(response)
            if content:
                return content
            if not doubled:
                doubled = True
                budget *= config.LLM_TOKEN_BUDGET_MULTIPLIER  # reasoning tokens ate the budget
                continue
            record_failure(
                "empty_content", "EmptyContentError",
                "model returned no content even with a doubled token budget",
            )
            raise LLMError("model returned empty content even with a doubled token budget")
        raise LLMError(f"LLM call failed after retries: {last_error}")

    # ── JSON path ─────────────────────────────────────────────────────
    @staticmethod
    def _strip_fences(content: str) -> str:
        """Remove markdown code fences and keep the outermost JSON object."""
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
            raise LLMValidationError(
                f"invalid JSON ({exc}); raw head: {content[:200]!r}"
            ) from exc
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc

    def _parse_with_repair(
        self, messages: List[Dict[str, str]], content: str, model_cls: Type[TModel]
    ) -> TModel:
        """Validate; on failure make ONE repair call carrying the exact error."""
        error: Optional[LLMValidationError] = None
        for _ in range(config.LLM_REPAIR_ATTEMPTS + 1):
            try:
                return self._parse(content, model_cls)
            except LLMValidationError as exc:
                error = exc
                repair_messages = list(messages) + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": prompts.REPAIR_SUFFIX.format(
                            error=str(exc)[:500],
                            schema=json.dumps(model_cls.model_json_schema()),
                        ),
                    },
                ]
                content = self._complete(repair_messages, config.LLM_MIN_MAX_TOKENS)
        record_failure("validate", type(error).__name__, str(error))  # type: ignore[union-attr]
        raise error  # type: ignore[misc]

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model_cls: Type[TModel],
        max_tokens: Optional[int] = None,
    ) -> TModel:
        """Chat and return a validated Pydantic model instance.

        Cache hits are validated exactly like fresh responses; an entry that no
        longer validates is treated as a miss and recomputed.
        """
        budget = max_tokens or config.LLM_MIN_MAX_TOKENS
        key = self._cache_key(self.settings.model, messages)
        if self.use_cache:
            cached = self._read_cache(key)
            if cached is not None:
                try:
                    self.cache_hits += 1
                    return self._parse(cached, model_cls)
                except LLMValidationError:
                    self.cache_hits -= 1  # stale/incompatible entry: recompute
                    record_failure("cache_validate", "LLMValidationError", cached[:200])
        content = self._complete(messages, budget)
        value = self._parse_with_repair(messages, content, model_cls)
        if self.use_cache:
            self._write_cache(key, content)
        return value


def client_from_env(use_cache: bool = True) -> LLMClient:
    """Build a client from environment variables (the standard entry point)."""
    return LLMClient(settings=settings_from_env(), use_cache=use_cache)
