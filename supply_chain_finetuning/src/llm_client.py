"""OpenAI-compatible LLM client shared by the teacher and the judge.

- Configuration comes from environment variables with a configurable prefix
  (``TEACHER_*`` / ``JUDGE_*``) so the two roles can use different providers
  and models without code changes.
- Robustness: exponential backoff on timeouts/429/5xx, one retry with a
  doubled token budget on empty content (thinking models), code-fence
  stripping, optional schema-repair attempt.
- Reproducibility: response cache keyed by model + messages (+ optional salt,
  used for teacher attempt numbers); every real call appends a usage row
  (tokens, latency) to a CSV so API spend is auditable.
"""
from __future__ import annotations

import csv
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

from . import config

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
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass

@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout_s: float
    temperature: float

def settings_from_env(
    prefix: str,
    default_base_url: str,
    default_model: str,
    temperature: float,
) -> LLMSettings:
    """Read ``<PREFIX>_BASE_URL/_API_KEY/_MODEL`` from the environment."""
    load_env()
    api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError(
            f"{prefix}_API_KEY is not set. Copy supply_chain_finetuning/.env.example "
            f"to '.env' at the repository root and fill in the {prefix}_* variables."
        )
    base_url = os.environ.get(f"{prefix}_BASE_URL", default_base_url).strip()
    model = os.environ.get(f"{prefix}_MODEL", default_model).strip()
    return LLMSettings(
        base_url=base_url, api_key=api_key, model=model,
        timeout_s=float(os.environ.get(f"{prefix}_TIMEOUT_S", 120.0) or 120.0),
        temperature=temperature,
    )

def client_from_env(
    prefix: str, temperature: float, cache_path: Path, usage_log_path: Path,
    disable_thinking: bool = False,
) -> "LLMClient":
    """Standard entry point for the teacher and the judge roles."""
    if prefix == config.ENV_TEACHER_MODEL.removesuffix("_MODEL"):
        settings = settings_from_env(
            prefix, config.DEFAULT_TEACHER_BASE_URL, config.DEFAULT_TEACHER_MODEL, temperature
        )
    else:  # judge has no hard default model: it must be configured
        settings = settings_from_env(prefix, config.DEFAULT_TEACHER_BASE_URL, "", temperature)
    return LLMClient(
        settings=settings, cache_path=cache_path, usage_log_path=usage_log_path,
        disable_thinking=disable_thinking,
    )

class LLMClient:
    """OpenAI-compatible chat client with retry, cache, and usage logging."""

    def __init__(
        self,
        settings: LLMSettings,
        cache_path: Path,
        usage_log_path: Path,
        repair_attempts: int = 0,
        disable_thinking: bool = False,
    ) -> None:
        self.settings = settings
        self.cache_path = Path(cache_path)
        self.usage_log_path = Path(usage_log_path)
        self.repair_attempts = repair_attempts
        self.disable_thinking = disable_thinking
        self._client: Optional[OpenAI] = None
        self.calls_made = 0
        self.cache_hits = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_s,
            )
        return self._client

    # plumbing
    def _cache_key(self, messages: List[Dict[str, str]], salt: str = "", max_tokens: int = 0) -> str:
        normalized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        # The token budget is part of the key: responses produced under a smaller
        # budget (e.g. truncated JSON) must never be replayed after a re-run.
        return hashlib.sha256(
            f"{self.settings.model}|{self.settings.temperature}|{max_tokens}|{salt}|{normalized}".encode("utf-8")
        ).hexdigest()

    def _read_cache(self, key: str) -> Optional[str]:
        try:
            cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
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
            self.cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:  # the cache is an optimisation; never fatal
            _LOGGER.warning("cache write failed: %s", exc)

    def _log_usage(self, usage: Dict[str, Any], latency_s: float, cache_state: str) -> None:
        """Append one row per real API call (spend is auditable per role)."""
        try:
            self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self.usage_log_path.exists()
            with open(self.usage_log_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if is_new:
                    writer.writerow(["timestamp_utc", "model", "prompt_tokens",
                                     "completion_tokens", "total_tokens", "latency_s", "cache"])
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    self.settings.model, usage.get("prompt_tokens", ""),
                    usage.get("completion_tokens", ""), usage.get("total_tokens", ""),
                    round(latency_s, 2), cache_state,
                ])
        except OSError as exc:
            _LOGGER.warning("usage log write failed: %s", exc)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        return getattr(exc, "status_code", None) in config.RETRYABLE_STATUS_CODES

    def _create(self, messages: List[Dict[str, str]], max_tokens: int):
        kwargs: Dict[str, Any] = dict(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=max_tokens,
        )
        if self.disable_thinking:
            # Structured generation does not need the reasoning pass; disabling it
            # cuts latency sharply. The acceptance gates remain the quality control.
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            return self.client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 400:
                # some gateways reject response_format; plain JSON instructions still work
                fallback = self.client.chat.completions.create(**kwargs)
                return fallback
            raise

    @staticmethod
    def _extract(response: Any) -> tuple[str, Dict[str, Any]]:
        choices = getattr(response, "choices", None) or []
        content = ""
        if choices:
            content = (getattr(choices[0].message, "content", None) or "").strip()
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", ""),
            "completion_tokens": getattr(usage, "completion_tokens", ""),
            "total_tokens": getattr(usage, "total_tokens", ""),
        }
        return content, usage_dict

    def _complete(
        self, messages: List[Dict[str, str]], max_tokens: int, salt: str = ""
    ) -> str:
        """Complete with backoff, the empty-content budget retry, cache, usage log."""
        key = self._cache_key(messages, salt=salt, max_tokens=max_tokens)
        if (cached := self._read_cache(key)) is not None:
            self.cache_hits += 1
            return cached
        budget = max_tokens
        doubled = False
        last_error: Optional[Exception] = None
        for attempt in range(config.LLM_MAX_ATTEMPTS * 2):
            started = time.monotonic()
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
            content, usage = self._extract(response)
            self._log_usage(usage, time.monotonic() - started, "miss")
            if content:
                self._write_cache(key, content)
                return content
            if not doubled:
                doubled = True
                budget *= 2  # thinking models spend the budget on reasoning tokens
                continue
            raise LLMError("model returned empty content even with a doubled token budget")
        raise LLMError(f"LLM call failed after retries: {last_error}")

    # JSON path
    @staticmethod
    def _strip_fences(content: str) -> str:
        """Remove markdown fences and keep the outermost JSON object."""
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

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model_cls: Type[TModel],
        max_tokens: Optional[int] = None,
        salt: str = "",
    ) -> TModel:
        """Chat and return a validated Pydantic instance.

        ``salt`` differentiates retries of the same prompt (teacher attempts)
        so a rejected sample is genuinely regenerated, not replayed.
        """
        budget = max_tokens or config.LLM_MAX_TOKENS
        content = self._complete(messages, budget, salt=salt)
        try:
            return self._parse(content, model_cls)
        except LLMValidationError as first_error:
            for _ in range(self.repair_attempts):
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
                content = self._complete(repair_messages, budget, salt=salt + "|repair")
                try:
                    return self._parse(content, model_cls)
                except LLMValidationError as exc:
                    first_error = exc
            raise
