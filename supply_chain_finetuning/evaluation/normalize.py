# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'JSON extraction, schema validation and canonicalization for model outputs', Date: 2026-10-06
"""Normalize raw model outputs into canonical, comparable JSON text.

Unparseable outputs are preserved verbatim and marked ``schema_valid=False``
so they can still be scored honestly (as schema failures) instead of dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import ValidationError

from src.schemas import AnomalyAssessment


@dataclass
class NormalizedOutput:
    """Result of normalization for one raw model response."""

    raw: str
    parsed: Optional[Dict[str, Any]]
    canonical_text: Optional[str]
    schema_valid: bool
    error: str = ""


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the outermost JSON object out of possibly fenced/prose-wrapped text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", cleaned, 0)
    return json.loads(cleaned[start:end + 1])


def canonicalize(obj: Dict[str, Any]) -> str:
    """Canonical text: sorted keys, compact separators, floats rounded."""
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return json.dumps(clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_output(raw: str) -> NormalizedOutput:
    """Extract, validate against the assessment schema, and canonicalize."""
    try:
        obj = extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return NormalizedOutput(raw=raw, parsed=None, canonical_text=None,
                                schema_valid=False, error=f"invalid JSON: {exc}")
    try:
        assessment = AnomalyAssessment.model_validate(obj)
    except ValidationError as exc:
        return NormalizedOutput(raw=raw, parsed=obj, canonical_text=None,
                                schema_valid=False, error=str(exc)[:300])
    canonical = canonicalize(assessment.model_dump())
    return NormalizedOutput(raw=raw, parsed=assessment.model_dump(),
                            canonical_text=canonical, schema_valid=True)
