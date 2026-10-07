"""Pydantic models for the supply-chain anomaly pipeline.

The output contract is enforced mechanically: taxonomy enum, severity band,
confidence range, root-cause sentence count, action count, and the
is_anomaly <-> class consistency rule all fail fast at validation time.
"""
from __future__ import annotations

import re
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

def count_sentences(text: str) -> int:
    """Capital-letter look-ahead split so decimals ("1.5") and "U.S." don't inflate."""
    parts = re.split(config.SENTENCE_SPLIT_REGEX, (text or "").strip())
    return len([p for p in parts if p.strip()])

def _to_float(value: Any) -> float:
    """Coerce ints/numeric strings to float (feeds sometimes quote numbers)."""
    return float(value)

class AnomalyInput(BaseModel):
    """One warehouse/SKU-day event feed (the model's input)."""

    product_category: str
    region: str
    order_volume_vs_forecast_pct: float
    inventory_days_on_hand: float
    on_time_delivery_pct: float
    supplier_status: str
    transit_days_normal: float
    transit_days_observed: float
    port_congestion_index: float
    weather_event: str
    labor_document_note: str
    unit_cost_vs_last_quarter_pct: float
    notes: str

    @field_validator(
        "order_volume_vs_forecast_pct", "inventory_days_on_hand", "on_time_delivery_pct",
        "transit_days_normal", "transit_days_observed", "port_congestion_index",
        "unit_cost_vs_last_quarter_pct", mode="before",
    )
    @classmethod
    def _numeric(cls, v: Any) -> float:
        try:
            return _to_float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expected a number, got {v!r}") from exc

    @field_validator("product_category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        if v not in config.PRODUCT_CATEGORIES:
            raise ValueError(f"unknown product_category {v!r}")
        return v

class AnomalyAssessment(BaseModel):
    """The structured output the student model must produce."""

    is_anomaly: bool
    anomaly_class: str
    severity: int = Field(ge=config.SEVERITY_MIN, le=config.SEVERITY_MAX)
    root_cause: str
    evidence_fields: List[str] = Field(default_factory=list)
    corrective_actions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=config.CONFIDENCE_MIN, le=config.CONFIDENCE_MAX)

    @field_validator("anomaly_class")
    @classmethod
    def _in_taxonomy(cls, v: str) -> str:
        if v not in config.TAXONOMY:
            raise ValueError(f"anomaly_class {v!r} outside the taxonomy")
        return v

    @field_validator("root_cause")
    @classmethod
    def _sentence_range(cls, v: str) -> str:
        n = count_sentences(v)
        low, high = config.ROOT_CAUSE_SENTENCE_RANGE
        if not (low <= n <= high):
            raise ValueError(f"root_cause must contain {low}-{high} sentences, found {n}")
        return v

    @field_validator("corrective_actions")
    @classmethod
    def _action_range(cls, v: List[str]) -> List[str]:
        low, high = config.ACTIONS_COUNT_RANGE
        if not (low <= len(v) <= high):
            raise ValueError(f"corrective_actions must contain {low}-{high} items, found {len(v)}")
        if any(not isinstance(a, str) or not a.strip() for a in v):
            raise ValueError("corrective_actions entries must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "AnomalyAssessment":
        if self.is_anomaly != (self.anomaly_class != config.NONE_CLASS):
            raise ValueError(
                "is_anomaly must match the class: normal cases use "
                f"anomaly_class={config.NONE_CLASS!r} with is_anomaly=false"
            )
        if self.is_anomaly and not self.evidence_fields:
            raise ValueError("an anomalous assessment must cite at least one evidence field")
        return self

class TeacherResponse(BaseModel):
    """One teacher call returns the feed and its grounded gold assessment."""

    input_feed: AnomalyInput
    gold_assessment: AnomalyAssessment
