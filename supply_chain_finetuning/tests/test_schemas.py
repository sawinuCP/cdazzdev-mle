"""Offline verification for the Pydantic models (no network)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import config
from src.schemas import AnomalyAssessment, AnomalyInput, count_sentences

BASE_INPUT = {
    "product_category": "electronics",
    "region": "Europe",
    "order_volume_vs_forecast_pct": 5.0,
    "inventory_days_on_hand": 30.0,
    "on_time_delivery_pct": 96.0,
    "supplier_status": "normal",
    "transit_days_normal": 7.0,
    "transit_days_observed": 7.5,
    "port_congestion_index": 35.0,
    "weather_event": "none",
    "labor_document_note": "none",
    "unit_cost_vs_last_quarter_pct": 1.2,
    "notes": "Steady week with no disruptions reported.",
}

def _assessment(**overrides):
    payload = {
        "is_anomaly": False,
        "anomaly_class": config.NONE_CLASS,
        "severity": 1,
        "root_cause": "All metrics sit inside their normal ranges. No disruption is indicated by the feed.",
        "evidence_fields": [],
        "corrective_actions": [
            "Continue routine monitoring of inbound schedules.",
            "Keep safety stock at the current level.",
            "Review next week's forecast with planning.",
        ],
        "confidence": 0.85,
    }
    payload.update(overrides)
    return AnomalyAssessment.model_validate(payload)

def test_valid_normal_assessment_passes():
    assessment = _assessment()
    assert assessment.anomaly_class == config.NONE_CLASS
    assert assessment.is_anomaly is False

def test_severity_bounds():
    with pytest.raises(ValidationError):
        _assessment(severity=0)
    with pytest.raises(ValidationError):
        _assessment(severity=6)

def test_confidence_bounds():
    with pytest.raises(ValidationError):
        _assessment(confidence=1.4)
    with pytest.raises(ValidationError):
        _assessment(confidence=-0.2)

def test_taxonomy_enum_enforced():
    with pytest.raises(ValidationError):
        _assessment(is_anomaly=True, anomaly_class="planet_alignment", severity=3,
                    evidence_fields=["port_congestion_index"])

def test_root_cause_sentence_range():
    two = "First sentence here. Second sentence follows it."
    assert 2 <= count_sentences(two) <= 4
    with pytest.raises(ValidationError):
        _assessment(root_cause="Only one sentence about the feed state.")
    five = ("One. Two. Three. Four. Five.")
    with pytest.raises(ValidationError):
        _assessment(root_cause=five)

def test_actions_count_range():
    with pytest.raises(ValidationError):
        _assessment(corrective_actions=["Only one action."])
    with pytest.raises(ValidationError):
        _assessment(corrective_actions=[f"Action number {i}" for i in range(10)])

def test_consistency_between_flag_and_class():
    with pytest.raises(ValidationError):
        _assessment(is_anomaly=True, anomaly_class=config.NONE_CLASS)
    with pytest.raises(ValidationError):
        _assessment(is_anomaly=False, anomaly_class="labor_strike")

def test_anomaly_requires_evidence():
    with pytest.raises(ValidationError):
        _assessment(is_anomaly=True, anomaly_class="demand_spike", severity=4)

def test_input_numeric_coercion_and_unknown_category():
    coerced = AnomalyInput.model_validate({**BASE_INPUT, "transit_days_observed": "9"})
    assert coerced.transit_days_observed == 9.0
    with pytest.raises(ValidationError):
        AnomalyInput.model_validate({**BASE_INPUT, "product_category": "spaceships"})

def test_sentence_counter_edge_cases():
    assert count_sentences("Revenue grew 1.5 times. The U.S. plant slowed. Costs held.") == 3
    assert count_sentences("One sentence only") == 1
