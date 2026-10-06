# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'JSONL builder verification: roles, exact split sizes, stratification, no leakage', Date: 2026-10-06
"""Offline verification for the JSONL builder (synthetic records, no network)."""
from __future__ import annotations

import json

import pytest

from src import config
from scripts.build_jsonl import build_example, canonical_json, stratified_split

FAMILIES = list(config.SCENARIO_FAMILIES)  # 12


def _synthetic_record(index: int) -> dict:
    family = FAMILIES[index % len(FAMILIES)]
    anomaly_class = config.FAMILY_TO_CLASS[family]
    is_anomaly = anomaly_class != config.NONE_CLASS
    severity = 1 if not is_anomaly else (index % 5) + 1
    return {
        "tuple_id": f"tuple-{index:03d}",
        "family": family,
        "anomaly_class": anomaly_class,
        "product_category": config.PRODUCT_CATEGORIES[index % len(config.PRODUCT_CATEGORIES)],
        "region": config.REGIONS[index % len(config.REGIONS)],
        "severity_level": "mild",
        "severity_band": [1, 5],
        "input_feed": {
            "product_category": config.PRODUCT_CATEGORIES[index % len(config.PRODUCT_CATEGORIES)],
            "region": config.REGIONS[index % len(config.REGIONS)],
            "order_volume_vs_forecast_pct": float(index),
            "inventory_days_on_hand": 20.0 + index,
            "on_time_delivery_pct": 90.0,
            "supplier_status": "normal",
            "transit_days_normal": 6.0,
            "transit_days_observed": 6.5,
            "port_congestion_index": 30.0,
            "weather_event": "none",
            "labor_document_note": "none",
            "unit_cost_vs_last_quarter_pct": 0.5,
            "notes": f"Synthetic feed number {index}.",
        },
        "gold_assessment": {
            "is_anomaly": is_anomaly,
            "anomaly_class": anomaly_class,
            "severity": severity,
            "root_cause": f"Synthetic root cause number {index}. It spans the required sentence count.",
            "evidence_fields": ["order_volume_vs_forecast_pct"] if is_anomaly else [],
            "corrective_actions": [
                f"Investigate SKU batch {index}.",
                "Adjust safety stock levels.",
                "Align with the planning team.",
            ],
            "confidence": 0.8,
        },
    }


def _records(count: int = 150) -> list:
    return [_synthetic_record(i) for i in range(count)]


def test_canonical_json_sorts_keys_and_compacts():
    text = canonical_json({"b": 1, "a": 2})
    assert text == '{"a":2,"b":1}'


def test_build_example_has_three_roles_with_canonical_content():
    example = build_example(_synthetic_record(0))
    roles = [message["role"] for message in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert example["messages"][0]["content"]  # system prompt present
    user_content = example["messages"][1]["content"]
    assert json.loads(user_content)["notes"] == "Synthetic feed number 0."
    assistant_content = example["messages"][2]["content"]
    assert assistant_content == json.dumps(
        json.loads(assistant_content), sort_keys=True, separators=(",", ":")
    )


def test_split_sizes_are_exact():
    records = _records(150)
    examples = [build_example(record) for record in records]
    splits = stratified_split(examples)
    assert {name: len(items) for name, items in splits.items()} == config.SPLIT_SIZES


def test_split_is_stratified_by_family():
    records = _records(150)
    examples = [build_example(record) for record in records]
    splits = stratified_split(examples)
    for split_name, items in splits.items():
        families = {example["meta"]["family"] for example in items}
        assert families == set(FAMILIES), f"{split_name} missed families"


def test_no_leakage_between_splits():
    records = _records(150)
    examples = [build_example(record) for record in records]
    splits = stratified_split(examples)
    seen = []
    for split_name in ("train", "valid", "test"):
        for example in splits[split_name]:
            seen.append(example["meta"]["tuple_id"])
    assert len(seen) == len(set(seen)) == 150


def test_split_rejects_wrong_record_count():
    with pytest.raises(ValueError):
        stratified_split([build_example(r) for r in _records(10)])
