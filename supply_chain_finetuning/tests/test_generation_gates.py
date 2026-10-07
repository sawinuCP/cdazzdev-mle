"""Offline verification for the acceptance gates and resumability (mocked teacher)."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src import config
from src.scenario_matrix import ScenarioTuple
from src.schemas import AnomalyAssessment, AnomalyInput
from scripts.generate_dataset import GenerationRunner, gate_check

TUPLE = ScenarioTuple(
    family="port_logistics_congestion",
    anomaly_class="port_logistics_congestion",
    product_category="electronics",
    region="Asia-Pacific",
    severity_level="severe",
    severity_band=(3, 4),
    is_control=False,
)

FEED = {
    "product_category": "electronics",
    "region": "Asia-Pacific",
    "order_volume_vs_forecast_pct": 2.0,
    "inventory_days_on_hand": 12.0,
    "on_time_delivery_pct": 71.0,
    "supplier_status": "delayed",
    "transit_days_normal": 6.0,
    "transit_days_observed": 14.0,
    "port_congestion_index": 88.0,
    "weather_event": "none",
    "labor_document_note": "none",
    "unit_cost_vs_last_quarter_pct": 3.1,
    "notes": "Container dwell time at the hub port keeps climbing this week.",
}

GOLD = {
    "is_anomaly": True,
    "anomaly_class": "port_logistics_congestion",
    "severity": 4,
    "root_cause": "Observed transit time is more than double the normal window. The port congestion index near 90 confirms the delay originates at the gateway port.",
    "evidence_fields": ["transit_days_observed", "transit_days_normal", "port_congestion_index"],
    "corrective_actions": ["Reroute via the secondary port.", "Book expedited drayage capacity.", "Notify downstream customers."],
    "confidence": 0.92,
}

def _gate(gold_overrides=None, feed=None, accepted=None, family_counts=None, target_total=150):
    input_feed = AnomalyInput.model_validate(feed or FEED)
    gold = AnomalyAssessment.model_validate({**GOLD, **(gold_overrides or {})})
    return gate_check(
        input_feed, gold, TUPLE, accepted or [], family_counts or {}, target_total,
    )

def test_gate_accepts_a_good_sample():
    ok, reason = _gate()
    assert ok and reason == "ok"

def test_gate_rejects_wrong_class():
    ok, reason = _gate({"anomaly_class": "weather_disruption"})
    assert not ok and "class" in reason

def test_gate_rejects_severity_outside_band():
    ok, reason = _gate({"severity": 2})  # band for "severe" is 3-4
    assert not ok and "severity" in reason

def test_gate_rejects_unknown_evidence_field():
    ok, reason = _gate({"evidence_fields": ["hurricane_magnitude"]})
    assert not ok and "outside the input schema" in reason

def test_empty_evidence_for_anomaly_is_rejected_by_the_schema():
    """The schema itself blocks the empty-evidence case before the gate runs.

    The gate keeps its own check as defence in depth for raw-dict paths.
    """
    with pytest.raises(ValidationError):
        _gate({"evidence_fields": []})

def test_gate_rejects_near_duplicate():
    ok, reason = _gate(accepted=[json.dumps(FEED, sort_keys=True)])
    assert not ok and "near-duplicate" in reason

def test_gate_enforces_family_share_cap():
    # with a 150-sample target the per-family cap is 150 * 0.15 = 22.5 samples
    ok, reason = _gate(family_counts={"port_logistics_congestion": 22}, target_total=150)
    assert not ok and "family share" in reason
    ok_again, _ = _gate(family_counts={"port_logistics_congestion": 21}, target_total=150)
    assert ok_again

class FakeTeacher:
    """Scripted teacher: returns queued payloads, one per chat_json call."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat_json(self, messages, model_cls, max_tokens=None, salt=""):
        self.calls.append(salt)
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return model_cls.model_validate(item)

def _runner(tmp_path, payloads):
    from src.llm_client import LLMSettings

    fake = FakeTeacher(payloads)
    fake.settings = LLMSettings("http://fake", "key", "fake-model", 1.0, 0.7)
    runner = GenerationRunner(client=fake, out_path=tmp_path / "raw.jsonl")
    return runner, fake

def test_runner_generates_and_stops_at_target(tmp_path):
    runner, fake = _runner(tmp_path, [dict(input_feed=FEED, gold_assessment=GOLD)])
    summary = runner.run([TUPLE], target=1)
    assert summary["accepted"] == 1
    assert summary["per_family"]["port_logistics_congestion"] == 1
    assert fake.calls == ["0|port_logistics_congestion|electronics|Asia-Pacific|severe|1"]
    lines = (tmp_path / "raw.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["tuple_id"] == TUPLE.tuple_id

def test_runner_is_resumable_and_skips_completed_tuples(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps({
            "tuple_id": TUPLE.tuple_id, "family": TUPLE.family,
            "input_feed": FEED, "gold_assessment": GOLD,
        }) + "\n",
        encoding="utf-8",
    )

    other = ScenarioTuple(
        family="labor_strike", anomaly_class="labor_strike",
        product_category="apparel", region="Europe",
        severity_level="mild", severity_band=(1, 2), is_control=False,
    )
    strike_gold = {
        "is_anomaly": True, "anomaly_class": "labor_strike", "severity": 2,
        "root_cause": "The labour note documents a dock worker walkout. Throughput is consequently reduced.",
        "evidence_fields": ["labor_document_note", "on_time_delivery_pct"],
        "corrective_actions": [
            "Activate temp labour agency.",
            "Re-sequence shipments by margin.",
            "Alert downstream customers.",
        ],
        "confidence": 0.8,
    }
    strike_feed = {
        "product_category": "apparel",
        "region": "Europe",
        "order_volume_vs_forecast_pct": 96.0,
        "inventory_days_on_hand": 41.0,
        "on_time_delivery_pct": 48.0,
        "supplier_status": "partial",
        "transit_days_normal": 4.0,
        "transit_days_observed": 5.0,
        "port_congestion_index": 22.0,
        "weather_event": "none",
        "labor_document_note": "Dock workers walkout entered day 3",
        "unit_cost_vs_last_quarter_pct": 0.4,
        "notes": "Union talks stalled overnight and picketing started at gate two.",
    }
    runner, fake = _runner(tmp_path, [dict(input_feed=strike_feed, gold_assessment=strike_gold)])
    accepted_before = runner.load_completed()
    assert accepted_before == 1

    summary = runner.run([TUPLE, other], target=2)
    assert summary["accepted"] == 2
    assert fake.calls == ["0|" + other.tuple_id + "|1"]  # the completed tuple was skipped

def test_runner_retries_after_failure_with_new_salt(tmp_path):
    runner, fake = _runner(tmp_path, [
        RuntimeError("transient gateway error"),
        dict(input_feed=FEED, gold_assessment=GOLD),
    ])
    summary = runner.run([TUPLE], target=1)
    assert summary["accepted"] == 1
    assert len(fake.calls) == 2
    assert fake.calls[0].endswith("|1") and fake.calls[1].endswith("|2")

def test_runner_abandons_after_max_attempts(tmp_path):
    runner, fake = _runner(tmp_path, [RuntimeError("down")] * 6)
    summary = runner.run([TUPLE], target=1)
    assert summary["accepted"] == 0
    assert summary["abandoned"] == [TUPLE.tuple_id]
