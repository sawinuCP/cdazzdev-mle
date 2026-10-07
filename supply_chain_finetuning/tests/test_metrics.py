"""Offline verification for the evaluation modules (mocked judge, no network)."""
from __future__ import annotations

import json

import pytest

from src import config
from src.schemas import AnomalyAssessment
from evaluation import judge as judge_module
from evaluation.judge import JudgeResult, blind, unblind
from evaluation.manual_audit import compute_rate, make_template
from evaluation.metrics import evaluate_arm, ground_guard_violations, rouge_l
from evaluation.normalize import normalize_output

VALID_OUTPUT = {
    "is_anomaly": True,
    "anomaly_class": "port_logistics_congestion",
    "severity": 4,
    "root_cause": "Observed transit time doubled versus the normal window. The port congestion index near 90 confirms the gateway port as the source.",
    "evidence_fields": ["transit_days_observed", "transit_days_normal", "port_congestion_index"],
    "corrective_actions": ["Reroute via the secondary port.", "Book expedited drayage.", "Notify customers."],
    "confidence": 0.91,
}

INPUT_FEED = {
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
    "notes": "Container dwell time keeps climbing at the hub port.",
}

def test_normalize_extracts_fenced_json():
    fenced = "```json\n" + json.dumps(VALID_OUTPUT) + "\n```"
    normalized = normalize_output(fenced)
    assert normalized.schema_valid is True
    assert normalized.canonical_text is not None
    assert normalized.parsed["anomaly_class"] == "port_logistics_congestion"

def test_normalize_marks_prose_wrapped_and_broken_outputs():
    noisy = "Here is my answer:\n" + json.dumps(VALID_OUTPUT) + "\nHope that helps!"
    assert normalize_output(noisy).schema_valid is True
    broken = normalize_output("{ this is not json")
    assert broken.schema_valid is False
    assert broken.canonical_text is None
    assert "invalid JSON" in broken.error

def test_normalize_flags_out_of_taxonomy_class():
    invalid = {**VALID_OUTPUT, "anomaly_class": "planet_alignment"}
    normalized = normalize_output(json.dumps(invalid))
    assert normalized.schema_valid is False
    assert "taxonomy" in normalized.error

def test_rouge_l_identical_texts_score_one():
    text = "the quick brown fox jumps over the lazy dog"
    assert rouge_l(text, text) == pytest.approx(1.0)

def test_ground_guard_flags_unbacked_claims():
    violations = ground_guard_violations(
        {"root_cause": "A dock worker strike stopped all outbound flows at the warehouse."},
        INPUT_FEED,  # labor_document_note is "none"
    )
    assert violations and "strike" in violations[0]
    assert ground_guard_violations(VALID_OUTPUT, INPUT_FEED) == []

def test_judge_result_total_must_match_dimensions():
    dimensions = {name: 4 for name in judge_module.JUDGE_DIMENSIONS}
    assert JudgeResult(**dimensions, total=24, one_line_reason="ok").total == 24
    with pytest.raises(Exception):
        JudgeResult(**dimensions, total=30, one_line_reason="inflated")

def test_blinding_hides_arms_and_unblind_restores():
    items = [
        {"tuple_id": f"tuple-{i}", "arm": "base" if i % 2 else "tuned",
         "input_feed": INPUT_FEED, "raw_output": json.dumps(VALID_OUTPUT)}
        for i in range(6)
    ]
    blinded, truths = blind(items, seed=config.SEED)
    assert all("arm" not in item for item in blinded)          # labels hidden
    assert sorted(mapping["slot"] for mapping in truths) == list(range(6))
    scored = [{**item, "result": JudgeResult(**{name: 3 for name in judge_module.JUDGE_DIMENSIONS},
                                             total=18, one_line_reason="r")}
              for item in blinded]
    report = unblind(scored, truths)
    assert set(report["summary"]) == {"base", "tuned"}
    restored_arms = {item["tuple_id"]: item["arm"] for item in report["items"]}
    assert restored_arms == {item["tuple_id"]: item["arm"] for item in items}

def test_evaluate_arm_programmatic_rates():
    records = {
        "tuple-001": {"input_feed": INPUT_FEED, "gold_assessment": VALID_OUTPUT},
        "tuple-002": {"input_feed": INPUT_FEED, "gold_assessment": VALID_OUTPUT},
    }
    generations = [
        {"tuple_id": "tuple-001", "arm": "base", "raw_output": json.dumps(VALID_OUTPUT)},
        {"tuple_id": "tuple-002", "arm": "base", "raw_output": "not json at all"},
    ]
    report = evaluate_arm(generations, records, normalize_output, include_bertscore=False)
    assert report["checks"]["valid_json_pct"] == 50.0
    assert report["checks"]["class_exact_pct"] == 50.0
    assert report["rouge_l"]["n"] == 1  # only the schema-valid item gets ROUGE
    broken_row = next(row for row in report["rows"] if row["tuple_id"] == "tuple-002")
    assert broken_row["rouge_l"] is None

def test_manual_audit_template_and_pending_rate(tmp_path):
    records = {"tuple-001": {"input_feed": INPUT_FEED, "gold_assessment": VALID_OUTPUT}}
    generations = [{"tuple_id": "tuple-001", "arm": "tuned", "raw_output": json.dumps(VALID_OUTPUT)}]
    target = tmp_path / "manual_audit_template.csv"
    written = make_template(generations, records, target)
    assert written.exists()
    content = target.read_text(encoding="utf-8")
    assert content.count(",,") >= 1  # label and note columns are empty
    assert compute_rate(target) is None  # labels pending -> no fabricated rate
