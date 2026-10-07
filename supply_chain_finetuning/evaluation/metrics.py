"""Metrics: ROUGE-L, BERTScore, and deterministic programmatic checks.

All metrics run on the SAME held-out test set for both arms, on the
canonicalized text form produced by ``normalize.normalize_output``.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict
from typing import Any, Dict, List

from rouge_score import rouge_scorer

from src import config
from src.schemas import AnomalyAssessment

_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure (stemmed) between two canonical texts."""
    return _SCORER.score(reference, prediction)["rougeL"].fmeasure

def bertscore_f1(predictions: List[str], references: List[str]) -> List[float]:
    """BERTScore F1 (default English model). Imported lazily: heavy download."""
    from bert_score import score

    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return [round(value, 4) for value in f1.tolist()]

def ground_guard_violations(assessment: Dict[str, Any], input_feed: Dict[str, Any]) -> List[str]:
    """Claims in the root cause that the input cannot back (hallucination signal).

    Example: a root cause mentioning a strike while ``labor_document_note`` is
    empty, or a storm while ``weather_event`` is "none". The keyword map lives
    in ``config.GROUND_GUARD_RULES``.
    """
    root_cause = str(assessment.get("root_cause", "")).lower()
    violations: List[str] = []
    for rule in config.GROUND_GUARD_RULES:
        field_value = str(input_feed.get(rule["field"], "")).strip().lower()
        if field_value in config.BLANK_FIELD_VALUES:
            for keyword in rule["keywords"]:
                if keyword in root_cause:
                    violations.append(f"{keyword!r} claimed but {rule['field']} is empty")
                    break
    return violations

def programmatic_check(
    normalized: Dict[str, Any], gold: Dict[str, Any], input_feed: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministic checks for one output (no LLM involved)."""
    schema_valid = bool(normalized["schema_valid"])
    parsed = normalized.get("parsed")
    result: Dict[str, Any] = {
        "tuple_id": normalized["tuple_id"],
        "schema_valid": schema_valid,
        "class_exact": False,
        "severity_within_1": False,
        "evidence_subset": False,
        "ground_guard_violations": [],
    }
    if not schema_valid:
        return result
    result["class_exact"] = parsed["anomaly_class"] == gold["anomaly_class"]
    result["severity_within_1"] = abs(int(parsed["severity"]) - int(gold["severity"])) <= 1
    result["evidence_subset"] = set(parsed["evidence_fields"]).issubset(set(config.INPUT_KEYS))
    result["ground_guard_violations"] = ground_guard_violations(parsed, input_feed)
    return result

def evaluate_arm(
    generations: List[Dict[str, Any]],
    records_by_id: Dict[str, Dict[str, Any]],
    normalize_fn,
    include_bertscore: bool = True,
) -> Dict[str, Any]:
    """Score one arm (base or tuned) on the held-out set.

    ``normalize_fn`` is injected so callers (and tests) can reuse the shared
    normalizer or a stub. Returns per-item rows plus aggregate ROUGE-L /
    BERTScore / check rates.
    """
    rows: List[Dict[str, Any]] = []
    rouge_scores: List[float] = []
    root_cause_scores: List[float] = []
    raw_by_id = {generation["tuple_id"]: generation["raw_output"] for generation in generations}
    for generation in generations:
        record = records_by_id[generation["tuple_id"]]
        gold = record["gold_assessment"]
        gold_canonical = _canonical_gold(gold)
        normalized = asdict(normalize_fn(generation["raw_output"]))
        normalized["tuple_id"] = generation["tuple_id"]
        normalized["arm"] = generation.get("arm", "unknown")

        rouge_item = rouge_cause = None
        if normalized["schema_valid"]:
            gold_assessment = AnomalyAssessment.model_validate(gold)
            rouge_item = rouge_l(normalized["canonical_text"] or "", gold_canonical)
            rouge_cause = rouge_l(
                canonicalize({"root_cause": normalized["parsed"].get("root_cause", "")}),
                canonicalize({"root_cause": gold_assessment.root_cause}),
            )
            rouge_scores.append(rouge_item)
            root_cause_scores.append(rouge_cause)
        rows.append({
            "tuple_id": generation["tuple_id"],
            "rouge_l": rouge_item,
            "rouge_l_root_cause": rouge_cause,
            "programmatic": programmatic_check(normalized, gold, record["input_feed"]),
            "bertscore_f1": None,
        })

    if include_bertscore:
        references = [_canonical_gold(records_by_id[row["tuple_id"]]["gold_assessment"]) for row in rows]
        bert_values = bertscore_f1([raw_by_id[row["tuple_id"]] for row in rows], references)
        for row, value in zip(rows, bert_values):
            row["bertscore_f1"] = value

    def _mean(values: List[Any]) -> Dict[str, Any]:
        clean = [v for v in values if v is not None]
        if not clean:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(statistics.mean(clean), 4),
            "std": round(statistics.pstdev(clean), 4) if len(clean) > 1 else 0.0,
            "n": len(clean),
        }

    return {
        "rows": rows,
        "rouge_l": _mean(rouge_scores),
        "rouge_l_root_cause": _mean(root_cause_scores),
        "bertscore_f1": _mean([row["bertscore_f1"] for row in rows]),
        "checks": {
            "valid_json_pct": _pct([row["programmatic"]["schema_valid"] for row in rows]),
            "class_exact_pct": _pct([row["programmatic"]["class_exact"] for row in rows]),
            "severity_within_1_pct": _pct([row["programmatic"]["severity_within_1"] for row in rows]),
            "evidence_subset_pct": _pct([row["programmatic"]["evidence_subset"] for row in rows]),
            "ground_guard_violation_count": sum(
                len(row["programmatic"]["ground_guard_violations"]) for row in rows
            ),
        },
    }

def _pct(flags: List[bool]) -> float:
    return round(100.0 * sum(1 for flag in flags if flag) / len(flags), 1) if flags else 0.0

def _canonical_gold(gold: Dict[str, Any]) -> str:
    from evaluation.normalize import canonicalize  # sibling package, avoids a cycle

    assessment = AnomalyAssessment.model_validate(gold)
    return canonicalize(assessment.model_dump())

def canonicalize(obj: Dict[str, Any]) -> str:
    """Re-export of the shared canonicalizer (kept here for import ergonomics)."""
    from evaluation.normalize import canonicalize as _canonicalize

    return _canonicalize(obj)
