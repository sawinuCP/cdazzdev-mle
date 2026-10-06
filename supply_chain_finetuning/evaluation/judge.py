# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'blinded LLM-as-judge with schema-validated scoring, shuffling and consistency gauge', Date: 2026-10-06
"""Blinded LLM-as-judge.

Design choices that keep the comparison honest:
- The judge sees the INPUT and ONE output at a time; it never sees the gold.
- Outputs from both arms are shuffled into random slots with arm labels hidden,
  then un-shuffled for reporting.
- A different judge model than the teacher is strongly preferred; when the two
  are identical a visible self-preference warning is printed.
- Temperature 0 and a response cache make re-runs reproducible; a consistency
  gauge re-scores three outputs and reports per-dimension differences.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, model_validator

from src import config, prompts
from src.llm_client import LLMClient, LLMSettings

JUDGE_DIMENSIONS = (
    "schema_adherence",
    "class_correct",
    "severity_calibration",
    "root_cause_causality",
    "action_quality",
    "evidence_grounding",
)


class JudgeResult(BaseModel):
    """Validated judge response (six dimensions 0-5 + total 0-30)."""

    schema_adherence: int = Field(ge=0, le=5)
    class_correct: int = Field(ge=0, le=5)
    severity_calibration: int = Field(ge=0, le=5)
    root_cause_causality: int = Field(ge=0, le=5)
    action_quality: int = Field(ge=0, le=5)
    evidence_grounding: int = Field(ge=0, le=5)
    total: int = Field(ge=0, le=30)
    one_line_reason: str = ""

    @model_validator(mode="after")
    def _total_is_sum(self) -> "JudgeResult":
        dimensions = [getattr(self, name) for name in JUDGE_DIMENSIONS]
        if self.total != sum(dimensions):
            raise ValueError(f"total {self.total} != sum of dimensions {sum(dimensions)}")
        return self


def warn_if_judge_is_teacher() -> bool:
    """Visible self-preference warning when judge and teacher are the same model."""
    judge = os.environ.get(config.ENV_JUDGE_MODEL, "").strip()
    teacher = os.environ.get(config.ENV_TEACHER_MODEL, "").strip()
    if judge and teacher and judge.lower() == teacher.lower():
        print(
            "[warn] the judge model is identical to the teacher model "
            f"({judge}); self-preference bias is possible. Prefer a different "
            "judge (e.g. Groq llama-3.3-70b-versatile) via JUDGE_MODEL."
        )
        return True
    return False


def judge_messages(input_feed: Dict[str, Any], output_json_text: str) -> List[Dict[str, str]]:
    import json

    return [
        {"role": "system", "content": prompts.JUDGE_SYSTEM},
        {
            "role": "user",
            "content": prompts.JUDGE_USER.format(
                input_json=json.dumps(input_feed, indent=2, ensure_ascii=False),
                output_json=output_json_text,
            ),
        },
    ]


def judge_item(judge: LLMClient, input_feed: Dict[str, Any], output_raw: str,
               salt: str = "") -> JudgeResult:
    """Score one output. The raw text is judged (schema failures get low scores)."""
    import json

    return judge.chat_json(
        judge_messages(input_feed, output_raw),
        JudgeResult,
        max_tokens=config.JUDGE_MAX_TOKENS,
        salt=salt,
    )


def blind(items: List[Dict[str, Any]], seed: int = config.SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shuffle items into anonymous slots; returns (blinded, slot->truth map)."""
    rng = random.Random(seed)
    truths = [{"slot": index, "tuple_id": item["tuple_id"], "arm": item["arm"]}
              for index, item in enumerate(items)]
    rng.shuffle(truths)
    blinded = [
        {"slot": mapping["slot"], "tuple_id": item["tuple_id"],
         "input_feed": item["input_feed"], "raw_output": item["raw_output"]}
        for mapping, item in zip(truths, items)
    ]
    return blinded, truths


def unblind(scored: List[Dict[str, Any]], truths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Restore arm labels and aggregate per-arm, per-dimension scores."""
    by_slot = {item["slot"]: item for item in scored}
    restored = []
    for mapping in truths:
        item = by_slot[mapping["slot"]]
        restored.append({**item, "tuple_id": mapping["tuple_id"], "arm": mapping["arm"]})
    summary: Dict[str, Any] = {}
    for arm in ("base", "tuned"):
        arm_items = [item for item in restored if item["arm"] == arm and "result" in item]
        if not arm_items:
            continue
        per_dimension = {
            name: round(sum(getattr(item["result"], name) for item in arm_items) / len(arm_items), 3)
            for name in JUDGE_DIMENSIONS
        }
        summary[arm] = {
            "n": len(arm_items),
            "mean_total": round(sum(item["result"].total for item in arm_items) / len(arm_items), 3),
            "per_dimension": per_dimension,
        }
    return {"items": restored, "summary": summary}


def consistency_gauge(judge: LLMClient, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-score up to three outputs; report per-dimension differences."""
    gauge: List[Dict[str, Any]] = []
    for item in items[:3]:
        first = judge_item(judge, item["input_feed"], item["raw_output"], salt=item["slot"] + "|first")
        second = judge_item(judge, item["input_feed"], item["raw_output"], salt=item["slot"] + "|second")
        diffs = {
            name: abs(getattr(first, name) - getattr(second, name))
            for name in JUDGE_DIMENSIONS
        }
        gauge.append({
            "slot": item["slot"],
            "diffs": diffs,
            "max_diff": max(diffs.values()),
            "within_tolerance": max(diffs.values()) <= 1,
        })
    return gauge
