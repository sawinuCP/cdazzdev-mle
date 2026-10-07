"""Teacher-based dataset generation with acceptance gates (resumable).

Gates (a rejected sample is regenerated, up to ``MAX_ATTEMPTS_PER_TUPLE``):
 1. Pydantic validation of the feed and the gold assessment.
 2. evidence_fields subset of the input keys, non-empty for anomalies.
 3. anomaly_class equals the scenario family's required class.
 4. severity within the tuple's band.
 5. root_cause 2-4 sentences and 3-5 corrective actions (schema-enforced too).
 6. near-duplicate rejection vs every accepted input (rapidfuzz token_set_ratio).
 7. running family-share cap.

Accepted samples are appended to ``data/raw.jsonl`` immediately (restart-safe)
and each request is usage-logged. The teacher system prompt is written verbatim
to ``data/teacher_system_prompt.txt``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, prompts  # noqa: E402
from src.llm_client import LLMError, LLMValidationError, client_from_env  # noqa: E402
from src.schemas import AnomalyAssessment, AnomalyInput, TeacherResponse  # noqa: E402
from src.scenario_matrix import ScenarioTuple, sample_tuples  # noqa: E402

_LOGGER = logging.getLogger("supply_chain.generate")

def build_teacher_messages(tup: ScenarioTuple) -> List[Dict[str, str]]:
    """Render the teacher prompt for one scenario tuple."""
    anomaly_instruction = (
        "This is a CONTROL tuple: produce a completely normal feed "
        f"(is_anomaly=false, anomaly_class={config.NONE_CLASS!r}, severity=1) with only "
        "benign fluctuations."
        if tup.is_control
        else "Produce a clear, realistic case of this anomaly class whose numbers tell the story."
    )
    return [
        {"role": "system", "content": prompts.TEACHER_SYSTEM},
        {
            "role": "user",
            "content": prompts.TEACHER_USER.format(
                family=tup.family,
                anomaly_class=tup.anomaly_class,
                product_category=tup.product_category,
                region=tup.region,
                severity_level=tup.severity_level,
                severity_low=tup.severity_band[0],
                severity_high=tup.severity_band[1],
                anomaly_instruction=anomaly_instruction,
            ),
        },
    ]

def gate_check(
    input_feed: AnomalyInput,
    gold: AnomalyAssessment,
    tup: ScenarioTuple,
    accepted_input_strings: List[str],
    family_counts: Dict[str, int],
    target_total: int,
) -> Tuple[bool, str]:
    """Pure acceptance gates 2-7 (gate 1 is Pydantic validation itself).

    The family-share cap is projected against the FINAL dataset size
    (``target_total``), not the current count: early in the run every family is
    at 1-of-N, and a share computed against the running total would wrongly
    reject the first sample of every family.
    """
    evidence = set(gold.evidence_fields)
    if gold.is_anomaly:
        if not evidence:
            return False, "gate: empty evidence_fields for an anomaly"
        unknown = evidence - set(config.INPUT_KEYS)
        if unknown:
            return False, f"gate: evidence fields outside the input schema: {sorted(unknown)}"
    if gold.anomaly_class != tup.anomaly_class:
        return False, f"gate: class {gold.anomaly_class!r} != required {tup.anomaly_class!r}"
    low, high = tup.severity_band
    if not low <= gold.severity <= high:
        return False, f"gate: severity {gold.severity} outside band {low}-{high}"

    serialized = json.dumps(input_feed.model_dump(), sort_keys=True, ensure_ascii=False)
    for previous in accepted_input_strings:
        if fuzz.token_set_ratio(serialized, previous) > config.NEAR_DUPLICATE_THRESHOLD:
            return False, "gate: near-duplicate of an accepted sample"

    family_count = family_counts.get(tup.family, 0)
    if family_count + 1 > config.MAX_FAMILY_SHARE * target_total:
        return False, "gate: family share cap reached"
    return True, "ok"

def load_completed(raw_path: Path) -> Dict[str, int]:
    """Count already-accepted samples per tuple id (restart safety)."""
    completed: Dict[str, int] = {}
    if not raw_path.exists():
        return completed
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn trailing line must not kill a resumed run
            tuple_id = record.get("tuple_id")
            if tuple_id:
                completed[tuple_id] = completed.get(tuple_id, 0) + 1
    return completed

class GenerationRunner:
    """Drives tuple-by-tuple generation with gates; restart-safe by design.

    ``nonce`` scopes the response cache per invocation: re-running the script
    after a partial pass generates FRESH samples for abandoned tuples instead
    of replaying the cached responses that were already rejected.
    """

    def __init__(
        self, client: Any, out_path: Path,
        target_total: int = config.SAMPLE_TARGET, nonce: str = "0",
        state_paths: Optional[List[Path]] = None,
    ) -> None:
        self.client = client
        self.out_path = Path(out_path)
        # Acceptance state may be shared across concurrent shard processes:
        # every shard re-reads ALL state files before each tuple, so the
        # near-duplicate and family-share gates stay correct globally.
        self.state_paths = [Path(p) for p in (state_paths or [out_path])]
        self.target_total = target_total
        self.nonce = nonce
        self.accepted: List[Dict[str, Any]] = []
        self.abandoned: List[str] = []
        self.rejected = 0
        self.accepted_input_strings: List[str] = []
        self.family_counts: Dict[str, int] = {}

    def refresh_state(self) -> int:
        """Rebuild acceptance state from every state file (shard coordination)."""
        self.accepted = []
        self.accepted_input_strings = []
        self.family_counts = {}
        seen_ids = set()
        for path in self.state_paths:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tuple_id = record.get("tuple_id")
                    if not tuple_id or tuple_id in seen_ids:
                        continue
                    seen_ids.add(tuple_id)
                    self.accepted.append(record)
                    self.accepted_input_strings.append(
                        json.dumps(record.get("input_feed", {}), sort_keys=True, ensure_ascii=False)
                    )
                    family = record.get("family", "?")
                    self.family_counts[family] = self.family_counts.get(family, 0) + 1
        return len(self.accepted)

    def load_completed(self) -> int:
        """Backwards-compatible alias for :meth:`refresh_state`."""
        total = self.refresh_state()
        _LOGGER.info("resuming: %d accepted samples already on disk", total)
        return total

    def _attempt(self, tup: ScenarioTuple, attempt: int) -> Optional[Dict[str, Any]]:
        """One teacher call plus gates; returns the record or None (rejected)."""
        messages = build_teacher_messages(tup)
        response = self.client.chat_json(
            messages, TeacherResponse,
            max_tokens=config.LLM_MAX_TOKENS,
            salt=f"{self.nonce}|{tup.tuple_id}|{attempt}",
        )
        ok, reason = gate_check(
            response.input_feed, response.gold_assessment, tup,
            self.accepted_input_strings, self.family_counts, self.target_total,
        )
        if not ok:
            self.rejected += 1
            _LOGGER.info("reject %s (attempt %d): %s", tup.tuple_id, attempt, reason)
            return None
        record = {
            "tuple_id": tup.tuple_id,
            "family": tup.family,
            "anomaly_class": tup.anomaly_class,
            "product_category": tup.product_category,
            "region": tup.region,
            "severity_level": tup.severity_level,
            "severity_band": list(tup.severity_band),
            "input_feed": response.input_feed.model_dump(),
            "gold_assessment": response.gold_assessment.model_dump(),
        }
        with open(self.out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.accepted.append(record)
        self.accepted_input_strings.append(
            json.dumps(response.input_feed.model_dump(), sort_keys=True, ensure_ascii=False)
        )
        self.family_counts[tup.family] = self.family_counts.get(tup.family, 0) + 1
        _LOGGER.info("accepted %s (attempt %d) | total %d", tup.tuple_id, attempt, len(self.accepted))
        return record

    def run(self, tuples: List[ScenarioTuple], target: int) -> Dict[str, Any]:
        """Generate until ``target`` accepted samples exist; returns a summary."""
        for tup in tuples:
            self.refresh_state()  # shard coordination: state may change under us
            if len(self.accepted) >= target:
                break
            done_ids = {record.get("tuple_id") for record in self.accepted}
            if tup.tuple_id in done_ids:
                continue  # restart safety: skip tuples that already produced a sample
            accepted_here = False
            for attempt in range(1, config.MAX_ATTEMPTS_PER_TUPLE + 1):
                try:
                    if self._attempt(tup, attempt) is not None:
                        accepted_here = True
                        break
                except Exception as exc:  # noqa: BLE001 - an unattended run must survive
                    _LOGGER.warning(
                        "attempt failed %s (attempt %d): %s: %s",
                        tup.tuple_id, attempt, type(exc).__name__, exc,
                    )
            if not accepted_here:
                self.abandoned.append(tup.tuple_id)
                _LOGGER.warning("tuple abandoned after max attempts: %s", tup.tuple_id)
        per_family = dict(sorted(self.family_counts.items()))
        return {
            "accepted": len(self.accepted),
            "rejected": self.rejected,
            "abandoned": self.abandoned,
            "per_family": per_family,
        }

def write_teacher_prompt(path: Path) -> None:
    """Persist the teacher system prompt verbatim (provenance requirement)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompts.TEACHER_SYSTEM, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-based dataset generation")
    parser.add_argument("--target", type=int, default=config.SAMPLE_TARGET)
    parser.add_argument("--overgenerate", type=int, default=config.SAMPLE_OVERGENERATE)
    parser.add_argument(
        "--nonce", type=str, default=None,
        help="cache-scope key; defaults to a timestamp so every invocation "
             "generates fresh samples for previously abandoned tuples",
    )
    parser.add_argument("--shard-index", type=int, default=0,
                        help="0-based index of this worker (for parallel runs)")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="number of parallel workers partitioning the tuples")
    parser.add_argument("--out", type=str, default=None,
                        help="output JSONL for this worker (defaults to data/raw.jsonl; "
                             "sharded runs write data/raw_shard<K>.jsonl)")
    args = parser.parse_args()
    nonce = args.nonce or str(int(time.time())) + f"|s{args.shard_index}"

    out_path = Path(args.out) if args.out else (
        config.DATA_DIR / f"raw_shard{args.shard_index}.jsonl"
        if args.shard_count > 1 else config.RAW_JSONL
    )
    state_paths = [out_path, config.RAW_JSONL] + [
        config.DATA_DIR / f"raw_shard{k}.jsonl" for k in range(args.shard_count)
    ]
    state_paths = sorted(set(state_paths))  # every worker sees every accepted sample

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config.ensure_dirs()
    write_teacher_prompt(config.TEACHER_PROMPT_TXT)
    _LOGGER.info("teacher system prompt written to %s", config.TEACHER_PROMPT_TXT)

    client = client_from_env(
        prefix=config.ENV_TEACHER_MODEL.removesuffix("_MODEL"),
        temperature=config.TEACHER_TEMPERATURE,
        cache_path=config.TEACHER_CACHE_JSON,
        usage_log_path=config.TEACHER_USAGE_CSV,
        disable_thinking=config.DISABLE_THINKING,
    )
    runner = GenerationRunner(
        client=client, out_path=out_path, nonce=nonce,
        target_total=args.target, state_paths=state_paths,
    )
    runner.load_completed()
    if len(runner.accepted) >= args.target:
        _LOGGER.info("target already reached (%d accepted); nothing to do", len(runner.accepted))
        return

    all_tuples = sample_tuples(args.overgenerate, seed=config.SEED)
    tuples = all_tuples[args.shard_index::args.shard_count]  # shard partition
    _LOGGER.info(
        "worker %d/%d: %d tuples in scope, %d accepted globally",
        args.shard_index, args.shard_count, len(tuples), len(runner.accepted),
    )
    summary = runner.run(tuples, target=args.target)
    _LOGGER.info("generation summary: %s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["accepted"] < args.target:
        print(
            f"[warn] only {summary['accepted']}/{args.target} accepted; "
            f"{len(summary['abandoned'])} tuples abandoned - re-run to resume."
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
