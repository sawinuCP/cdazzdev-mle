"""Build chat-format JSONL and the exact stratified 120/15/15 split.

Each line is ``{"messages": [system, user, assistant]}``. The gold assistant
content is canonicalised (sorted keys, compact separators) so the SFT target
and the metric comparisons share one canonical text form. The system turn is
the SAME text later given to the base model, keeping the comparison fair.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.prompts import STUDENT_SYSTEM  # noqa: E402

def canonical_json(obj: Dict[str, Any]) -> str:
    """Canonical JSON text: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def build_example(record: Dict[str, Any]) -> Dict[str, Any]:
    """One chat-format training example."""
    return {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": canonical_json(record["input_feed"])},
            {"role": "assistant", "content": canonical_json(record["gold_assessment"])},
        ],
        "meta": {
            "tuple_id": record["tuple_id"],
            "family": record["family"],
            "anomaly_class": record["anomaly_class"],
        },
    }

def stratified_split(
    records: List[Dict[str, Any]],
    sizes: Dict[str, int] = config.SPLIT_SIZES,
    seed: int = config.SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by family (round-robin over families per split), deterministically.

    With 12 families and 15-per-split quotas every family lands in every split
    at least once, so no class is unseen in validation or testing.
    """
    if len(records) != sum(sizes.values()):
        raise ValueError(
            f"expected exactly {sum(sizes.values())} records, got {len(records)}"
        )
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for example in records:
        family = example["meta"]["family"]
        by_family.setdefault(family, []).append(example)

    rng = random.Random(seed)
    queues = {
        family: rng.sample(members, len(members))
        for family, members in by_family.items()
    }
    family_names = sorted(queues)
    splits: Dict[str, List[Dict[str, Any]]] = {name: [] for name in sizes}
    for split_name in ("test", "valid"):  # smallest quotas first
        while len(splits[split_name]) < sizes[split_name]:
            progressed = False
            for family in family_names:
                if len(splits[split_name]) >= sizes[split_name]:
                    break
                if queues[family]:
                    splits[split_name].append(queues[family].pop())
                    progressed = True
            if not progressed:
                raise ValueError("not enough records to fill the stratified split")
    for family in family_names:
        splits["train"].extend(queues[family])
    if len(splits["train"]) != sizes["train"]:
        raise ValueError("train split did not fill exactly")
    return splits

def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    config.ensure_dirs()
    records = []
    with open(config.RAW_JSONL, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    expected = sum(config.SPLIT_SIZES.values())
    if len(records) != expected:
        raise SystemExit(
            f"[error] raw.jsonl holds {len(records)} records; the split requires exactly {expected}. "
            "Finish generation first (re-run generate_dataset.py to resume)."
        )

    examples = [build_example(record) for record in records]
    splits = stratified_split(examples, config.SPLIT_SIZES, config.SEED)
    write_jsonl(config.TRAIN_JSONL, splits["train"])
    write_jsonl(config.VALID_JSONL, splits["valid"])
    write_jsonl(config.TEST_JSONL, splits["test"])

    print(f"split sizes: train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")
    print("\nper-family distribution:")
    table: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "valid", "test"):
        for example in splits[split_name]:
            family = example["meta"]["family"]
            table.setdefault(family, {"train": 0, "valid": 0, "test": 0})[split_name] += 1
    for family, row in sorted(table.items()):
        print(f"  {family:32s} train={row['train']:3d} valid={row['valid']:2d} test={row['test']:2d}")
    ids = [e["meta"]["tuple_id"] for split in splits.values() for e in split]
    if len(ids) != len(set(ids)):
        raise SystemExit("[error] leakage detected: a tuple appears in more than one split")
    print("no leakage: every tuple appears in exactly one split")

if __name__ == "__main__":
    main()
