"""Manual hallucination audit: template generation + rate calculation.

The labels are filled by a HUMAN reviewer (the template is deliberately empty);
this module only prepares the CSV and computes the rate once labels exist.
Allowed labels: correct | partially_correct | hallucinated.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config

ALLOWED_LABELS = ("correct", "partially_correct", "hallucinated")
COLUMNS = ["id", "input", "gold", "tuned_output", "label", "note"]

def make_template(
    tuned_generations: List[Dict[str, Any]],
    records_by_id: Dict[str, Dict[str, Any]],
    path: Path = config.AUDIT_TEMPLATE_CSV,
) -> Path:
    """Write the audit CSV for all tuned outputs; label/note left empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for generation in tuned_generations:
            record = records_by_id[generation["tuple_id"]]
            writer.writerow({
                "id": generation["tuple_id"],
                "input": generation.get("input_text")
                         or json_compact(record["input_feed"]),
                "gold": json_compact(record["gold_assessment"]),
                "tuned_output": generation["raw_output"],
                "label": "",
                "note": "",
            })
    return path

def json_compact(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def compute_rate(path: Path = config.AUDIT_TEMPLATE_CSV) -> Optional[Dict[str, Any]]:
    """Read human labels and compute the hallucination rate.

    Returns ``None`` (after printing a status line) while labels are pending.
    """
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["label"].strip() for row in rows}
    if labels <= {""}:
        print("manual labels pending: fill the 'label' column in "
              f"{path} (values: {', '.join(ALLOWED_LABELS)}) and re-run.")
        return None

    unknown = labels - set(ALLOWED_LABELS) - {""}
    if unknown:
        raise ValueError(f"invalid labels present in {path}: {sorted(unknown)}")
    labelled = [row for row in rows if row["label"].strip()]
    counts = {name: 0 for name in ALLOWED_LABELS}
    for row in labelled:
        counts[row["label"].strip()] += 1
    reviewed = len(labelled)
    hallucinated = counts["hallucinated"]
    rate = round(100.0 * hallucinated / reviewed, 1) if reviewed else None
    print(f"manual audit matrix (n={reviewed}/{len(rows)} rows): {counts}")
    print(f"hallucination rate: {hallucinated}/{reviewed} x 100 = {rate}%")
    return {"reviewed": reviewed, "counts": counts, "hallucination_rate_pct": rate}
