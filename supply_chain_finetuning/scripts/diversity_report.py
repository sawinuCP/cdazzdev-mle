"""Dataset diversity report.

Produces the evidence that the dataset is diverse by construction:
- token-length histograms (student tokenizer; whitespace fallback) with the
  95th-percentile fit check against the training sequence budget,
- keyword/topic frequency (overall + per-family distinguishing terms),
- family x product-category coverage heatmap and the max family share,
- near-duplicate pair count (expected zero after the generation gate).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.prompts import STUDENT_SYSTEM  # noqa: E402

_TOKENIZER_INFO: Dict[str, str] = {}

def load_raw(path: Path) -> List[Dict[str, Any]]:
    """Read the accepted samples from raw.jsonl."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def get_tokenizer():
    """Student tokenizer with a documented whitespace fallback."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.STUDENT_MODEL_ID, trust_remote_code=True
        )
        _TOKENIZER_INFO["backend"] = f"transformers tokenizer for {config.STUDENT_MODEL_ID}"
        return tokenizer
    except Exception as exc:  # noqa: BLE001 - offline machines still need the report
        _TOKENIZER_INFO["backend"] = (
            f"whitespace fallback (tokenizer download failed: {type(exc).__name__})"
        )
        return None

def token_count(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())

def chat_example_text(record: Dict[str, Any]) -> str:
    """Full training example text (system + user + assistant) for length checks."""
    user = json.dumps(record["input_feed"], sort_keys=True, ensure_ascii=False)
    assistant = json.dumps(record["gold_assessment"], sort_keys=True, ensure_ascii=False)
    return f"{STUDENT_SYSTEM}\n{user}\n{assistant}"

def token_length_stats(records: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    lengths = [token_count(chat_example_text(r), tokenizer) for r in records]
    ordered = sorted(lengths)
    p95 = ordered[int(config.TOKEN_FIT_SHARE * (len(ordered) - 1))]
    share_fit = sum(1 for n in lengths if n <= config.MAX_SEQ_LENGTH) / len(lengths)
    return {
        "backend": _TOKENIZER_INFO["backend"],
        "min": min(lengths), "median": ordered[len(ordered) // 2], "max": max(lengths),
        "p95": p95, "share_within_budget": round(share_fit, 4),
        "budget": config.MAX_SEQ_LENGTH,
        "fits": share_fit >= config.TOKEN_FIT_SHARE,
    }

def plot_token_histogram(records: List[Dict[str, Any]], tokenizer, path: Path) -> None:
    lengths = [token_count(chat_example_text(r), tokenizer) for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(lengths, bins=24, color="#3f7cbf", alpha=0.85)
    ax.axvline(config.MAX_SEQ_LENGTH, color="#b02a37", ls="--", lw=1.2,
               label=f"training budget ({config.MAX_SEQ_LENGTH})")
    ax.set_xlabel("tokens (full chat example)")
    ax.set_ylabel("examples")
    ax.set_title("Token-length distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def _terms(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in config.STOPWORDS
            and len(t) > 2]

def keyword_frequencies(records: List[Dict[str, Any]]) -> Tuple[Counter, Dict[str, Counter]]:
    """Top terms overall (from the free-text fields) and per family."""
    overall: Counter = Counter()
    per_family: Dict[str, Counter] = {}
    for record in records:
        family = record["family"]
        texts = [
            record["input_feed"].get("notes", ""),
            record["gold_assessment"].get("root_cause", ""),
            " ".join(record["gold_assessment"].get("corrective_actions", [])),
        ]
        terms = _terms(" ".join(texts))
        overall.update(terms)
        per_family.setdefault(family, Counter()).update(terms)
    return overall, per_family

def distinguishing_terms(per_family: Dict[str, Counter], overall: Counter,
                         per_family_totals: Dict[str, int], top_n: int = 5) -> Dict[str, List[str]]:
    """Terms that over-index in one family versus the whole corpus."""
    result: Dict[str, List[str]] = {}
    total_terms = sum(overall.values()) or 1
    for family, counter in per_family.items():
        scored = []
        for term, count in counter.items():
            if count < 2:
                continue
            global_share = overall[term] / total_terms
            local_share = count / per_family_totals[family]
            if local_share > max(2.0 * global_share, 0.01):
                scored.append((local_share / max(global_share, 1e-9), term))
        scored.sort(reverse=True)
        result[family] = [term for _, term in scored[:top_n]]
    return result

def family_category_matrix(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[List[int]]]:
    families = sorted({r["family"] for r in records})
    categories = sorted({r["product_category"] for r in records})
    grid = [[0] * len(categories) for _ in families]
    for record in records:
        grid[families.index(record["family"])][
            categories.index(record["product_category"])
        ] += 1
    return families, categories, grid

def plot_heatmap(families: List[str], categories: List[str],
                 grid: List[List[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(categories)), categories, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(families)), families, fontsize=8)
    for y in range(len(families)):
        for x in range(len(categories)):
            ax.text(x, y, grid[y][x], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Family x product-category coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def near_duplicate_pairs(records: List[Dict[str, Any]]) -> int:
    serialized = [
        json.dumps(r["input_feed"], sort_keys=True, ensure_ascii=False) for r in records
    ]
    count = 0
    for i in range(len(serialized)):
        for j in range(i + 1, len(serialized)):
            if fuzz.token_set_ratio(serialized[i], serialized[j]) > config.NEAR_DUPLICATE_THRESHOLD:
                count += 1
    return count

def main() -> None:
    config.ensure_dirs()
    records = load_raw(config.RAW_JSONL)
    print(f"records: {len(records)}")

    tokenizer = get_tokenizer()
    stats = token_length_stats(records, tokenizer)
    print(f"tokenizer backend: {stats['backend']}")
    print(
        f"token lengths: min={stats['min']} median={stats['median']} "
        f"p95={stats['p95']} max={stats['max']} | within {stats['budget']}-token "
        f"budget: {stats['share_within_budget']:.1%} (fits: {stats['fits']})"
    )
    if not stats["fits"]:
        print(
            f"[warn] fewer than {config.TOKEN_FIT_SHARE:.0%} of examples fit "
            f"{config.MAX_SEQ_LENGTH} tokens - shorten prompts or raise the budget."
        )
    plot_token_histogram(records, tokenizer, config.DIVERSITY_DIR / "token_lengths.png")

    overall, per_family = keyword_frequencies(records)
    print(f"\ntop {config.TOP_TERMS_COUNT} terms overall:")
    print(", ".join(f"{term}({count})" for term, count in overall.most_common(config.TOP_TERMS_COUNT)))
    per_family_totals = {family: sum(counter.values()) for family, counter in per_family.items()}
    print("\ndistinguishing terms per family:")
    for family, terms in distinguishing_terms(per_family, overall, per_family_totals).items():
        print(f"  {family}: {', '.join(terms) or '(none distinct)'}")

    families, categories, grid = family_category_matrix(records)
    plot_heatmap(families, categories, grid, config.DIVERSITY_DIR / "family_category_heatmap.png")
    max_share = max(Counter(r["family"] for r in records).values()) / len(records)
    print(f"\nmax family share: {max_share:.1%} (cap {config.MAX_FAMILY_SHARE:.0%})")

    duplicates = near_duplicate_pairs(records)
    print(f"near-duplicate pairs above threshold: {duplicates} (expected 0)")
    print(f"charts written to {config.DIVERSITY_DIR}")

if __name__ == "__main__":
    main()
