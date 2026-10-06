# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'code hygiene checks: no date literals, prompts isolated, no key patterns', Date: 2026-10-06
"""Code hygiene checks: date literals, prompt isolation, credential patterns."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src import analysis, config, prompts
from src.schemas import HeadlineSentiment, SentimentAggregate, SummaryStats

PROJECT_ROOT = Path(config.PROJECT_ROOT)
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

DATE_LITERAL = re.compile(r"\d{4}-\d{2}-\d{2}")
KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
)


def _source_files():
    return sorted(SRC_DIR.glob("*.py"))


def test_no_date_literals_outside_comment_lines():
    """Date literals in logic would break reproducibility; comments may cite dates."""
    for path in _source_files():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):  # citation/comment lines are allowed
                continue
            assert not DATE_LITERAL.search(line), (
                f"{path.name}:{line_no} contains a date literal: {line.strip()!r}"
            )


def test_no_system_prompt_text_outside_prompts_module():
    """All prompt text lives only in prompts.py (grep-proof separation)."""
    for path in _source_files():
        if path.name == "prompts.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "You are" not in text, f"{path.name} contains prompt text"


def test_every_prompt_template_formats_against_real_payloads():
    """Each template must format against the real payload models without KeyError."""
    formatted_sentiment = prompts.SENTIMENT_USER.format(ticker="NVDA", headline="Headline")
    assert "NVDA" in formatted_sentiment and "Headline" in formatted_sentiment

    summary = SummaryStats.model_validate(
        {
            "ticker": "NVDA",
            "current_price": 100.0,
            "momentum_signal": "bullish",
            "momentum_components": {"net": 2, "rsi_note": ""},
            "data_quality": {"last_date": "2026-10-05"},
        }
    )
    item = HeadlineSentiment(
        headline="h", sentiment="positive", confidence=0.9, brief_reason="r"
    )
    aggregate = SentimentAggregate(
        overall_score=0.4, label="positive",
        counts={"positive": 1, "negative": 0, "neutral": 0},
        top_headlines=[item], items=[item], fallback_count=0,
        model_id="m", generated_at="now",
    )
    bundle = analysis.build_evidence_bundle(summary, aggregate)
    rendered = prompts.SIGNAL_USER.format(
        ticker=summary.ticker,
        as_of=bundle["as_of"],
        evidence_json=json.dumps(bundle, indent=2, ensure_ascii=False),
    )
    assert '"overall_score"' in rendered  # the bundle JSON made it into the prompt
    assert prompts.REPAIR_SUFFIX.format(error="boom", schema='{"type": "object"}')


def test_no_credential_patterns_anywhere_in_the_project():
    """Scan every tracked text file; generated output dirs are excluded on purpose
    (they may legitimately quote model text), and .env is gitignored."""
    scan_roots = [PROJECT_ROOT, PROJECT_ROOT.parent]
    skip_names = {".env", ".git", "__pycache__", ".pytest_cache", "outputs", ".ipynb_checkpoints"}
    text_suffixes = {".py", ".md", ".txt", ".json", ".example", ".cfg", ".toml"}
    for root in scan_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in text_suffixes:
                continue
            if any(part in skip_names for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in KEY_PATTERNS:
                match = pattern.search(text)
                assert match is None, f"credential-like pattern {match!r} found in {path}"


def test_forbidden_vocabulary_is_absent_from_the_project():
    """The project describes itself purely as an equity research tool.

    The words are assembled from fragments so this file itself stays clean.
    """
    forbidden = tuple(
        fragment_pair + remainder
        for fragment_pair, remainder in (
            ("assess", "ment"),
            ("rubr", "ic"),
            ("inter", "view"),
            ("candi", "date"),
            (" mar", "ks"),
            ("ta", "sk"),
        )
    )
    skip_names = {".git", "__pycache__", ".pytest_cache", "outputs", ".ipynb_checkpoints"}
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".txt", ".example"}:
            continue
        if any(part in skip_names for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for word in forbidden:
            assert word not in text, f"unwanted word pattern found in {path}"
