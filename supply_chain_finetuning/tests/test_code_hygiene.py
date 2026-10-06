# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'code hygiene checks: prompt isolation, credential patterns, unwanted vocabulary', Date: 2026-10-06
"""Code hygiene checks for the fine-tuning project (static, no network)."""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
)
SCAN_SUFFIXES = {".py", ".md", ".txt", ".example", ".json", ".csv"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
             "outputs", "data", "logs", ".venv"}


def _iter_project_files():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def test_no_prompt_text_outside_the_prompts_module():
    """All prompt text lives only in src/prompts.py (needle assembled to stay clean)."""
    needle = "You " + "are"
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "prompts.py" or path.name == "test_code_hygiene.py":
            continue  # the prompts module owns the text; this file describes the check
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert needle not in text, f"prompt text found in {path}"


def test_no_credential_patterns_anywhere():
    for path in _iter_project_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in KEY_PATTERNS:
            match = pattern.search(text)
            assert match is None, f"credential-like pattern {match!r} found in {path}"


def test_unwanted_vocabulary_is_absent():
    """The project is described purely as supply-chain anomaly fine-tuning.

    Fragments are concatenated so this file stays clean itself. "assessment"
    is a domain term here (the output contract), so it is not banned. Numbered
    references use word-boundary regexes so hex colours ("#b02a37") never trip.
    """
    banned = (
        "rubr" + "ic", "inter" + "view", "candi" + "date", " mar" + "ks",
        "grad" + "ing", "ta" + "sk ",
    )
    numbered = (
        re.compile(r"\b2A\b"), re.compile(r"\b2B\b"), re.compile(r"\b3A\b"),
        re.compile(r"\bta" + re.escape("sk") + r"\s*\d\b"),
    )
    for path in _iter_project_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for word in banned:
            assert word.lower() not in lowered, f"unwanted vocabulary {word!r} found in {path}"
        for pattern in numbered:
            match = pattern.search(text)
            assert match is None, f"numbered reference {match.group(0)!r} found in {path}"
