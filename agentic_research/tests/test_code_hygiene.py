# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'code hygiene checks: prompt isolation, credentials, forbidden vocabulary, date literals', Date: 2026-10-06
"""Code hygiene checks for the agentic research project (static, no network)."""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
)
DATE_LITERAL = re.compile(r"\d{4}-\d{2}-\d{2}")
SCAN_SUFFIXES = {".py", ".md", ".txt", ".example", ".ipynb"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
             "outputs", "cache", "logs"}


def _project_files():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def test_no_prompt_text_outside_the_prompts_module():
    """All prompt text lives only in src/prompts.py (needle assembled to stay clean)."""
    needle = "You " + "are"
    for path in PROJECT_ROOT.joinpath("src").glob("*.py"):
        if path.name == "prompts.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert needle not in text, f"prompt text found in {path.name}"


def test_no_credential_patterns_anywhere():
    for path in _project_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in KEY_PATTERNS:
            match = pattern.search(text)
            assert match is None, f"credential-like pattern in {path}"


def test_no_date_literals_in_src_logic():
    """Date literals in src/ logic break reproducibility; comments may cite dates."""
    for path in PROJECT_ROOT.joinpath("src").glob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            assert not DATE_LITERAL.search(line), \
                f"date literal in {path.name}:{line_no}"


def test_unwanted_vocabulary_is_absent():
    """The project is described purely as a multi-agent research system.

    Word-boundary regexes keep look-alikes safe: '\\bcandi'+'date\\b' does not
    match 'candidates' (the 's' kills the trailing boundary), and '\\bta'+ 'sk'+'\\b'
    does not match 'task_type' (underscore is a word character).
    """
    banned_word_patterns = (
        re.compile(r"\brubr" + re.escape("ic") + r"\b"),
        re.compile(r"\binter" + re.escape("view") + r"\b"),
        re.compile(r"\bcandi" + re.escape("date") + r"\b"),
        re.compile(r"\bmar" + re.escape("ks") + r"\b"),
        re.compile(r"\bgrad" + re.escape("ing") + r"\b"),
    )
    banned_patterns = banned_word_patterns + (
        re.compile(r"\bta" + re.escape("sk") + r"\b"),
        re.compile(r"\b3A\b"), re.compile(r"\b3B\b"), re.compile(r"\b3C\b"),
    )
    for path in _project_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in banned_patterns:
            match = pattern.search(text)
            assert match is None, f"pattern {pattern.pattern!r} in {path}"
