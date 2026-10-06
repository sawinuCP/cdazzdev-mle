# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'pytest bootstrap adding the project root to sys.path', Date: 2026-10-06
"""Pytest bootstrap: make ``src`` importable regardless of invocation directory."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
