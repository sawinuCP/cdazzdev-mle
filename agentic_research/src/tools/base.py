"""Tool plumbing shared by every tool module."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from ..schemas import ToolResult

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _finite(value: Any) -> Optional[float]:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if (isinstance(as_float, float) and math.isnan(as_float)) else as_float

def make_result(ok: bool, data: Any = None, error: Optional[str] = None,
                hint: Optional[str] = None, source: str = "") -> ToolResult:
    """Single construction point for every ToolResult."""
    return ToolResult(ok=ok, data=data, error=error, hint=hint,
                      source=source, fetched_at=_now_iso())
