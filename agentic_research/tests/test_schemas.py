# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'research report schema verification: three risks, evidence requirements, hedge data basis', Date: 2026-10-06
"""ResearchReport validator verification."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    Evidence, FinancialHealthSummary, HedgePlan, ReportMeta, ResearchReport, Risk,
)


def _report(**overrides):
    payload = {
        "ticker": "NVDA",
        "financial_health_summary": {"narrative": "Solid. Momentum positive.",
                                     "evidence_refs": ["get_price_data"]},
        "risks": [
            {"title": "R1", "detail": "d", "likelihood": 3,
             "evidence": [{"source_tool": "get_price_data", "key_datum": "RSI 67"}]},
            {"title": "R2", "detail": "d", "likelihood": 3,
             "evidence": [{"source_tool": "calculate_volatility",
                           "key_datum": "ann 0.42"}]},
            {"title": "R3", "detail": "d", "likelihood": 2,
             "evidence": [{"source_tool": "get_news", "key_datum": "10 headlines"}]},
        ],
        "hedge": {"strategy": "Collar", "instruments": ["put", "call"],
                  "rationale": "band-sized", "data_basis": ["1sd band ±12.3", "price 238"],
                  "cost_note": "n/a"},
        "meta": {"run_id": "r1", "run_at": "now", "tool_call_count": 5,
                 "replans": 1, "critique_cycles": 1, "memory_hits": 0,
                 "degraded": False},
        "disclaimer": "not investment advice",
    }
    payload.update(overrides)
    return ResearchReport.model_validate(payload)


def test_valid_report_passes():
    report = _report()
    assert len(report.risks) == 3 and report.meta.degraded is False


def test_risk_count_enforced():
    risks = _report().risks
    with pytest.raises(ValidationError):
        _report(risks=risks[:2])


def test_every_risk_needs_evidence():
    risky = _report()
    risky.risks[0].evidence.clear()
    with pytest.raises(ValidationError):
        ResearchReport.model_validate(risky.model_dump())


def test_hedge_data_basis_needs_two():
    with pytest.raises(ValidationError):
        _report(hedge={"strategy": "s", "instruments": ["put"],
                       "rationale": "r", "data_basis": ["only one"],
                       "cost_note": "c"})


def test_likelihood_bounds():
    with pytest.raises(ValidationError):
        _report(risks=[
            {"title": "R1", "detail": "d", "likelihood": 9,
             "evidence": [{"source_tool": "get_price_data", "key_datum": "x"}]},
        ] + [risk.model_dump() for risk in _report().risks[1:]])
