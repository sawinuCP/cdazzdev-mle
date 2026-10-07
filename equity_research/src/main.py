# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'CLI entry point: full pipeline with friendly failures and stale-cache fallback', Date: 2026-10-06
"""Command-line entry point.

Usage (from the ``equity_research`` folder):

    python -m src.main --ticker NVDA [--no-cache]

The CLI never shows a raw traceback: every expected failure becomes a friendly
message and a non-zero exit code, with a stale-cache fallback when possible.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from . import config
from .analysis import analysis
from .data import data_pipeline, news
from .llm.llm_client import (
    LLMConfigurationError,
    client_from_env,
    failure_records,
)
from .reporting import report
from .schemas import SummaryStats


def _write_json(path, payload) -> None:
    """Persist an artifact atomically enough for our purposes (single writer)."""
    config.ensure_dirs()
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def run(ticker: str, use_cache: bool = True) -> int:
    """Full pipeline: data -> news -> sentiment -> signal -> report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("equity_research.main")
    summary: SummaryStats
    frame = None

    # ── 1. market data (with the stale-cache fallback) ────────────────
    try:
        frame, quality = data_pipeline.fetch_history(ticker)
        info = data_pipeline.fetch_info(ticker)
        summary = data_pipeline.build_summary(ticker, frame, quality, info)
    except data_pipeline.MarketDataError as exc:
        cached = data_pipeline.load_cached_summary()
        if cached is None:
            logger.error("Market data unavailable: %s", exc)
            print(f"[error] market data unavailable: {exc}")
            return 1
        summary = cached
        logger.warning("Using the previous cached summary (stale): %s", exc)
    _write_json(config.SUMMARY_JSON, summary.model_dump())
    span = summary.data_quality
    print(
        f"[data] {summary.ticker}: {span.get('bars', '?')} bars, "
        f"{span.get('first_date', '?')} .. {span.get('last_date', '?')} "
        f"({span.get('span_days', '?')} days){'  [STALE]' if summary.stale else ''}"
    )
    print(
        f"[data] price={summary.current_price} daily={summary.daily_return_pct}% "
        f"ytd={summary.ytd_return_pct}% momentum={summary.momentum_signal}"
    )

    # ── 2. news + sentiment + signal (LLM optional if the key is missing) ──
    try:
        client = client_from_env(use_cache=use_cache)
    except LLMConfigurationError as exc:
        logger.warning("%s", exc)
        print(f"[warn] {exc}")
        print(
            "[warn] Continuing without LLM output: summary and chart were written; "
            "sentiment/signal/report need the key."
        )
        return 1

    headlines_result = news.fetch_headlines(summary.ticker)
    headlines = headlines_result["headlines"]
    coverage = headlines_result["coverage"]
    print(f"[news] {len(headlines)} unique headlines (coverage: {coverage})")
    if not headlines:
        print("[warn] no headlines from any source; skipping sentiment/signal/report")

    sentiment = analysis.score_headlines(summary.ticker, headlines, client)
    _write_json(config.SENTIMENT_JSON, sentiment.model_dump())
    print(
        f"[sentiment] overall={sentiment.overall_score:+.3f} ({sentiment.label}), "
        f"fallbacks={sentiment.fallback_count}"
    )

    signal = analysis.generate_signal(summary, sentiment, client)
    _write_json(config.SIGNAL_JSON, signal.model_dump())
    print(f"[signal] {signal.action} ({signal.generated_by})")
    print(f"[signal] {signal.justification}")

    # ── 3. report ──────────────────────────────────────────────────────
    written = report.write_report(
        frame, summary, sentiment, signal, len(headlines), coverage
    )
    for key, path in written.items():
        if path:
            print(f"[report] {key}: {path}")
    if failure_records():
        print(f"[llm] {len(failure_records())} failure(s) logged to {config.LLM_FAILURE_LOG}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated equity research assistant")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER, help="stock ticker symbol")
    parser.add_argument(
        "--no-cache", action="store_true", help="bypass the LLM response cache"
    )
    args = parser.parse_args()
    try:
        code = run(args.ticker, use_cache=not args.no_cache)
    except KeyboardInterrupt:
        print("\n[stopped] interrupted by user")
        code = 130
    except Exception as exc:  # noqa: BLE001 - the user never sees a raw traceback
        print(f"[error] pipeline failed: {type(exc).__name__}: {exc}")
        logging.getLogger("equity_research.main").exception("pipeline failed")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
