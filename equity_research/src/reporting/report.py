"""One-page equity research brief: Markdown source + styled HTML + chart.

The HTML is a single self-contained file (inline CSS, no JavaScript, chart
embedded as base64) so it renders identically offline and prints to PDF
straight from the browser.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering; the notebook displays the HTML instead
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from .. import config
from .schemas import SentimentAggregate, SummaryStats, TechnicalSignal

DISCLAIMER = (
    "This brief is generated automatically for informational purposes only and is "
    "not investment advice. Market data may be delayed or incomplete, model output "
    "can be wrong, and past performance never guarantees future results. Consult a "
    "licensed financial advisor before making investment decisions."
)

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1a2332; margin: 0;
       background: #f4f6f9; }
.page { max-width: 880px; margin: 24px auto; background: #ffffff; padding: 32px 40px;
        box-shadow: 0 1px 6px rgba(20,40,80,.15); border-radius: 8px; }
h1 { font-size: 26px; margin: 0 0 2px; color: #0f2a52; }
h2 { font-size: 17px; margin: 26px 0 8px; color: #0f2a52;
     border-bottom: 2px solid #e3e9f2; padding-bottom: 4px; }
.meta { color: #66738a; font-size: 12px; margin-bottom: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { border: 1px solid #e3e9f2; padding: 6px 10px; text-align: left; }
th { background: #eef3fa; color: #33475e; }
.pos { color: #1c7c3c; font-weight: 600; }
.neg { color: #b02a37; font-weight: 600; }
.signal { display: inline-block; padding: 4px 14px; border-radius: 14px; color: #fff;
          font-weight: 700; letter-spacing: .5px; }
.buy { background: #1c7c3c; } .hold { background: #b58a1c; } .sell { background: #b02a37; }
.chart { width: 100%; border: 1px solid #e3e9f2; border-radius: 6px; margin-top: 8px; }
.drivers li { margin: 3px 0; }
.disclaimer { margin-top: 26px; padding: 12px 16px; background: #fff7e8;
              border-left: 4px solid #d99b1c; font-size: 12px; color: #5c4a17; }
footer { margin-top: 18px; font-size: 11px; color: #93a1b5; }
"""

def _fmt(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "N/A (source unavailable)" if suffix == "pe" else "N/A"
    return f"{value:,.{digits}f}{suffix}"

def _pe_display(summary: SummaryStats) -> str:
    return "N/A (source unavailable)" if summary.pe_trailing is None else f"{summary.pe_trailing:,.2f}"

def _signed(value: Optional[float], digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}%"

def build_chart(frame, summary: SummaryStats, path=config.CHART_PNG) -> str:
    """Render the 3-panel chart (price/SMAs/Bollinger, RSI, MACD); return base64."""
    from . import indicators  # pure recompute keeps this module decoupled

    view = frame.tail(config.CHART_WINDOW_BARS)
    ind = indicators.compute_all(frame["Close"]).tail(config.CHART_WINDOW_BARS)

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1.2]},
    )
    ax = axes[0]
    ax.plot(view.index, view["Close"], color="#17456e", lw=1.4, label="Close")
    ax.plot(ind.index, ind["sma_50"], color="#d99b1c", lw=1.1, label="SMA 50")
    ax.plot(ind.index, ind["sma_200"], color="#7a4ea8", lw=1.1, label="SMA 200")
    ax.fill_between(
        ind.index, ind["lower"], ind["upper"], color="#3f7cbf", alpha=0.12,
        label="Bollinger (20, 2σ)",
    )
    ax.set_title(f"{summary.ticker} — price, moving averages and Bollinger bands")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(ind.index, ind["rsi_14"], color="#17456e", lw=1.2)
    for level, style in ((config.RSI_OVERSOLD, "--"), (50.0, ":"), (config.RSI_OVERBOUGHT, "--")):
        ax.axhline(level, color="#98a5b8", lw=0.8, ls=style)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI 14")
    ax.grid(alpha=0.25)

    ax = axes[2]
    colors = ["#1c7c3c" if v >= 0 else "#b02a37" for v in ind["hist"].fillna(0.0)]
    ax.bar(ind.index, ind["hist"], color=colors, width=1.0, alpha=0.55, label="Histogram")
    ax.plot(ind.index, ind["macd"], color="#17456e", lw=1.1, label="MACD")
    ax.plot(ind.index, ind["signal"], color="#d99b1c", lw=1.1, label="Signal")
    ax.axhline(0.0, color="#98a5b8", lw=0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"{summary.ticker} — technical outlook "
        f"({summary.data_quality.get('first_date', '')} → {summary.data_quality.get('last_date', '')})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    config.ensure_dirs()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode("ascii")

def _snapshot_rows(summary: SummaryStats) -> str:
    ind = summary.indicators
    rows = [
        ("Last price", _fmt(summary.current_price)),
        ("Previous close", _fmt(summary.prev_close)),
        ("Daily return", _signed(summary.daily_return_pct)),
        ("Year-to-date", _signed(summary.ytd_return_pct)),
        ("52-week high / low", f"{_fmt(summary.week52_high)} / {_fmt(summary.week52_low)}"),
        ("% from 52-week high", _signed(summary.pct_from_52w_high)),
        ("Trailing P/E", _pe_display(summary)),
        (
            "Momentum vote",
            f"{summary.momentum_signal} (net {summary.momentum_components.get('net', 0):+d})",
        ),
        ("SMA 50 / SMA 200", f"{_fmt(ind.sma_50)} / {_fmt(ind.sma_200)}"),
        ("RSI 14", f"{_fmt(ind.rsi_14)} {ind.rsi_note}".strip()),
        (
            "MACD / signal / histogram",
            f"{_fmt(ind.macd)} / {_fmt(ind.macd_signal)} / {_fmt(ind.macd_hist)}",
        ),
        ("%B / bandwidth", f"{_fmt(ind.pct_b)} / {_fmt(ind.bandwidth)}"),
    ]
    return "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)

def _headline_rows(sentiment: SentimentAggregate) -> str:
    rows = []
    for item in sentiment.top_headlines:
        css = (
            "pos" if item.sentiment == "positive"
            else "neg" if item.sentiment == "negative"
            else ""
        )
        rows.append(
            f"<tr><td>{item.headline}</td>"
            f'<td class="{css}">{item.sentiment}</td>'
            f"<td>{item.confidence:.2f}</td></tr>"
        )
    return "".join(rows)

def build_html(
    summary: SummaryStats,
    sentiment: Optional[SentimentAggregate],
    signal: TechnicalSignal,
    chart_b64: Optional[str],
    headline_count: int,
    coverage: str,
) -> str:
    """Compose the single-file HTML brief (inline CSS, no JavaScript)."""
    stale_note = (
        "<p><strong>Note:</strong> live market data was unavailable; figures come from the "
        "previous cached summary and may be outdated.</p>"
        if summary.stale
        else ""
    )
    chart_html = (
        f'<img class="chart" alt="technical outlook chart" '
        f'src="data:image/png;base64,{chart_b64}">'
        if chart_b64
        else ""
    )
    if sentiment:
        label_css = (
            "pos" if sentiment.label == "positive"
            else "neg" if sentiment.label == "negative"
            else ""
        )
        counts = sentiment.counts
        sentiment_html = (
            "<table>"
            f"<tr><th>Overall (confidence-weighted)</th><td class=\"{label_css}\">"
            f"{sentiment.overall_score:+.3f} ({sentiment.label})</td></tr>"
            f"<tr><th>Headlines analysed</th><td>{headline_count} "
            f"(coverage: {coverage}; positive {counts.get('positive', 0)} / "
            f"negative {counts.get('negative', 0)} / neutral {counts.get('neutral', 0)})</td></tr>"
            "</table>"
            "<h2>Top headlines</h2>"
            "<table><tr><th>Headline</th><th>Reading</th><th>Confidence</th></tr>"
            f"{_headline_rows(sentiment)}</table>"
        )
    else:
        sentiment_html = "<p>News sentiment unavailable for this run.</p>"
    fallback_note = (
        f"<p><em>Generated by: {signal.generated_by}.</em></p>"
        if signal.generated_by == "rules_fallback"
        else ""
    )
    drivers_html = "".join(
        f"<li>{driver.replace('_', ' ')}</li>" for driver in signal.key_drivers
    )
    badge_css = (
        "buy" if signal.action == "BUY" else "hold" if signal.action == "HOLD" else "sell"
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        f"<title>{summary.ticker} — equity research brief</title>"
        f"<style>{_CSS}</style></head>\n<body><div class=\"page\">"
        f"<h1>{summary.ticker} — Equity Research Brief</h1>"
        f"<div class=\"meta\">Generated {generated_at} · daily bars "
        f"{summary.data_quality.get('first_date', 'N/A')} → "
        f"{summary.data_quality.get('last_date', 'N/A')} · "
        f"{summary.data_quality.get('bars', 0)} bars</div>"
        f"{stale_note}"
        "<h2>Company snapshot</h2>"
        f"<table>{_snapshot_rows(summary)}</table>"
        "<h2>Technical outlook</h2>"
        "<p>The momentum vote counts five simple conditions (net "
        f"{summary.momentum_components.get('net', 0):+d}); the recommendation below "
        "reasons over this evidence together with the news sentiment.</p>"
        f"{chart_html}"
        "<h2>News sentiment</h2>"
        f"{sentiment_html}"
        "<h2>Recommendation</h2>"
        f"<p><span class=\"signal {badge_css}\">{signal.action}</span></p>"
        f"<p>{signal.justification}</p>"
        f"<ul class=\"drivers\">{drivers_html}</ul>"
        f"<p><strong>Risks:</strong> {signal.risk_notes}</p>"
        f"{fallback_note}"
        f"<div class=\"disclaimer\">{DISCLAIMER}</div>"
        "<footer>Produced by the automated equity research assistant. "
        "Data: Yahoo Finance (yfinance). Not investment advice.</footer>"
        "</div></body></html>"
    )

def build_markdown(
    summary: SummaryStats,
    sentiment: Optional[SentimentAggregate],
    signal: TechnicalSignal,
    headline_count: int,
    coverage: str,
) -> str:
    """Markdown source of the same brief (kept alongside the HTML file)."""
    if sentiment:
        top_lines = "\n".join(
            f"- *{item.headline}* — {item.sentiment} ({item.confidence:.2f})"
            for item in sentiment.top_headlines
        )
        sentiment_block = (
            f"- Overall: **{sentiment.overall_score:+.3f} ({sentiment.label})**\n"
            f"- Headlines: {headline_count} (coverage: {coverage})\n\n{top_lines}"
        )
    else:
        sentiment_block = "News sentiment unavailable for this run."
    return (
        f"# {summary.ticker} — Equity Research Brief\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
        "## Snapshot\n\n"
        f"- Last price: **{_fmt(summary.current_price)}**\n"
        f"- Previous close: {_fmt(summary.prev_close)}\n"
        f"- Daily return: {_signed(summary.daily_return_pct)}\n"
        f"- Year-to-date: {_signed(summary.ytd_return_pct)}\n"
        f"- 52-week high / low: {_fmt(summary.week52_high)} / {_fmt(summary.week52_low)}\n"
        f"- % from 52-week high: {_signed(summary.pct_from_52w_high)}\n"
        f"- Trailing P/E: {_pe_display(summary)}\n"
        f"- Momentum vote: **{summary.momentum_signal}** "
        f"(net {summary.momentum_components.get('net', 0):+d})\n\n"
        "## Technical outlook\n\n"
        f"- SMA 50 / SMA 200: {_fmt(summary.indicators.sma_50)} / "
        f"{_fmt(summary.indicators.sma_200)}\n"
        f"- RSI 14: {_fmt(summary.indicators.rsi_14)} {summary.indicators.rsi_note}\n"
        f"- MACD / signal / histogram: {_fmt(summary.indicators.macd)} / "
        f"{_fmt(summary.indicators.macd_signal)} / {_fmt(summary.indicators.macd_hist)}\n"
        f"- %B / bandwidth: {_fmt(summary.indicators.pct_b)} / "
        f"{_fmt(summary.indicators.bandwidth)}\n\n"
        f"## News sentiment\n\n{sentiment_block}\n\n"
        "## Recommendation\n\n"
        f"**{signal.action}** — {signal.justification}\n\n"
        f"Key drivers: {', '.join(signal.key_drivers) or 'n/a'}\n\n"
        f"Risks: {signal.risk_notes}\n\n"
        "---\n\n"
        f"{DISCLAIMER}\n"
    )

def write_report(
    frame,
    summary: SummaryStats,
    sentiment: Optional[SentimentAggregate],
    signal: TechnicalSignal,
    headline_count: int,
    coverage: str,
) -> Dict[str, str]:
    """Produce chart.png, report.md and report.html; return the written paths."""
    config.ensure_dirs()
    chart_b64: Optional[str] = None
    if frame is not None and not frame.empty:
        try:
            chart_b64 = build_chart(frame, summary)
        except Exception as exc:  # noqa: BLE001 - a chart failure must not kill the report
            chart_b64 = None
    html_text = build_html(
        summary, sentiment, signal, chart_b64, headline_count, coverage
    )
    markdown_text = build_markdown(summary, sentiment, signal, headline_count, coverage)
    config.REPORT_HTML.write_text(html_text, encoding="utf-8")
    config.REPORT_MD.write_text(markdown_text, encoding="utf-8")
    return {
        "report_html": str(config.REPORT_HTML),
        "report_md": str(config.REPORT_MD),
        "chart_png": str(config.CHART_PNG) if chart_b64 else "",
    }
