# Reflection

All three projects were built as engineering systems where correctness is
structural, not aspirational. The decisions below are shared across them.

**Contracts move correctness into code.** Every stage that produces or
consumes data is typed: Pydantic schemas validate LLM output (summary,
signal, sentiment order, anomaly extraction, agent actions, the final report),
with bounds and inter-field rules enforced mechanically — three risks exactly,
likelihood 1–5, evidence present, hedge basis with at least two data
disciplines. When the composer in the agentic system returned an invalid
report, the degraded fallback built a *smaller but valid* artifact from tool
data instead of crashing: invalid output degrades gracefully rather than
propagating.

**Numbers belong to Python.** In the equity research assistant the LLM sees a
numeric summary and writes narrative, never arithmetic; in the agentic system
the same principle governs the one-sigma hedging band (derived in Python from
tool data) and the handlers never invent figures. The agents' handoff is a
typed brief verified lossless by a JSON round-trip test.

**Agents are constrained by evidence, not instructions.** The critique loop in
the two-agent design is structural: agent A's tools cannot obtain news, agent
B's cannot score sentiment, so the gap A declares is real by construction —
the clarification cycle closes exactly the missing pieces (scored sentiment,
short-window volatility). Replans are stamped from explicit reasons at the
moment they happen.

**What I would improve with more time.** (1) A real streaming chat-server
client for incremental observation-reset — the client-side JSON repair helper
spends budget on retries that a streaming-completion-mode endpoint design
could trim to a single full reload with tool reuse. (2) The QLoRA runbook
needs the Colab T4 execution results filled in (loss curves, RED vs base
rates) plus the manual audit labels. (3) A small gold-set benchmark for
indicator verification beyond the unit tests. (4) The dashboard would gain a
live tail (websocket) rather than a reload-to-refresh JSONL read.

**Limitations encountered.** Teacher-generated gold labels for the
fine-tuning dataset are themselves model output — that is why the manual
hallucination audit exists and stays human-labeled. Free-tier rate limits
dictated the batched per-headline sentiment design (fewer, larger requests
were rejected); the supply-chain generation pipeline needed resumability and
a usage log to survive 429s. Colab session variance (T4 vs L4, memory limits)
required an OOM playbook with fallbacks rather than one fixed configuration.
Provider JSON-mode differences (not all gateways implement structured output)
pushed the design to handle malformed JSON with explicit repair retries
instead of assuming response_format support. And yfinance's news schema
changed mid-project — the ladder now normalises both the legacy flat shape and
the nested content shape, which is exactly the brittleness a thin data
contract review should catch.
