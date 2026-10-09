"""Robustness + output checks (no graph dependency).

Kept separate from the guardrail *nodes* (``nodes/guardrails.py``) so these pure
helpers stay trivially testable:

- ``RETRY_POLICY``      — a LangGraph retry policy that backs off on transient
                          failures *including 429 rate limits*.
- ``LLMCallCounter``    — a callback that counts model calls, so the researcher
                          sub-agent can report its real cost to the budget guard.
- ``verify_citations``  — the hard citation gate: which cited URLs are NOT backed
                          by a gathered source.
- ``scan_content``      — content checks: leaked system prompts and PII.
"""

from __future__ import annotations

import re

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import RetryPolicy

from research_agent.config import settings


# ── Robustness: retry policy ─────────────────────────────────────────────────
def _is_transient(exc: Exception) -> bool:
    """Retry on rate limits (429), server errors (5xx), and connection drops.

    The provider SDKs don't share a base class, so we match structurally:
    anything carrying a 429 / 5xx status code, plus any error whose name screams
    "rate limit" or "timeout". Deterministic bugs (ValueError, TypeError, …) are
    deliberately *not* retried — retrying them just wastes money.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "timeout" in name


# One policy, reused on every LLM node in the graph. `initial_interval` doubles
# by `backoff_factor` each attempt → 1s, 2s, 4s, 8s, exactly the schedule the
RETRY_POLICY = RetryPolicy(
    max_attempts=settings.retry_max_attempts,
    initial_interval=settings.retry_initial_interval,
    backoff_factor=settings.retry_backoff_factor,
    retry_on=_is_transient,
)


# ── Budget: count real LLM calls ─────────────────────────────────────────────
class LLMCallCounter(BaseCallbackHandler):
    """Counts model invocations within a single ``invoke``.

    Plain LLMs fire ``on_llm_start``; chat models fire ``on_chat_model_start`` —
    a given call triggers exactly one of them, so counting both is safe and never
    double-counts. Used by the researcher node, whose internal tool-loop makes an
    unknown number of calls, to report its true cost to the budget guard.
    """

    def __init__(self) -> None:
        self.count = 0

    def on_llm_start(self, *args, **kwargs) -> None:
        self.count += 1

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.count += 1


# ── Output guardrail: citation verification ──────────────────────────────────
def _findings_text(findings) -> str:
    """Flatten accumulated findings (strings or Finding objects) into one blob."""
    parts = []
    for f in findings or []:
        parts.append(f if isinstance(f, str) else str(getattr(f, "model_dump", lambda: f)()))
    return "\n\n".join(parts)


def verify_citations(findings, report) -> tuple[float, list[str]]:
    """Hard citation gate.

    Returns ``(coverage, unsupported)`` where ``coverage`` is the fraction of the
    report's cited URLs that actually appear in the gathered findings, and
    ``unsupported`` lists the URLs that do NOT — i.e. citations the model may have
    invented. A report with no citations scores 0.0: an ungrounded report should
    fail the gate.
    """
    citations = report.get("citations", []) if isinstance(report, dict) else []
    if not citations:
        return 0.0, []
    haystack = _findings_text(findings)
    unsupported = [url for url in citations if not url or url not in haystack]
    coverage = (len(citations) - len(unsupported)) / len(citations)
    return coverage, unsupported


# ── Output guardrail: content checks ─────────────────────────────────────────
# Markers that would mean our own instructions leaked into the user-facing text.
_PROMPT_LEAK_MARKERS = (
    "you are a strict evaluator",
    "system prompt",
    "every claim must reference",
    "break this into research subtasks",
    "as an ai language model",
)

# Conservative PII patterns. Emails and long digit runs (phone / card numbers)
# are the realistic leaks for a web-research report; kept simple on purpose.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LONG_DIGITS_RE = re.compile(r"\b(?:\d[ -]?){9,}\b")


def scan_content(text: str) -> list[str]:
    """Return a list of content violations found in the report text (empty = clean)."""
    flags: list[str] = []
    low = text.lower()
    for marker in _PROMPT_LEAK_MARKERS:
        if marker in low:
            flags.append(f"possible leaked system prompt: '{marker}'")
    if _EMAIL_RE.search(text):
        flags.append("possible PII: email address in report")
    if _LONG_DIGITS_RE.search(text):
        flags.append("possible PII: phone/card-like number in report")
    return flags
