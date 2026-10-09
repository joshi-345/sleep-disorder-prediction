"""
Three scorers, each returning a float in [0, 1]:

- ``faithfulness``      — LLM-as-judge: are the report's claims supported by the
                          findings the researchers actually gathered?
- ``coverage``          — LLM-as-judge: does the report address every part of the
                          query (checked against the dataset's ``key_points``)?
- ``citation_accuracy`` — PROGRAMMATIC (no judge, no cost): do the URLs the report
                          cites actually appear in the gathered findings? This is a
                          direct, deterministic measurement.

The two LLM judges go through the same provider-agnostic factory as the rest of
the system (``get_model("judge")``), pointed at a cheap spec via ``model_judge``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_agent.models import get_model


class Score(BaseModel):
    """A judge's verdict: a 0..1 score plus a one-line justification (for trace logs)."""

    score: float = Field(ge=0.0, le=1.0, description="0.0 = fails the rubric, 1.0 = fully satisfies it")
    reasoning: str = Field(description="One sentence justifying the score")


def _findings_text(findings) -> str:
    """Flatten the accumulated findings (strings or Finding-like objects) into one blob."""
    parts = []
    for f in findings or []:
        parts.append(f if isinstance(f, str) else getattr(f, "model_dump", lambda: f)().__str__())
    return "\n\n".join(parts)


def _report_text(report) -> str:
    """Render the report dict (summary + sections) as plain text for a judge to read."""
    if isinstance(report, dict):
        summary = report.get("summary", "")
        sections = "\n".join(report.get("sections", []) or [])
        return f"{summary}\n\n{sections}".strip()
    return str(report)


def faithfulness(findings, report) -> Score:
    """Are the report's claims grounded in the gathered findings, or invented?"""
    judge = get_model("judge").with_structured_output(Score)
    verdict = judge.invoke(
        "You are a strict evaluator scoring GROUNDEDNESS.\n"
        "Score from 0 to 1 how well EVERY claim in the report is supported by the "
        "findings below. Penalise claims that are not backed by the findings.\n\n"
        f"FINDINGS:\n{_findings_text(findings)}\n\n"
        f"REPORT:\n{_report_text(report)}"
    )
    assert isinstance(verdict, Score)
    return verdict


def coverage(query: str, key_points: list[str], report) -> Score:
    """Does the report address the whole query and the expected key points?"""
    judge = get_model("judge").with_structured_output(Score)
    points = "\n".join(f"- {p}" for p in key_points)
    verdict = judge.invoke(
        "You are a strict evaluator scoring COVERAGE.\n"
        "Score from 0 to 1 the fraction of the expected key points that the report "
        "meaningfully addresses for the given query.\n\n"
        f"QUERY:\n{query}\n\n"
        f"EXPECTED KEY POINTS:\n{points}\n\n"
        f"REPORT:\n{_report_text(report)}"
    )
    assert isinstance(verdict, Score)
    return verdict


def citation_accuracy(findings, report) -> float:
    """Fraction of cited URLs that actually appear in the gathered findings.

    Deterministic and free — no LLM call. A report with no citations scores 0,
    which is the behaviour we want: an ungrounded report should fail the gate.
    """
    citations = report.get("citations", []) if isinstance(report, dict) else []
    if not citations:
        return 0.0
    haystack = _findings_text(findings)
    present = sum(1 for url in citations if url and url in haystack)
    return present / len(citations)
