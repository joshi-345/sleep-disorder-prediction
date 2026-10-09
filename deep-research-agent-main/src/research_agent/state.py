from typing import Annotated, TypedDict
from operator import add

from pydantic import BaseModel

class Finding(BaseModel):
    claim: str
    source_url: str
    source_title: str

class ResearchState(TypedDict, total=False):
    query: str                                # set at input
    subtasks: list[str]                       # written by planner
    findings: Annotated[list[Finding], add]   # ← reducer: ACCUMULATE
    report: str                               # written by synthesizer
    iterations: int                           # bumped each critic pass
    gaps: list[str]                           # critic's notes for the next round

    # Guardrails
    rejected: bool                            # input guard verdict
    rejection_reason: str                     # message delivered when a query is refused
    llm_calls: Annotated[int, add]            # budget accounting (summed across fan-out)
    repairs: int                              # output-guard repair attempts so far
    repair_notes: str                         # output guard → synthesizer feedback on a repair
    guardrail_flags: list[str]                # citation/content issues surfaced on the report

    # Human-in-the-loop
    plan_feedback: str                        # human's notes when a plan is rejected → planner redrafts
    _plan_rejected: bool                      # transient routing flag set by the approval gate
    _aborted: bool                            # human aborted at the gate → route straight to END