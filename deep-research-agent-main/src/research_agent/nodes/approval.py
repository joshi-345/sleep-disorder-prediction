"""Human-in-the-loop approval gate.

This node sits between the planner and the parallel fan-out — the natural place
for a human to sign off *before* the agent spends money on N concurrent
researcher branches. It uses LangGraph's ``interrupt()`` primitive:

* ``interrupt(payload)`` **pauses** the whole graph and surfaces ``payload`` to
  the caller (the operator/UI). The current state is saved to the checkpointer,
  so the process can even die here and resume later.
* The caller inspects the payload, gets a human decision, and resumes with
  ``Command(resume=decision)``. On resume LangGraph **re-runs this node from the
  top**, and this time ``interrupt()`` returns ``decision`` instead of pausing.

The human can choose one of four actions:

* ``approve`` — run the plan as-is → fall through to the fan-out;
* ``edit``    — replace the subtasks with the human's edited list → fan-out;
* ``reject``  — send it back to the planner to redraft, optionally with feedback;
* ``abort``   — stop the run for good → route straight to END with an aborted report.

Because ``interrupt()`` requires durable state, this gate only does anything when
a checkpointer is configured AND ``settings.hitl_approval_enabled`` is True. When
disabled it is a transparent pass-through, so evals/CI/the default API path stay
fully autonomous.
"""

from __future__ import annotations

from langgraph.types import interrupt

from research_agent.config import settings


def approval_node(state):
    """Pause for human approval of the FIRST plan; pass through otherwise."""
    # Two reasons to skip the gate entirely (return a clean, non-rejected state):
    #   1. HITL is turned off → stay autonomous.
    #   2. We're on a critic-loop replan (iterations > 0) → the human already
    #      approved this run's direction; don't nag them every loop. Only the
    #      first, costly fan-out needs sign-off.
    if not settings.hitl_approval_enabled or state.get("iterations", 0) > 0:
        return {"_plan_rejected": False}

    # Snapshot EXACTLY what the human is approving into the interrupt payload, so
    # the operator sees precisely what they're signing off on (a best practice:
    # never make them approve something they can't see).
    decision = interrupt(
        {
            "type": "plan_approval",
            "message": (
                "Review the proposed research plan, then resume with a decision: "
                "{'action': 'approve'} | "
                "{'action': 'edit', 'subtasks': [...]} | "
                "{'action': 'reject', 'feedback': '...'} | "
                "{'action': 'abort', 'reason': '...'}"
            ),
            "query": state.get("query"),
            "proposed_subtasks": state.get("subtasks", []),
        }
    )

    # ── Below here only runs AFTER the human resumes ────────────────────────────
    # `decision` is whatever was passed to Command(resume=...). Default to
    # "approve" so a bare/empty resume is treated as a green light.
    action = (decision or {}).get("action", "approve")

    if action == "edit":
        # Human handed us a corrected list of subtasks → use it, then fan out.
        return {"subtasks": decision["subtasks"], "_plan_rejected": False, "plan_feedback": ""}

    if action == "reject":
        # Route back to the planner to redraft. Their free-text feedback (if any)
        # is stored so the planner can address it on the next pass.
        return {"_plan_rejected": True, "plan_feedback": decision.get("feedback", "")}

    if action == "abort":
        # Stop the run for good. We route straight to END (see route_after_approval)
        # and write a report-shaped result so callers get a uniform output, mirroring
        # how the input guard refuses a query. The graph reaches a terminal state, so
        # the final checkpoint records the abort rather than leaving a dangling pause.
        reason = decision.get("reason") or "Run aborted by the operator at the approval gate."
        return {
            "_aborted": True,
            "report": {"summary": f"Run aborted: {reason}", "sections": [], "citations": []},
        }

    # approve (or unknown action) → proceed as-is. We still set the flag
    # explicitly to False so a stale `True` from a prior reject can't linger in
    # the persisted state and re-trigger the reject route.
    return {"_plan_rejected": False, "plan_feedback": ""}
