"""API tests for FastAPI wrapper.

Like the persistence/HITL tests, these exercise OUR code (the endpoints, the
background runner, status reporting) against a STUB graph — no real LLM calls, no
API key — so they run in normal CI.

The stub graph is compiled with an in-memory SQLite checkpointer and injected via
``create_api(graph=...)``, so the app never opens the real ``checkpoints.db``.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import START, END, StateGraph

from research_agent.api import create_api
from research_agent.config import settings
from research_agent.graph import route_after_approval
from research_agent.nodes.approval import approval_node
from research_agent.persistence import sqlite_saver
from research_agent.state import ResearchState


@pytest.fixture
def hitl_on():
    """Turn the approval gate on for a test, then restore the default (off)."""
    original = settings.hitl_approval_enabled
    settings.hitl_approval_enabled = True
    yield
    settings.hitl_approval_enabled = original


def _report_node(state):
    """Stand in for the whole graph: emit a report-shaped result + a call count."""
    return {
        "report": {
            "summary": f"Report for: {state['query']}",
            "sections": ["section one"],
            "citations": ["https://example.com"],
        },
        "llm_calls": 3,
    }


def _build_stub(saver):
    g = StateGraph(ResearchState)
    g.add_node("run", _report_node)
    g.add_edge(START, "run")
    g.add_edge("run", END)
    return g.compile(checkpointer=saver)


def _poll_until(client, thread_id, targets, timeout=5.0):
    """Background runs change status asynchronously; poll GET until one of
    ``targets`` is reached (e.g. {"completed", "error"} or {"interrupted"})."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/research/{thread_id}").json()
        if body["status"] in targets:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {thread_id} did not reach {targets} within {timeout}s")


def _poll_until_done(client, thread_id, timeout=5.0):
    return _poll_until(client, thread_id, {"completed", "error"}, timeout)


# A HITL stub mirroring the real graph's shape: planner → approval (the REAL
# gate) → researcher (fans out, one branch per subtask) → synthesizer. No LLM
# calls. The planner returns ['edited!'] after it sees feedback (a rejected
# plan), else ['a','b']. The researcher writes the *reduced* `findings` channel
# (so the parallel branches don't clash), and the synthesizer writes the single
# `report` once — exactly why route_after_approval needs that fan-out structure.
def _hitl_planner(state):
    feedback = state.get("plan_feedback")
    return {"subtasks": ["edited!"] if feedback else ["a", "b"], "plan_feedback": ""}


def _hitl_researcher(state):
    return {"findings": []}


def _hitl_synth(state):
    return {"report": {"summary": "done: " + ",".join(state["subtasks"]), "sections": [], "citations": []}}


def _build_hitl_stub(saver):
    g = StateGraph(ResearchState)
    g.add_node("planner", _hitl_planner)
    g.add_node("approval", approval_node)
    g.add_node("researcher", _hitl_researcher)
    g.add_node("synthesizer", _hitl_synth)
    g.add_edge(START, "planner")
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile(checkpointer=saver)


def test_health():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.get("/health").json() == {"status": "ok"}


def test_research_lifecycle():
    """POST queues a run (202 + thread_id); GET eventually returns its report."""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            resp = client.post("/research", json={"query": "what is RAG?"})
            assert resp.status_code == 202
            thread_id = resp.json()["thread_id"]
            assert resp.json()["status"] == "running"

            done = _poll_until_done(client, thread_id)
            assert done["status"] == "completed"
            assert done["report"]["summary"] == "Report for: what is RAG?"
            assert done["report"]["citations"] == ["https://example.com"]
            assert done["llm_calls"] == 3


def test_query_validation_rejects_empty_and_oversized():
    """The request schema caps query size (cost/abuse guard) — both an empty query
    and one over the limit are rejected with 422 before any run is queued."""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.post("/research", json={"query": ""}).status_code == 422
            assert client.post("/research", json={"query": "x" * 2001}).status_code == 422
            # A query at the boundary is accepted.
            assert client.post("/research", json={"query": "x" * 2000}).status_code == 202


def test_unknown_thread_is_404():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.get("/research/does-not-exist").status_code == 404


def test_failed_run_reports_error():
    """If the graph raises, the run surfaces as status='error', not a hang."""

    def _boom(state):
        raise RuntimeError("planner exploded")

    with sqlite_saver(":memory:") as saver:
        g = StateGraph(ResearchState)
        g.add_node("run", _boom)
        g.add_edge(START, "run")
        g.add_edge("run", END)
        app = g.compile(checkpointer=saver)

        with TestClient(create_api(graph=app)) as client:
            thread_id = client.post("/research", json={"query": "x"}).json()["thread_id"]
            done = _poll_until_done(client, thread_id)
            assert done["status"] == "error"
            assert "planner exploded" in done["error"]


# ── Human-in-the-loop over HTTP ───────────────────────────────────────────────
def test_hitl_pauses_then_resumes_approve(hitl_on):
    """With HITL on, POST pauses at the gate; resume(approve) runs the plan."""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]

            paused = _poll_until(client, thread_id, {"interrupted"})
            assert paused["status"] == "interrupted"
            assert paused["report"] is None
            assert paused["interrupt"]["proposed_subtasks"] == ["a", "b"]

            resume = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            assert resume.status_code == 202

            done = _poll_until_done(client, thread_id)
            assert done["status"] == "completed"
            assert done["report"]["summary"] == "done: a,b"  # ran the proposed plan


def test_hitl_resume_edit_uses_human_subtasks(hitl_on):
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})

            client.post(
                f"/research/{thread_id}/resume",
                json={"action": "edit", "subtasks": ["x", "y", "z"]},
            )
            done = _poll_until_done(client, thread_id)
            assert done["report"]["summary"] == "done: x,y,z"  # human's list won


def test_hitl_reject_repauses_then_approve(hitl_on):
    """A reject re-plans and pauses AGAIN, so the client resumes a second time."""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})

            client.post(
                f"/research/{thread_id}/resume",
                json={"action": "reject", "feedback": "too broad"},
            )
            repaused = _poll_until(client, thread_id, {"interrupted"})
            # Looped back to the planner (which saw the feedback) and re-paused.
            assert repaused["interrupt"]["proposed_subtasks"] == ["edited!"]

            client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            done = _poll_until_done(client, thread_id)
            assert done["report"]["summary"] == "done: edited!"


def test_resume_rejected_when_not_interrupted(hitl_on):
    """Resuming a finished run is a 409, not a silent no-op."""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})
            client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            _poll_until_done(client, thread_id)

            again = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            assert again.status_code == 409


def test_resume_unknown_thread_is_404():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            resp = client.post("/research/nope/resume", json={"action": "approve"})
            assert resp.status_code == 404
