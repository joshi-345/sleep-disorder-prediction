"""FastAPI service that wraps the durable research graph.

An HTTP interface so the agent is a real, callable service instead of a script. 
Per the project's priorities the surface is deliberately small: start a run, 
poll for its result, and a health check.

--------------------------------------------------------------------------------
Background execution + polling (not a blocking request)
--------------------------------------------------------------------------------
A research run is many LLM calls and can take tens of seconds. Holding an HTTP
connection open that long is fragile (proxies and load balancers time out). So:

    POST /research            -> start a run in a background thread, return a
                                 thread_id immediately (HTTP 202 Accepted)
    GET  /research/{thread_id} -> poll status; once done, the report is read back
                                 from the checkpointer

--------------------------------------------------------------------------------
One checkpointer for the whole server (not one per request)
--------------------------------------------------------------------------------
``sqlite_saver()`` is a context manager that owns a SQLite connection. We open it
once in the FastAPI *lifespan* and keep it open for the server's lifetime, then
build the graph once on top of it. Running the graph from background threads is
safe because ``SqliteSaver.from_conn_string`` opens with ``check_same_thread=False``
and serializes every write behind a ``threading.Lock`` — concurrent runs queue on
that lock rather than corrupting the file. (For a multi-process deployment you'd
swap in ``PostgresSaver``; see ``research_agent/persistence.py``. No graph code
changes — only the saver.)
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.graph import build_graph
from research_agent.persistence import sqlite_saver
from research_agent.state import ResearchState


# ── Request / response schemas ───────────────────────────────────────────────
class ResearchRequest(BaseModel):
    # Cap the input: the graph fans out to several LLM workers, so an unbounded
    # query is a cost amplifier (and a cheap abuse vector). 2000 chars is generous
    # for a research question; min_length rejects an empty query with a 422 before
    # any work is queued.
    query: str = Field(min_length=1, max_length=2000)


class ResearchAccepted(BaseModel):
    """Returned by POST /research the instant the run is queued."""

    thread_id: str
    status: str  # always "running" here; poll GET for the rest


class Report(BaseModel):
    """The shape the synthesizer/guardrails write into ``state['report']``."""

    summary: str
    sections: list[str]
    citations: list[str]


class ResearchStatus(BaseModel):
    """Returned by GET /research/{thread_id}."""

    thread_id: str
    status: str  # running | completed | error | interrupted
    report: Report | None = None
    guardrail_flags: list[str] = []
    llm_calls: int | None = None
    error: str | None = None
    # When status == "interrupted", the payload the graph surfaced for the human
    # (the proposed plan + the decision schema). None otherwise.
    interrupt: dict | None = None


class ResumeDecision(BaseModel):
    """Body of POST /research/{thread_id}/resume — the human's call on the plan.

    Mirrors what the approval node reads off ``Command(resume=...)``:
        approve  -> {"action": "approve"}
        edit     -> {"action": "edit", "subtasks": ["...", "..."]}
        reject   -> {"action": "reject", "feedback": "..."}
        abort    -> {"action": "abort", "reason": "..."}
    """

    action: str
    subtasks: list[str] | None = None
    feedback: str | None = None
    reason: str | None = None


# ── In-memory job registry ───────────────────────────────────────────────────
# The checkpointer is the source of truth for the *result*, but it can't tell us
# whether a run is still executing or crashed mid-way (a half-finished run just
# looks like a checkpoint with no report yet). This small thread-safe registry
# tracks live status for runs started by *this* process. It's intentionally
# ephemeral: on restart, GET falls back to inferring status from the durable
# checkpoint (see _read_status).
class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def start(self, thread_id: str) -> None:
        # Also used to flip an interrupted run back to running on resume, so we
        # reset the interrupt payload here too.
        with self._lock:
            self._jobs[thread_id] = {"status": "running", "error": None, "interrupt": None}

    def finish(self, thread_id: str) -> None:
        with self._lock:
            if thread_id in self._jobs:
                self._jobs[thread_id].update(status="completed", interrupt=None)

    def interrupt(self, thread_id: str, payload: dict) -> None:
        """Record that the run paused at the human-in-the-loop gate."""
        with self._lock:
            if thread_id in self._jobs:
                self._jobs[thread_id].update(status="interrupted", interrupt=payload)

    def fail(self, thread_id: str, error: str) -> None:
        with self._lock:
            if thread_id in self._jobs:
                self._jobs[thread_id].update(status="error", error=error)

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(thread_id)
            return dict(job) if job else None


# ── Helpers ──────────────────────────────────────────────────────────────────
def _run_config(thread_id: str) -> RunnableConfig:
    """Build the per-run config. ``thread_id`` is what makes the run addressable
    and resumable; the metadata mirrors graph.py's demo so traces are labelled."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "deep-research",
        "metadata": {
            "model_planner": settings.model_planner,
            "model_researcher": settings.model_researcher,
            "model_synthesizer": settings.model_synthesizer,
            "model_critic": settings.model_critic,
        },
    }


def _run_graph(graph, registry: JobRegistry, thread_id: str, inputs) -> None:
    """The background job: drive the graph forward, recording status.

    ``inputs`` is the initial state (``{"query": ...}``) on a fresh run, or a
    ``Command(resume=decision)`` when continuing a run paused at the HITL gate —
    the graph treats both the same way.

    Runs in a thread-pool worker. Three outcomes:
      * the graph paused at ``interrupt()`` — invoke RETURNS (it does not raise or
        block) with an ``__interrupt__`` payload → record "interrupted";
      * it ran to the end → "completed";
      * it raised → "error" (caught so the run never silently vanishes).

    The graph writes its report into the checkpointer as it goes, so we don't
    return anything — GET reads the result back from there.
    """
    try:
        result = graph.invoke(inputs, config=_run_config(thread_id))
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            registry.interrupt(thread_id, interrupts[0].value)
        else:
            registry.finish(thread_id)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the caller
        registry.fail(thread_id, f"{type(exc).__name__}: {exc}")


def _pending_interrupt(snapshot) -> dict | None:
    """Best-effort extraction of a pending interrupt payload from a checkpoint.

    Used only for the post-restart path, where the live JobRegistry doesn't know
    about a run started by a previous process. Defensive: snapshot internals can
    shift across LangGraph versions, so a failure just yields None.
    """
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0].value
    return None


def _read_status(graph, registry: JobRegistry, thread_id: str) -> ResearchStatus:
    """Assemble a status response from live tracking + the durable checkpoint."""
    snapshot = graph.get_state(_run_config(thread_id))
    values = snapshot.values or {}
    job = registry.get(thread_id)

    if job is not None:
        status, error = job["status"], job["error"]
        interrupt_payload = job.get("interrupt")
    elif values:
        # Not tracked in this process (e.g. started before a restart): infer from
        # the checkpoint. Pending nodes mean it's waiting (paused at the HITL
        # gate); no pending nodes + a report means it finished.
        if snapshot.next == () and values.get("report"):
            status, interrupt_payload = "completed", None
        else:
            status, interrupt_payload = "interrupted", _pending_interrupt(snapshot)
        error = None
    else:
        # No live job and no checkpoint for this id — it never existed here.
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")

    report = values.get("report")
    return ResearchStatus(
        thread_id=thread_id,
        status=status,
        report=Report(**report) if report else None,
        guardrail_flags=values.get("guardrail_flags", []),
        llm_calls=values.get("llm_calls"),
        error=error,
        interrupt=interrupt_payload,
    )


# ── App factory ──────────────────────────────────────────────────────────────
def create_api(graph=None) -> FastAPI:
    """Build the FastAPI app.

    ``graph`` lets tests inject a stub graph (already compiled with an in-memory
    checkpointer) so the suite makes no LLM calls and touches no real DB file.
    Left ``None`` (production/dev), the lifespan opens the SQLite checkpointer and
    builds the real graph on top of it.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Shared per-server resources, created once at startup, torn down at exit.
        app.state.registry = JobRegistry()
        app.state.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="research")
        try:
            if graph is not None:
                app.state.graph = graph
                yield
            else:
                # Keep the checkpointer connection open for the whole server life.
                with sqlite_saver() as saver:
                    app.state.graph = build_graph(checkpointer=saver)
                    yield
        finally:
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    api = FastAPI(
        title="Deep Research Agent",
        version="0.1.0",
        summary="Durable, citation-backed research agent exposed over HTTP.",
        lifespan=lifespan,
    )

    @api.get("/health")
    def health() -> dict:
        """Liveness probe for hosting platforms."""
        return {"status": "ok"}

    @api.post("/research", response_model=ResearchAccepted, status_code=202)
    def start_research(req: ResearchRequest, request: Request) -> ResearchAccepted:
        """Queue a research run and return its thread_id immediately.

        202 Accepted = "we took the work, it isn't finished" — poll GET for the
        result.
        """
        thread_id = uuid.uuid4().hex
        request.app.state.registry.start(thread_id)
        request.app.state.executor.submit(
            _run_graph,
            request.app.state.graph,
            request.app.state.registry,
            thread_id,
            cast(ResearchState, {"query": req.query}),
        )
        return ResearchAccepted(thread_id=thread_id, status="running")

    @api.get("/research/{thread_id}", response_model=ResearchStatus)
    def get_research(thread_id: str, request: Request) -> ResearchStatus:
        """Fetch a run's status and, once finished, its report.

        When ``status == "interrupted"`` the response carries an ``interrupt``
        payload (the proposed plan) — POST it back to the resume endpoint.
        """
        return _read_status(request.app.state.graph, request.app.state.registry, thread_id)

    @api.post("/research/{thread_id}/resume", response_model=ResearchAccepted, status_code=202)
    def resume_research(thread_id: str, decision: ResumeDecision, request: Request) -> ResearchAccepted:
        """Resume a run paused at the human-in-the-loop gate with the human's call.

        Only valid when the run is currently ``interrupted``. Like POST /research
        this returns 202 and continues the graph in the background — a ``reject``
        re-plans and pauses *again*, so the client may need to resume more than
        once (poll GET between calls).
        """
        graph, registry = request.app.state.graph, request.app.state.registry

        # The run must exist and actually be waiting for input. Check the live
        # registry first; fall back to the durable checkpoint (post-restart).
        job = registry.get(thread_id)
        if job is not None:
            if job["status"] != "interrupted":
                raise HTTPException(
                    status_code=409,
                    detail=f"Run is '{job['status']}', not waiting for input.",
                )
        else:
            snapshot = graph.get_state(_run_config(thread_id))
            if not snapshot.values:
                raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
            if snapshot.next == ():
                raise HTTPException(status_code=409, detail="Run is finished, not waiting for input.")

        # Flip back to running and feed the decision in via Command(resume=...).
        # exclude_none drops the fields irrelevant to this action (e.g. no
        # 'subtasks' on an approve), matching what the approval node reads.
        registry.start(thread_id)
        request.app.state.executor.submit(
            _run_graph, graph, registry, thread_id, Command(resume=decision.model_dump(exclude_none=True))
        )
        return ResearchAccepted(thread_id=thread_id, status="running")

    return api


# Module-level app for `uvicorn research_agent.api:api`.
api = create_api()
