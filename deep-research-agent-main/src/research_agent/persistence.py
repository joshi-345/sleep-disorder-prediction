"""Durable checkpointing for the research graph.

A **checkpointer** persists the full graph state after every superstep, keyed by
a ``thread_id``. That single addition is what makes the graph durable, and it
unlocks four first-class features at once:

* **pause/resume** — stop mid-run and continue later with the same ``thread_id``;
* **crash recovery** — if the process dies, the last checkpoint survives on disk;
* **time-travel** — replay the graph from any past checkpoint to debug;
* **human-in-the-loop** — ``interrupt()`` can only pause/resume because the state
  is durably saved between the pause and the human's reply.

This module is the one place that decides *which* backend stores those
checkpoints. The rest of the codebase just calls ``build_graph(checkpointer=...)``
and never cares about the storage details — that is the whole point of the
checkpointer abstraction.

--------------------------------------------------------------------------------
DEV (what this project uses): SqliteSaver
--------------------------------------------------------------------------------
A single local file (``checkpoints.db`` by default). Zero setup, zero cost,
perfect for development, a single-process server, and this portfolio's scope.

    from research_agent.persistence import sqlite_saver
    from research_agent.graph import build_graph

    with sqlite_saver() as saver:                 # opens the SQLite connection
        app = build_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "run-123"}}
        app.invoke({"query": "..."}, config=cfg)  # state is now persisted

--------------------------------------------------------------------------------
PRODUCTION: use PostgresSaver instead
--------------------------------------------------------------------------------
SQLite is a single-file, single-writer store. A real deployment with multiple
workers/processes hammering the same database needs a real database. LangGraph's
supported production backend is Postgres, and — crucially — the *only* thing that
changes is the saver; the graph code is identical.

    # pip install langgraph-checkpoint-postgres
    import os
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        saver.setup()                             # first run: create the tables
        app = build_graph(checkpointer=saver)
        app.invoke({"query": "..."}, config={"configurable": {"thread_id": "run-123"}})

Any free Postgres works (a hosted free tier, or a local ``docker run postgres``).
For an async web server there is also ``AsyncPostgresSaver`` with the same API.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.config import settings


@contextmanager
def sqlite_saver(db_path: str | None = None) -> Iterator[SqliteSaver]:
    """Yield a ready-to-use ``SqliteSaver`` bound to ``db_path``.

    Use it as a context manager so the underlying SQLite connection is opened for
    the duration of the work and closed cleanly on exit:

        with sqlite_saver() as saver:
            app = build_graph(checkpointer=saver)
            ...

    ``db_path`` defaults to ``settings.checkpoint_db_path``. The connection is
    owned by the ``with`` block — which is exactly why a long-lived web server
    (Stage 10) keeps the block open for its whole lifetime instead of per-request.

    Note: ``SqliteSaver.from_conn_string`` creates the checkpoint tables lazily on
    first use, so there is no separate ``setup()`` call to remember here.
    """
    path = db_path or settings.checkpoint_db_path
    with SqliteSaver.from_conn_string(path) as saver:
        yield saver
