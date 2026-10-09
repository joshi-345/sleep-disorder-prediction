from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from research_agent.config import settings
from research_agent.guardrails import RETRY_POLICY
from research_agent.state import ResearchState
from research_agent.nodes.planner import planner_node
from research_agent.nodes.approval import approval_node
from research_agent.nodes.researcher import researcher_node
from research_agent.nodes.critic import critic_node
from research_agent.nodes.synthesizer import synthesizer_node
from research_agent.nodes.guardrails import input_guard_node, output_guard_node
from typing import cast

def route_after_input_guard(state) -> str:
    """Input guardrail: refuse out-of-scope/unsafe queries before any spend."""
    return END if state.get("rejected") else "planner"

def route_to_researchers(state) -> list[Send]:
    """Emit one Send per subtask → N parallel researcher branches."""
    return [
        Send("researcher", {"subtasks": [task], "findings": []})
        for task in state["subtasks"]
    ]

def route_after_approval(state):
    """After the human-in-the-loop gate: abort, redraft, or fan out.

    A conditional edge may return a node name (string), the END sentinel, or a
    list of Sends. We use all three: END to stop a run the human aborted,
    'planner' to loop back on a reject, or the fan-out Sends on approve/edit.
    Every possible target is declared in the edge's path map."""
    if state.get("_aborted"):
        return END
    if state.get("_plan_rejected"):
        return "planner"
    return route_to_researchers(state)

def route_after_critic(state) -> str:
    """Stop the research loop when the findings are sufficient, the iteration cap
    is reached, OR the budget guard trips — whichever comes first. Any of these
    falls through to the synthesizer so we still ship a report from what we have."""
    budget_spent = state.get("llm_calls", 0) >= settings.max_llm_calls_per_run
    if state.get("_sufficient") or state["iterations"] >= settings.max_research_iterations or budget_spent:
        return "synthesizer"
    return "planner"

def route_after_output_guard(state) -> str:
    """Output guardrail: send a failing report back to be repaired, else deliver."""
    return "synthesizer" if state.get("_needs_repair") else END

def build_graph(checkpointer=None):
    """Compile the research graph.

    ``checkpointer`` pass a saver (e.g. from ``research_agent.persistence.sqlite_saver``)
    and the graph persists its state after every superstep, making runs durable, 
    resumable, and able to pause fora human.
    Left ``None`` (the default), the graph behaves exactly as before —
    in-memory and ephemeral — which keeps evals/CI unchanged. Note: the
    human-in-the-loop approval gate only actually pauses when a checkpointer is
    present, because ``interrupt()`` needs durable state to resume from."""
    g = StateGraph(ResearchState)
    g.add_node("input_guard", input_guard_node, retry_policy=RETRY_POLICY)
    g.add_node("planner", planner_node, retry_policy=RETRY_POLICY)
    g.add_node("approval", approval_node)   # human-in-the-loop gate; no LLM call
    g.add_node("researcher", researcher_node, retry_policy=RETRY_POLICY)   # backoff handles 429s on fan-out
    g.add_node("critic", critic_node, retry_policy=RETRY_POLICY)
    g.add_node("synthesizer", synthesizer_node, retry_policy=RETRY_POLICY)
    g.add_node("output_guard", output_guard_node)   # no LLM call → no retry needed

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input_guard, ["planner", END])
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["planner", "synthesizer"])
    g.add_edge("synthesizer", "output_guard")
    g.add_conditional_edges("output_guard", route_after_output_guard, ["synthesizer", END])
    return g.compile(checkpointer=checkpointer)


def _print_result(result):
    print(result["report"])
    if result.get("guardrail_flags"):
        print("\n⚠ guardrail flags:", result["guardrail_flags"])
    print(f"\nLLM calls this run: {result.get('llm_calls')}")


def _ask_human(payload):
    """Render the paused plan and BLOCK on stdin for a real human decision.

    Returns the dict expected by the approval node's interrupt():
    {"action": "approve"} | {"action": "edit", "subtasks": [...]}
                          | {"action": "reject", "feedback": "..."}
                          | {"action": "abort", "reason": "..."}
    """
    print("\n⏸  Graph paused for human approval.")
    print(f"Query: {payload.get('query')}")
    print("Proposed subtasks:")
    for i, task in enumerate(payload["proposed_subtasks"], 1):
        print(f"  {i}. {task}")

    # input() blocks until you type something and press Enter — THIS is the wait.
    choice = input("\nDecision [approve / edit / reject / abort] (default approve): ").strip().lower()

    if choice == "edit":
        raw = input("New subtasks (comma-separated): ")
        subtasks = [s.strip() for s in raw.split(",") if s.strip()]
        return {"action": "edit", "subtasks": subtasks}
    if choice == "reject":
        feedback = input("Feedback for the planner: ").strip()
        return {"action": "reject", "feedback": feedback}
    if choice == "abort":
        reason = input("Reason for aborting (optional): ").strip()
        return {"action": "abort", "reason": reason}
    return {"action": "approve"}


if __name__ == "__main__":
    # HITL (it's off by default) and wrap the graph in a SQLite checkpointer 
    # so the run survives the pause. A `thread_id` names this one run; 
    # re-using it is what makes a run resumable.
    import uuid

    from langgraph.types import Command

    from research_agent.persistence import sqlite_saver

    settings.hitl_approval_enabled = True   # demo-only; normally set via env/config

    query = "What are the trade-offs of RAG vs fine-tuning?"
    config: RunnableConfig = {
        "configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:8]}"},
        "run_name": "deep-research",
        "metadata": {
            "model_planner": settings.model_planner,
            "model_researcher": settings.model_researcher,
            "model_synthesizer": settings.model_synthesizer,
            "model_critic": settings.model_critic,
        },
    }

    with sqlite_saver() as saver:
        app = build_graph(checkpointer=saver)

        # Start the run. It executes input_guard → planner → approval, then the
        # approval node calls interrupt(), which PAUSES and returns control here.
        result = app.invoke(cast("ResearchState", {"query": query}), config=config)

        # The pause shows up as an __interrupt__ payload in the result. Loop while
        # the graph is paused: ask the human, resume with their decision, repeat.
        # (A "reject" replans and pauses AGAIN, which is why this is a loop.)
        while result.get("__interrupt__"):
            decision = _ask_human(result["__interrupt__"][0].value)
            print(f"\n▶  Resuming with: {decision['action']}\n")
            result = app.invoke(Command(resume=decision), config=config)

        _print_result(result)