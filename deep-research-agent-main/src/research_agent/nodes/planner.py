from pydantic import BaseModel, Field
from research_agent.models import get_model

class Plan(BaseModel):
    subtasks: list[str] = Field(description="3-5 focused, non-overlapping research questions")

def planner_node(state):
    model = get_model("planner").with_structured_output(Plan)

    feedback = state.get("plan_feedback")   # ← set when a human REJECTED the previous plan (HITL)
    gaps = state.get("gaps")                # ← empty on first pass, populated after a critic loop
    if feedback:
        prompt = (
            f"Original query: {state['query']}\n\n"
            f"A human reviewer rejected your previous plan with this feedback:\n{feedback}\n\n"
            "Produce a revised set of subtasks that addresses their feedback."
        )
    elif gaps:
        prompt = (
            f"Original query: {state['query']}\n\n"
            f"A first research pass left these gaps:\n{gaps}\n\n"
            "Produce focused subtasks that close ONLY these gaps."
        )
    else:
        prompt = f"Break this into research subtasks:\n{state['query']}"

    plan = model.invoke(prompt)
    assert isinstance(plan, Plan)
    # Clear plan_feedback so it doesn't leak into a later critic-loop replan.
    return {"subtasks": plan.subtasks, "llm_calls": 1, "plan_feedback": ""}   # budget accounting