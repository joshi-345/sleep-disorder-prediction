from pydantic import BaseModel
from research_agent.models import get_model

class Assessment(BaseModel):
    sufficient: bool
    missing: list[str]   # gaps to research next, empty if sufficient

def critic_node(state):
    model = get_model("critic").with_structured_output(Assessment)
    assessment = model.invoke(
        f"Query: {state['query']}\n\nFindings:\n{state['findings']}\n\n"
        "Are these sufficient and well-sourced? If not, list what's missing."
    )
    assert isinstance(assessment, Assessment)
    return {"gaps": assessment.missing, "iterations": state.get("iterations", 0) + 1,
            "llm_calls": 1,                          # budget accounting
            "_sufficient": assessment.sufficient}   # transient flag for routing
