from pydantic import BaseModel

from research_agent.models import get_model

class CitedReport(BaseModel):
    summary: str
    sections: list[str]
    citations: list[str]   # the source URLs actually used

def synthesizer_node(state):
    model = get_model("synthesizer").with_structured_output(CitedReport)
    prompt = (
        "Write a report. Every claim must reference one of these sources "
        f"by URL.\n\nFindings:\n{state['findings']}"
    )
    # On an output-guardrail repair pass, append the guard's specific feedback.
    repair_notes = state.get("repair_notes")
    if repair_notes:
        prompt += f"\n\n{repair_notes}"

    report = model.invoke(prompt)
    if not isinstance(report, CitedReport):
        raise ValueError(f"Expected CitedReport, got {type(report)}")
    return {"report": report.model_dump(), "llm_calls": 1}
