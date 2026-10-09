from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from research_agent.guardrails import LLMCallCounter
from research_agent.models import get_model
from research_agent.tools.search import web_search

_agent = create_agent(get_model("researcher"), tools=[web_search])

def researcher_node(state):
    task = state["subtasks"][0]
    # Count the agent's internal model calls (tool loop → unknown count) so the
    # budget guard sees this branch's true cost. The reducer sums across fan-out.
    counter = LLMCallCounter()
    result = _agent.invoke(
        {"messages": [HumanMessage(content=f"Research: {task}")]},
        config={"callbacks": [counter]},
    )
    finding = result["messages"][-1].content
    return {"findings": [finding], "llm_calls": counter.count}