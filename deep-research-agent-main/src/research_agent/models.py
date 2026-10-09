from functools import lru_cache
from langchain.chat_models import init_chat_model
from research_agent.config import settings

_SPEC_BY_ROLE = {
    "planner": settings.model_planner,
    "researcher": settings.model_researcher,
    "synthesizer": settings.model_synthesizer,
    "critic": settings.model_critic,
    "judge": settings.model_judge,
}


@lru_cache(maxsize=None)
def get_model(role: str):
    # Return a chat model for a logical role, provider-agnostic.
    spec = _SPEC_BY_ROLE[role]
    return init_chat_model(spec, temperature=settings.temperature)