from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_planner: str = "openai:gpt-4o"
    model_researcher: str = "openai:gpt-4o-mini"
    model_synthesizer: str = "openai:gpt-4o"
    model_critic: str = "openai:gpt-4o-mini"
    model_judge: str = "openai:gpt-4o-mini"   # cheap model for LLM-as-judge evals

    temperature: float = 0.0
    max_research_iterations: int = 2

    # Eval gates: the agent must stay at/above these mean scores in CI.
    eval_min_faithfulness: float = 0.7
    eval_min_citation_accuracy: float = 0.8
    eval_min_coverage: float = 0.5
    eval_max_cases: int = 0   # 0 = run the whole dataset; >0 caps it (cheap CI runs)

    # ── Guardrails & robustness ─────────────────────────────────────
    # Input guardrail. The classifier decides if a query belongs to this domain
    # and is safe to run. `guardrail_domain` is injected into its prompt so the
    # scope is configurable without code changes.
    guardrail_input_enabled: bool = True
    guardrail_domain: str = (
        "factual research questions answerable from public web sources "
        "(technology, science, business, history, current events, and similar)"
    )

    # Output guardrail. A report must cite real gathered sources for at least
    # this fraction of its citations, or it is sent back to the synthesizer to
    # be repaired — at most `max_output_repairs` times before we flag and ship.
    guardrail_output_enabled: bool = True
    min_citation_coverage: float = 0.8
    max_output_repairs: int = 1

    # Budget guard. A hard ceiling on LLM calls per run. Combined with
    # `max_research_iterations`, this guarantees a single query can't run away.
    max_llm_calls_per_run: int = 22

    # Robustness. Retry policy applied to every LLM node so transient failures
    # (timeouts, 429 rate limits) back off and retry instead of crashing the run.
    retry_max_attempts: int = 4
    retry_initial_interval: float = 1.0   # seconds; doubles each attempt (1, 2, 4, 8)
    retry_backoff_factor: float = 2.0

    # ── Persistence & human-in-the-loop ───────────────────
    # Where the SQLite checkpointer writes durable state. A single local file;
    # zero setup, zero cost. For production swap in PostgresSaver (see
    # research_agent/persistence.py). Relative paths resolve from the cwd, so
    # like .env it expects you to run from the project root.
    checkpoint_db_path: str = "checkpoints.db"

    # Human-in-the-loop approval gate. When True, the graph pauses right after
    # the FIRST plan (before the costly fan-out) and waits for a human to
    # approve / edit / reject the subtasks. Default OFF so the agent stays fully
    # autonomous for evals, CI, and the API's default path — turn it on per-run
    # for interactive/operator use. NOTE: HITL only works with a checkpointer;
    # interrupt() needs durable state to pause and resume.
    hitl_approval_enabled: bool = False


settings = Settings()