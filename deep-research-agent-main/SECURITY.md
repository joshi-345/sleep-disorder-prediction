# Security Policy

## Reporting a vulnerability

This is a portfolio project, not a hosted service. If you spot a security issue,
please open a GitHub issue (or email the maintainer) with steps to reproduce.
There's no bug-bounty program and no SLA, but reports are welcome and will be
addressed on a best-effort basis.

## Deployment model & threat model

**The HTTP API is intentionally unauthenticated.** It is built to run locally or
on a private network for demos, evaluation, and development — *not* to be exposed
directly to the public internet. There is, by design, no built-in auth, CORS
policy, or rate limiting on the application itself.

If you deploy the running service somewhere reachable, put it behind your own
control plane:

- **Authentication / authorization** — an API gateway, reverse proxy, or auth
  middleware in front of the service.
- **Rate limiting & quotas** — the agent fans out to multiple LLM calls per run,
  so an open endpoint with live provider keys is a direct cost-and-abuse vector.
  A per-run hard ceiling exists (`MAX_LLM_CALLS_PER_RUN`), and the API caps the
  request `query` length, but neither is a substitute for network-level limits.
- **CORS** — add a policy matching the origins you actually serve.
- **TLS** — terminate HTTPS at your proxy.

## Secrets

- All credentials are supplied via environment variables / a local `.env`; none
  are committed. `.env` and local state (`*.db`, logs) are git-ignored, and the
  `.dockerignore` keeps them out of the image — keys are passed at `docker run`
  time, never baked in.
- The repository's git history has been checked for committed secrets.

## Data handling

User queries and gathered findings are sent to the configured LLM and search
providers (OpenAI, Tavily, or whatever you swap in) to produce a report. Review
those providers' data-use policies before sending sensitive data. The output
guardrail scans reports for leaked system prompts and basic PII, but it is a
backstop, not a guarantee — don't feed the agent data you wouldn't send to a
third-party API.
