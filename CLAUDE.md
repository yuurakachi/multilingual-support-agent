# Project context

Portfolio project: a multilingual (Spanish, Japanese, English) customer support AI agent for a fictional online store. The owner is learning, so the goal is understanding as much as a working result.

## Hard constraints

- Python 3.10+.
- Claude API through the official `anthropic` SDK, with tool use.
- **No agent frameworks** (LangChain, etc.). The agent loop is written by hand: model decides -> tool runs -> result is returned to the model -> repeat until it answers. The loop must have an iteration limit.
- The model name comes from `ANTHROPIC_MODEL` in `.env`. Never hardcode it.
- The API key lives only in `.env`, which is git-ignored. Never commit it, print it or copy it elsewhere.
- Simulated data in SQLite. Interface is a CLI chat for now.
- Tools return clear error results (order not found, email mismatch, outside policy) instead of raising.
- Every conversation is saved as structured JSON (messages, tool calls, arguments, results, token usage, latency). An eval system will be built on these logs later, so keep them stable and machine-readable.

## Language

- Code, comments, commit messages and README: **English**.
- Explanations to the project owner: **Spanish**, simple, with analogies.

## Workflow

Work phase by phase. At the end of each phase: run the tests, commit, push to GitHub, explain in Spanish what was done and why, then **stop and wait for approval** before starting the next phase. Small, frequent commits inside each phase. If something is ambiguous, ask instead of assuming.

## Phases

0. Repo setup (done)
1. (done) SQLite data (~20 customers, ~50 orders; statuses `processing`, `shipped`, `delivered`, `cancelled`) and tools, each with tests:
   - `get_order_status(order_id, email)`
   - `update_shipping_address(order_id, email, new_address)` — only if the order has not shipped
   - `request_refund(order_id, email, reason)` — only within 30 days after delivery
   - `create_support_ticket(email, summary)`
   - `escalate_to_human(reason)`
2. (done) Agent loop and system prompt with store policies: verify identity (order number + email) before giving or changing information; always answer in the customer's language; never invent information or promise anything outside policy; escalate to a human if the customer asks, is very upset, or the tools do not cover the case.
3. (done) Interactive CLI and JSON conversation logging.
4. (done) Ten scenario conversations (mixed languages: happy paths, out-of-policy requests, a customer trying to see someone else's order, a customer asking for a human) and a script that runs them and reports what happened.
5. Final README: what it does, Mermaid architecture diagram, how to run, design decisions and trade-offs, known limitations.

## Commands

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python -m support_agent.seed   # rebuild data/store.db from scratch
pytest
python -m support_agent        # interactive chat against the live API (--reset-db, --quiet)
python -m support_agent.run_scenarios   # ten scripted conversations against the live API (costs ~$0.12 per full run)
```

## Layout

- `support_agent/db.py` — SQLite connection and schema
- `support_agent/seed.py` — deterministic simulated data; ORD-1001..ORD-1008 are hand-picked anchor orders with a known state
- `support_agent/tools.py` — the five support tools
- `support_agent/tool_registry.py` — JSON Schema tool definitions sent to the model, and `execute_tool` dispatcher
- `support_agent/prompts.py` — system prompt with the store policies (store name: Kumo Market)
- `support_agent/config.py` — `Settings` from env (`ANTHROPIC_MODEL`, optional `ANTHROPIC_EFFORT`, `AGENT_MAX_ITERATIONS`)
- `support_agent/agent.py` — `SupportAgent.reply()`, the hand-written loop; returns `AgentReply` (text, stop_reason, iterations, latency_ms, model_calls, tool_calls, usage, error). `build_agent(conn, settings)` creates the live one
- `support_agent/conversation_log.py` — `ConversationLog`: one JSON file per conversation (`record_turn`, `save(transcript)`); accepts free-form `metadata`
- `support_agent/cli.py` + `__main__.py` — terminal chat; `run_chat` takes injectable `read`/`write` so tests can script it
- `scenarios/scenarios.json` — the ten scripted conversations and their expectations
- `support_agent/scenarios.py` — loading/validating scenarios, `check_expectations`, `run_scenario` (fresh in-memory seeded store per scenario)
- `support_agent/run_scenarios.py` — runner CLI: prints each conversation, writes `logs/scenarios/<run id>/` with `summary.json`
- `tests/` — unit tests; `tests/fakes.py` has the scripted `FakeClient` and response builders
- `data/` — SQLite database (generated, git-ignored)
- `logs/` — conversation logs (generated, git-ignored)

## Conventions

- Tools take the SQLite connection as first argument and return a dict: `{"ok": True, ...}` or `{"ok": False, "error_code", "message"}`. Error codes are stable identifiers the evals will rely on; do not rename them casually.
- Date-dependent tools accept a keyword-only `now` so tests never depend on the real clock.
- Tool unit tests use the small hand-written store in `tests/conftest.py`, not the seed data.
- Tests never call the real API: they script a `FakeClient` (`tests/fakes.py`). For a live check without typing, pipe lines into the chat: `printf '%s
' "message" "/exit" | python -m support_agent`.
- The conversation history is append-only and model turns are stored unchanged (`response.content`, thinking blocks included). Do not edit or strip earlier turns.
- A test fails if any file in `support_agent/` contains a model name; the model only comes from `.env`.
- `SupportAgent.reply()` does not raise on API failures: it returns `stop_reason="api_error"` with `reply.error` set, so the turn is still logged.
- The log format is a contract for the evals: bump `SCHEMA_VERSION` in `conversation_log.py` when a field is renamed or removed. Adding fields is fine.
- On this project the API client must be created after `load_settings()` (which loads `.env`); see `build_agent`.
- Scenario checks stay deterministic and narrow (tool outcomes, forbidden strings). Judging language, tone or accuracy belongs to the future eval system, not to `check_expectations`.
- Scenarios depend on the anchor orders ORD-1001..ORD-1008 in `seed.py`; a test fails if a scenario mentions an order or email the seed does not have.
- Do not tune the system prompt just to make a scenario pass without telling the owner; a failing scenario is information.
