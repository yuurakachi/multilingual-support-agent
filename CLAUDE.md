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
3. Interactive CLI and JSON conversation logging.
4. Ten scenario conversations (mixed languages: happy paths, out-of-policy requests, a customer trying to see someone else's order, a customer asking for a human) and a script that runs them and reports what happened.
5. Final README: what it does, Mermaid architecture diagram, how to run, design decisions and trade-offs, known limitations.

## Commands

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python -m support_agent.seed   # rebuild data/store.db from scratch
pytest
```

## Layout

- `support_agent/db.py` — SQLite connection and schema
- `support_agent/seed.py` — deterministic simulated data; ORD-1001..ORD-1008 are hand-picked anchor orders with a known state
- `support_agent/tools.py` — the five support tools
- `support_agent/tool_registry.py` — JSON Schema tool definitions sent to the model, and `execute_tool` dispatcher
- `support_agent/prompts.py` — system prompt with the store policies (store name: Kumo Market)
- `support_agent/config.py` — `Settings` from env (`ANTHROPIC_MODEL`, optional `ANTHROPIC_EFFORT`, `AGENT_MAX_ITERATIONS`)
- `support_agent/agent.py` — `SupportAgent.reply()`, the hand-written loop; returns `AgentReply` (text, stop_reason, iterations, tool_calls, usage)
- `tests/` — unit tests
- `data/` — SQLite database (generated, git-ignored)
- `logs/` — conversation logs (generated, git-ignored)

## Conventions

- Tools take the SQLite connection as first argument and return a dict: `{"ok": True, ...}` or `{"ok": False, "error_code", "message"}`. Error codes are stable identifiers the evals will rely on; do not rename them casually.
- Date-dependent tools accept a keyword-only `now` so tests never depend on the real clock.
- Tool unit tests use the small hand-written store in `tests/conftest.py`, not the seed data.
- Agent tests never call the real API: `tests/test_agent.py` scripts a `FakeClient`. Live checks go through `python -m support_agent.agent "message"`.
- The conversation history is append-only and model turns are stored unchanged (`response.content`, thinking blocks included). Do not edit or strip earlier turns.
- A test fails if any file in `support_agent/` contains a model name; the model only comes from `.env`.
