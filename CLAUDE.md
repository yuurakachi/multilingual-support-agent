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
1. SQLite data (~20 customers, ~50 orders; statuses `processing`, `shipped`, `delivered`, `cancelled`) and tools, each with tests:
   - `get_order_status(order_id, email)`
   - `update_shipping_address(order_id, email, new_address)` — only if the order has not shipped
   - `request_refund(order_id, email, reason)` — only within 30 days after delivery
   - `create_support_ticket(email, summary)`
   - `escalate_to_human(reason)`
2. Agent loop and system prompt with store policies: verify identity (order number + email) before giving or changing information; always answer in the customer's language; never invent information or promise anything outside policy; escalate to a human if the customer asks, is very upset, or the tools do not cover the case.
3. Interactive CLI and JSON conversation logging.
4. Ten scenario conversations (mixed languages: happy paths, out-of-policy requests, a customer trying to see someone else's order, a customer asking for a human) and a script that runs them and reports what happened.
5. Final README: what it does, Mermaid architecture diagram, how to run, design decisions and trade-offs, known limitations.

## Commands

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Layout

- `support_agent/` — agent source code
- `tests/` — unit tests
- `data/` — SQLite database (generated, git-ignored)
- `logs/` — conversation logs (generated, git-ignored)
