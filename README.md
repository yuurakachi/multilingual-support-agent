# Multilingual Support Agent

A customer support AI agent for a fictional online store. It answers in Spanish, Japanese and English, and uses tools to look up orders, change shipping addresses, request refunds, open support tickets and escalate to a human.

The agent loop is written by hand on top of the Claude API (no agent framework), so every step — model decides, tool runs, result goes back — is visible in the code.

> **Status:** work in progress. Phase 1 (store data and tools) is done; the agent loop comes next.

## Features

_To be completed as the phases land._

- [x] Simulated store data in SQLite (customers, products, orders)
- [x] Support tools with policy checks and clear error results
- [ ] Hand-written tool-use loop with an iteration limit
- [ ] Interactive CLI chat
- [ ] Structured JSON conversation logs (messages, tool calls, tokens, latency)
- [ ] Scenario test runner

## Tools

| Tool | What it does | Policy |
| ---- | ------------ | ------ |
| `get_order_status(order_id, email)` | Order status, dates, items and total | Email must match the order |
| `update_shipping_address(order_id, email, new_address)` | Changes the delivery address | Only while the order is `processing` |
| `request_refund(order_id, email, reason)` | Opens a refund request | Delivered orders, within 30 days of delivery, once per order |
| `create_support_ticket(email, summary)` | Opens a ticket for follow-up by email | Valid email required |
| `escalate_to_human(reason)` | Queues the conversation for a human agent | — |

Tools never raise for expected problems. They return `{"ok": false, "error_code": "...", "message": "..."}` (for example `order_not_found`, `email_mismatch`, `refund_window_expired`) so the model can explain the problem and the logs record it.

## Architecture

_Diagram coming in Phase 5._

## Getting started

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # then add your ANTHROPIC_API_KEY
python -m support_agent.seed    # build the simulated store database
```

Run the tests:

```bash
pytest
```

## Configuration

| Variable            | Description                                  |
| ------------------- | -------------------------------------------- |
| `ANTHROPIC_API_KEY` | Your Anthropic API key. Never commit it.     |
| `ANTHROPIC_MODEL`   | Claude model the agent uses.                 |

## Project structure

```
support_agent/   Agent source code
tests/           Unit tests
data/            SQLite database (generated, not committed)
logs/            Conversation logs in JSON (generated, not committed)
```

## Design decisions and trade-offs

_Coming in Phase 5._

## Known limitations

_Coming in Phase 5._
