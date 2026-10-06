# Multilingual Support Agent

A customer support AI agent for a fictional online store. It answers in Spanish, Japanese and English, and uses tools to look up orders, change shipping addresses, request refunds, open support tickets and escalate to a human.

The agent loop is written by hand on top of the Claude API (no agent framework), so every step — model decides, tool runs, result goes back — is visible in the code.

> **Status:** work in progress. Phase 3 (terminal chat and conversation logs) is done; scenario tests come next.

## Features

_To be completed as the phases land._

- [x] Simulated store data in SQLite (customers, products, orders)
- [x] Support tools with policy checks and clear error results
- [x] Hand-written tool-use loop with an iteration limit
- [x] Interactive CLI chat
- [x] Structured JSON conversation logs (messages, tool calls, tokens, latency)
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

## How the loop works

`SupportAgent.reply()` in `support_agent/agent.py` handles one customer message:

1. Send the full conversation, the system prompt and the tool definitions to the model.
2. If the model stops with `tool_use`, run every requested tool, append the model's turn and one message with all the results, and go back to step 1.
3. If it stops with `end_turn`, its text is the reply.
4. Anything else (`refusal`, `max_tokens`, an API failure, or reaching the iteration limit) ends the turn with a fixed fallback message, and no tool from an incomplete turn is ever run.

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

Run the tests (they use a scripted fake client, so no API key is needed):

```bash
pytest
```

Start the chat:

```bash
python -m support_agent
```

```text
You> Where is my order ORD-1001? My email is yuki.tanaka@example.jp
  [tool] get_order_status {"order_id": "ORD-1001", "email": "yuki.tanaka@example.jp"} -> ok
  [2 model call(s), 4.1 s, 5187 tokens in / 240 out, stop: end_turn]
Agent> Your order ORD-1001 is still being processed, so it has not shipped yet...
```

| Option | Effect |
| ------ | ------ |
| `--quiet` | Hide the tool, timing and token lines |
| `--reset-db` | Rebuild the simulated store first (order dates are relative to the day it is built) |
| `--log-dir PATH` | Save conversation logs somewhere other than `logs/` |

Inside the chat, `/new` starts a fresh conversation and `/exit` leaves.

## Conversation logs

Every conversation is saved to `logs/<timestamp>_<id>.json` and rewritten after each turn. The format is meant for automated evaluation:

```jsonc
{
  "schema_version": 1,
  "conversation_id": "20261006T191317Z_10dd59f6",
  "model": "...", "effort": "medium", "max_iterations": 8,
  "system_prompt_sha256": "...",        // which prompt version produced this conversation
  "metadata": {},                       // free-form labels, e.g. a scenario name
  "totals": { "turns": 2, "model_calls": 3, "tool_calls": 1, "latency_ms": 8058, "usage": { ... } },
  "turns": [
    {
      "index": 1,
      "user_message": "...",
      "assistant_message": "...",
      "stop_reason": "end_turn",        // or max_iterations, refusal, max_tokens, api_error...
      "completed": true,
      "iterations": 2,
      "latency_ms": 4669,
      "usage": { "input_tokens": 6, "output_tokens": 277, "cache_creation_input_tokens": 301, "cache_read_input_tokens": 4880 },
      "error": null,
      "model_calls": [ { "iteration": 1, "stop_reason": "tool_use", "latency_ms": 1616, "usage": { ... } } ],
      "tool_calls": [
        { "iteration": 1, "id": "toolu_...", "name": "request_refund",
          "input": { "order_id": "ORD-1007", "email": "...", "reason": "..." },
          "result": { "ok": true, "refund_id": "RF-0001", "status": "requested" },
          "is_error": false, "latency_ms": 19 }
      ]
    }
  ],
  "transcript": [ ... ]                 // raw message history exactly as sent to the API
}
```

Logs contain whatever the customer typed (emails, addresses), so `logs/` is git-ignored.

## Configuration

| Variable            | Description                                  |
| ------------------- | -------------------------------------------- |
| `ANTHROPIC_API_KEY` | Your Anthropic API key. Never commit it.     |
| `ANTHROPIC_MODEL`   | Claude model the agent uses.                 |
| `ANTHROPIC_EFFORT`  | Optional thinking effort (`low` … `max`). Empty = API default. |
| `AGENT_MAX_ITERATIONS` | Optional cap on model calls per customer message (default 8). |

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
