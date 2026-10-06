# Multilingual Support Agent

A customer support AI agent for a fictional online store. It answers in Spanish, Japanese and English, and uses tools to look up orders, change shipping addresses, request refunds, open support tickets and escalate to a human.

The agent loop is written by hand on top of the Claude API (no agent framework), so every step — model decides, tool runs, result goes back — is visible in the code.

> **Status:** work in progress. Phase 0 (repository setup) is done.

## Features

_To be completed as the phases land._

- [ ] Simulated store data in SQLite (customers, products, orders)
- [ ] Support tools with policy checks and clear error results
- [ ] Hand-written tool-use loop with an iteration limit
- [ ] Interactive CLI chat
- [ ] Structured JSON conversation logs (messages, tool calls, tokens, latency)
- [ ] Scenario test runner

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
