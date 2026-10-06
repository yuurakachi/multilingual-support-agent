# Project context

Portfolio project: a multilingual (Spanish, Japanese, English) customer support AI agent for a fictional online store. The owner is learning, so the goal is understanding as much as a working result.

The text agent is finished (phases 0-5). **Current work: adding a voice channel** on the `voice` branch (phases 6-11 below).

## Voice channel goal

voice -> STT (speech-to-text) -> existing agent -> TTS (text-to-speech) -> audio, in Spanish, Japanese and English. The brain (tools, policies, tool-use loop) is **not duplicated**: voice is one more interface around the same `SupportAgent`, selected through a `Channel` (see Conventions).

## Hard constraints

- Python 3.10+.
- Claude API through the official `anthropic` SDK, with tool use.
- **No agent frameworks** (LangChain, etc.). The agent loop is written by hand: model decides -> tool runs -> result is returned to the model -> repeat until it answers. The loop must have an iteration limit.
- The model name comes from `ANTHROPIC_MODEL` in `.env`. Never hardcode it.
- The API key lives only in `.env`, which is git-ignored. Never commit it, print it or copy it elsewhere.
- Simulated data in SQLite. Interfaces: a CLI text chat, and a push-to-talk voice mode in the terminal that must work on Windows.
- All API keys (Anthropic and any STT/TTS provider) live only in `.env`. When a new variable is added, update `.env.example` in the same commit.
- Tools return clear error results (order not found, email mismatch, outside policy) instead of raising.
- Every conversation is saved as structured JSON (messages, tool calls, arguments, results, token usage, latency). An eval system will be built on these logs later, so keep them stable and machine-readable.

## Language

- Code, comments, commit messages and README: **English**.
- Explanations to the project owner: **Spanish**, simple, with analogies.

## Workflow

Work phase by phase. At the end of each phase:

1. Run the tests.
2. Commit with a clear message in English.
3. Push.
4. Explain to the owner what was done and why, in simple Spanish with analogies.
5. **Stop and wait for approval** before starting the next phase.

Also:

- Small, frequent commits inside each phase.
- If something is ambiguous, ask instead of assuming.
- At the end of each phase, update this file (these rules, the voice goal, phases done, next phase) so the work can continue in a new session without losing context.
- Voice work happens on the `voice` branch. At the end (phase 11) a Pull Request to `main` is opened with `gh pr create`. Do not merge into `main` before that.

## Phases

Phases 0-5 (text agent) and phase 6 are complete. **Next: phase 7.** The eval system built on the conversation logs is still planned for after the voice channel.

### Text agent

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
5. (done) Final README: what it does, Mermaid architecture diagram, how to run, design decisions and trade-offs, known limitations.

### Voice channel (branch `voice`)

6. (done) Separate the brain from the interface: the agent core no longer depends on the channel. The text CLI works exactly as before.
7. **(next)** Provider research, no code: propose 2-3 STT and 2-3 TTS options, at least one local or free. Comparison table: quality in Spanish/Japanese/English, latency, approximate cost, whether it runs on Windows, ease of integration. Give a recommendation and wait for the owner to choose.
8. Basic voice loop: push-to-talk in the terminal (press a key, speak, release, the agent answers with audio), working on Windows. Add a voice mode to the system prompt: short, conversational answers, no Markdown or lists, numbers and IDs written so they read well aloud.
9. Measure latency: per turn, log the time of each stage (STT, LLM including tool calls, TTS, total) in the JSON logs, plus a script that prints average and worst case per stage.
10. Voice-specific problems: the agent repeats and confirms key data (order numbers, emails) before using tools, because dictation garbles them; empty audio, noise or unintelligible transcripts make it ask the customer to repeat; test audio files generated with the TTS from the existing scenarios in the three languages, and a script that runs them through the whole pipeline without a microphone.
11. README and Pull Request: Mermaid diagram of the voice flow, table of measured latencies, decisions and trade-offs (providers chosen, why push-to-talk), limitations, a "Next steps" section (streaming to cut latency, barge-in, connecting to a real phone line). Open the PR to `main` with a clear description.

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
- `support_agent/prompts.py` — `build_system_prompt(setting, style)`: the store policies shared by every channel plus the two parts a channel supplies (store name: Kumo Market). `SYSTEM_PROMPT` is the text chat's prompt
- `support_agent/channels.py` — `Channel` (name, system prompt, fallback message) and `TEXT_CHANNEL`
- `support_agent/config.py` — `Settings` from env (`ANTHROPIC_MODEL`, optional `ANTHROPIC_EFFORT`, `AGENT_MAX_ITERATIONS`)
- `support_agent/agent.py` — `SupportAgent.reply()`, the hand-written loop; returns `AgentReply` (text, stop_reason, iterations, latency_ms, model_calls, tool_calls, usage, error). `build_agent(conn, settings, channel)` creates the live one
- `support_agent/conversation_log.py` — `ConversationLog`: one JSON file per conversation (`record_turn`, `save(transcript)`); accepts free-form `metadata` and the `channel`
- `support_agent/cli.py` + `__main__.py` — terminal chat, the interface of the text channel; `run_chat` takes injectable `read`/`write` so tests can script it
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
- Tests never call the real API: they script a `FakeClient` (`tests/fakes.py`). For a live check without typing, pipe lines into the chat: `printf '%s\n' "message" "/exit" | python -m support_agent`.
- The conversation history is append-only and model turns are stored unchanged (`response.content`, thinking blocks included). Do not edit or strip earlier turns.
- A test fails if any file in `support_agent/` contains a model name; the model only comes from `.env`.
- `SupportAgent.reply()` does not raise on API failures: it returns `stop_reason="api_error"` with `reply.error` set, so the turn is still logged.
- The log format is a contract for the evals: bump `SCHEMA_VERSION` in `conversation_log.py` when a field is renamed or removed. Adding fields is fine.
- On this project the API client must be created after `load_settings()` (which loads `.env`); see `build_agent`.
- Scenario checks stay deterministic and narrow (tool outcomes, forbidden strings). Judging language, tone or accuracy belongs to the future eval system, not to `check_expectations`.
- Scenarios depend on the anchor orders ORD-1001..ORD-1008 in `seed.py`; a test fails if a scenario mentions an order or email the seed does not have.
- The agent core is channel-independent. `SupportAgent.reply(text)` takes text and returns an `AgentReply`; it never reads input or produces output itself. Whatever depends on the channel goes in a `Channel` passed to both `SupportAgent` and `ConversationLog` (the same one to both), never in `agent.py`, `tools.py` or the shared policies of `prompts.py`. A new channel is a new `Channel` plus an interface that calls `reply()`.
- The text channel is the default everywhere and its system prompt must stay byte-identical unless the owner agrees to a change: logs identify the prompt by `system_prompt_sha256`. Logs also carry the channel name in `channel`.
- Do not tune the system prompt just to make a scenario pass without telling the owner; a failing scenario is information.
