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
- **The voice channel must cost nothing** (owner's decision, 2026-10-06): only free STT/TTS providers that need no payment method. Never add a paid provider or a paid tier. If one of the three languages does not work well in voice with free providers, it may be skipped for now: tell the owner instead of paying for a fix. The Claude API calls of the agent itself are the only paid part and already existed.
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

Phases 0-5 (text agent) and phases 6-8 are complete. **Next: phase 9.** The eval system built on the conversation logs is still planned for after the voice channel.

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
7. (done) Provider research, no code: propose 2-3 STT and 2-3 TTS options, at least one local or free. Comparison table: quality in Spanish/Japanese/English, latency, approximate cost, whether it runs on Windows, ease of integration. Give a recommendation and wait for the owner to choose.
   - STT shortlist: Groq `whisper-large-v3-turbo` (free tier, fastest, OpenAI-compatible API, returns the detected language), OpenAI `gpt-4o-mini-transcribe` / `gpt-transcribe` (paid, no detected language), local `faster-whisper` (free, offline; on this CPU `small` is usable, `large-v3-turbo` is slower than real time).
   - TTS shortlist: OpenAI `gpt-4o-mini-tts` (paid, one voice for all three languages, WAV/PCM output), `edge-tts` (free, unofficial Microsoft endpoint, native neural voice per language, MP3 only), Windows SAPI voices (local, free, robotic; es-MX, en-US and ja-JP are already installed on the owner's machine).
   - **Chosen: Groq free tier for STT + `edge-tts` for TTS**, because the owner wants everything free (OpenAI TTS was ruled out for being paid). Both go behind a small interface so a provider can be swapped from `.env`. Free fallbacks if one of them stops working: local `faster-whisper` for STT, the Windows SAPI voices for TTS.
   - Known risks of the choice: `edge-tts` is unofficial and may break; it needs the language to pick a voice (taken from the STT result); Whisper can misdetect the language of very short clips; Norton TLS interception may break new HTTP libraries on the owner's machine (fix: OS trust store or the Norton pem, never disabling verification).
   - Owner's machine: Ryzen 7 4700U (8 cores), 15 GB RAM, no NVIDIA GPU, about 23 GB free disk. Local models run on CPU only.
   - Phase 8 needs a `GROQ_API_KEY` in `.env`. The owner creates the Groq account and key (free, no card); never create accounts or type keys for them.
8. (done) Basic voice loop: push-to-talk in the terminal (hold SPACE, speak, release, the agent answers with audio), working on Windows. Voice mode in the system prompt: short, conversational answers, no Markdown or lists, numbers and IDs written so they read well aloud.
   - Verified without a microphone in the three languages (synthesized customer audio -> Groq -> live agent -> edge-tts). The real microphone and key were smoke-tested on the owner's machine but a spoken conversation has to be tried by the owner.
   - Left for later phases on purpose: stage timings in the log (phase 9); confirming order numbers and emails, and spoken handling of silence, noise or an unrecognised language (phase 10). Today silence only prints a message, and an unrecognised language keeps the previous turn's voice.
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
python -m support_agent --voice   # push-to-talk voice chat: hold SPACE to talk, N new conversation, Q or Esc to leave. Needs a microphone, GROQ_API_KEY and STT_MODEL in .env
python -m support_agent.run_scenarios   # ten scripted conversations against the live API (costs ~$0.12 per full run)
```

## Layout

- `support_agent/db.py` — SQLite connection and schema
- `support_agent/seed.py` — deterministic simulated data; ORD-1001..ORD-1008 are hand-picked anchor orders with a known state
- `support_agent/tools.py` — the five support tools
- `support_agent/tool_registry.py` — JSON Schema tool definitions sent to the model, and `execute_tool` dispatcher
- `support_agent/prompts.py` — `build_system_prompt(setting, identifiers, style)`: the store policies shared by every channel plus the three parts a channel supplies (store name: Kumo Market). `SYSTEM_PROMPT` is the text chat's prompt, `VOICE_SYSTEM_PROMPT` the voice call's
- `support_agent/channels.py` — `Channel` (name, system prompt, fallback message, optional per-language fallbacks), `TEXT_CHANNEL` and `VOICE_CHANNEL`
- `support_agent/config.py` — `Settings` from env (`ANTHROPIC_MODEL`, optional `ANTHROPIC_EFFORT`, `AGENT_MAX_ITERATIONS`) and `VoiceSettings` (`GROQ_API_KEY`, `STT_MODEL`, optional `TTS_VOICE_ES/JA/EN`)
- `support_agent/agent.py` — `SupportAgent.reply()`, the hand-written loop; returns `AgentReply` (text, stop_reason, iterations, latency_ms, model_calls, tool_calls, usage, error). `build_agent(conn, settings, channel)` creates the live one
- `support_agent/conversation_log.py` — `ConversationLog`: one JSON file per conversation (`record_turn`, `save(transcript)`); accepts free-form `metadata` and the `channel`
- `support_agent/cli.py` + `__main__.py` — entry point and terminal chat, the interface of the text channel; `run_chat` takes injectable `read`/`write` so tests can script it. `--voice` hands over to `voice/chat.py`
- `support_agent/voice/` — the voice channel's interface (imported only with `--voice`):
  - `speech.py` — `SpeechToText` / `TextToSpeech` protocols, `Transcript`, `SpeechError`, `LANGUAGES`
  - `audio.py` — `Audio` (mono int16 + sample rate, `to_wav`, `from_encoded`), `Microphone` (stream kept open, `record_while(is_held)`), `Speaker`
  - `stt_groq.py` — `GroqSpeechToText`: one HTTP POST with `httpx2`, returns text and language
  - `tts_edge.py` — `EdgeTextToSpeech`: one voice per language, MP3 decoded with `soundfile`
  - `keys.py` — `PushToTalkKey`: `wait()` -> TALK / NEW / QUIT, `is_held()`, `drain()`; Windows only
  - `chat.py` — `run_voice_chat` (every part injectable) and `start_voice_chat` (wires the real ones)
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
- Voice code never decides what to answer: it only turns sound into the text passed to `reply()` and the reply back into sound. `agent.reply(text, language=...)` takes the language only to choose the fallback message; it is not sent to the model.
- Speech providers sit behind the `SpeechToText` / `TextToSpeech` protocols and raise `SpeechError`; the voice loop catches it and keeps the chat alive. Unit tests use fakes (`tests/test_voice_chat.py`) and never touch the microphone, the speaker or the network.
- The speech-to-text model name comes from `STT_MODEL` in `.env`, like the Claude model.
- `tts_edge.py` calls `truststore.inject_into_ssl()` before importing `edge_tts`: the owner's antivirus re-signs HTTPS and edge-tts would fail certificate checks otherwise. Do not remove it or disable verification. `Communicate.stream_sync()` hung on that error, so the async API is used with a time limit.
- The voice prompt (`VOICE_*` in `prompts.py`) is new and may be tuned, but tell the owner what changed and why. It was adjusted once in phase 8: number words in the language being spoken, and tracking numbers offered instead of recited.
- The text channel is the default everywhere and its system prompt must stay byte-identical unless the owner agrees to a change: logs identify the prompt by `system_prompt_sha256`. Logs also carry the channel name in `channel`.
- Do not tune the system prompt just to make a scenario pass without telling the owner; a failing scenario is information.
