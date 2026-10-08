"""Tests for channels: what depends on how the customer reaches the agent."""

import hashlib
import json

from fakes import FakeClient, response, text, thinking

from support_agent.agent import SupportAgent
from support_agent.channels import (
    FALLBACK_MESSAGE,
    TEXT_CHANNEL,
    VOICE_CHANNEL,
    VOICE_FALLBACK_MESSAGES,
    Channel,
)
from support_agent.config import Settings
from support_agent.conversation_log import ConversationLog
from support_agent.prompts import (
    SYSTEM_PROMPT,
    TEXT_IDENTIFIERS,
    TEXT_SETTING,
    TEXT_STYLE,
    VOICE_IDENTIFIERS,
    VOICE_SETTING,
    VOICE_STYLE,
    VOICE_SYSTEM_PROMPT,
    build_system_prompt,
)

SETTINGS = Settings(model="test-model")

# A channel that is neither text nor voice, to prove nothing is tied to them.
OTHER_CHANNEL = Channel(
    name="other",
    system_prompt=build_system_prompt(
        "You are answering letters.", "Copy codes carefully.", "Keep every answer short."
    ),
    fallback_message="Sorry, please write again.",
)

SHARED_SECTIONS = (
    "# Language",
    "# Verifying identity",
    "# Staying accurate",
    "# Store policies",
    "# Tickets and human agents",
)


def sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_text_channel_keeps_the_original_prompt_and_fallback():
    assert TEXT_CHANNEL.name == "text"
    assert TEXT_CHANNEL.system_prompt == SYSTEM_PROMPT
    assert TEXT_CHANNEL.fallback_message == FALLBACK_MESSAGE
    # The text chat does not know the customer's language: one message for all.
    assert TEXT_CHANNEL.fallback_for("ja") == FALLBACK_MESSAGE
    for part in (TEXT_SETTING, TEXT_IDENTIFIERS, TEXT_STYLE):
        assert part in SYSTEM_PROMPT


def test_channels_differ_only_in_their_three_parts():
    as_other = (
        SYSTEM_PROMPT.replace(TEXT_SETTING, "You are answering letters.")
        .replace(TEXT_IDENTIFIERS, "Copy codes carefully.")
        .replace(TEXT_STYLE, "Keep every answer short.")
    )

    assert OTHER_CHANNEL.system_prompt == as_other
    assert OTHER_CHANNEL.system_prompt != SYSTEM_PROMPT
    for section in SHARED_SECTIONS:
        assert section in OTHER_CHANNEL.system_prompt


def test_voice_channel_shares_the_policies_and_has_its_own_style():
    assert VOICE_CHANNEL.name == "voice"
    assert VOICE_CHANNEL.system_prompt == VOICE_SYSTEM_PROMPT
    for part in (VOICE_SETTING, VOICE_IDENTIFIERS, VOICE_STYLE):
        assert part in VOICE_SYSTEM_PROMPT
    for part in (TEXT_SETTING, TEXT_IDENTIFIERS, TEXT_STYLE):
        assert part not in VOICE_SYSTEM_PROMPT
    for section in SHARED_SECTIONS:
        assert section in VOICE_SYSTEM_PROMPT

    # Same policies, word for word: swapping the three parts gives the text prompt back.
    as_text = (
        VOICE_SYSTEM_PROMPT.replace(VOICE_SETTING, TEXT_SETTING)
        .replace(VOICE_IDENTIFIERS, TEXT_IDENTIFIERS)
        .replace(VOICE_STYLE, TEXT_STYLE)
    )
    assert as_text == SYSTEM_PROMPT


def test_voice_fallback_is_spoken_in_one_language():
    for language in ("es", "ja", "en"):
        assert VOICE_CHANNEL.fallback_for(language) == VOICE_FALLBACK_MESSAGES[language]
        # A single language, so no line breaks between translations.
        assert "\n" not in VOICE_FALLBACK_MESSAGES[language]

    assert VOICE_CHANNEL.fallback_for(None) == VOICE_FALLBACK_MESSAGES["en"]
    assert VOICE_CHANNEL.fallback_for("pt") == VOICE_FALLBACK_MESSAGES["en"]


def test_agent_uses_the_text_channel_by_default(conn):
    assert SupportAgent(FakeClient(), conn, SETTINGS).channel is TEXT_CHANNEL


def test_agent_sends_the_prompt_of_its_channel(conn):
    client = FakeClient(response(text("Hello.")))

    SupportAgent(client, conn, SETTINGS, channel=OTHER_CHANNEL).reply("hi")

    assert client.requests[0]["system"] == OTHER_CHANNEL.system_prompt


def test_agent_falls_back_with_the_message_of_its_channel(conn):
    agent = SupportAgent(FakeClient(response(thinking())), conn, SETTINGS, channel=OTHER_CHANNEL)

    reply = agent.reply("hi")

    assert reply.stop_reason == "empty_response"
    assert reply.text == "Sorry, please write again."
    assert agent.messages[-1] == {"role": "assistant", "content": "Sorry, please write again."}


def test_agent_falls_back_in_the_language_it_was_told(conn):
    agent = SupportAgent(FakeClient(response(thinking())), conn, SETTINGS, channel=VOICE_CHANNEL)

    reply = agent.reply("注文はどこですか", language="ja")

    assert reply.text == VOICE_FALLBACK_MESSAGES["ja"]
    assert agent.messages[-1] == {"role": "assistant", "content": VOICE_FALLBACK_MESSAGES["ja"]}


def test_language_is_not_sent_to_the_model(conn):
    client = FakeClient(response(text("Hola.")))

    SupportAgent(client, conn, SETTINGS, channel=VOICE_CHANNEL).reply("hola", language="es")

    assert client.requests[0]["messages"] == [{"role": "user", "content": "hola"}]


def test_log_records_the_channel_and_its_prompt(tmp_path):
    log = ConversationLog(SETTINGS, log_dir=tmp_path, channel=VOICE_CHANNEL)

    saved = json.loads(log.save().read_text(encoding="utf-8"))

    assert saved["channel"] == "voice"
    assert saved["system_prompt_sha256"] == sha256(VOICE_SYSTEM_PROMPT)


def test_log_defaults_to_the_text_channel(tmp_path):
    saved = json.loads(ConversationLog(SETTINGS, log_dir=tmp_path).save().read_text("utf-8"))

    assert saved["channel"] == "text"
    assert saved["system_prompt_sha256"] == sha256(SYSTEM_PROMPT)
