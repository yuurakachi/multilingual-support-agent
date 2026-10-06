"""Tests for channels: what depends on how the customer reaches the agent."""

import hashlib
import json

from fakes import FakeClient, response, text, thinking

from support_agent.agent import SupportAgent
from support_agent.channels import FALLBACK_MESSAGE, TEXT_CHANNEL, Channel
from support_agent.config import Settings
from support_agent.conversation_log import ConversationLog
from support_agent.prompts import SYSTEM_PROMPT, TEXT_SETTING, TEXT_STYLE, build_system_prompt

SETTINGS = Settings(model="test-model")

# A channel that is not the text chat, to prove nothing is tied to it.
OTHER_CHANNEL = Channel(
    name="other",
    system_prompt=build_system_prompt("You are on a phone call.", "Keep every answer short."),
    fallback_message="Sorry, please say that again.",
)


def sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_text_channel_keeps_the_original_prompt_and_fallback():
    assert TEXT_CHANNEL.name == "text"
    assert TEXT_CHANNEL.system_prompt == SYSTEM_PROMPT
    assert TEXT_CHANNEL.fallback_message == FALLBACK_MESSAGE
    assert TEXT_SETTING in SYSTEM_PROMPT
    assert TEXT_STYLE in SYSTEM_PROMPT


def test_channels_differ_only_in_setting_and_style():
    as_other = SYSTEM_PROMPT.replace(TEXT_SETTING, "You are on a phone call.").replace(
        TEXT_STYLE, "Keep every answer short."
    )

    assert OTHER_CHANNEL.system_prompt == as_other
    assert OTHER_CHANNEL.system_prompt != SYSTEM_PROMPT
    for section in ("# Verifying identity", "# Store policies", "# Tickets and human agents"):
        assert section in OTHER_CHANNEL.system_prompt


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
    assert reply.text == "Sorry, please say that again."
    assert agent.messages[-1] == {"role": "assistant", "content": "Sorry, please say that again."}


def test_log_records_the_channel_and_its_prompt(tmp_path):
    log = ConversationLog(SETTINGS, log_dir=tmp_path, channel=OTHER_CHANNEL)

    saved = json.loads(log.save().read_text(encoding="utf-8"))

    assert saved["channel"] == "other"
    assert saved["system_prompt_sha256"] == sha256(OTHER_CHANNEL.system_prompt)


def test_log_defaults_to_the_text_channel(tmp_path):
    saved = json.loads(ConversationLog(SETTINGS, log_dir=tmp_path).save().read_text("utf-8"))

    assert saved["channel"] == "text"
    assert saved["system_prompt_sha256"] == sha256(SYSTEM_PROMPT)
