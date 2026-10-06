"""Channels: the ways a customer can reach the agent.

The loop, the tools and the store policies are the same on every channel. A
Channel holds the little that depends on how the conversation travels: the
system prompt (shared policies plus that channel's setting and style) and the
message used when the loop has to stop without an answer.

The agent takes a Channel and never looks at where the customer's text came
from or where its reply goes: reading a keyboard or a microphone, and printing
or speaking the answer, is the job of the interface built around it.
"""

from __future__ import annotations

from dataclasses import dataclass

from support_agent.prompts import SYSTEM_PROMPT

# Shown when the loop has to stop without an answer from the model. It cannot be
# written in the customer's language by the model, so it carries all three.
FALLBACK_MESSAGE = (
    "Sorry, I could not complete your request. Please try again, or ask to speak with a "
    "human agent.\n"
    "Lo siento, no pude completar tu solicitud. Inténtalo de nuevo o pide hablar con un "
    "agente humano.\n"
    "申し訳ございません。ご依頼を完了できませんでした。もう一度お試しいただくか、"
    "担当者との会話をご希望の旨をお伝えください。"
)


@dataclass(frozen=True)
class Channel:
    # Stable identifier, written to the conversation log.
    name: str
    system_prompt: str
    fallback_message: str


TEXT_CHANNEL = Channel(
    name="text", system_prompt=SYSTEM_PROMPT, fallback_message=FALLBACK_MESSAGE
)
