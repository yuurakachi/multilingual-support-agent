"""Voice channel: speech in, speech out, around the same agent as the text chat.

    microphone -> speech-to-text -> SupportAgent.reply() -> text-to-speech -> speaker

Nothing in here decides what to answer. These modules only turn sound into the
text the agent reads, and the agent's text back into sound.
"""
