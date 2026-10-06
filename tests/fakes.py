"""Scripted stand-ins for the Anthropic client, shared by the agent and CLI tests."""

from types import SimpleNamespace

import anthropic
import httpx2


def text(value):
    return SimpleNamespace(type="text", text=value)


def thinking():
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def tool_use(block_id, name, **tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(*content, stop_reason="end_turn", input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=3,
        ),
    )


class FakeClient:
    """Stands in for anthropic.Anthropic(): returns scripted responses in order."""

    def __init__(self, *responses, repeat_last=False):
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        # The agent keeps mutating its history, so snapshot what was sent.
        self.requests.append({**request, "messages": list(request["messages"])})
        if self._repeat_last and len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


def api_connection_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="Connection error.", request=request)


class FailingClient(FakeClient):
    """Returns its scripted responses, then fails like a dropped connection."""

    def _create(self, **request):
        if not self._responses:
            self.requests.append(request)
            raise api_connection_error()
        return super()._create(**request)
