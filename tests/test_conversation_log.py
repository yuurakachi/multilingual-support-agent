import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from support_agent.agent import AgentReply, ModelCall, ToolCall
from support_agent.config import Settings
from support_agent.conversation_log import SCHEMA_VERSION, ConversationLog

SETTINGS = Settings(model="test-model", effort="medium", max_iterations=4)


class FakeNow:
    """A clock that starts at a fixed instant and moves one second per reading."""

    def __init__(self):
        self.current = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        value = self.current
        self.current += timedelta(seconds=1)
        return value


def usage(input_tokens, output_tokens):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def reply_with_tool():
    return AgentReply(
        text="ご注文は発送済みです。",
        stop_reason="end_turn",
        iterations=2,
        latency_ms=1800,
        model_calls=[
            ModelCall(1, "tool_use", 900, usage(100, 20)),
            ModelCall(2, "end_turn", 800, usage(150, 30)),
        ],
        tool_calls=[
            ToolCall(
                iteration=1,
                id="toolu_1",
                name="get_order_status",
                input={"order_id": "ORD-2", "email": "yuki@example.jp"},
                result={"ok": True, "status": "shipped", "customer_name": "田中 由紀"},
                is_error=False,
                latency_ms=3,
            )
        ],
        usage=usage(250, 50),
    )


def new_log(tmp_path, **kwargs):
    return ConversationLog(SETTINGS, log_dir=tmp_path, now=FakeNow(), **kwargs)


def test_nothing_is_written_until_save(tmp_path):
    log = new_log(tmp_path)

    assert log.path.parent == tmp_path
    assert list(tmp_path.iterdir()) == []


def test_saved_file_has_the_conversation_header(tmp_path):
    log = new_log(tmp_path, metadata={"scenario": "happy_path_ja"})
    log.record_turn("注文はどこですか？", reply_with_tool())

    saved = json.loads(log.save().read_text(encoding="utf-8"))

    assert saved["schema_version"] == SCHEMA_VERSION
    assert saved["conversation_id"] == log.conversation_id
    assert log.conversation_id.startswith("20261001T120000Z_")
    assert log.path.name == f"{log.conversation_id}.json"
    assert saved["started_at"] == "2026-10-01T12:00:00+00:00"
    assert saved["model"] == "test-model"
    assert saved["effort"] == "medium"
    assert saved["max_iterations"] == 4
    assert len(saved["system_prompt_sha256"]) == 64
    assert saved["metadata"] == {"scenario": "happy_path_ja"}


def test_turn_records_messages_tools_tokens_and_latency(tmp_path):
    log = new_log(tmp_path)
    log.record_turn("注文はどこですか？", reply_with_tool())

    (turn,) = json.loads(log.save().read_text(encoding="utf-8"))["turns"]

    assert turn["index"] == 1
    assert turn["user_message"] == "注文はどこですか？"
    assert turn["assistant_message"] == "ご注文は発送済みです。"
    assert turn["stop_reason"] == "end_turn"
    assert turn["completed"] is True
    assert turn["iterations"] == 2
    assert turn["latency_ms"] == 1800
    assert turn["usage"] == usage(250, 50)
    assert turn["error"] is None
    assert turn["model_calls"] == [
        {"iteration": 1, "stop_reason": "tool_use", "latency_ms": 900, "usage": usage(100, 20)},
        {"iteration": 2, "stop_reason": "end_turn", "latency_ms": 800, "usage": usage(150, 30)},
    ]
    assert turn["tool_calls"] == [
        {
            "iteration": 1,
            "id": "toolu_1",
            "name": "get_order_status",
            "input": {"order_id": "ORD-2", "email": "yuki@example.jp"},
            "result": {"ok": True, "status": "shipped", "customer_name": "田中 由紀"},
            "is_error": False,
            "latency_ms": 3,
        }
    ]


def test_typed_turn_has_no_voice_data(tmp_path):
    log = new_log(tmp_path)
    log.record_turn("hello", reply_with_tool())

    (turn,) = json.loads(log.save().read_text(encoding="utf-8"))["turns"]

    assert turn["voice"] is None


def test_spoken_turn_keeps_its_voice_data_and_later_additions(tmp_path):
    log = new_log(tmp_path)
    timings = {"stt": 900, "llm": 1800, "tts": None, "total": None}
    voice = {"language": "ja", "recording_seconds": 4.2, "timings_ms": timings}
    log.record_turn("注文はどこですか？", reply_with_tool(), voice=voice)
    log.save()

    # The text-to-speech time is only known after the turn was first saved.
    timings.update(tts=1200, total=3950)
    (turn,) = json.loads(log.save().read_text(encoding="utf-8"))["turns"]

    assert turn["voice"] == {
        "language": "ja",
        "recording_seconds": 4.2,
        "timings_ms": {"stt": 900, "llm": 1800, "tts": 1200, "total": 3950},
    }


def test_failed_turn_keeps_its_error(tmp_path):
    log = new_log(tmp_path)
    error = {"type": "APIConnectionError", "message": "Connection error.", "status_code": None}
    log.record_turn("hello", AgentReply(text="Sorry", stop_reason="api_error", error=error))

    (turn,) = json.loads(log.save().read_text(encoding="utf-8"))["turns"]

    assert turn["completed"] is False
    assert turn["stop_reason"] == "api_error"
    assert turn["error"] == error


def test_totals_add_up_every_turn(tmp_path):
    log = new_log(tmp_path)
    log.record_turn("first", reply_with_tool())
    log.record_turn("second", reply_with_tool())

    totals = json.loads(log.save().read_text(encoding="utf-8"))["totals"]

    assert totals == {
        "turns": 2,
        "model_calls": 4,
        "tool_calls": 2,
        "latency_ms": 3600,
        "usage": usage(500, 100),
    }


def test_non_ascii_text_is_stored_readable(tmp_path):
    log = new_log(tmp_path)
    log.record_turn("¿Dónde está mi pedido?", reply_with_tool())

    raw = log.save().read_text(encoding="utf-8")

    assert "¿Dónde está mi pedido?" in raw
    assert "田中 由紀" in raw
    assert "\\u" not in raw


def test_transcript_serialises_sdk_content_blocks(tmp_path):
    class PydanticLikeBlock:
        def model_dump(self, mode):
            assert mode == "json"
            return {"type": "text", "text": "from the SDK"}

    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [PydanticLikeBlock()]},
        {"role": "assistant", "content": [SimpleNamespace(type="text", text="from a test fake")]},
    ]
    log = new_log(tmp_path)

    saved = json.loads(log.save(transcript).read_text(encoding="utf-8"))

    assert saved["transcript"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "from the SDK"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "from a test fake"}]},
    ]


def test_saving_again_updates_the_same_file(tmp_path):
    log = new_log(tmp_path)
    log.record_turn("first", reply_with_tool())
    first_path = log.save()
    log.record_turn("second", reply_with_tool())
    second_path = log.save()

    assert first_path == second_path
    assert [path.name for path in tmp_path.iterdir()] == [first_path.name]
    assert len(json.loads(second_path.read_text(encoding="utf-8"))["turns"]) == 2


def test_each_conversation_gets_its_own_file(tmp_path):
    first, second = new_log(tmp_path), new_log(tmp_path)

    assert first.path != second.path
