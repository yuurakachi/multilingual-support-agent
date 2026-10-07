"""Tests for the spoken scenarios: their script, sound files, checks and runner.

Nothing here calls a real API: speech-to-text, text-to-speech and the model are fakes.
"""

import json
from datetime import date

import pytest
from fakes import FakeClient, response, text, tool_use
from voice_fakes import FakeSpeechToText, FakeTextToSpeech, speech

from support_agent.agent import AgentReply, ToolCall
from support_agent.config import Settings
from support_agent.run_voice_scenarios import main, run_all
from support_agent.scenarios import Expectations, Scenario, ScenarioError, load_scenarios
from support_agent.voice.pipeline import SILENCE, Answer, Heard
from support_agent.voice.scenarios import (
    CONFIRMATIONS,
    CUSTOMER_VOICES,
    IDENTITY_TOOLS,
    MANIFEST_NAME,
    Exchange,
    SpokenLine,
    audio_problems,
    check_voice,
    expected_clips,
    make_audio,
    run_voice_scenario,
    spoken_lines,
)
from support_agent.voice.speech import Transcript

SETTINGS = Settings(model="test-model")
TODAY = date(2026, 10, 1)

STATUS = Scenario(
    id="status",
    language="es",
    category="happy_path",
    description="Order status.",
    turns=("¿Cómo va mi pedido ORD-1002? Mi correo es maria.garcia@example.com",),
    expect=Expectations(must_succeed=("get_order_status",)),
    voice_confirm_after=(1,),
)
CHAT = Scenario(
    id="chat",
    language="en",
    category="happy_path",
    description="Two lines, nothing to confirm.",
    turns=("Hello.", "Thanks, bye."),
    expect=Expectations(),
)


class ToneTextToSpeech(FakeTextToSpeech):
    """Makes audible clips, so files made from it count as someone talking."""

    def synthesize(self, text, language):
        self.spoken.append((text, language))
        return speech(seconds=1.0, sample_rate=24_000)


@pytest.fixture
def audio_dir(tmp_path):
    folder = tmp_path / "audio"
    make_audio([STATUS, CHAT], ToneTextToSpeech(), folder, write=lambda line: None)
    return folder


# --- the script ---


def test_the_customer_confirms_after_the_listed_turns():
    scenario = Scenario(
        id="demo",
        language="ja",
        category="happy_path",
        description="",
        turns=("one", "two", "three"),
        expect=Expectations(),
        voice_confirm_after=(1, 3),
    )

    lines = spoken_lines(scenario)

    assert [line.text for line in lines] == [
        "one",
        CONFIRMATIONS["ja"],
        "two",
        "three",
        CONFIRMATIONS["ja"],
    ]
    assert [line.file for line in lines] == [
        "demo_1.mp3",
        "confirm_ja.mp3",
        "demo_2.mp3",
        "demo_3.mp3",
        "confirm_ja.mp3",
    ]
    assert [line.is_confirmation for line in lines] == [False, True, False, False, True]
    assert {line.language for line in lines} == {"ja"}


def test_a_scenario_without_confirmations_is_spoken_as_written():
    assert [line.text for line in spoken_lines(CHAT)] == list(CHAT.turns)


def test_confirmation_turns_must_exist(tmp_path):
    path = tmp_path / "scenarios.json"
    scenario = {
        "id": "demo",
        "language": "en",
        "category": "happy_path",
        "description": "",
        "turns": ["hello"],
        "expect": {},
        "voice_confirm_after": [2],
    }
    path.write_text(json.dumps([scenario]), encoding="utf-8")

    with pytest.raises(ScenarioError, match="voice_confirm_after"):
        load_scenarios(path)


def test_every_tool_that_takes_an_order_or_email_waits_for_confirmation():
    assert IDENTITY_TOOLS == {
        "get_order_status",
        "update_shipping_address",
        "request_refund",
        "create_support_ticket",
    }


# --- the sound files ---


def test_every_language_has_a_customer_voice_and_a_confirmation():
    assert set(CUSTOMER_VOICES) == set(CONFIRMATIONS) == {"es", "ja", "en"}


def test_one_confirmation_clip_serves_every_scenario_of_a_language():
    other = Scenario("other", "es", "happy_path", "", ("hola",), Expectations(), (1,))

    clips = expected_clips([STATUS, other])

    assert list(clips) == ["status_1.mp3", "confirm_es.mp3", "other_1.mp3"]
    assert clips["confirm_es.mp3"] == {"language": "es", "text": CONFIRMATIONS["es"]}


def test_make_audio_writes_every_clip_and_a_manifest(tmp_path):
    tts = ToneTextToSpeech()
    printed = []

    make_audio([STATUS], tts, tmp_path / "audio", write=printed.append)

    assert sorted(path.name for path in (tmp_path / "audio").iterdir()) == [
        "confirm_es.mp3",
        MANIFEST_NAME,
        "status_1.mp3",
    ]
    assert tts.spoken == [(STATUS.turns[0], "es"), (CONFIRMATIONS["es"], "es")]
    manifest = json.loads((tmp_path / "audio" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["clips"]["status_1.mp3"] == {"language": "es", "text": STATUS.turns[0]}
    assert len(printed) == 2
    assert audio_problems([STATUS], tmp_path / "audio") == []


def test_stale_or_missing_sound_files_are_reported(audio_dir, tmp_path):
    edited = Scenario("status", "es", "happy_path", "", ("Otra frase.",), Expectations(), (1,))
    assert audio_problems([edited], audio_dir) == ["status_1.mp3 was made from a different text"]

    (audio_dir / "chat_2.mp3").unlink()
    assert audio_problems([CHAT], audio_dir) == ["chat_2.mp3 is missing"]

    assert "no usable manifest.json" in audio_problems([CHAT], tmp_path / "nowhere")[0]


def test_the_committed_sound_files_match_the_scenarios():
    """Guards against editing scenarios.json without regenerating the audio."""
    assert audio_problems(load_scenarios()) == []


# --- the voice checks ---


def exchange(line_text, *tool_names, is_confirmation=False, understood=True, voiced=True):
    line = SpokenLine(line_text, "en", "x.mp3", is_confirmation)
    heard = Heard(speech(), 0.0, 5000, Transcript(line_text, "en"), 100)
    if not understood:
        heard.problem = SILENCE
        return Exchange(line, heard)
    calls = [ToolCall(1, f"toolu_{name}", name, {}, {"ok": True}, False, 1) for name in tool_names]
    reply = AgentReply(text="ok", stop_reason="end_turn", tool_calls=calls)
    return Exchange(line, heard, Answer(reply, speech() if voiced else None, {}))


def descriptions(checks):
    return {check.description: check.passed for check in checks}


def test_a_tool_called_after_the_confirmation_passes():
    checks = check_voice(
        [exchange("ORD-1 a@b.c"), exchange("yes", "get_order_status", is_confirmation=True)]
    )

    assert all(check.passed for check in checks)
    assert "no order number or email was used before the customer confirmed it" in descriptions(
        checks
    )


def test_a_tool_called_before_the_confirmation_fails():
    checks = check_voice(
        [exchange("ORD-1 a@b.c", "get_order_status"), exchange("yes", is_confirmation=True)]
    )

    (failed,) = [check for check in checks if not check.passed]
    assert "used early by: get_order_status" in failed.description


def test_handing_over_to_a_human_does_not_need_a_confirmation():
    checks = check_voice(
        [exchange("I am furious", "escalate_to_human"), exchange("yes", is_confirmation=True)]
    )

    assert all(check.passed for check in checks)


def test_without_confirmation_lines_there_is_no_confirmation_check():
    checks = check_voice([exchange("hello", "get_order_status")])

    assert list(descriptions(checks)) == [
        "every customer line was understood",
        "every reply was turned into speech",
    ]


def test_a_line_that_was_not_understood_fails_the_scenario():
    checks = check_voice([exchange("hello", understood=False)])

    assert descriptions(checks) == {
        "every customer line was understood (not understood: silence)": False,
        "every reply was turned into speech": True,
    }


def test_a_reply_that_could_not_be_voiced_fails_the_scenario():
    checks = check_voice([exchange("hello", voiced=False)])

    assert descriptions(checks)["every reply was turned into speech"] is False


# --- running ---


def status_responses():
    return (
        response(text("Entendí O, R, D, uno, cero, cero, dos. ¿Es correcto?")),
        response(
            tool_use(
                "toolu_1",
                "get_order_status",
                order_id="ORD-1002",
                email="maria.garcia@example.com",
            ),
            stop_reason="tool_use",
        ),
        response(text("Tu pedido ya fue enviado.")),
    )


def status_transcripts():
    return [
        Transcript("¿Cómo va mi pedido ORD1002? Mi correo es maria.garcia.example.com", "es"),
        Transcript("Sí, es correcto.", "es"),
    ]


def test_a_spoken_scenario_runs_through_the_pipeline_and_is_logged(audio_dir, tmp_path):
    stt = FakeSpeechToText(status_transcripts())
    tts = FakeTextToSpeech()

    result = run_voice_scenario(
        STATUS,
        FakeClient(*status_responses()),
        SETTINGS,
        stt,
        tts,
        tmp_path / "logs",
        audio_dir,
        run_id="run-1",
        metadata={"stt_provider": "fake"},
        today=TODAY,
    )

    assert result.passed
    assert [exchange.line.text for exchange in result.exchanges] == [
        STATUS.turns[0],
        CONFIRMATIONS["es"],
    ]
    assert result.replies[1].tool_calls[0].result["status"] == "shipped"
    # The agent's replies went through text-to-speech too.
    assert [language for _, language in tts.spoken] == ["es", "es"]

    saved = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert saved["channel"] == "voice"
    assert saved["metadata"]["scenario_id"] == "status"
    assert saved["metadata"]["run_id"] == "run-1"
    assert saved["metadata"]["stt_provider"] == "fake"
    assert saved["metadata"]["passed"] is True
    # What the recogniser wrote is what the agent was given.
    assert saved["turns"][0]["user_message"].startswith("¿Cómo va mi pedido ORD1002?")
    assert saved["turns"][1]["voice"]["timings_ms"]["total"] is not None


def test_using_the_order_before_the_confirmation_fails_the_scenario(audio_dir, tmp_path):
    eager = (
        response(
            tool_use(
                "toolu_1",
                "get_order_status",
                order_id="ORD-1002",
                email="maria.garcia@example.com",
            ),
            stop_reason="tool_use",
        ),
        response(text("Tu pedido ya fue enviado.")),
        response(text("De nada.")),
    )

    result = run_voice_scenario(
        STATUS,
        FakeClient(*eager),
        SETTINGS,
        FakeSpeechToText(status_transcripts()),
        FakeTextToSpeech(),
        tmp_path / "logs",
        audio_dir,
        today=TODAY,
    )

    assert not result.passed
    (failed,) = [check for check in result.checks if not check.passed]
    assert "before the customer confirmed" in failed.description


def test_a_line_nobody_understood_is_noted_and_the_script_goes_on(audio_dir, tmp_path):
    stt = FakeSpeechToText([Transcript(".", "en"), Transcript("Thanks, bye.", "en")])

    result = run_voice_scenario(
        CHAT,
        FakeClient(response(text("Bye!"))),
        SETTINGS,
        stt,
        FakeTextToSpeech(),
        tmp_path / "logs",
        audio_dir,
        today=TODAY,
    )

    assert not result.passed
    assert [exchange.answer is None for exchange in result.exchanges] == [True, False]
    saved = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert saved["events"][0]["reason"] == "no_words"
    assert len(saved["turns"]) == 1


def test_run_all_prints_each_conversation_and_ends_with_the_latency(audio_dir, tmp_path):
    output = []

    summary = run_all(
        [STATUS],
        FakeClient(*status_responses()),
        SETTINGS,
        FakeSpeechToText(status_transcripts()),
        FakeTextToSpeech(),
        tmp_path / "run",
        "run-1",
        audio_dir,
        metadata={"stt_provider": "fake"},
        write=output.append,
    )

    printed = "\n".join(output)
    assert f"  Said>  {STATUS.turns[0]}" in output
    assert "  Heard> ¿Cómo va mi pedido ORD1002? Mi correo es maria.garcia.example.com" in output
    assert "[tool] get_order_status" in printed
    assert "PASS  no order number or email was used before the customer confirmed it" in printed
    assert "1 of 1 scenarios passed." in printed
    assert "Voice latency: 2 spoken turn(s) in 1 conversation(s)" in output

    assert summary["passed"] == 1 and summary["failed"] == 0
    assert summary["unheard_lines"] == 0
    saved = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert saved["stt_provider"] == "fake"
    assert saved["latency"]["turns"] == 2
    assert saved["results"][0]["tools"] == ["get_order_status:ok"]


# --- the command line ---


def test_list_shows_the_spoken_lines_without_any_key(capsys):
    assert main(["--list", "es_order_status"]) == 0

    printed = capsys.readouterr().out
    assert "es_order_status_1.mp3" in printed
    assert "confirm_es.mp3" in printed
    assert "ja_address_change" not in printed


def test_unknown_scenario_id_is_reported(capsys):
    assert main(["no_such_scenario"]) == 2
    assert "Unknown scenario id(s): no_such_scenario" in capsys.readouterr().err


def test_missing_sound_files_stop_the_run_before_any_api_call(tmp_path, capsys):
    assert main(["--audio-dir", str(tmp_path)]) == 2

    printed = capsys.readouterr().err
    assert "do not match the scenarios" in printed
    assert "--make-audio" in printed
