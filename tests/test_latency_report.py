"""Tests for the latency summary built from conversation logs."""

import json

from support_agent.latency_report import (
    find_logs,
    format_report,
    load_voice_turns,
    main,
    stage_stats,
    summarize,
)


def spoken(stt, llm, tts, total, language="es", recording=4.0, speech=10.0):
    return {
        "language": language,
        "recording_seconds": recording,
        "speech_seconds": speech,
        "timings_ms": {"stt": stt, "llm": llm, "tts": tts, "total": total},
    }


def write_log(folder, name, *voices):
    """A conversation log with one turn per entry; None makes a typed turn."""
    folder.mkdir(parents=True, exist_ok=True)
    turns = [{"index": number, "voice": voice} for number, voice in enumerate(voices, start=1)]
    path = folder / name
    path.write_text(json.dumps({"schema_version": 1, "turns": turns}), encoding="utf-8")
    return path


def test_average_and_worst_case_per_stage(tmp_path):
    write_log(tmp_path, "a.json", spoken(1000, 4000, 1500, 6600), spoken(2000, 6000, 2500, 10700))
    write_log(tmp_path, "b.json", spoken(900, 2000, 500, 3500))

    summary = summarize(load_voice_turns(find_logs([tmp_path])))

    assert summary["turns"] == 3
    assert summary["conversations"] == 2
    assert summary["overall"] == {
        "stt": {"turns": 3, "average_ms": 1300, "worst_ms": 2000},
        "llm": {"turns": 3, "average_ms": 4000, "worst_ms": 6000},
        "tts": {"turns": 3, "average_ms": 1500, "worst_ms": 2500},
        "total": {"turns": 3, "average_ms": 6933, "worst_ms": 10700},
    }
    assert summary["slowest_turn"] == {
        "file": "a.json",
        "index": 2,
        "timings_ms": {"stt": 2000, "llm": 6000, "tts": 2500, "total": 10700},
    }


def test_a_stage_that_did_not_happen_is_left_out_of_its_average():
    turns = [
        {"timings_ms": {"stt": 1000, "llm": 3000, "tts": 1000, "total": 5000}},
        # Text-to-speech failed on this turn.
        {"timings_ms": {"stt": 3000, "llm": 5000, "tts": None, "total": None}},
    ]

    stats = stage_stats(turns)

    assert stats["stt"] == {"turns": 2, "average_ms": 2000, "worst_ms": 3000}
    assert stats["tts"] == {"turns": 1, "average_ms": 1000, "worst_ms": 1000}
    assert stats["total"] == {"turns": 1, "average_ms": 5000, "worst_ms": 5000}


def test_stages_are_also_reported_per_language(tmp_path):
    write_log(
        tmp_path,
        "a.json",
        spoken(1000, 4000, 1000, 6000, language="es"),
        spoken(1000, 4000, 3000, 8000, language="ja"),
        spoken(1000, 4000, 5000, 10000, language="ja"),
    )

    by_language = summarize(load_voice_turns(find_logs([tmp_path])))["by_language"]

    assert list(by_language) == ["es", "ja"]
    assert by_language["es"]["tts"] == {"turns": 1, "average_ms": 1000, "worst_ms": 1000}
    assert by_language["ja"]["tts"] == {"turns": 2, "average_ms": 4000, "worst_ms": 5000}


def test_typed_turns_and_other_files_are_ignored(tmp_path):
    write_log(tmp_path, "typed.json", None, None)
    write_log(tmp_path / "scenarios" / "run1", "voice.json", spoken(1000, 2000, 1000, 4000))
    (tmp_path / "scenarios" / "run1" / "summary.json").write_text('{"scenarios": 10}', "utf-8")
    (tmp_path / "broken.json").write_text("{not json", "utf-8")
    (tmp_path / "list.json").write_text("[1, 2]", "utf-8")
    (tmp_path / "notes.txt").write_text("hello", "utf-8")

    turns = load_voice_turns(find_logs([tmp_path]))

    assert [(turn["file"], turn["index"]) for turn in turns] == [("voice.json", 1)]


def test_report_shows_seconds_per_stage(tmp_path):
    write_log(tmp_path, "a.json", spoken(1000, 4000, 1500, 6600), spoken(2000, 6000, 2500, 10700))

    lines = format_report(summarize(load_voice_turns(find_logs([tmp_path]))))

    assert lines[0] == "Voice latency: 2 spoken turn(s) in 1 conversation(s)"
    assert "STAGE                    TURNS   AVERAGE     WORST" in lines
    assert "speech-to-text               2    1.50 s    2.00 s" in lines
    assert "agent (model + tools)        2    5.00 s    6.00 s" in lines
    assert "text-to-speech               2    2.00 s    2.50 s" in lines
    assert "total wait                   2    8.65 s   10.70 s" in lines
    assert "Language: es" in lines
    assert "Slowest turn: 10.70 s (turn 2 of a.json)" in lines


def test_report_without_spoken_turns(tmp_path):
    write_log(tmp_path, "typed.json", None)

    lines = format_report(summarize(load_voice_turns(find_logs([tmp_path]))))

    assert lines == ["No spoken turns with timings were found."]


def test_main_prints_the_report_for_a_folder_or_a_file(tmp_path, capsys):
    path = write_log(tmp_path, "a.json", spoken(1000, 4000, 1500, 6600))

    assert main([str(tmp_path)]) == 0
    from_folder = capsys.readouterr().out
    assert main([str(path)]) == 0
    from_file = capsys.readouterr().out

    assert "total wait                   1    6.60 s    6.60 s" in from_folder
    assert from_file == from_folder


def test_main_rejects_a_path_that_does_not_exist(tmp_path, capsys):
    assert main([str(tmp_path / "missing")]) == 2
    assert "Not found" in capsys.readouterr().err
