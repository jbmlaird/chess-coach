"""The tool-arm metrics on hand-built transcripts shaped like the real server's (empty content and a
prefixed error message on failures). Engine facts are the ones tests/test_stockfish_mcp.py pins
(Wj3vh: after Qxc4?? the engine's reply is f8a3, mate in 2)."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ModelOutput, ModelUsage
from inspect_ai.tool import ToolCall, ToolCallError

import render_results
import tool_metrics

FEN = "r4q1k/6pp/1p3n2/5N2/P1b2P2/1Q2P2P/K5P1/2bR2R1 w - - 0 31"
GIVEN = {"best_move": "d1d6"}
AFTER = {"best_move": "f8a3"}


def call(i, moves):
    return ToolCall(id=str(i), function="analyse", arguments={"fen": FEN, "moves": moves})


def reply(i, result=None, error=None):
    return ChatMessageTool(content=json.dumps(result) if result else "", tool_call_id=str(i), function="analyse",
                           error=ToolCallError(type="unknown", message="Error executing tool analyse: " + error) if error else None)


def sample(turns, answer, arm="blunder"):
    messages = [ChatMessageUser(content="prompt"), *turns, ChatMessageAssistant(content=answer)]
    return SimpleNamespace(messages=messages, output=ModelOutput.from_content(model="m", content=answer),
                           metadata={"FEN": FEN, "PlayedMove": "b3c4", "Arm": arm})


RELAYED = sample([ChatMessageAssistant(content="", tool_calls=[call(1, []), call(2, ["b3c4"])]), reply(1, GIVEN), reply(2, AFTER)],
                 "VERDICT: BLUNDER\nREFUTATION: f8a3\nBEST_MOVE: d1d6\nEXPLANATION: x")
FRAMED = sample([ChatMessageAssistant(content="", tool_calls=[call(1, ["b3c4"])]), reply(1, AFTER)],
                "VERDICT: BLUNDER\nREFUTATION: f8a3\nBEST_MOVE: Qa3+\nEXPLANATION: the engine's reply, reported as White's move")
MIXED = sample([ChatMessageAssistant(content="", tool_calls=[call(1, [])]), reply(1, GIVEN)],
               "VERDICT: BLUNDER\nREFUTATION: f8a3\nBEST_MOVE: rd1\nEXPLANATION: queried the given position, then an invalid best move")
ERRORED = sample([ChatMessageAssistant(content="", tool_calls=[call(1, ["e2e4"])]), reply(1, error="e2e4 is not legal")],
                 "VERDICT: BLUNDER\nREFUTATION: f8a3\nBEST_MOVE: d1c1\nEXPLANATION: own idea")
UNTOOLED = sample([], "VERDICT: BEST\nREFUTATION: NONE\nBEST_MOVE: b3c4\nEXPLANATION: x", arm="best")


def test_a_faithful_relay_is_counted_as_one():
    assert tool_metrics.sample_stats(RELAYED) == {
        "calls": 2, "errors": 0, "answered_after_tool": True, "queried_given": True, "best_legal": True,
        "relay": True, "legal_fields": 2, "beyond_engine": 0, "frame_error": False}


def test_the_opponents_reply_reported_as_the_students_move_is_a_frame_error_even_in_san():
    t = tool_metrics.sample_stats(FRAMED)
    assert t["frame_error"] and not t["relay"] and not t["queried_given"]
    assert t["legal_fields"] == 1  # Qa3+ is not White's move; only the refutation parses


def test_a_queried_position_with_an_unusable_best_move_is_not_a_relay_candidate():
    t = tool_metrics.sample_stats(MIXED)
    assert t["queried_given"] and not t["best_legal"] and not t["relay"] and t["legal_fields"] == 1


def test_a_tool_error_and_own_moves_are_counted():
    t = tool_metrics.sample_stats(ERRORED)
    assert t["errors"] == 1 and t["calls"] == 1 and not t["answered_after_tool"]
    assert t["beyond_engine"] == 2 and t["legal_fields"] == 2  # the engine answered nothing, so both moves are its own


def test_a_sample_that_never_called_the_tool():
    t = tool_metrics.sample_stats(UNTOOLED)
    assert t["calls"] == 0 and not t["answered_after_tool"] and t["beyond_engine"] == 1


def test_the_mock_model_costs_nothing_even_when_it_records_usage():
    usage = ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert tool_metrics.sample_cost(SimpleNamespace(model_usage={"mockllm/model": usage}), "mockllm/model") == 0.0
    assert tool_metrics.sample_cost(SimpleNamespace(model_usage={}), "anthropic/claude-haiku-4-5") == 0.0


def test_tool_rows_read_dash_on_a_no_tool_column():
    table = render_results.Table("### scratch", {"Haiku 4.5": render_results.HAIKU_V2}, render_results.TOOL_ROWS)
    lines = render_results.render(table).splitlines()[2:]
    assert all(line.rstrip("| ").endswith("-") for line in lines) and len(lines) == len(render_results.TOOL_ROWS)


@pytest.mark.parametrize("run, expected", [
    ("golden-v2/sonnet-4-6-tool-required", "relay fidelity 200/200; beyond-engine answers 0/400 legal fields; frame errors 0; unfinished 0"),
    ("golden-v2/haiku-4-5-tool-optional", "answered after a tool call 169/200; relay fidelity 163/167"),
], ids=["sonnet-perfect-relay", "haiku-optional-limit-hits"])
def test_published_tool_lines_reproduce(run, expected):
    """The tool-row figures the README quotes in prose come from this CLI's printed lines."""
    log = render_results.column(run).log
    out = subprocess.run([sys.executable, "scripts/tool_metrics.py", "--log_file", str(log)],
                         cwd=render_results.REPO, capture_output=True, text=True, check=True).stdout
    assert expected in out, out
