"""Tests for the Stockfish MCP tool. Chess fixtures reuse positions test_engine.py
already proves with the pinned engine - no new chess claims are made here."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import chess
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import LATEST_PROTOCOL_VERSION

from engine import Engine, MATE_SCORE_CP, STOCKFISH_PATH
from stockfish_mcp import analyse

SERVER = Path(__file__).parent.parent / "stockfish_mcp.py"
needs_stockfish = pytest.mark.skipif(STOCKFISH_PATH is None, reason="stockfish binary not installed")

# Wj3vh: after white's blunder Qxc4?? black has the certified mate Qa3+
WJ3VH_FEN = "r4q1k/6pp/1p3n2/5N2/P1b2P2/1Q2P2P/K5P1/2bR2R1 w - - 0 31"
# f16cc: after the blunder Qd8 and the certified mating reply Qxg7# (both
# proven in test_engine.py) black is checkmated
F16CC_FEN = "r1b2rk1/ppqn1pbp/3p3B/2p3Qp/3pP3/5P2/PPP1N1P1/2KR1BN1 b - - 1 14"


@needs_stockfish
def test_analyse_after_moves_returns_the_grader_verdict():
    with patch.object(Engine, "grader", wraps=Engine.grader) as grader:
        verdict = analyse(WJ3VH_FEN, ["b3c4"])
    grader.assert_called_once()  # the oracle is the ruler
    assert verdict.keys() == {"side_to_move", "win_percent", "best_move",
                              "score_centipawns", "mate_in", "principal_variation"}
    assert verdict["side_to_move"] == "black"
    assert (verdict["best_move"], verdict["mate_in"]) == ("f8a3", 2)
    assert verdict["win_percent"] == 97.5  # rounded to one decimal
    for thread in threading.enumerate():  # a leaked SimpleEngine thread hangs pytest at exit
        if thread is not threading.main_thread():
            thread.join(timeout=5)
            assert not thread.is_alive(), thread.name


@needs_stockfish
def test_moves_apply_in_order_and_a_mated_side_gets_no_best_move():
    verdict = analyse(F16CC_FEN, ["c7d8", "g5g7"])
    assert verdict["side_to_move"] == "black"
    assert verdict["best_move"] is None
    assert (verdict["score_centipawns"], verdict["mate_in"]) == (-MATE_SCORE_CP, 0)


@pytest.mark.parametrize("fen, moves, message", [
    (chess.STARTING_FEN, ["e2e5"], "not legal"),  # well-formed but illegal
    (chess.STARTING_FEN, ["0000"], "not legal"),  # null move: python-chess pushes it silently
    ("8/8/8/8/8/8/8/8 w - - 0 1", [], "invalid position"),  # parses, but no kings
    ("not a fen", [], None),  # python-chess's own message suffices
])
def test_rejects_unusable_input(fen, moves, message):
    with pytest.raises(ValueError, match=message):
        analyse(fen, moves)


@needs_stockfish
@pytest.mark.anyio
async def test_serves_analyse_over_mcp_stdio():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)],
                                   env={**os.environ, "STOCKFISH_PATH": STOCKFISH_PATH})
    async with stdio_client(params) as streams, ClientSession(*streams) as session:
        await session.initialize()
        tools = await session.list_tools()
        good = await session.call_tool("analyse", {"fen": WJ3VH_FEN, "moves": ["b3c4"]})
        bad = await session.call_tool("analyse", {"fen": WJ3VH_FEN, "moves": ["e2e5"]})
    assert {t.name for t in tools.tools} == {"analyse"}
    assert not good.isError
    assert json.loads(good.content[0].text)["best_move"] == "f8a3"
    assert bad.isError and "not legal" in bad.content[0].text


def test_server_refuses_to_start_without_an_engine():
    # Inspect gives stdio servers a restricted environment; a tool whose engine
    # fails per call would hand the model error text and score the sample anyway
    run = subprocess.run([sys.executable, str(SERVER)], input="", capture_output=True,
                         text=True, timeout=10, env={"PATH": "/usr/bin:/bin"})
    assert run.returncode != 0 and "stockfish not found" in run.stderr


@needs_stockfish
def test_a_call_leaves_stdout_to_the_protocol_and_stderr_silent():
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "analyse", "arguments": {"fen": WJ3VH_FEN}}},
    ]
    run = subprocess.run([sys.executable, str(SERVER)], capture_output=True, text=True,
                         timeout=10, input="".join(json.dumps(m) + "\n" for m in messages))
    replies = [json.loads(line) for line in run.stdout.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2]
    assert "best_move" in replies[1]["result"]["content"][0]["text"]
    assert run.stderr == ""
