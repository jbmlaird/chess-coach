"""Stockfish as an MCP tool - the eval's tool arm.

Runs Engine.grader(), so the oracle's answers are the frozen reference's by
construction. Speaks MCP over stdio: stdout is the protocol channel.
move_review.SERVER is how Inspect launches it.
"""

import json
import sys
from dataclasses import asdict

import chess
from mcp.server.fastmcp import FastMCP

from engine import Engine

mcp = FastMCP("stockfish", log_level="WARNING")


@mcp.tool()
def analyse(fen: str, moves: list[str] = []) -> dict:
    """Stockfish's verdict on the position reached by playing `moves` (UCI, in
    order) from `fen`. Scores are from the perspective of the side to move in
    the RESULTING position: positive means that side is better; a forced mate is
    +/-10000 with the distance in mate_in, and win_percent saturates at 97.5.
    best_move is the engine's choice there (null on checkmate or stalemate)."""
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"invalid position: {board.status()!r}")
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if not board.is_legal(move):
            raise ValueError(f"{uci} is not legal in {board.fen()}")
        board.push(move)
    with Engine.grader() as engine:
        verdict = engine.analyse(board)
    return {"side_to_move": chess.COLOR_NAMES[board.turn],
            "win_percent": round(verdict.win_percent, 1), **asdict(verdict)}


if __name__ == "__main__":
    with Engine.grader() as engine:  # no engine: die at spawn, not as per-call error text
        if "--provenance" in sys.argv:  # what this process's engine is, for the eval log's metadata
            print(json.dumps(engine.provenance))
            sys.exit()
    mcp.run()
