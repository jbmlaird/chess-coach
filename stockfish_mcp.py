"""Stockfish as an MCP tool - the eval's tool arm.

Runs Engine.grader(), so the oracle's answers are the frozen reference's by
construction. Speaks MCP over stdio: stdout is the protocol channel. Register
with Inspect via mcp_server_stdio(command="uv", args=["run", "python",
"stockfish_mcp.py"], cwd=<repo root>); stdio servers get a restricted
environment, so pass env={"STOCKFISH_PATH": ...} if you rely on that override.
"""

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
    Engine.grader().close()  # no engine: die at spawn, not as per-call error text
    mcp.run()
