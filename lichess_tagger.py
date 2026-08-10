"""Adapter around the vendored Lichess puzzle tagger.

cook.py wants a chess.pgn.Game and an engine eval (cp); the dump gives FEN +
UCI moves and no eval. make_puzzle bridges the formats with cp stubbed to 0,
and generated_themes drops the cp-derived outcome tags that the stub would
corrupt (the mate-family tags are line-derived and unaffected).
"""

import sys
from pathlib import Path

import chess
import chess.pgn

VENDOR = Path(__file__).parent / "vendor" / "lichess_puzzler"

sys.path.insert(0, str(VENDOR))
try:
    import cook
    from model import Puzzle
finally:
    sys.path.remove(str(VENDOR))

CP_DERIVED = {"equality", "advantage", "crushing"}


def make_puzzle(puzzle_id: str, fen: str, moves: str) -> Puzzle:
    game = chess.pgn.Game()
    game.setup(chess.Board(fen))
    game.add_line(chess.Move.from_uci(m) for m in moves.split())
    return Puzzle(puzzle_id, game, 0)


def generated_themes(puzzle_id: str, fen: str, moves: str) -> list[str]:
    return [t for t in cook.cook(make_puzzle(puzzle_id, fen, moves)) if t not in CP_DERIVED]
