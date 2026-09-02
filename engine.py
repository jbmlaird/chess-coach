"""Shared Stockfish access for the grader and the model-facing MCP tool.

Scores are always reported from the perspective of the side to move in the
analysed position; callers comparing before/after a move must flip the sign
themselves.
"""

import os
import shutil
from dataclasses import dataclass
from types import TracebackType

import chess
import chess.engine
import math

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")

# Lichess's win model ceils centipawns at +-1000 before the sigmoid and maps
# any forced mate to the ceiling (~97.5%), never 100 - see
# https://github.com/lichess-org/scalachess/blob/master/core/src/main/scala/eval.scala
CP_CEILING = 1000

# Sentinel for forced mates in score_cp; beyond CP_CEILING, so after clamping
# it reproduces Lichess's mate handling exactly. True mate distance is
# preserved separately in mate_in.
MATE_SCORE_CP = 10_000

GRADER_CONFIG: chess.engine.ConfigMapping = {"Threads": 1, "Hash": 128}
GRADER_NODES = 1_000_000


def win_percent(score_cp: int) -> float:
    """Lichess's win-probability model for a centipawn score (side to move).

    https://lichess.org/page/accuracy - implemented to match scalachess:
    centipawns are ceiled to +-CP_CEILING first, so evals saturate at ~97.5%
    and a 200cp slip in a crushing position costs less than one in an equal
    position.
    """
    clamped = max(-CP_CEILING, min(CP_CEILING, score_cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * clamped)) - 1)


def move_damage_pp(before_cp: int, after_cp: int) -> float:
    """Win% a move gave away, from the mover's perspective.

    before_cp is the mover's view of the position; after_cp is the score of
    the position after the move - the OPPONENT'S view (side to move flipped) -
    hence the 100-minus. Shared by the certify and grading scripts so the
    harness has exactly one statement of its core metric.
    """
    return win_percent(before_cp) - (100 - win_percent(after_cp))


@dataclass(frozen=True)
class EngineEval:
    """Engine verdict on one position, from the side to move's perspective.

    On terminal positions (checkmate/stalemate) there is no move to
    make: best_move is None and pv is empty, with score_cp/mate_in carrying
    the terminal verdict (mated -> -MATE_SCORE_CP with mate_in=0, drawn -> 0).
    """

    best_move: str | None  # UCI - None on terminal positions
    score_centipawns: int  # forced mates mapped to +-MATE_SCORE_CP
    mate_in: int | None  # signed moves-to-mate, 0 = already mated, None = no mate
    principal_variation: list[str]  # UCI

    @property
    def win_percent(self) -> float:
        return win_percent(self.score_centipawns)


class Engine:
    """Context-managed Stockfish wrapper with a fixed search limit.

    >>> with Engine.grader() as engine:
    ...     verdict = engine.analyse(board)
    """

    def __init__(self, limit: chess.engine.Limit, config: chess.engine.ConfigMapping,
                 path: str | None = STOCKFISH_PATH) -> None:
        if path is None:
            raise FileNotFoundError(
                "stockfish not found on PATH - install it (brew install stockfish) "
                "or set the STOCKFISH_PATH environment variable")
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self._engine.configure(config)
        except Exception:
            # SimpleEngine runs a non-daemon thread: without quit() the
            # process leaks and the interpreter hangs at exit.
            self._engine.quit()
            raise
        self.path = path
        self.limit = limit
        self.config = dict(config)
        self.name = self._engine.id.get("name", "unknown")

    @classmethod
    def grader(cls) -> "Engine":
        return cls(limit=chess.engine.Limit(nodes=GRADER_NODES), config=GRADER_CONFIG)

    @property
    def provenance(self) -> dict:
        return {"name": self.name, "path": self.path,
                "limit": repr(self.limit), "config": self.config}

    def analyse(self, board: chess.Board) -> EngineEval:
        """Verdict on a valid position (board.is_valid()): Stockfish aborts on e.g. a kingless board."""
        # A fresh game token per call forces ucinewgame
        info = self._engine.analyse(board, self.limit, game=object())
        pov_score = info["score"].pov(board.turn)
        mate_in = pov_score.mate()
        if mate_in is not None:
            # mate_in == 0 means the side to move is already checkmated
            score_cp = MATE_SCORE_CP if mate_in > 0 else -MATE_SCORE_CP
        else:
            score = pov_score.score()
            assert score is not None  # non-mate scores are always concrete
            score_cp = score
        pv = [move.uci() for move in info.get("pv", [])]
        if not pv:
            if board.is_game_over():
                return EngineEval(best_move=None, score_centipawns=score_cp,
                                  mate_in=mate_in, principal_variation=[])
            raise ValueError(f"engine returned no principal variation for {board.fen()!r}")
        return EngineEval(best_move=pv[0], score_centipawns=score_cp, mate_in=mate_in, principal_variation=pv)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()
