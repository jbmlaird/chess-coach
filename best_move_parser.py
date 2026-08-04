import re
from dataclasses import dataclass
from enum import auto, Enum

import chess
from chess import InvalidMoveError, IllegalMoveError


class Outcome(Enum):
    LEGAL = auto()
    ILLEGAL = auto()
    INVALID = auto()
    REJECTED = auto()
    PARSE_ERROR = auto()


@dataclass(frozen=True)
class ParsedBestMove:
    outcome: Outcome
    explanation: str
    answer: str | None


def parse_best_move(board_fen: str, output: str) -> ParsedBestMove:
    matches = re.findall(
        r"BEST_MOVE:\s?(\S+)$", output, re.MULTILINE | re.IGNORECASE
    )

    if len(matches) > 1:
        return ParsedBestMove(
            outcome=Outcome.PARSE_ERROR,
            explanation="Best move was suggested more than once.",
            answer=','.join(matches),
        )

    best_move = matches[-1].rstrip(".,;") if matches else None
    if best_move is None:
        return ParsedBestMove(
            outcome=Outcome.PARSE_ERROR,
            explanation="No best move found",
            answer=best_move,
        )

    board = chess.Board(board_fen)
    try:
        board.parse_san(best_move)
    except InvalidMoveError as e:
        return ParsedBestMove(
            outcome=Outcome.INVALID,
            explanation=f"Move suggested is not valid: {e}",
            answer=best_move,
        )
    except IllegalMoveError as e:
        return ParsedBestMove(
            outcome=Outcome.ILLEGAL,
            explanation=f"Move suggested is illegal: {e}",
            answer=best_move,
        )
    except ValueError as e:
        return ParsedBestMove(
            outcome=Outcome.REJECTED,
            explanation=f"Unable to play move suggested: {e}",
            answer=best_move,
        )
    return ParsedBestMove(
        outcome=Outcome.LEGAL,
        explanation="Legal move found via regex.",
        answer=best_move,
    )
