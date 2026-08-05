import re
from dataclasses import dataclass
from enum import auto, Enum

import chess
from chess import InvalidMoveError, IllegalMoveError, AmbiguousMoveError


class Outcome(Enum):
    LEGAL = auto()
    ILLEGAL = auto()
    INVALID = auto()
    AMBIGUOUS = auto()
    PARSE_ERROR = auto()


@dataclass(frozen=True)
class ParsedBestMove:
    outcome: Outcome
    explanation: str
    answer: str | None


BEST_MOVE_RE = re.compile(r"BEST_MOVE\s*:[ \t]*(.+)", re.IGNORECASE)


def extract_best_move(completion: str) -> str | None:
    cleaned = completion.replace("*", "").replace("`", "")

    matches = BEST_MOVE_RE.findall(cleaned)
    if not matches:
        return None

    words = matches[-1].split()
    if not words:
        return None

    move = words[0].rstrip(".,;:!?")
    return move or None


def parse_best_move(board_fen: str, output: str) -> ParsedBestMove:
    best_move = extract_best_move(output)
    if best_move is None:
        return ParsedBestMove(
            outcome=Outcome.PARSE_ERROR,
            explanation="No best move found in LLM output.",
            answer=None,
        )
    board = chess.Board(board_fen)
    try:
        move = board.parse_san(best_move)
        if not board.is_legal(move):
            return ParsedBestMove(
                outcome=Outcome.ILLEGAL,
                explanation=f"Null move provided: {best_move}",
                answer=best_move,
            )
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
    except AmbiguousMoveError as e:
        return ParsedBestMove(
            outcome=Outcome.AMBIGUOUS,
            explanation=f"Unable to play move suggested: {e}",
            answer=best_move,
        )
    return ParsedBestMove(
        outcome=Outcome.LEGAL,
        explanation="Legal move found.",
        answer=best_move,
    )
