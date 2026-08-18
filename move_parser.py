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
class ParsedMove:
    outcome: Outcome
    explanation: str
    answer: str | None
    uci: str | None = None


BEST_MOVE_RE = re.compile(r"BEST_MOVE\s*:[ \t]*(.+)", re.IGNORECASE)
REFUTATION_RE = re.compile(r"REFUTATION\s*:[ \t]*(.+)", re.IGNORECASE)


def extract_move(completion: str, field_name: str) -> str | None:
    cleaned = completion.replace("*", "").replace("`", "")

    matches = None
    if field_name == "BEST_MOVE":
        matches = BEST_MOVE_RE.findall(cleaned)
    elif field_name == "REFUTATION":
        matches = REFUTATION_RE.findall(cleaned)

    if not matches:
        return None

    words = matches[-1].split()
    if not words:
        return None

    move = words[0].rstrip(".,;:!?")
    return move or None


def parse_move_field(board_fen: str, output: str, field_name: str) -> ParsedMove:
    parsed_move = extract_move(output, field_name)
    if parsed_move is None:
        return ParsedMove(
            outcome=Outcome.PARSE_ERROR,
            explanation="No move found in LLM output.",
            answer=None,
        )
    board = chess.Board(board_fen)
    try:
        move = board.parse_san(parsed_move)
        if not board.is_legal(move):
            return ParsedMove(
                outcome=Outcome.ILLEGAL,
                explanation=f"Null move provided: {parsed_move}",
                answer=parsed_move,
            )
    except InvalidMoveError as e:
        return ParsedMove(
            outcome=Outcome.INVALID,
            explanation=f"Move suggested is not valid: {e}",
            answer=parsed_move,
        )
    except IllegalMoveError as e:
        return ParsedMove(
            outcome=Outcome.ILLEGAL,
            explanation=f"Move suggested is illegal: {e}",
            answer=parsed_move,
        )
    except AmbiguousMoveError as e:
        return ParsedMove(
            outcome=Outcome.AMBIGUOUS,
            explanation=f"Unable to play move suggested: {e}",
            answer=parsed_move,
        )
    return ParsedMove(
        outcome=Outcome.LEGAL,
        explanation="Legal move found.",
        answer=parsed_move,
        uci=move.uci(),
    )
