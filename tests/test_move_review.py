import asyncio
import string

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target, CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import TaskState

from move_review import METADATA_FIELDS, PROMPT, ground_truth, legal_move


def make_state(board_fen: str, completion: str, metadata: dict | None = None):
    return TaskState(
        model="mockllm/model",
        sample_id="test",
        epoch=0,
        input="unused",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content=completion),
        metadata={"FEN": board_fen, **(metadata or {})},
    )


def run_scorer(board_fen: str, completion: str):
    score = legal_move()
    return asyncio.run(score(make_state(board_fen, completion), Target("")))


def test_legal_move():
    score = run_scorer("8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", "BEST_MOVE: Ra3+")
    assert CORRECT == score.value
    assert "LEGAL" == score.metadata["outcome"]
    assert "Ra3+" == score.answer


def test_illegal_move():
    score = run_scorer("8/1B2k3/4pp2/N5p1/Pp1bP2p/1P3K1P/n5P1/8 b - - 0 42", "BEST_MOVE: Nb4")
    assert INCORRECT == score.value
    assert "ILLEGAL" == score.metadata["outcome"]
    assert "Nb4" == score.answer


def test_parse_error():
    score = run_scorer("8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", "")
    assert None == score.answer
    assert NOANSWER == score.value
    assert "PARSE_ERROR" == score.metadata["outcome"]


BLUNDER_FEN = "r4q1k/6pp/1p3n2/5N2/P1b2P2/1Q2P2P/K5P1/2bR2R1 w - - 0 31"
BLUNDER_META = {"PlayedMove": "b3c4", "GroundTruth": "blunder",
                "Continuation": "f8a3 a2b1 a3b2"}
BEST_META = {"PlayedMove": "d3h7", "GroundTruth": "best", "Continuation": ""}


def run_ground_truth(completion: str, metadata: dict, board_fen: str = BLUNDER_FEN):
    score = ground_truth()
    return asyncio.run(score(make_state(board_fen, completion, metadata), Target("")))


def test_ground_truth_no_verdict_line():
    score = run_ground_truth("Bad move.", BLUNDER_META)
    assert NOANSWER == score.value
    assert "PARSE_ERROR" == score.metadata["outcome"]


def test_ground_truth_refutation_cannot_substitute_for_verdict():
    score = run_ground_truth("REFUTATION: Qa3+\nBEST_MOVE: Rd8", BLUNDER_META)
    assert NOANSWER == score.value
    assert "PARSE_ERROR" == score.metadata["outcome"]


def test_ground_truth_wrong_verdict_ignores_refutation():
    score = run_ground_truth("VERDICT: BLUNDER\nREFUTATION: Qa3+", BEST_META)
    assert INCORRECT == score.value
    assert "WRONG_VERDICT" == score.metadata["outcome"]


def test_ground_truth_missed_blunder():
    score = run_ground_truth("VERDICT: BEST\nREFUTATION: NONE", BLUNDER_META)
    assert INCORRECT == score.value
    assert "WRONG_VERDICT" == score.metadata["outcome"]


def test_ground_truth_correct_best_verdict():
    score = run_ground_truth("VERDICT: BEST\nREFUTATION: NONE", BEST_META)
    assert CORRECT == score.value
    assert "CORRECT_VERDICT" == score.metadata["outcome"]


def test_ground_truth_blunder_with_certified_refutation():
    score = run_ground_truth("VERDICT: BLUNDER\nREFUTATION: Qa3+\nBEST_MOVE: Rd8", BLUNDER_META)
    assert CORRECT == score.value
    assert "CORRECT_REFUTATION" == score.metadata["outcome"]
    assert "Qa3+" == score.answer


def test_ground_truth_blunder_with_wrong_refutation():
    score = run_ground_truth("VERDICT: BLUNDER\nREFUTATION: Rb8", BLUNDER_META)
    assert INCORRECT == score.value
    assert "WRONG_REFUTATION" == score.metadata["outcome"]


def test_ground_truth_blunder_with_illegal_refutation():
    score = run_ground_truth("VERDICT: BLUNDER\nREFUTATION: Qh1", BLUNDER_META)
    assert INCORRECT == score.value
    assert "ILLEGAL" == score.metadata["outcome"]


def test_prompt_placeholders_are_metadata_fields():
    # ensures all placeholders are correctly set
    placeholders = {name for _, name, _, _ in string.Formatter().parse(PROMPT) if name}
    assert placeholders <= set(METADATA_FIELDS)
