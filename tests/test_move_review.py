import asyncio

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target, CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import TaskState

from move_review import legal_move


def make_state(board_fen: str, completion: str):
    return TaskState(
        model="mockllm/model",
        sample_id="test",
        epoch=0,
        input="unused",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content=completion),
        metadata={"fen": board_fen}
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
