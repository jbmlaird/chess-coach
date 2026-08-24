import asyncio
import string

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target, CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import TaskState
from move_review import METADATA_FIELDS, PROMPT, ground_truth, legal_move

BLUNDER_FEN = "r4q1k/6pp/1p3n2/5N2/P1b2P2/1Q2P2P/K5P1/2bR2R1 w - - 0 31"
BLUNDER_META = {"PlayedMove": "b3c4", "GroundTruth": "blunder",
                "Continuation": "f8a3 a2b1 a3b2"}
BEST_META = {"PlayedMove": "d3h7", "GroundTruth": "best", "Continuation": ""}


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


@pytest.mark.parametrize(
    "completion,expected_value,expected_outcome,expected_answer",
    [
        pytest.param("BEST_MOVE: Ra3+", CORRECT, "LEGAL", "Ra3+",
                     id="legal-move-scores-correct"),
        pytest.param("BEST_MOVE: Qh4", INCORRECT, "ILLEGAL", "Qh4",
                     id="illegal-move-scores-incorrect"),
        pytest.param("", NOANSWER, "PARSE_ERROR", None,
                     id="missing-field-abstains"),
    ],
)
def test_legal_move(completion, expected_value, expected_outcome, expected_answer):
    score = asyncio.run(
        legal_move()(make_state("8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", completion), Target("")))
    assert score.value == expected_value
    assert score.metadata["outcome"] == expected_outcome
    assert score.answer == expected_answer


@pytest.mark.parametrize(
    "completion,metadata,expected_value,expected_outcome",
    [
        pytest.param("Bad move.", BLUNDER_META, NOANSWER, "PARSE_ERROR",
                     id="no-verdict-line-abstains"),
        pytest.param("REFUTATION: Qa3+\nBEST_MOVE: Rd8", BLUNDER_META, NOANSWER, "PARSE_ERROR",
                     id="certified-refutation-cannot-substitute-for-missing-verdict"),
        pytest.param("VERDICT: BLUNDER\nREFUTATION: Qa3+", BEST_META, INCORRECT, "WRONG_VERDICT",
                     id="criticising-a-best-move-is-wrong-even-with-plausible-refutation"),
        pytest.param("VERDICT: BEST\nREFUTATION: NONE", BLUNDER_META, INCORRECT, "WRONG_VERDICT",
                     id="endorsing-a-blunder-is-the-other-wrong-verdict"),
        pytest.param("VERDICT: BLUNDER\nBEST_MOVE: d1c1", BLUNDER_META, INCORRECT, "PARSE_ERROR",
                     id="blunder-call-without-refutation-is-incorrect-not-abstain"),
        pytest.param("VERDICT: BLUNDER\nREFUTATION: NONE", BLUNDER_META, INCORRECT, "INVALID",
                     id="blunder-call-refusing-its-own-refutation-obligation"),
        pytest.param("VERDICT: BEST\nREFUTATION: NONE", BEST_META, CORRECT, "CORRECT_VERDICT",
                     id="endorsing-a-best-move-is-correct"),
        pytest.param("VERDICT: BLUNDER\nREFUTATION: Qa3+\nBEST_MOVE: Rd8", BLUNDER_META,
                     CORRECT, "CORRECT_REFUTATION",
                     id="blunder-call-backed-by-certified-refutation"),
        pytest.param("VERDICT: BLUNDER\nREFUTATION: Rb8", BLUNDER_META, INCORRECT, "WRONG_REFUTATION",
                     id="blunder-call-with-legal-but-wrong-refutation"),
        pytest.param("VERDICT: BLUNDER\nREFUTATION: Qh1", BLUNDER_META, INCORRECT, "ILLEGAL",
                     id="blunder-call-with-illegal-refutation"),
    ],
)
def test_ground_truth(completion, metadata, expected_value, expected_outcome):
    score = asyncio.run(
        ground_truth()(make_state(BLUNDER_FEN, completion, metadata), Target("")))
    assert score.value == expected_value
    assert score.metadata["outcome"] == expected_outcome


def test_prompt_placeholders_are_metadata_fields():
    # ensures all placeholders are correctly set
    placeholders = {name for _, name, _, _ in string.Formatter().parse(PROMPT) if name}
    assert placeholders <= set(METADATA_FIELDS)