from inspect_ai import task, Task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.scorer import Score, scorer, accuracy, stderr, Target, CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import TaskState, prompt_template, generate

from best_move_parser import Outcome, parse_best_move

PROMPT = """
You are a chess coach reviewing a game with a club-level student.

Your student had this position in front of them. It is given in
Forsyth-Edwards Notation, and the side to move in the FEN is your
student's side:

{fen}

It was their move, and they played:

{played_move}

That move is in UCI notation: the square the piece started on followed
by the square it moved to.

Review that move. Decide whether it was the strongest move available.
If it was not, work out what your student overlooked and what they
should have played instead.

Before you answer, reason through the position: what each side is
threatening, what your student's move allows the opponent to do, and
what the strongest legal alternative is. Consider only moves that are
legal for the side to move.

Then respond in exactly this format, with nothing after it:

VERDICT: <BLUNDER | MISTAKE | INACCURACY | GOOD | BEST>
BEST_MOVE: <the strongest legal move in SAN notation; repeat your
student's move here if they already found the best one>
EXPLANATION: <two or three sentences aimed at a club player. Name the
concrete tactic or positional point at stake — the specific piece,
square or line — rather than a general principle.>
"""


@scorer(metrics=[accuracy(), stderr()])
def legal_move():
    async def score(state: TaskState, _: Target) -> Score:
        parsed_move = parse_best_move(state.metadata['fen'], state.output.completion)

        if parsed_move.outcome == Outcome.LEGAL:
            move_score = CORRECT
        elif parsed_move.outcome == Outcome.PARSE_ERROR:
            move_score = NOANSWER
        else:
            move_score = INCORRECT

        return Score(
            value=move_score,
            answer=parsed_move.answer,
            explanation=parsed_move.explanation,
            metadata={"outcome": parsed_move.outcome.name},
        )

    return score


@task
def positions() -> Task:
    return Task(
        name='Positions',
        dataset=json_dataset(
            'positions.jsonl',
            sample_fields=FieldSpec(
                input="fen",
                id="id",
                metadata=["fen", "played_move"],
            )
        ),
        solver=[prompt_template(PROMPT), generate()],
        scorer=legal_move(),
    )
