from inspect_ai import task, Task
from inspect_ai.dataset import json_dataset, Sample
from typing import Any

PROMPT = """
You are a chess coach reviewing a game with a club-level student.

Your student had this position in front of them. It is given in
Forsyth-Edwards Notation, and the side to move in the FEN is your
student's side:

{board}

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
BEST_MOVE: <the strongest legal move in UCI notation; repeat your
student's move here if they already found the best one>
EXPLANATION: <two or three sentences aimed at a club player. Name the
concrete tactic or positional point at stake — the specific piece,
square or line — rather than a general principle.>
"""


@task
def initial_boards() -> 'Task':
    return Task(
        name='Initial Boards',
        dataset=json_dataset(
            'initialboards.jsonl',
            sample_fields=record_to_sample
        ),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(id=record['id'],
                  input=PROMPT.format(
                      board=record['fen'],
                      played_move=record['played_move'],
                  ),
                  metadata={
                      "board": record['fen'],
                      "played_move": record['played_move']}
                  )
