import chess
from ground_truth_parser import Outcome as GroundTruthOutcome, parse_ground_truth
from inspect_ai import task, Task
from inspect_ai.dataset import FieldSpec, csv_dataset
from inspect_ai.scorer import Score, scorer, accuracy, stderr, Target, CORRECT, INCORRECT, NOANSWER, grouped
from inspect_ai.solver import TaskState, prompt_template, generate
from move_parser import Outcome as MoveParserOutcome, parse_move_field

METADATA_FIELDS = ["FEN", "PlayedMove", "GroundTruth", "Arm", "Category", "Band", "Rating", "Continuation"]

WRONG_VERDICT = "WRONG_VERDICT"
CORRECT_VERDICT = "CORRECT_VERDICT"
CORRECT_REFUTATION = "CORRECT_REFUTATION"
WRONG_REFUTATION = "WRONG_REFUTATION"

PROMPT = """
You are a chess coach reviewing a game with a club-level student.

Your student had this position in front of them. It is given in
Forsyth-Edwards Notation, and the side to move in the FEN is your
student's side:

{FEN}

It was their move, and they played:

{PlayedMove}

That move is in UCI notation: the square the piece started on followed
by the square it moved to.

Review that move. It is either a serious mistake or the strongest move
available - decide which. Do not assume either in advance: a sharp-looking
move can be correct, and a quiet-looking move can lose the game.

Before you answer, reason through the position: what each side is
threatening, what your student's move allows the opponent to do, and
how the opponent would punish it if it is a mistake. Consider only
moves that are legal.

Then respond in exactly this format, with nothing after it:

VERDICT: <BLUNDER | BEST>
REFUTATION: <if BLUNDER: the opponent's reply that punishes the move, in
the same UCI notation (from-square then to-square, e.g. e7e5, or a7a8q
for a promotion). If BEST: write NONE>
BEST_MOVE: <the strongest legal move for your student in UCI notation;
repeat their move here if it was already the best one>
EXPLANATION: <two or three sentences aimed at a club player. Name the
concrete tactic or positional point at stake - the specific piece,
square or line - rather than a general principle.>
"""


@scorer(metrics=[grouped(accuracy(), "Arm"),
                 grouped(accuracy(), "Category"), grouped(accuracy(), "Band"), stderr()])
def legal_move():
    async def score(state: TaskState, _: Target) -> Score:
        parsed_move = parse_move_field(state.metadata['FEN'], state.output.completion, "BEST_MOVE")

        if parsed_move.outcome == MoveParserOutcome.LEGAL:
            move_score = CORRECT
        elif parsed_move.outcome == MoveParserOutcome.PARSE_ERROR:
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


@scorer(metrics=[grouped(accuracy(), "Arm"),
                 grouped(accuracy(), "Category"), grouped(accuracy(), "Band"), stderr()])
def ground_truth():
    async def score(state: TaskState, _: Target) -> Score:
        verdict = parse_ground_truth(state.output.completion)
        if verdict.outcome == GroundTruthOutcome.PARSE_ERROR:
            return Score(value=NOANSWER, answer=None,
                         explanation="No VERDICT line found in the output.",
                         metadata={"outcome": "PARSE_ERROR"})

        truth = state.metadata['GroundTruth']
        claimed = "blunder" if verdict.outcome == GroundTruthOutcome.BLUNDER else "best"
        if claimed != truth:
            return Score(value=INCORRECT, answer=claimed,
                         explanation=f"Verdict was {claimed} but the played move was {truth}.",
                         metadata={"outcome": WRONG_VERDICT})

        if truth == "best":
            return Score(value=CORRECT, answer=claimed,
                         explanation="Correctly endorsed the best move.",
                         metadata={"outcome": CORRECT_VERDICT})

        refutation_board = chess.Board(state.metadata['FEN'])
        refutation_board.push_uci(state.metadata['PlayedMove'])
        refutation = parse_move_field(refutation_board.fen(), state.output.completion, "REFUTATION")
        if refutation.outcome != MoveParserOutcome.LEGAL:
            return Score(value=INCORRECT, answer=refutation.answer,
                         explanation=f"Blunder called without a usable refutation: {refutation.explanation}",
                         metadata={"outcome": refutation.outcome.name})

        expected = state.metadata['Continuation'].split()[0]
        if refutation.uci == expected:
            return Score(value=CORRECT, answer=refutation.answer,
                         explanation="Blunder identified with the certified refutation.",
                         metadata={"outcome": CORRECT_REFUTATION})
        return Score(value=INCORRECT, answer=refutation.answer,
                     explanation=f"Refutation {refutation.answer} does not match certified {expected}.",
                     metadata={"outcome": WRONG_REFUTATION})

    return score


@task
def positions() -> Task:
    return Task(
        name='Positions',
        version=2,
        dataset=csv_dataset(
            'golden_candidates.csv',
            sample_fields=FieldSpec(
                input="FEN",
                id="PuzzleId",
                metadata=METADATA_FIELDS,
            )
        ),
        solver=[prompt_template(PROMPT), generate()],
        scorer=[legal_move(), ground_truth()],
    )
