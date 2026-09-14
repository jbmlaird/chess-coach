import json
import subprocess
import sys
from pathlib import Path

import chess
from inspect_ai import task, Task
from inspect_ai.dataset import FieldSpec, csv_dataset
from inspect_ai.scorer import Score, scorer, accuracy, stderr, Target, CORRECT, INCORRECT, NOANSWER, grouped
from inspect_ai.solver import Generate, Solver, TaskState, generate, prompt_template, solver
from inspect_ai.tool import mcp_connection, mcp_server_stdio
from mcp.client.stdio import get_default_environment

from engine import STOCKFISH_PATH
from ground_truth_parser import Outcome as GroundTruthOutcome, parse_ground_truth
from move_parser import Outcome as MoveParserOutcome, parse_move_field

HERE = Path(__file__).parent

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

TOOL_PARAGRAPH = """You also have a tool, analyse(fen, moves), which returns Stockfish's
verdict on the position reached by playing moves (UCI, in order) from
fen: its best move, principal variation, score and any forced mate, all
from the perspective of the side to move in that resulting position.
{call}
Your final message must be the answer in the format below and nothing
else.

"""
CALL_SENTENCE = {"optional": "You may call it as often as you like before answering.",
                 "required": "Call it at least once before you answer."}
ARMS = ("none", "silent", "optional", "required")


def prompt(tool_use: str) -> str:
    if tool_use not in CALL_SENTENCE:
        return PROMPT
    paragraph = TOOL_PARAGRAPH.replace("{call}", CALL_SENTENCE[tool_use])
    return PROMPT.replace("Then respond in exactly", paragraph + "Then respond in exactly")


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
                         metadata={"outcome": MoveParserOutcome.PARSE_ERROR.name})

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

        # Strict single-answer match. Known edge case: a position can have more than one mating move; the
        # certification audit only flags it when the engine prefers the other one (refutation_disagreements,
        # alternate_mate), so a certified line can still have an unflagged twin.
        expected = state.metadata['Continuation'].split()[0]
        if refutation.uci == expected:
            return Score(value=CORRECT, answer=refutation.answer,
                         explanation="Blunder identified with the certified refutation.",
                         metadata={"outcome": CORRECT_REFUTATION})
        return Score(value=INCORRECT, answer=refutation.answer,
                     explanation=f"Refutation {refutation.answer} does not match certified {expected}.",
                     metadata={"outcome": WRONG_REFUTATION})

    return score


SERVER = dict(command=sys.executable, args=[str(HERE / "stockfish_mcp.py")], cwd=HERE,
              env={"STOCKFISH_PATH": STOCKFISH_PATH or ""})


def tool_provenance() -> dict:
    run = subprocess.run([SERVER["command"], *SERVER["args"], "--provenance"], cwd=SERVER["cwd"],
                         env={**get_default_environment(), **SERVER["env"]},
                         stdout=subprocess.PIPE, text=True, check=True)
    return json.loads(run.stdout)


@solver
def grounded() -> Solver:
    """The tool loop with one MCP server per sample, connected for the whole sample: without the held
    connection Inspect spawns a fresh server (and engine) for every tool call."""
    server = mcp_server_stdio(name="stockfish", **SERVER)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with mcp_connection([server]):
            state.tools = await server.tools()
            return await generate(state)

    return solve


@task
def positions(tool_use: str = "none") -> Task:
    if tool_use not in ARMS:
        raise ValueError(f"tool_use must be one of {ARMS}, not {tool_use!r}")
    return Task(
        name='Positions',
        version=3,
        dataset=csv_dataset(
            'golden/v2/golden_candidates.csv',
            sample_fields=FieldSpec(
                input="FEN",
                id="PuzzleId",
                metadata=METADATA_FIELDS,
            )
        ),
        solver=[prompt_template(prompt(tool_use)), grounded() if tool_use != "none" else generate()],
        scorer=[legal_move(), ground_truth()],
        turn_limit=8,  # a faithful sample needs 2-3 generations; the 9th is billed and discarded
        working_limit=1800,
        metadata={"tool_engine": tool_provenance() if tool_use != "none" else None},
    )
