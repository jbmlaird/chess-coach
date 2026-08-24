"""Certify the golden set with a pinned engine and freeze its reference evals.

One pass of Engine.grader() over golden_candidates.csv, producing:

- golden_engine.csv: per PuzzleId, the engine's raw verdict on up to three
  positions - BEFORE the played move, AFTER the played move, and (blunder rows
  only) after the certified refutation.
- golden_engine.meta.json: engine provenance (what actually ran), the frozen
  input's sha256, column semantics, and the audit findings with their deciding
  numbers.

The audit asks: does an independent, pinned engine co-sign the dataset's
labels? Disagreements are findings to document, not failures - four classes:

- best_arm_disagreements: engine best != played move on a best row (the
  generator claims solution uniqueness with a 0.7 win-chances margin).
- already_lost_before: blunder rows where the player was in a FORCED MATE
  before the played move (before mate_in < 0) - every legal move loses, so
  the binary blunder label is engine-indefensible. Parameter-free bright line.
- weak_swing: blunder rows whose played move gave away under 30 win% points
  and were not already lost - margin findings, usually near-equal positions
  punished by "winning advantage".
- refutation_disagreements: engine's best reply != certified refutation.
  Flagged alternate_mate when both mate in one - the generator explicitly
  allows multiple mates on the final move, so those are benign.

Threshold provenance: the Lichess generator's 0.6 win-chances swing gate
applies only to ADVANTAGE-path candidate selection; solution uniqueness uses
0.7; mate-path puzzles have NO setup-swing gate at all (a mate puzzle may
arise from an already-lost position - that is what already_lost_before
detects). The generator's win_chances scale maps mate to +-1 unclamped; our
win_percent clamps at +-1000cp, so swings >= 30pp are unreachable once the
before eval is at or below about -200cp.

Score convention: every score is from the side to move's perspective in THAT
position (engine.py's contract). BEFORE scores are the student's view; AFTER
scores are the opponent's. The played move's damage in win% terms is
    before.win_percent - (100 - after_played.win_percent)
via the symmetry win%(-cp) = 100 - win%(cp). Damage can be slightly negative:
before/after come from two independent searches. Terminal positions (a move
that ends the game) are legitimate rows: BestMove is empty and the score
carries the verdict.

Usage:
    uv run python scripts/certify_golden.py
    uv run python scripts/certify_golden.py --limit 5 --output /tmp/smoke.csv
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import chess

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from engine import Engine, EngineEval  # noqa: E402

BLUNDER_SWING_THRESHOLD_PP = 30.0

COLUMNS_NOTE = (
    "Scores are centipawns from the side to move's perspective in that position; "
    "forced mates are stored as +-10000 with the true distance in the MateIn "
    "column (0 = already mated). Empty BestMove = terminal position; empty "
    "AfterRefutation cells = best-arm row (not analysed)."
)


def eval_columns(prefix: str, verdict: EngineEval | None) -> dict[str, str]:
    """Flatten an EngineEval into CSV cells; empty cells when not analysed."""
    if verdict is None:
        return {f"{prefix}BestMove": "", f"{prefix}ScoreCentipawns": "", f"{prefix}MateIn": ""}
    return {
        f"{prefix}BestMove": verdict.best_move or "",
        f"{prefix}ScoreCentipawns": str(verdict.score_centipawns),
        f"{prefix}MateIn": "" if verdict.mate_in is None else str(verdict.mate_in),
    }


def played_move_damage_pp(before: EngineEval, after_played: EngineEval) -> float:
    """Win% the played move gave away, from the player's perspective."""
    return before.win_percent - (100 - after_played.win_percent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=REPO / "golden_candidates.csv")
    parser.add_argument("--output", type=Path, default=REPO / "golden_engine.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="only analyse the first N rows (smoke testing; "
                             "requires a non-default --output)")
    args = parser.parse_args()

    if args.limit is not None:
        if args.limit <= 0:
            sys.exit(f"--limit must be positive, got {args.limit}")
        if args.output == REPO / "golden_engine.csv":
            sys.exit("--limit runs must use a non-default --output: the default "
                     "path is the frozen reference and a partial file must never "
                     "land there")
    meta_path = args.output.with_suffix(".meta.json")
    for frozen in (args.output, meta_path):
        if frozen.exists():
            sys.exit(f"{frozen} already exists - it is a frozen reference; "
                     f"delete it deliberately to re-certify")

    golden_bytes = args.golden.read_bytes()
    input_sha256 = hashlib.sha256(golden_bytes).hexdigest()
    golden_meta_file = args.golden.parent / "golden_candidates.meta.json"
    if golden_meta_file.exists():
        frozen_sha = json.loads(golden_meta_file.read_text()).get("csv_sha256")
        if frozen_sha and frozen_sha != input_sha256:
            sys.exit(f"{args.golden} does not match the frozen csv_sha256 in "
                     f"{golden_meta_file.name} - refusing to certify drifted bytes")

    rows = list(csv.DictReader(golden_bytes.decode().splitlines()))
    if not rows:
        sys.exit(f"{args.golden} contains no rows")
    if args.limit is not None:
        rows = rows[:args.limit]

    best_arm_disagreements: list[dict] = []
    already_lost_before: list[dict] = []
    weak_swing: list[dict] = []
    refutation_disagreements: list[dict] = []

    out_rows: list[dict[str, str]] = []
    with Engine.grader() as engine:
        provenance = engine.provenance
        for index, row in enumerate(rows, 1):
            board = chess.Board(row["FEN"])
            before = engine.analyse(board)

            board.push_uci(row["PlayedMove"])
            after_played = engine.analyse(board)

            after_refutation = None
            if row["Arm"] == "blunder":
                certified = row["Continuation"].split()[0]
                refutation_board = board.copy()
                refutation_board.push_uci(certified)
                after_refutation = engine.analyse(refutation_board)

                # after_played can't be terminal on a self-consistent blunder
                # row (the certified reply above just pushed legally), so
                # best_move is always present here.
                if after_played.best_move != certified:
                    refutation_disagreements.append({
                        "puzzle_id": row["PuzzleId"],
                        "engine_move": after_played.best_move,
                        "certified": certified,
                        # the generator allows multiple mates on the final move
                        "alternate_mate": after_played.mate_in == 1,
                    })
                damage = played_move_damage_pp(before, after_played)
                if before.mate_in is not None and before.mate_in < 0:
                    already_lost_before.append({
                        "puzzle_id": row["PuzzleId"],
                        "before_mate_in": before.mate_in,
                        "damage_pp": round(damage, 1),
                    })
                elif damage < BLUNDER_SWING_THRESHOLD_PP:
                    weak_swing.append({
                        "puzzle_id": row["PuzzleId"],
                        "before_cp": before.score_centipawns,
                        "before_win_pct": round(before.win_percent, 1),
                        "damage_pp": round(damage, 1),
                    })
            else:
                if before.best_move != row["PlayedMove"]:
                    best_arm_disagreements.append({
                        "puzzle_id": row["PuzzleId"],
                        "engine_move": before.best_move,
                        "played": row["PlayedMove"],
                    })

            out_rows.append({
                "PuzzleId": row["PuzzleId"],
                "Arm": row["Arm"],
                **eval_columns("Before", before),
                **eval_columns("AfterPlayed", after_played),
                **eval_columns("AfterRefutation", after_refutation),
            })
            if index % 25 == 0:
                print(f"{index}/{len(rows)} rows analysed", flush=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    audit = {
        "best_arm_disagreements": {"count": len(best_arm_disagreements), "rows": best_arm_disagreements},
        "already_lost_before": {"count": len(already_lost_before), "rows": already_lost_before},
        "weak_swing": {"count": len(weak_swing), "rows": weak_swing},
        "refutation_disagreements": {"count": len(refutation_disagreements), "rows": refutation_disagreements},
    }
    meta = {
        "engine": provenance,
        "generated": date.today().isoformat(),
        "input": args.golden.name,
        "input_sha256": input_sha256,
        "rows": len(out_rows),
        "limit": args.limit,
        "columns": COLUMNS_NOTE,
        "thresholds": {
            "already_lost_before": "before mate_in < 0 (forced mate regardless of move played)",
            "weak_swing_pp": BLUNDER_SWING_THRESHOLD_PP,
            "provenance": "generator's advantage-path setup gate is win_chances +0.6; "
                          "solution uniqueness is +0.7; mate-path puzzles have no setup gate",
        },
        "audit": audit,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n{len(out_rows)} rows certified -> {args.output}")
    print("audit counts:", {name: block["count"] for name, block in audit.items()})
    for name, block in audit.items():
        if block["rows"]:
            print(f"{name}:")
            for entry in block["rows"]:
                details = " ".join(f"{k}={v}" for k, v in entry.items() if k != "puzzle_id")
                print(f"  {entry['puzzle_id']} {details} "
                      f"https://lichess.org/training/{entry['puzzle_id']}")


if __name__ == "__main__":
    main()
