"""Sample the golden-set candidates from the labeled post-cutoff puzzles.

Draws 250 puzzles into two disjoint arms for manual review:

- blunder arm (200): the eval position is the dump FEN and the played move is
  the puzzle's setup move - a blunder by construction, with the solver's reply
  as its refutation. Stratified evenly: 10 motif categories x 4 rating bands
  x 5 puzzles.
- best arm (50): the eval position is AFTER the setup move and the played move
  is the solver's first reply - the engine-verified winning move, so the
  ground truth is "best". 10 categories x 5, spread over rating bands
  round-robin. No overlap with the blunder arm.

Each puzzle gets ONE primary category via a precedence list (most specific
tactic first), because generated themes are multi-label. Sampling is seeded
and reproducible. Ratings are provisional for most post-cutoff puzzles (median
RatingDeviation ~116), so bands are approximate; the meta sidecar records this.

Usage:
    uv run python scripts/sample_golden.py
"""

import argparse
import csv
import json
import random
import sys
from datetime import date
from itertools import cycle, islice
from pathlib import Path

import chess

PRECEDENCE = [
    ("mate", lambda t: "mate" in t),
    ("fork", lambda t: "fork" in t),
    ("pin", lambda t: "pin" in t),
    ("skewer", lambda t: "skewer" in t),
    ("discoveredAttack", lambda t: "discoveredAttack" in t),
    ("hangingPiece", lambda t: "hangingPiece" in t),
    ("sacrifice", lambda t: "sacrifice" in t),
    ("promotion", lambda t: t & {"promotion", "advancedPawn", "underPromotion"}),
    ("defensive", lambda t: t & {"defensiveMove", "quietMove"}),
    ("other", lambda t: True),
]
BANDS = [(0, 1200, "<1200"), (1200, 1600, "1200-1599"), (1600, 2000, "1600-1999"), (2000, 9999, "2000+")]
PER_CELL = 5      # blunder arm: 10 categories x 4 bands x 5 = 200
PER_CATEGORY = 5  # best arm: 10 categories x 5 = 50


def category(themes: set[str]) -> str:
    for name, predicate in PRECEDENCE:
        if predicate(themes):
            return name
    raise AssertionError("PRECEDENCE ends with a catch-all")


def band(rating: int) -> str:
    for lo, hi, name in BANDS:
        if lo <= rating < hi:
            return name
    raise ValueError(f"rating {rating} outside all bands")


def eval_item(row: dict, arm: str) -> dict:
    """The eval-facing view of a puzzle: a position, the move played in it,
    and the ground truth for that move."""
    moves = row["Moves"].split()
    if arm == "blunder":
        fen, played, continuation = row["FEN"], moves[0], moves[1:]
    else:
        board = chess.Board(row["FEN"])
        board.push_uci(moves[0])
        fen, played, continuation = board.fen(), moves[1], moves[2:]
    return {
        "Arm": arm,
        "PuzzleId": row["PuzzleId"],
        "FEN": fen,
        "PlayedMove": played,
        "GroundTruth": arm,  # arm names double as the ground-truth labels
        "Continuation": " ".join(continuation),
        "Category": row["Category"],
        "Band": row["Band"],
        "Rating": row["Rating"],
        "RatingDeviation": row["RatingDeviation"],
        "GeneratedThemes": row["GeneratedThemes"],
        "GameUrl": row["GameUrl"],
        "TrainingUrl": f"https://lichess.org/training/{row['PuzzleId']}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puzzles", type=Path, default=Path("post_cutoff_puzzles.csv"))
    parser.add_argument("--themes", type=Path, default=Path("post_cutoff_themes.csv"))
    parser.add_argument("--output", type=Path, default=Path("golden_candidates.csv"))
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} already exists and may contain manual review marks; "
                 f"delete it (or pass a different --output) to resample")

    with open(args.themes, newline="") as f:
        themes = {r["PuzzleId"]: set(r["GeneratedThemes"].split()) for r in csv.DictReader(f)}
    cells: dict[tuple[str, str], list[dict]] = {}
    with open(args.puzzles, newline="") as f:
        for row in csv.DictReader(f):
            row_themes = themes[row["PuzzleId"]]
            row["GeneratedThemes"] = " ".join(sorted(row_themes))
            row["Category"] = category(row_themes)
            row["Band"] = band(int(row["Rating"]))
            cells.setdefault((row["Category"], row["Band"]), []).append(row)

    rng = random.Random(args.seed)
    items: list[dict] = []

    for cat, _ in PRECEDENCE:
        for _, _, band_name in BANDS:
            pool = cells.get((cat, band_name), [])
            if len(pool) < PER_CELL:
                sys.exit(f"cell ({cat}, {band_name}) has only {len(pool)} puzzles; can't fill the grid")
            items += [eval_item(r, "blunder") for r in rng.sample(pool, PER_CELL)]
    blunder_total = len(items)

    taken = {item["PuzzleId"] for item in items}
    for offset, (cat, _) in enumerate(PRECEDENCE):
        # start each category's band rotation at a different point, otherwise
        # the 5th pick of every category lands in the same (first) band
        picked = 0
        for _, _, band_name in islice(cycle(BANDS), offset, offset + len(BANDS) * PER_CATEGORY):
            if picked >= PER_CATEGORY:
                break
            pool = [r for r in cells[cat, band_name] if r["PuzzleId"] not in taken]
            if not pool:
                continue
            row = rng.choice(pool)
            taken.add(row["PuzzleId"])
            items.append(eval_item(row, "best"))
            picked += 1
        if picked < PER_CATEGORY:
            sys.exit(f"category {cat} ran out of best-arm candidates ({picked}/{PER_CATEGORY})")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)

    deviations = sorted(int(i["RatingDeviation"]) for i in items)
    meta = {
        "generated": date.today().isoformat(),
        "seed": args.seed,
        "puzzles": args.puzzles.name, "puzzles_size": args.puzzles.stat().st_size,
        "themes": args.themes.name, "themes_size": args.themes.stat().st_size,
        "blunder_arm": blunder_total,
        "best_arm": len(items) - blunder_total,
        "grid": f"{len(PRECEDENCE)} categories x {len(BANDS)} bands x {PER_CELL} (blunder); "
                f"{len(PRECEDENCE)} categories x {PER_CATEGORY} round-robin bands (best)",
        "columns": "PlayedMove is UCI in FEN's position; Continuation is the engine line after "
                   "PlayedMove, space-separated UCI; Category/Band are sampling strata baked at generation",
        "rating_caveat": f"post-cutoff ratings are provisional; median RatingDeviation of the "
                         f"selection is {deviations[len(deviations) // 2]}, "
                         f"{sum(1 for d in deviations if d > 100)}/{len(deviations)} over 100",
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(items)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
