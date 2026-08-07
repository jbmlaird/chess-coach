"""Calibrate my motif detectors against Lichess's own theme tags.

For a detector (e.g. hanging_piece) and its corresponding Lichess theme
(e.g. hangingPiece), samples tagged and untagged puzzles from an old Lichess
dump that is tagged and reports:

- recall vs tag: of puzzles Lichess tagged with the theme, how many we detect
- broader-fire rate: how often we detect tags on puzzles tagged with OTHER tactics
- example disagreements in both directions, with lichess.org/training URLs
  for hand-verification
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motif_detector import hanging_piece

DETECTORS = {
    "hanging": (hanging_piece, "hangingPiece"),
}

TACTIC_THEMES = {
    "hangingPiece", "pin", "fork", "skewer", "discoveredAttack", "sacrifice",
    "deflection", "attraction", "mateIn1", "mateIn2", "mateIn3", "trappedPiece",
}


def sample_puzzles(dump: Path, target_theme: str, n: int, seed: int):
    """Use coin flip sampling. Make a different --seed draw a different sample."""
    rng = random.Random(seed)
    tagged, untagged = [], []
    with open(dump, newline="") as f:
        for row in csv.DictReader(f):
            themes = set(row["Themes"].split())
            if target_theme in themes:
                if len(tagged) < n and rng.random() < 0.05:
                    tagged.append(row)
            elif themes & TACTIC_THEMES:
                if len(untagged) < n and rng.random() < 0.005:
                    untagged.append(row)
            if len(tagged) >= n and len(untagged) >= n:
                break
    return tagged, untagged


def url(row: dict) -> str:
    return f"https://lichess.org/training/{row['PuzzleId']}"


def calibrate(name: str, dump: Path, n: int, seed: int, examples: int) -> None:
    detector, theme = DETECTORS[name]
    tagged, untagged = sample_puzzles(dump, theme, n, seed)

    misses = [r for r in tagged if not detector(r["FEN"], r["Moves"].split())]
    fires = [r for r in untagged if detector(r["FEN"], r["Moves"].split())]

    hits = len(tagged) - len(misses)
    print(f"[{name}] vs Lichess '{theme}' tag ({dump.name})")
    print(f"  tagged sample n={len(tagged)}: recall {hits}/{len(tagged)} ({100 * hits / len(tagged):.2f}%)")
    print(f"  other-tactic sample n={len(untagged)}: fires on {len(fires)}/{len(untagged)} "
          f"({100 * len(fires) / len(untagged):.2f}%)")

    if misses:
        print(f"  tagged-but-not-fired (recall gaps, hand-check {min(examples, len(misses))}):")
        for row in misses[:examples]:
            print(f"    {row['PuzzleId']} [{row['Themes'][:55]}] {url(row)}")
    if fires:
        print(f"  fired-but-untagged (definition boundary, hand-check {min(examples, len(fires))}):")
        for row in fires[:examples]:
            print(f"    {row['PuzzleId']} [{row['Themes'][:55]}] {url(row)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path,
                        default=Path.home() / ".cache" / "chess-coach-evals" / "lichess_db_puzzle.csv",
                        help="an OLD, fully-theme-tagged puzzle dump (not the post-cutoff slice)")
    parser.add_argument("--detector", choices=DETECTORS, default=None,
                        help="calibrate one detector (default: all registered)")
    parser.add_argument("--sample", type=int, default=3000,
                        help="sample size for each of tagged/untagged pools (default: 3000)")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--examples", type=int, default=6,
                        help="how many disagreement URLs to print per direction")
    args = parser.parse_args()

    names = [args.detector] if args.detector else list(DETECTORS)
    for name in names:
        calibrate(name, args.dump, args.sample, args.seed, args.examples)


if __name__ == "__main__":
    main()
