"""Acceptance checks for the vendored Lichess tagger, run BEFORE trusting its labels.

Both checks sample the fully-tagged May dump:

1. Per-theme agreement: of the sampled puzzles the dump tags with a theme, how
   many does the vendored cook.py reproduce, and how often does it fire where
   the dump doesn't. Disagreement here is expected, measurable noise - dump
   themes are tagger output refined by player votes and older tagger versions -
   and this puts a number on it.

2. hanging_piece cross-check: motif_detector.hanging_piece vs the vendored
   hanging_piece on the same rows. Every disagreement must fall into the one
   documented divergence class (pawn / en passant victims, which I keep and
   Lichess excludes). Anything unexplained means the adapter is wired wrong,
   and the script exits nonzero.

Usage:
    uv run python scripts/validate_vendored_tagger.py
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import chess

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from calibrate_motifs import DUMP_DEFAULT, url  # noqa: E402
from lichess_tagger import generated_themes  # noqa: E402
from motif_detector import VALUES, hanging_piece, victim_value  # noqa: E402

SAMPLE_RATE = 0.002  # per-row keep probability; interacts with --sample, which caps the total

THEMES = [
    "hangingPiece", "fork", "pin", "skewer", "discoveredAttack", "sacrifice",
    "deflection", "attraction", "trappedPiece", "intermezzo", "exposedKing",
    "quietMove", "defensiveMove", "advancedPawn", "promotion", "enPassant",
    "backRankMate", "mateIn1", "mateIn2", "mateIn3",
]


def sample_rows(dump: Path, n: int, seed: int) -> list[dict]:
    """Coin-flip on raw lines, CSV-parsing only the keepers. Stopping at n reads
    a prefix of the file, which is fine here: the dump is ordered by random
    PuzzleId, so a prefix sample is effectively uniform."""
    rng = random.Random(seed)
    kept: list[str] = []
    with open(dump, newline="") as f:
        header = next(f)
        for line in f:
            if rng.random() < SAMPLE_RATE:
                kept.append(line)
                if len(kept) >= n:
                    break
    return list(csv.DictReader([header] + kept))


def pawn_or_ep_victim(fen: str, moves: list[str]) -> bool:
    """The one class where my hanging_piece deliberately out-fires Lichess's.
    Only valid on rows where hanging_piece fired, which guarantees moves[1]
    is a capture (victim_value's precondition)."""
    board = chess.Board(fen)
    board.push_uci(moves[0])
    reply = chess.Move.from_uci(moves[1])
    return victim_value(board, reply) == VALUES[chess.PAWN]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path, default=DUMP_DEFAULT)
    parser.add_argument("--sample", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    rows = sample_rows(args.dump, args.sample, args.seed)
    print(f"sampled {len(rows)} rows from {args.dump.name}\n")

    dump_n: Counter[str] = Counter()
    agree: Counter[str] = Counter()
    extra: Counter[str] = Counter()
    mine_only, theirs_only, unexplained = [], [], []

    for row in rows:
        dump_themes = set(row["Themes"].split())
        generated = set(generated_themes(row["PuzzleId"], row["FEN"], row["Moves"]))
        for theme in THEMES:
            if theme in dump_themes:
                dump_n[theme] += 1
                if theme in generated:
                    agree[theme] += 1
            elif theme in generated:
                extra[theme] += 1

        moves = row["Moves"].split()
        mine = hanging_piece(row["FEN"], moves)
        theirs = "hangingPiece" in generated
        if mine and not theirs:
            if pawn_or_ep_victim(row["FEN"], moves):
                mine_only.append(row)
            else:
                unexplained.append(row)
        elif theirs and not mine:
            theirs_only.append(row)

    print("check 1: vendored cook.py vs dump themes (vote/version noise)")
    print(f"  {'theme':18s} {'dump':>5s} {'reproduced':>10s} {'extra-fires':>11s}")
    for theme in THEMES:
        if dump_n[theme] or extra[theme]:
            reproduced = f"{100 * agree[theme] / dump_n[theme]:9.1f}%" if dump_n[theme] else f"{'-':>10s}"
            print(f"  {theme:18s} {dump_n[theme]:5d} {reproduced} {extra[theme]:11d}")

    print("\ncheck 2: my hanging_piece vs vendored hanging_piece")
    print(f"  mine-only, explained (pawn/ep victim): {len(mine_only)}")
    print(f"  theirs-only: {len(theirs_only)}")
    print(f"  UNEXPLAINED (adapter bug if nonzero): {len(unexplained)}")
    for row in (unexplained + theirs_only)[:10]:
        print(f"    {row['PuzzleId']} {url(row)}")
    if unexplained or theirs_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
