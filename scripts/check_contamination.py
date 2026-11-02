"""Is any golden position in a training set? A training set must clear this by puzzle id and by position before it
is used, and a public checkpoint's data must clear it before that model's column is published.

Usage:
    uv run python scripts/check_contamination.py --hf lucasdino/chess-reasoning-data --column fen_board
    uv run python scripts/check_contamination.py --file data/train.csv --column fen [--id-column PuzzleId]

Positions compare on board, side to move and castling rights (clocks and en-passant squares vary between sources)
against the three positions every golden row exposes: the Lichess puzzle position before its setup move, the position
the student saw, and the position after their move. Exits 1 on any hit.
"""

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import chess
import duckdb

REPO = Path(__file__).parent.parent
GOLDEN = sorted((REPO / "golden").glob("v*/golden_candidates.csv"))


def key(board: chess.Board) -> str:
    return f"{board.board_fen()} {'w' if board.turn else 'b'} {board.castling_xfen()}"


def golden_positions() -> dict[str, str]:
    """position key -> the golden puzzle id it belongs to."""
    source = {r["PuzzleId"]: r["FEN"] for r in csv.DictReader((REPO / "post_cutoff_puzzles.csv").open())}
    positions = {}
    for path in GOLDEN:
        for row in csv.DictReader(path.open()):
            given = chess.Board(row["FEN"])
            after = given.copy()
            after.push_uci(row["PlayedMove"])
            for board in (chess.Board(source[row["PuzzleId"]]), given, after):
                positions[key(board)] = row["PuzzleId"]
    return positions


def scan(source: str, column: str, id_column: str | None = None) -> dict:
    """Rows scanned, and the golden ids hit by position and (if the set carries them) by puzzle id."""
    positions = golden_positions()
    wanted = ", ".join(f"'{k}'" for k in positions)  # 1,500 literals: the match runs inside duckdb, next to the data
    con = duckdb.connect()
    rows = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    hits = con.execute(f'SELECT k, count(*) FROM (SELECT array_to_string(string_split("{column}", \' \')[1:3], \' \') AS k '
                       f"FROM {source}) WHERE k IN ({wanted}) GROUP BY k").fetchall()
    ids = ", ".join(f"'{i}'" for i in set(positions.values()))
    by_id = con.execute(f'SELECT "{id_column}", count(*) FROM {source} WHERE "{id_column}" IN ({ids}) GROUP BY 1').fetchall() if id_column else []
    return {"rows": rows, "positions": {positions[k]: n for k, n in hits}, "ids": dict(by_id)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf", help="a Hugging Face dataset id: every parquet file in it is scanned")
    parser.add_argument("--file", type=Path, help="a local csv or jsonl training set")
    parser.add_argument("--column", required=True, help="the FEN column")
    parser.add_argument("--id-column", help="a Lichess puzzle id column, if the set has one")
    args = parser.parse_args()
    if args.hf:
        sha = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{args.hf}"))["sha"]
        source, name = f"read_parquet('hf://datasets/{args.hf}/**/*.parquet', union_by_name=true)", f"{args.hf} @ {sha[:12]}"
    else:
        reader = "read_json_auto" if args.file.suffix in (".json", ".jsonl") else "read_csv_auto"
        source, name = f"{reader}('{args.file}')", str(args.file)
    result = scan(source, args.column, args.id_column)
    hits = {**result["positions"], **result["ids"]}
    print(f"{name}: {result['rows']:,} rows scanned against {sum(1 for _ in GOLDEN) * 250} golden rows; "
          f"{len(hits)} golden ids hit" + (f": {hits}" if hits else ""))
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
