"""The contamination check finds a golden position however its clocks are written, and misses nothing else."""

import csv

import chess

import check_contamination


def test_a_golden_position_is_found_by_board_not_by_string(tmp_path):
    row = next(csv.DictReader(open("golden/v2/golden_candidates.csv")))
    board = chess.Board(row["FEN"])
    board.push_uci(row["PlayedMove"])  # the position after the student's move counts too
    disguised = " ".join(board.fen().split()[:4] + ["7", "99"])  # different clocks, same position
    path = tmp_path / "train.csv"
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["fen", "answer"])
        w.writerow([disguised, "x"])
        w.writerow([chess.STARTING_FEN, "y"])
    result = check_contamination.scan(f"read_csv_auto('{path}')", "fen")
    assert result == {"rows": 2, "positions": {row["PuzzleId"]: 1}, "ids": {}}


def test_a_clean_set_reports_nothing(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("PuzzleId,fen\nzzzzz," + chess.STARTING_FEN + "\n")
    assert check_contamination.scan(f"read_csv_auto('{path}')", "fen", "PuzzleId") == {"rows": 1, "positions": {}, "ids": {}}
