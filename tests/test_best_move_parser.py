import pytest

import best_move_parser
from best_move_parser import Outcome


@pytest.mark.parametrize(
    "completion,board_fen,outcome",
    [
        # 2 hallucinations
        ("BEST_MOVE: Nb4", "8/1B2k3/4pp2/N5p1/Pp1bP2p/1P3K1P/n5P1/8 b - - 0 42", Outcome.ILLEGAL),
        ("BEST_MOVE: Qh7+", "r4rk1/1pq1np1p/p3pBp1/3pP1Q1/3n2P1/2N5/PPP2P1P/R4RK1 w - - 1 16", Outcome.ILLEGAL),

        ("BEST_MOVE: Ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # legal SAN
        ("BEST_MOVE: a5a3", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # legal UCI
        ("BEST_MOVE: g7=Q", "8/1k4P1/8/8/8/8/2K5/8 w - - 0 1", Outcome.ILLEGAL),  # illegal promotion
        ("BEST_MOVE: g8=Q", "8/1k4P1/8/8/8/8/2K5/8 w - - 0 1", Outcome.LEGAL),  # legal promotion
        # short castle
        ("BEST_MOVE: O-O", "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", Outcome.LEGAL),
        # long castle
        ("BEST_MOVE: O-O-O", "r1bqkb1r/ppp1pppp/5n2/3p4/1n1P1B2/2NQ4/PPP1PPPP/R3KBNR w KQkq - 0 1", Outcome.LEGAL),

        ("", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.PARSE_ERROR),
        ("BEST_MOVE: Ra3+\nBEST_MOVE: Ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.PARSE_ERROR),
        ("BEST_MOVE: ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.INVALID),  # notation failure
        ("BEST_MOVE: Ra3+..", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # strip dots
    ]
)
def test_eval(completion, board_fen, outcome):
    parsed_verdict = best_move_parser.parse_best_move(board_fen, completion)
    assert parsed_verdict.outcome is outcome
