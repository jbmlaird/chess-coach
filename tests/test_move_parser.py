import pytest

import move_parser
from move_parser import Outcome


@pytest.mark.parametrize(
    "completion,board_fen,outcome",
    [
        # 2 hallucinations
        ("BEST_MOVE: Nb4", "8/1B2k3/4pp2/N5p1/Pp1bP2p/1P3K1P/n5P1/8 b - - 0 42", Outcome.ILLEGAL),
        ("BEST_MOVE: Qh7+", "r4rk1/1pq1np1p/p3pBp1/3pP1Q1/3n2P1/2N5/PPP2P1P/R4RK1 w - - 1 16", Outcome.ILLEGAL),

        ("BEST_MOVE: Ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # legal SAN
        ("BEST_MOVE: a5a3", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # legal UCI
        ("BEST_MOVE: g7=Q", "8/1k4P1/8/8/8/8/2K5/8 w - - 0 1", Outcome.ILLEGAL),  # g7 isn't a promotion square
        ("BEST_MOVE: g8=Q", "8/1k4P1/8/8/8/8/2K5/8 w - - 0 1", Outcome.LEGAL),  # legal promotion
        # short castle
        ("BEST_MOVE: O-O", "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", Outcome.LEGAL),
        # long castle
        ("BEST_MOVE: O-O-O", "r1bqkb1r/ppp1pppp/5n2/3p4/1n1P1B2/2NQ4/PPP1PPPP/R3KBNR w KQkq - 0 1", Outcome.LEGAL),

        ("", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.PARSE_ERROR),
        ("BEST_MOVE: Ra3+\nBEST_MOVE: Ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # last match wins
        ("BEST_MOVE: Ra3+\nBEST_MOVE: Rb3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.ILLEGAL),
        # last match illegal
        ("BEST_MOVE: ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.INVALID),  # notation failure
        ("BEST_MOVE: Ra3+..", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # strip dots
        ("BEST_MOVE: 0000", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.ILLEGAL),  # null move
        ("BEST_MOVE: --", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.ILLEGAL),  # null move
        ("BEST_MOVE: e3", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", Outcome.ILLEGAL),
        # checkmate. No moves possible
        ("BEST_MOVE: Rc1", "7k/8/8/7K/8/8/8/R3R3 w - - 0 ", Outcome.AMBIGUOUS),  # two pieces can move there

        # incorrectly labeled illegal due to parsing issues, now parse
        ("**BEST_MOVE:** Qf6+", "8/pp6/6k1/2q5/7P/6p1/P5P1/5Q1K w - - 7 42", Outcome.LEGAL),
        ("**BEST_MOVE: Rh1+**", "8/6p1/3k2P1/R7/7K/8/8/6r1 b - - 10 56", Outcome.LEGAL),

        ("BEST_MOVE: Ra3+!", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # annotation stripped
        ("BEST_MOVE:  Ra3+", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # extra spaces after colon
        ("BEST_MOVE: Ra3+ wins the rook", "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # trailing commentary
        ("VERDICT: GOOD\r\nBEST_MOVE: Ra3+\r\nEXPLANATION: text here.\r\n",
         "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),  # CRLF line endings

        ("The rook is attacked; Kf5 walks into danger.\n\n"
         "VERDICT: BLUNDER\n"
         "BEST_MOVE: Ra3+\n"
         "EXPLANATION: The check on a3 wins a tempo against the exposed king.",
         "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81", Outcome.LEGAL),
    ]
)
def test_eval(completion, board_fen, outcome):
    parsed_verdict = move_parser.parse_move_field(board_fen, completion, "BEST_MOVE")
    assert parsed_verdict.outcome == outcome


ROOK_FEN = "8/8/7R/r3k3/8/5K2/8/8 b - - 19 81"
BOTH_FIELDS = "VERDICT: BLUNDER\nREFUTATION: Ra3+\nBEST_MOVE: Rb5\nEXPLANATION: text."


@pytest.mark.parametrize(
    "completion,board_fen,outcome,uci",
    [
        (BOTH_FIELDS, ROOK_FEN, Outcome.LEGAL, "a5a3"),

        ("BEST_MOVE: Ra3+", ROOK_FEN, Outcome.PARSE_ERROR, None),

        ("REFUTATION: NONE", ROOK_FEN, Outcome.INVALID, None),
        ("**REFUTATION:** Ra3+", ROOK_FEN, Outcome.LEGAL, "a5a3"),  # markdown stripped
        ("REFUTATION: Ra3+\nREFUTATION: Rb3+", ROOK_FEN, Outcome.ILLEGAL, None),  # last match wins
        # uci carries the promotion suffix
        ("REFUTATION: g8=Q", "8/1k4P1/8/8/8/8/2K5/8 w - - 0 1", Outcome.LEGAL, "g7g8q"),
    ]
)
def test_refutation_field(completion, board_fen, outcome, uci):
    parsed = move_parser.parse_move_field(board_fen, completion, "REFUTATION")
    assert parsed.outcome == outcome
    assert parsed.uci == uci


def test_best_move_field_selectivity():
    parsed = move_parser.parse_move_field(ROOK_FEN, BOTH_FIELDS, "BEST_MOVE")
    assert parsed.outcome == Outcome.LEGAL
    assert parsed.uci == "a5b5"
