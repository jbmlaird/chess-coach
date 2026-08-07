"""Fixtures for the hanging-piece motif detector.

Sampled from my May 27 2026 Lichess puzzle dump. Rows with IDs are from the dump,
anything with *-handmade was written specifically to test certain board positions:
- 3 free hangs: a piece is left with no defender and gets taken
- 2 en passant hangs
- 5 fake defenses: the only "defender" aims through the square the capturing
  pawn stands on, so the piece is really hanging
- 1 recapture: queen takes back the queen that just took it - a trade, not a hang
- 5 other tactics (fork, skewer, two mates, discovered attack): the solution
  doesn't start with a capture, so no hang
- 1 defended en passant
- 1 king grabs a free pawn
- 1 pinned defender: counted as a defender even though it can't legally
  recapture - wrong but not captured
- 1 battery: no direct defender, but a rook behind the attacking rook
  recaptures, so not a hang
"""

import pytest

from motif_detector import hanging_piece


@pytest.mark.parametrize(
    "fen,moves,expected",
    [
        pytest.param(
            "8/pk2Q2p/3n2p1/p2Pp3/4P2P/1q1p2PB/8/6K1 b - - 5 39",
            "b7a6 e7d6 b3b6 d6b6",
            True,
            id="hang-free-Xa2zo-r1108",
        ),
        pytest.param(
            "r4rk1/pp2qppp/8/3b4/3np3/2PB4/PP3PPP/R1BQ1RK1 w - - 0 16",
            "d3e4 e7e4 f2f3 d4e2",
            True,
            id="hang-free-OXCLK-r1697",
        ),
        pytest.param(
            "r1q3k1/p1p3pp/bp1pN1r1/2nPp3/Q1P2p2/P1P3PB/4PP1P/R3R1K1 w - - 2 21",
            "e6c5 c8h3 a4d7 h3d7",
            True,
            id="hang-free-vsSfR-r1868",
        ),
        pytest.param(
            "7k/3p4/8/4P3/8/8/8/K7 b - - 0 1",
            "d7d5 e5d6",
            True,
            id="hang-ep-white-captures-handmade",
        ),
        pytest.param(
            "k7/8/8/8/3p4/8/4P3/7K w - - 0 1",
            "e2e4 d4e3",
            True,
            id="hang-ep-black-captures-handmade",
        ),
        pytest.param(
            "2br1r2/1p2nkpQ/4p2p/pR2P2P/P1q4B/2Pp4/1PB2PP1/5RK1 w - - 0 28",
            "h4e7 d3c2 h7c2 f7e7",
            True,
            id="hang-value-pawnxbishop-30DQF-r1813",
        ),
        pytest.param(
            "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",
            "d1d8 e8d8",
            False,
            id="no-hang-queen-recapture-handmade"
        ),
        pytest.param(
            "1r3rk1/1bR2ppp/pB6/1q2p3/N2pn3/1B2QP1P/PP4P1/2R3K1 w - - 0 24",
            "c7b7 d4e3 b7b8 f8b8",
            True,
            id="hang-value-pawnxqueen-3oF2X-r1803",
        ),
        pytest.param(
            "5r1k/1pp4p/3p4/p2PnB2/2P3P1/1P1nQ1Nq/P7/1R4K1 b - - 3 30",
            "f8f5 g4f5 e5g4 e3e8",
            True,
            id="hang-value-pawnxrook-4QT5q-r2208",
        ),
        pytest.param(
            "5rk1/1b3ppp/pQP5/5P2/P3q3/1P1n4/6PP/2R2R1K b - - 2 31",
            "d3c1 c6b7 f8b8 f1c1",
            True,
            id="hang-value-pawnxbishop-6MoA4-r1672",
        ),
        pytest.param(
            "1r1r2k1/pB3ppp/2P5/3q4/P2P1Q1b/4P1nP/6P1/R3R1K1 b - - 1 29",
            "b8b7 c6b7 d5b7 f4h4",
            True,
            id="hang-value-pawnxrook-DLLsr-r1546",
        ),
        pytest.param(
            "8/3pk3/8/4P3/8/8/8/K7 b - - 0 1",
            "d7d5 e5d6",
            False,
            id="no-hang-en-passant-handmade",
        ),
        pytest.param(
            "1k6/1pp2p2/p1n5/6R1/7r/4Bpp1/PPP5/2K5 b - - 1 32",
            "f3f2 g5g8 c6d8 g8d8",
            False,
            id="no-hang-mateIn2-iMPRM-r663",
        ),
        pytest.param(
            "r2r2k1/p1qnbppp/1p2p3/1Pp5/N1P3n1/4Pb1P/PBQ1BPP1/R4RK1 w - - 0 15",
            "e2f3 c7h2",
            False,
            id="no-hang-mateIn1-Lm4Hl-r933",
        ),
        pytest.param(
            "2r3k1/5Rpp/1p6/3p1N2/3r4/2P5/q5PP/5R1K b - - 1 30",
            "d4g4 f5e7 g8h8 f7f8 c8f8 f1f8",
            False,
            id="no-hang-fork-eSlHH-r1475",
        ),
        pytest.param(
            "8/8/8/2p2p1R/r3k1p1/4P1KP/6P1/8 b - - 0 43",
            "g4h3 h5h4 e4e3 h4a4",
            False,
            id="no-hang-skewer-071SG-r1283",
        ),
        pytest.param(
            "6B1/2q3b1/3p2rk/p1p2p1p/P1N2R1P/2PPQ3/5KP1/1r6 b - - 2 38",
            "g7c3 f4g4 h6g7 g4g6",
            False,
            id="no-hang-discoveredAttack-iCT5I-r2391",
        ),
        pytest.param(
            "k7/8/8/8/8/8/4p3/4K3 b - - 0 1",
            "a8a7 e1e2",
            True,
            id="hang-king-captures-free-pawn-handmade",
        ),
        pytest.param(
            # Bd6 is defended only by Ne4, which is absolutely pinned to the
            # e-file by the queen - Nxd6 is illegal, so the bishop is free.
            # The detector still counts the pinned defender, the pin detector
            # will capture this.
            "k2rq3/8/8/2B5/4N3/8/8/4K3 w - - 0 1",
            "c5d6 d8d6",
            False,
            id="no-hang-pinned-only-defender-handmade-KNOWN-LIMITATION",
        ),
        pytest.param(
            # The d5 knight looks free: Rd8's defense is blocked by the white
            # rook on d6. is_defended lifts the ray attacker off the line and
            # finds the battery - after Rxd5, Rd8 recaptures through the
            # square the capturer vacated. Deleting the lift branch flips
            # this row to a (wrong) True.
            "3r2k1/8/1n1R4/8/8/8/8/6K1 b - - 0 1",
            "b6d5 d6d5",
            False,
            id="no-hang-battery-defense-through-attacker-handmade",
        ),
    ]
)
def test_hanging_piece(fen, moves, expected):
    assert hanging_piece(fen, moves.split()) == expected


def test_single_move_raises():
    with pytest.raises(ValueError, match="setup move and a solution reply"):
        hanging_piece("k6r/8/8/8/8/8/8/KR6 w - - 0 1", ["b1b2"])


def test_illegal_solution_move_raises():
    with pytest.raises(ValueError, match="not legal"):
        hanging_piece("k6r/8/8/8/8/8/8/KR6 w - - 0 1", ["b1b2", "b2a7"])
