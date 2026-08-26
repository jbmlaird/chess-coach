"""Tests for the shared engine wrapper.

Engine-backed tests need a Stockfish binary. Assertions avoid anything version-sensitive except the one golden
cross-check, where any strong engine agrees (f8a3 is the unique mate; the second-best move is half a pawn worse).
"""

import chess
import pytest

from engine import (
    CP_CEILING,
    Engine,
    EngineEval,
    GRADER_CONFIG,
    GRADER_NODES,
    MATE_SCORE_CP,
    STOCKFISH_PATH,
    win_percent,
)


@pytest.fixture(scope="module")
def engine():
    if STOCKFISH_PATH is None:
        pytest.skip("stockfish binary not installed")
    with Engine.grader() as e:
        yield e


def test_finds_certified_refutation_on_golden_row(engine):
    # Wj3vh after the blunder Qxc4??: the certified refutation is Qa3+ (f8a3)
    board = chess.Board("r4q1k/6pp/1p3n2/5N2/P1b2P2/1Q2P2P/K5P1/2bR2R1 w - - 0 31")
    board.push_uci("b3c4")
    verdict = engine.analyse(board)
    assert verdict.best_move == "f8a3"
    assert verdict.mate_in == 2
    assert verdict.score_centipawns == MATE_SCORE_CP
    # Lichess maps forced mates to the +-1000cp ceiling, ~97.5%, never 100
    assert verdict.win_percent == pytest.approx(97.5, abs=0.1)


def test_score_is_side_to_move_relative(engine):
    board = chess.Board("3qk3/8/8/8/8/8/8/2QQK3 w - - 0 1")
    assert engine.analyse(board).score_centipawns > 500
    board = chess.Board("2qqk3/8/8/8/8/8/8/3QK3 b - - 0 1")
    assert engine.analyse(board).score_centipawns > 500


def test_checkmated_position_is_a_terminal_verdict(engine):
    # golden row f16cc after blunder and the certified mating reply Qxg7#:
    # the grader analyses this exact shape whenever a model answers perfectly
    board = chess.Board("r1b2rk1/ppqn1pbp/3p3B/2p3Qp/3pP3/5P2/PPP1N1P1/2KR1BN1 b - - 1 14")
    board.push_uci("c7d8")
    board.push_uci("g5g7")
    verdict = engine.analyse(board)
    assert verdict.best_move is None
    assert verdict.principal_variation == []
    assert verdict.mate_in == 0
    assert verdict.score_centipawns == -MATE_SCORE_CP  # side to move is checkmated


def test_stalemate_is_a_draw_verdict(engine):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    verdict = engine.analyse(board)
    assert verdict.best_move is None
    assert verdict.score_centipawns == 0


def test_repeated_analysis_is_deterministic(engine):
    # the fresh-game token clears the transposition table per call: the same
    # position must produce the identical verdict on a warm engine
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    first = engine.analyse(board)
    engine.analyse(chess.Board())
    second = engine.analyse(board)
    assert first == second


def test_win_percent_shape():
    assert win_percent(0) == pytest.approx(50)
    assert win_percent(100) > 55
    assert win_percent(-100) < 45
    # symmetric around equality
    assert win_percent(300) + win_percent(-300) == pytest.approx(100)
    # Lichess ceiling: beyond +-CP_CEILING everything saturates at ~97.5
    assert win_percent(CP_CEILING) == win_percent(5000) == win_percent(MATE_SCORE_CP)
    assert win_percent(CP_CEILING) == pytest.approx(97.5, abs=0.1)
    assert EngineEval("e2e4", 300, None, ["e2e4"]).win_percent == win_percent(300)


def test_engine_reports_provenance(engine):
    assert "Stockfish" in engine.name
    prov = engine.provenance
    assert prov["config"] == GRADER_CONFIG
    assert f"nodes={GRADER_NODES}" in prov["limit"]
    assert prov["path"] == STOCKFISH_PATH


def test_move_damage_pp():
    from engine import move_damage_pp
    # an even position where the move hands the opponent an even position: no damage
    assert move_damage_pp(0, 0) == pytest.approx(0)
    # perfect play preserves the eval: before +300 (my view) -> after -300 (their view)
    assert move_damage_pp(300, -300) == pytest.approx(0)
    # blundering mate away: before winning +773... e.g. Wj3vh-style collapse
    assert move_damage_pp(-773, MATE_SCORE_CP) == pytest.approx(3.0, abs=0.1)
    # from equal to mated: the full price of an equal-position blunder
    assert move_damage_pp(0, MATE_SCORE_CP) == pytest.approx(47.5, abs=0.1)
