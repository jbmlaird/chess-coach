import chess

VALUES = {
    chess.PAWN: 1,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

RAY_PIECE_TYPES = {chess.BISHOP, chess.ROOK, chess.QUEEN}


def victim_value(board, move):
    if board.is_en_passant(move):
        return VALUES[chess.PAWN]
    return VALUES[board.piece_at(move.to_square).piece_type]


def is_defended(board, color, square):
    """Mirrors lichess-puzzler util.is_defended: any friendly attacker of the
    square counts, and so does one revealed by lifting an enemy ray piece off
    the line (battery defense through the attacker). A defense line blocked by
    an enemy pawn or knight does NOT count - those attackers don't clear the
    line, so the defense only appears after the capture is already made."""
    if board.attackers(color, square):
        return True
    for attacker in board.attackers(not color, square):
        if board.piece_at(attacker).piece_type in RAY_PIECE_TYPES:
            lifted = board.copy(stack=False)
            lifted.remove_piece_at(attacker)
            if lifted.attackers(color, square):
                return True
    return False


def material_diff(board, color):
    return sum(
        value * (len(board.pieces(piece_type, color)) - len(board.pieces(piece_type, not color)))
        for piece_type, value in VALUES.items()
    )


def hanging_piece(fen, moves):
    """Mirrors lichess-puzzler cook.hanging_piece, with two deliberate
    divergences: pawn victims count (Lichess excludes them by curation, which
    also hides every en passant hang since an ep victim is always a pawn), and
    there is no in-check exclusion.

    The solution's first move must capture a piece that:
    - is undefended in the is_defended sense, judged BEFORE the capture. For
      en passant this is the landing square, where any recapture would happen;
    - is not a recapture completing an even-or-better trade for the opponent
      (same square, and the setup move's victim was worth at least as much);
    - stays won: the solver's material edge after their second move is no
      worse than after their first (lines under 4 plies pass automatically).

    This is driven by geometry not by an engine verdict.
    """
    if len(moves) < 2:
        raise ValueError(f"need a setup move and a solution reply, got {moves!r}")
    chessboard = chess.Board(fen)
    first_move = chess.Move.from_uci(moves[0])
    setup_victim = victim_value(chessboard, first_move) if chessboard.is_capture(first_move) else None
    chessboard.push_uci(moves[0])

    second_move = chess.Move.from_uci(moves[1])
    if not chessboard.is_legal(second_move):
        raise ValueError(f"move {moves[1]!r} is not legal after {moves[0]!r} in {fen!r}")
    if not chessboard.is_capture(second_move):
        return False

    solver = chessboard.turn
    if is_defended(chessboard, not solver, second_move.to_square):
        return False

    if (setup_victim is not None
            and second_move.to_square == first_move.to_square
            and setup_victim >= victim_value(chessboard, second_move)):
        return False

    if len(moves) < 4:
        return True
    chessboard.push(second_move)
    won = material_diff(chessboard, solver)
    chessboard.push_uci(moves[2])
    chessboard.push_uci(moves[3])
    return material_diff(chessboard, solver) >= won