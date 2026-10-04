"""Plain-language move names for the UI; chess notation remains internal."""

import re

import chess


PIECE_TYPES = {chess.piece_name(kind): kind for kind in chess.PIECE_TYPES}


def move_text(board: chess.Board, move: chess.Move, *, include_color=False) -> str:
    """Describe a legal move using the piece, origin, and destination."""
    if move not in board.legal_moves:
        raise ValueError(f"Move {move.uci()} is not legal in this position")
    piece = board.piece_at(move.from_square)
    side = ("White " if piece.color else "Black ") if include_color else ""
    start = chess.square_name(move.from_square)
    end = chess.square_name(move.to_square)
    if board.is_castling(move):
        kingside = board.is_kingside_castling(move)
        rook_start = ("h" if kingside else "a") + ("1" if piece.color else "8")
        rook_end = ("f" if kingside else "d") + ("1" if piece.color else "8")
        flank = "kingside" if kingside else "queenside"
        verb = "castles" if include_color else "castle"
        return f"{side}{verb} {flank}: king from {start} to {end}, rook from {rook_start} to {rook_end}"
    name = chess.piece_name(piece.piece_type)
    if board.is_capture(move):
        captured = ("pawn" if board.is_en_passant(move) else
                    chess.piece_name(board.piece_at(move.to_square).piece_type))
        result = f"{side}{name} from {start} captures the {captured} on {end}"
    else:
        result = f"{side}{name} from {start} to {end}"
    if move.promotion:
        result += f" and promotes to a {chess.piece_name(move.promotion)}"
    return result


def parse_move_text(board: chess.Board, text: str) -> chess.Move:
    """Accept notation or a simple spoken-style move for manual recovery."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a move, such as 'knight from g1 to f3'")
    phrase = text.strip().lower().rstrip(".!?")
    try:
        return board.parse_uci(phrase)
    except ValueError:
        pass
    try:
        return board.parse_san(text.strip())
    except ValueError:
        pass
    if phrase in ("castle kingside", "castle queenside"):
        kingside = phrase.endswith("kingside")
        matches = [move for move in board.legal_moves
                   if (board.is_kingside_castling(move) if kingside
                       else board.is_queenside_castling(move))]
    else:
        square_pair = re.fullmatch(r"([a-h][1-8])\s+(?:to|takes|captures|->|→)\s+([a-h][1-8])", phrase)
        named = re.fullmatch(
            r"(?:(?:white|black|my|the)\s+)?"
            r"(king|queen|rook|bishop|knight|pawn)\s+"
            r"(?:from\s+([a-h][1-8])\s+)?"
            r"(?:to|on|takes|captures)\s+([a-h][1-8])"
            r"(?:\s+(?:and\s+)?promot(?:e|es|ing)\s+to\s+(?:a\s+)?(queen|rook|bishop|knight))?",
            phrase)
        if square_pair:
            origin, destination = square_pair.groups()
            matches = [move for move in board.legal_moves
                       if chess.square_name(move.from_square) == origin
                       and chess.square_name(move.to_square) == destination]
        elif named:
            piece_name, origin, destination, promotion = named.groups()
            matches = [move for move in board.legal_moves
                       if board.piece_at(move.from_square).piece_type == PIECE_TYPES[piece_name]
                       and (origin is None or chess.square_name(move.from_square) == origin)
                       and chess.square_name(move.to_square) == destination
                       and (promotion is None or move.promotion == PIECE_TYPES[promotion])]
        else:
            raise ValueError("Use a move such as 'knight from g1 to f3' or 'pawn to e5'")
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("That move is not legal in the current position")
    if len({(move.from_square, move.to_square) for move in matches}) == 1:
        raise ValueError("Choose a promotion piece: queen, rook, bishop, or knight")
    raise ValueError("More than one piece can make that move; include its starting square")
