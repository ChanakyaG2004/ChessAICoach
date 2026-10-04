"""Atomic PGN storage for a camera-tracked chess game.

The custom PGN headers record the physical setup and its last accepted position.
On load, every move is replayed legally and checked against that position, so a
truncated or edited file cannot silently resume from the wrong board state.
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import tempfile

import chess
import chess.pgn


@dataclass
class SavedGame:
    initial_fen: str
    board: chess.Board
    human_color: str


def _starting_board(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid initial FEN: {exc}") from exc
    if not board.is_valid():
        raise ValueError("Invalid initial FEN: position is not legal")
    return board


def _check_color(human_color: str) -> None:
    if human_color not in ("white", "black"):
        raise ValueError("Human color must be 'white' or 'black'")


def save_game(path: str | Path, initial_fen: str, board: chess.Board,
              human_color: str) -> None:
    """Write all accepted moves as one PGN, replacing the file atomically."""
    _check_color(human_color)
    initial = _starting_board(initial_fen)
    if board.root().fen() != initial.fen():
        raise ValueError("Board history does not start from the initial FEN")

    replay = initial.copy()
    game = chess.pgn.Game()
    game.setup(initial)
    game.headers["Event"] = "Physical Chess Coach"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Human" if human_color == "white" else "Coach"
    game.headers["Black"] = "Human" if human_color == "black" else "Coach"
    game.headers["HumanColor"] = human_color
    game.headers["InitialFEN"] = initial.fen()
    node = game
    for move in board.move_stack:
        if move not in replay.legal_moves:
            raise ValueError(f"Board history contains an illegal move: {move.uci()}")
        node = node.add_variation(move)
        replay.push(move)
    if replay.fen() != board.fen():
        raise ValueError("Board position does not match its move history")
    game.headers["CurrentFEN"] = replay.fen()
    outcome = replay.outcome(claim_draw=False)
    game.headers["Result"] = outcome.result() if outcome else "*"

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            print(game, file=handle, end="\n\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_game(path: str | Path) -> SavedGame:
    """Load one saved game and reject missing, damaged, or inconsistent history."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
            if game is None:
                raise ValueError("Invalid saved game: PGN is empty")
            if game.errors:
                raise ValueError(f"Invalid saved game: {game.errors[0]}")
            if chess.pgn.read_game(handle) is not None:
                raise ValueError("Invalid saved game: expected exactly one PGN game")
    except UnicodeError as exc:
        raise ValueError("Invalid saved game: PGN is not UTF-8 text") from exc

    initial_fen = game.headers.get("InitialFEN")
    current_fen = game.headers.get("CurrentFEN")
    human_color = game.headers.get("HumanColor")
    if not initial_fen or not current_fen or not human_color:
        raise ValueError("Invalid saved game: missing InitialFEN, CurrentFEN, or HumanColor header")
    _check_color(human_color)
    initial = _starting_board(initial_fen)
    if game.board().fen() != initial.fen():
        raise ValueError("Invalid saved game: PGN setup differs from InitialFEN")

    board = initial.copy()
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise ValueError(f"Invalid saved game: illegal move {move.uci()}")
        board.push(move)
    if board.fen() != current_fen:
        raise ValueError("Invalid saved game: move history differs from CurrentFEN")
    outcome = board.outcome(claim_draw=False)
    result = outcome.result() if outcome else "*"
    if game.headers.get("Result") != result:
        raise ValueError("Invalid saved game: result differs from reconstructed position")
    return SavedGame(initial.fen(), board, human_color)
