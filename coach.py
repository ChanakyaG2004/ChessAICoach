"""Optional, local Stockfish coaching for a position tracked by python-chess.

The camera and move detector remain usable when Stockfish is unavailable.
Only positions and legal moves supplied by the caller are sent to the local
engine; this module does not try to recognize pieces in a camera image.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil

import chess
import chess.engine

from move_language import move_text


class EngineUnavailable(RuntimeError):
    """No runnable Stockfish executable was found or it could not be started."""


class CoachError(RuntimeError):
    """Stockfish started but could not analyze the requested position."""


def find_stockfish(engine_path: str | os.PathLike | None = None) -> Path:
    """Locate Stockfish using an explicit path, environment, PATH, or macOS paths.

    An explicit path or STOCKFISH_EXECUTABLE takes priority. A bad override is
    reported rather than silently choosing a different engine.
    """
    override = engine_path or os.environ.get("STOCKFISH_EXECUTABLE")
    if override:
        name = os.fspath(override)
        located = shutil.which(name)
        candidate = Path(located) if located else Path(name).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        raise EngineUnavailable(f"Stockfish is not executable at {name!r}")

    candidates = (
        shutil.which("stockfish"),
        "/opt/homebrew/bin/stockfish",  # Apple silicon Homebrew
        "/usr/local/bin/stockfish",     # Intel Homebrew
        "/usr/bin/stockfish",
        "/Applications/Stockfish.app/Contents/MacOS/stockfish",
    )
    for name in candidates:
        if name:
            candidate = Path(name)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
    raise EngineUnavailable(
        "Stockfish was not found. Install it with `brew install stockfish` "
        "or pass --engine-path /path/to/stockfish."
    )


@dataclass(frozen=True)
class Evaluation:
    """An engine score from the perspective of one player.

    Exactly one of centipawns, mate_in, or terminal is populated. A positive
    score favors the requested player. Mate signs follow UCI convention.
    """

    centipawns: int | None = None
    mate_in: int | None = None
    terminal: str | None = None

    @property
    def text(self) -> str:
        if self.terminal == "win":
            return "checkmate"
        if self.terminal == "loss":
            return "checkmated"
        if self.terminal == "draw":
            return "draw"
        if self.mate_in is not None:
            return (f"mate in {self.mate_in}" if self.mate_in > 0
                    else f"opponent mates in {abs(self.mate_in)}")
        if self.centipawns is not None:
            return f"{self.centipawns / 100:+.2f} pawns"
        return "unknown"


@dataclass(frozen=True)
class MoveSuggestion:
    move: chess.Move
    san: str
    evaluation: Evaluation
    plain: str = ""

    @property
    def summary(self) -> str:
        return (f"Suggested move: {self.plain or self.san}; "
                f"evaluation for the side to move: {self.evaluation.text}.")


@dataclass(frozen=True)
class MoveReview:
    move: chess.Move
    san: str
    grade: str
    best_move: chess.Move | None
    best_san: str | None
    before: Evaluation
    after: Evaluation
    centipawn_loss: int | None
    notes: tuple[str, ...] = ()
    reply_move: chess.Move | None = None
    reply_san: str | None = None
    plain: str = ""
    best_plain: str | None = None
    reply_plain: str | None = None

    @property
    def summary(self) -> str:
        played = self.plain or self.san
        if self.grade == "Checkmate":
            return f"{played.capitalize()}: checkmate."
        if self.grade == "Best":
            sentence = f"{played.capitalize()}: Stockfish's top choice"
        else:
            sentence = f"{played.capitalize()}: {self.grade.lower()}"
            if self.centipawn_loss is not None and self.centipawn_loss > 0:
                sentence += f"; about {self.centipawn_loss} cp lost"
        if self.notes:
            sentence += f" ({'; '.join(self.notes)})"
        sentence += "."
        if self.best_san and self.best_move != self.move:
            sentence += f" Stockfish preferred {self.best_plain or self.best_san}."
        if self.reply_san:
            sentence += f" Next reply: {self.reply_plain or self.reply_san}."
        return sentence


def _factual_notes(board_before: chess.Board, move: chess.Move,
                   board_after: chess.Board) -> tuple[str, ...]:
    """Describe only facts visible in the known legal position and move."""
    piece = board_before.piece_at(move.from_square)
    notes = []
    if board_before.is_kingside_castling(move):
        notes.append("castles kingside")
    elif board_before.is_queenside_castling(move):
        notes.append("castles queenside")
    if board_before.is_en_passant(move):
        notes.append("captures en passant")
    elif board_before.is_capture(move):
        notes.append(f"captures on {chess.square_name(move.to_square)}")
    if move.promotion is not None:
        notes.append(f"promotes to {chess.piece_name(move.promotion)}")
    if board_after.is_check() and not board_after.is_checkmate():
        notes.append("gives check")
    if piece and piece.piece_type == chess.PAWN and move.to_square in (
            chess.D4, chess.E4, chess.D5, chess.E5):
        notes.append("pawn to center")
    home_squares = {
        chess.WHITE: {chess.B1, chess.G1, chess.C1, chess.F1},
        chess.BLACK: {chess.B8, chess.G8, chess.C8, chess.F8},
    }
    if (piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and move.from_square in home_squares[piece.color]
            and board_before.root().board_fen() == chess.STARTING_BOARD_FEN
            and not any(past.from_square == move.from_square
                        for past in board_before.move_stack)):
        notes.append(f"develops {chess.piece_name(piece.piece_type)}")
    return tuple(notes[:3])


def _evaluation(info: chess.engine.InfoDict, color: chess.Color) -> Evaluation:
    score = info.get("score")
    if not isinstance(score, chess.engine.PovScore):
        raise CoachError("Stockfish returned no usable position score")
    pov = score.pov(color)
    mate = pov.mate()
    if mate is not None:
        return Evaluation(mate_in=mate)
    cp = pov.score()
    if cp is None:
        raise CoachError("Stockfish returned no usable position score")
    return Evaluation(centipawns=cp)


def _terminal_evaluation(board: chess.Board, color: chess.Color) -> Evaluation | None:
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return Evaluation(terminal="draw")
    return Evaluation(terminal="win" if outcome.winner == color else "loss")


def _loss_and_grade(before: Evaluation, after: Evaluation,
                    played: chess.Move, best: chess.Move | None) -> tuple[int | None, str]:
    if after.terminal == "win":
        return 0, "Checkmate"
    if played == best:
        # Scores from two short searches can differ even for the top move.
        return 0, "Best"
    if before.mate_in is not None or after.mate_in is not None or after.terminal == "loss":
        if before.mate_in is not None and before.mate_in > 0:
            if after.mate_in is None or after.mate_in <= 0:
                return None, "Missed forced mate"
            return None, "Good" if after.mate_in <= before.mate_in else "Inaccuracy"
        if after.terminal == "loss":
            return None, "Blunder"
        if after.mate_in is not None and after.mate_in < 0:
            if before.mate_in is None or before.mate_in >= 0:
                return None, "Blunder"
            return None, "Mistake" if after.mate_in > before.mate_in else "Good"
        return None, "Good"
    after_cp = 0 if after.terminal == "draw" else after.centipawns
    if before.centipawns is None or after_cp is None:
        return None, "Unrated"
    loss = max(0, before.centipawns - after_cp)
    if loss < 35:
        grade = "Good"
    elif loss < 100:
        grade = "Inaccuracy"
    elif loss < 250:
        grade = "Mistake"
    else:
        grade = "Blunder"
    return loss, grade


class ChessCoach:
    """Manage one local UCI engine and review accepted legal moves.

    Use ``with ChessCoach(...) as coach`` or call ``close()`` on shutdown.
    ``board_before`` is the position immediately before ``move``; neither
    public analysis method changes a caller-owned board.
    """

    def __init__(self, engine_path: str | os.PathLike | None = None,
                 analysis_time: float = 0.2, *, engine=None):
        if not math.isfinite(analysis_time) or analysis_time <= 0:
            raise ValueError("analysis_time must be a positive finite number of seconds")
        self.analysis_time = analysis_time
        self.engine_path: Path | None = None
        if engine is not None:
            self.engine = engine
        else:
            self.engine_path = find_stockfish(engine_path)
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path), timeout=30)
            except (OSError, TimeoutError, chess.engine.EngineError) as exc:
                detail = str(exc) or type(exc).__name__
                raise EngineUnavailable(f"Could not start Stockfish at {self.engine_path}: {detail}") from exc
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.engine.quit()
            except (OSError, chess.engine.EngineError):
                # A terminated engine has no remaining process to close.
                pass

    def _analyze(self, board: chess.Board) -> chess.engine.InfoDict:
        if self._closed:
            raise CoachError("Stockfish has already been closed")
        try:
            return self.engine.analyse(board, chess.engine.Limit(time=self.analysis_time))
        except (OSError, TimeoutError, chess.engine.EngineError) as exc:
            raise CoachError(f"Stockfish analysis failed: {exc}") from exc

    @staticmethod
    def _best_move(board: chess.Board, info: chess.engine.InfoDict) -> chess.Move | None:
        pv = info.get("pv") or []
        move = pv[0] if pv else None
        return move if move is not None and move in board.legal_moves else None

    def suggest_move(self, board: chess.Board) -> MoveSuggestion | None:
        """Suggest a legal move for the side to move, or None for a finished game."""
        if board.is_game_over(claim_draw=False):
            return None
        info = self._analyze(board)
        move = self._best_move(board, info)
        if move is None:
            raise CoachError("Stockfish returned no legal suggested move")
        return MoveSuggestion(move, board.san(move), _evaluation(info, board.turn),
                              move_text(board, move))

    def review_move(self, board_before: chess.Board, move: chess.Move) -> MoveReview:
        """Compare a played legal move with Stockfish's choice for that position."""
        if move not in board_before.legal_moves:
            raise ValueError(f"Move {move.uci()} is not legal in the supplied position")
        mover = board_before.turn
        san = board_before.san(move)
        info = self._analyze(board_before)
        before = _evaluation(info, mover)
        best = self._best_move(board_before, info)
        best_san = board_before.san(best) if best else None
        board_after = board_before.copy()
        board_after.push(move)
        notes = _factual_notes(board_before, move, board_after)
        after = _terminal_evaluation(board_after, mover)
        reply = None
        reply_san = None
        if after is None:
            after_info = self._analyze(board_after)
            after = _evaluation(after_info, mover)
            reply = self._best_move(board_after, after_info)
            if reply is not None:
                reply_san = board_after.san(reply)
        loss, grade = _loss_and_grade(before, after, move, best)
        return MoveReview(move, san, grade, best, best_san, before, after,
                          loss, notes, reply, reply_san,
                          move_text(board_before, move),
                          move_text(board_before, best) if best else None,
                          move_text(board_after, reply) if reply else None)
