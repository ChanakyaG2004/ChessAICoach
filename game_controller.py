"""Connect accepted camera moves to coaching, recovery, and saved game state."""

from pathlib import Path

import chess

from coach import CoachError, EngineUnavailable
from move_detector import MoveDetector
from move_language import move_text
from session_store import save_game


class GameController:
    def __init__(self, moves: MoveDetector, initial_fen: str, human_color: str,
                 save_path: str | Path, coach=None):
        if human_color not in ("white", "black"):
            raise ValueError("human_color must be white or black")
        self.moves = moves
        self.initial_fen = initial_fen
        self.human_color = human_color
        self.save_path = Path(save_path)
        self.coach = coach
        self.engine_status = "Local Stockfish ready" if coach else "Engine unavailable; move tracking remains active"
        self.storage_status = ""
        self.feedback = ""
        self.instruction = ""
        self.suggestion = None
        self._save()
        self.refresh_instruction()

    @property
    def human_turn(self) -> bool:
        return self.moves.board.turn == (self.human_color == "white")

    def _save(self) -> None:
        try:
            save_game(self.save_path, self.initial_fen, self.moves.board, self.human_color)
            self.storage_status = f"Saved: {self.save_path}"
        except (OSError, ValueError) as exc:
            self.storage_status = f"SAVE FAILED: {exc}"
            print(self.storage_status, flush=True)

    def save_now(self) -> None:
        """Retry the current PGN write, including after a transient disk error."""
        self._save()

    def _coach_failed(self, exc: Exception) -> None:
        self.engine_status = f"Stockfish unavailable: {exc}"
        print(self.engine_status, flush=True)
        if self.coach is not None:
            try:
                self.coach.close()
            except Exception:
                pass
        self.coach = None
        self.suggestion = None

    def refresh_instruction(self) -> None:
        board = self.moves.board
        outcome = board.outcome(claim_draw=False)
        if outcome:
            self.suggestion = None
            self.instruction = f"Game over: {outcome.result()} ({board.outcome().termination.name.lower()})."
            return
        side = "White" if board.turn else "Black"
        if self.human_turn:
            self.suggestion = None
            self.instruction = f"Your turn ({side}). Make one legal move, clear your hand, then press Check my move."
            return
        if self.coach is None:
            self.suggestion = None
            self.instruction = f"{side} to move. Stockfish is unavailable; play that side, clear your hand, then press Check my move."
            return
        try:
            self.suggestion = self.coach.suggest_move(board)
        except (CoachError, EngineUnavailable, OSError, TimeoutError) as exc:
            self._coach_failed(exc)
            self.instruction = f"{side} to move. Engine unavailable; play that side, clear your hand, then press Check my move."
            return
        if self.suggestion is None:
            self.instruction = f"{side} to move."
            return
        move = self.suggestion.move
        self.instruction = (f"Coach's move: {move_text(board, move, include_color=True)}. "
                            "Play it, clear your hand, then press Check my move.")
        print(self.instruction, flush=True)

    def on_accepted_move(self, move: chess.Move) -> None:
        """Handle the one move just committed by MoveDetector."""
        board_after = self.moves.board
        if not board_after.move_stack or board_after.peek() != move:
            raise ValueError("Accepted move does not match recorded game history")
        board_before = board_after.copy(stack=True)
        board_before.pop()
        plain = move_text(board_before, move, include_color=True)
        self._save()
        if board_before.turn == (self.human_color == "white"):
            if self.coach is not None:
                try:
                    review = self.coach.review_move(board_before, move)
                    self.feedback = review.summary
                except (CoachError, EngineUnavailable, OSError, TimeoutError) as exc:
                    self._coach_failed(exc)
                    self.feedback = f"Your {plain} was recorded; engine feedback unavailable."
            else:
                self.feedback = f"Your {plain} was recorded."
        elif self.suggestion is not None and move != self.suggestion.move:
            self.feedback = (f"Played {plain}, different from the suggested "
                             f"{move_text(board_before, self.suggestion.move, include_color=True)}. "
                             "Following the physical board.")
        elif self.suggestion is not None:
            self.feedback = f"Coach move {plain} confirmed on the physical board."
        else:
            self.feedback = f"{plain.capitalize()} recorded for the other side; no engine move was suggested."
        print(self.feedback, flush=True)
        self.refresh_instruction()

    def record_manual_move(self, move: chess.Move, warped) -> None:
        """Confirm a legal physical move that camera evidence could not isolate."""
        board = self.moves.board
        if move not in board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in the recorded position")
        plain = move_text(board, move)
        board.push(move)
        self.moves.rebaseline(warped)
        self.moves.last_move = f"{plain} manually confirmed"
        self.on_accepted_move(move)
        self.feedback = "Manually confirmed from the physical board. " + self.feedback

    def resync(self, warped) -> None:
        self.moves.rebaseline(warped)
        self.feedback = "Image baseline refreshed for the confirmed position."
        self.refresh_instruction()

    def undo(self, warped) -> None:
        self.moves.undo_and_rebaseline(warped)
        self._save()
        self.feedback = "Last accepted move undone; image baseline refreshed."
        self.refresh_instruction()

    def new_game(self, warped, save_path: str | Path) -> None:
        self.moves.rebaseline(warped, chess.STARTING_FEN)
        self.initial_fen = chess.STARTING_FEN
        self.save_path = Path(save_path)
        self._save()
        self.feedback = "New game started from the confirmed standard position."
        self.refresh_instruction()

    def close(self) -> None:
        if self.coach is not None:
            self.coach.close()
            self.coach = None
