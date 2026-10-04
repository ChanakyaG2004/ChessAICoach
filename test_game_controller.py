"""Game workflow tests without a camera or a Stockfish subprocess."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import chess
import numpy as np

from game_controller import GameController
from move_detector import MoveDetector
from session_store import load_game


class FakeCoach:
    def __init__(self):
        self.reviewed = []
        self.closed = False

    def suggest_move(self, board):
        move = chess.Move.from_uci("e2e4") if board.fullmove_number == 1 else next(iter(board.legal_moves))
        return SimpleNamespace(move=move, san=board.san(move))

    def review_move(self, board, move):
        self.reviewed.append((board.fen(), move.uci()))
        return SimpleNamespace(summary=f"{board.san(move)} reviewed")

    def close(self):
        self.closed = True


class ControllerTests(unittest.TestCase):
    def test_coach_move_human_feedback_and_saved_history(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "game.pgn"
            moves = MoveDetector(top_color="black")
            coach = FakeCoach()
            controller = GameController(moves, chess.STARTING_FEN, "black", path, coach)
            self.assertIn("e4", controller.instruction)
            self.assertEqual(load_game(path).board.fen(), chess.STARTING_FEN)

            white = chess.Move.from_uci("e2e4")
            moves.board.push(white)
            controller.on_accepted_move(white)
            self.assertIn("confirmed", controller.feedback)
            self.assertTrue(controller.human_turn)

            black = chess.Move.from_uci("e7e5")
            before = moves.board.fen()
            moves.board.push(black)
            controller.on_accepted_move(black)
            self.assertIn("e5 reviewed", controller.feedback)
            self.assertEqual(coach.reviewed, [(before, "e7e5")])
            self.assertEqual(load_game(path).board.fen(), moves.board.fen())

            image = np.zeros((512, 512, 3), np.uint8)
            controller.undo(image)
            self.assertEqual(len(moves.board.move_stack), 1)
            self.assertEqual(len(load_game(path).board.move_stack), 1)
            controller.resync(image)
            self.assertEqual(len(moves.board.move_stack), 1)
            moves.request_check()
            controller.new_game(image, Path(directory) / "new.pgn")
            self.assertFalse(moves.check_pending)
            self.assertIn("Check my move", controller.instruction)
            self.assertEqual(moves.board.fen(), chess.STARTING_FEN)
            self.assertEqual(load_game(Path(directory) / "new.pgn").board.fen(), chess.STARTING_FEN)
            controller.close()
            self.assertTrue(coach.closed)

    def test_manual_legal_move_recovers_a_missed_camera_match(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manual.pgn"
            moves = MoveDetector(top_color="white")
            controller = GameController(moves, chess.STARTING_FEN, "white", path)
            image = np.full((512, 512, 3), 120, np.uint8)
            controller.record_manual_move(chess.Move.from_uci("e2e4"), image)
            self.assertEqual(moves.board.peek().uci(), "e2e4")
            self.assertEqual(load_game(path).board.fen(), moves.board.fen())
            self.assertIn("Manually confirmed", controller.feedback)
            with self.assertRaisesRegex(ValueError, "not legal"):
                controller.record_manual_move(chess.Move.from_uci("e2e4"), image)


if __name__ == "__main__":
    unittest.main()
