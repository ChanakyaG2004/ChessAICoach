"""Round-trip and corruption checks for persisted physical chess games."""

from pathlib import Path
import tempfile
import unittest

import chess

from session_store import load_game, save_game


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "nested" / "game.pgn"

    def test_standard_game_round_trip_with_full_move_stack(self):
        board = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
            board.push_san(san)
        save_game(self.path, chess.STARTING_FEN, board, "black")

        saved = load_game(self.path)
        self.assertEqual(saved.initial_fen, chess.STARTING_FEN)
        self.assertEqual(saved.human_color, "black")
        self.assertEqual(saved.board.fen(), board.fen())
        self.assertEqual(saved.board.move_stack, board.move_stack)
        self.assertIn("[CurrentFEN", self.path.read_text())

    def test_custom_fen_round_trip_with_black_to_move(self):
        initial_fen = "4k3/8/8/8/8/8/4p3/K7 b - - 0 1"
        board = chess.Board(initial_fen)
        board.push_san("e1=Q+")
        save_game(self.path, initial_fen, board, "white")

        saved = load_game(self.path)
        self.assertEqual(saved.initial_fen, initial_fen)
        self.assertEqual(saved.board.fen(), board.fen())
        self.assertEqual(saved.board.move_stack, board.move_stack)
        self.assertEqual(saved.human_color, "white")
        self.assertIn("[SetUp \"1\"]", self.path.read_text())

    def test_invalid_inputs_do_not_replace_existing_file(self):
        board = chess.Board()
        save_game(self.path, chess.STARTING_FEN, board, "white")
        original = self.path.read_bytes()
        board.push_san("e4")
        with self.assertRaisesRegex(ValueError, "Human color"):
            save_game(self.path, chess.STARTING_FEN, board, "green")
        with self.assertRaisesRegex(ValueError, "does not start"):
            save_game(self.path, "4k3/8/8/8/8/8/8/4K3 w - - 0 1", board, "white")
        self.assertEqual(self.path.read_bytes(), original)

    def test_missing_and_corrupt_pgn_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_game(self.path)
        self.path.parent.mkdir(parents=True)
        self.path.write_text("")
        with self.assertRaisesRegex(ValueError, "empty"):
            load_game(self.path)

        board = chess.Board()
        board.push_san("e4")
        save_game(self.path, chess.STARTING_FEN, board, "white")
        valid = self.path.read_text()
        self.path.write_text(valid.replace("e4", "d4"))
        with self.assertRaisesRegex(ValueError, "CurrentFEN"):
            load_game(self.path)
        self.path.write_text(valid.replace('[HumanColor "white"]', '[HumanColor "green"]'))
        with self.assertRaisesRegex(ValueError, "Human color"):
            load_game(self.path)
        self.path.write_text(valid + valid)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_game(self.path)


if __name__ == "__main__":
    unittest.main()
