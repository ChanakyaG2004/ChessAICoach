"""Plain-language move display and recovery input."""

import unittest

import chess

from move_language import move_text, parse_move_text


class MoveLanguageTests(unittest.TestCase):
    def test_piece_name_and_origin_are_explicit(self):
        board = chess.Board()
        move = chess.Move.from_uci("g1f3")
        self.assertEqual(move_text(board, move, include_color=True),
                         "White knight from g1 to f3")
        self.assertEqual(parse_move_text(board, "Knight to f3"), move)
        self.assertEqual(parse_move_text(board, "knight from g1 to f3"), move)
        self.assertEqual(parse_move_text(board, "g1 to f3"), move)

    def test_capture_castling_and_promotion(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("d7d5")
        self.assertEqual(move_text(board, chess.Move.from_uci("e4d5")),
                         "pawn from e4 captures the pawn on d5")
        castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertIn("king from e1 to g1", move_text(castle, parse_move_text(castle, "castle kingside")))
        promotion = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        move = parse_move_text(promotion, "pawn from a7 to a8 promote to queen")
        self.assertEqual(move.uci(), "a7a8q")

    def test_ambiguous_destination_requires_origin(self):
        board = chess.Board("4k3/8/8/8/8/8/3N1N2/4K3 w - - 0 1")
        with self.assertRaisesRegex(ValueError, "starting square"):
            parse_move_text(board, "knight to e4")


if __name__ == "__main__":
    unittest.main()
