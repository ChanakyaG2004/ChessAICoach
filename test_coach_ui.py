"""Smoke tests for the OpenCV operator dashboard."""

import unittest

import chess
import numpy as np

from coach_ui import BOARD_X, BOARD_Y, HEIGHT, WIDTH, render_coach_view


class CoachUiTests(unittest.TestCase):
    def render(self, warped, board=None, **overrides):
        values = dict(human_color="black", detection_status="Tracked board",
                      instruction="Make a move and clear your hand",
                      feedback="Opening principle: control the center",
                      engine_status="Stockfish ready", save_path="session.pgn")
        values.update(overrides)
        return render_coach_view(warped, board or chess.Board(), **values)

    def test_render_contains_camera_board_without_mutating_input(self):
        board_image = np.full((256, 256, 3), (20, 80, 180), np.uint8)
        original = board_image.copy()
        view = self.render(board_image)
        self.assertEqual(view.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(view.dtype, np.uint8)
        self.assertTrue(np.array_equal(board_image, original))
        self.assertTrue(np.array_equal(view[BOARD_Y + 300, BOARD_X + 300],
                                       np.array([20, 80, 180])))
        self.assertGreater(np.count_nonzero(view[:, 700:] != view[0, 0]), 1000)

    def test_missing_board_and_long_messages_render_safely(self):
        view = self.render(None, human_color="white",
                           detection_status="A long camera status " * 40,
                           instruction="Return the physical pieces to the last accepted position " * 30,
                           feedback="A detailed suggestion " * 40,
                           engine_status="Engine is processing " * 30,
                           save_path="/a/very/long/path/" * 30,
                           pending_action="Click four outer corners " * 30)
        self.assertEqual(view.shape, (HEIGHT, WIDTH, 3))
        self.assertGreater(np.count_nonzero(view), 0)

    def test_game_over_is_visually_distinct(self):
        board_image = np.zeros((128, 128, 3), np.uint8)
        ongoing = self.render(board_image)
        finished = chess.Board()
        for san in ("f3", "e5", "g4", "Qh4#"):
            finished.push_san(san)
        complete = self.render(board_image, finished)
        self.assertTrue(finished.is_checkmate())
        self.assertFalse(np.array_equal(ongoing[35:80, 700:1240],
                                        complete[35:80, 700:1240]))

    def test_rejects_invalid_orientation(self):
        with self.assertRaisesRegex(ValueError, "human_color"):
            self.render(None, human_color="red")


if __name__ == "__main__":
    unittest.main()
