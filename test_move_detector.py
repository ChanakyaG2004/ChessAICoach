"""Camera-free regression tests: python -m unittest -v test_move_detector."""

import contextlib
import io
from pathlib import Path
import unittest

import chess
import cv2
import numpy as np

from move_detector import (MoveDetector, Orientation, align_image, changed_squares,
                           match_legal_moves, square_scores)
from perspective import rotate_board


def position(moves=(), fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    return board


def render(board, top_color="black"):
    """Draw a known test fixture; this is not a recognition algorithm."""
    orientation = Orientation(top_color)
    image = np.empty((512, 512, 3), np.uint8)
    for row in range(8):
        for col in range(8):
            image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = (
                (140, 165, 180) if (row + col) % 2 else (210, 220, 230))
            piece = board.piece_at(orientation.square(row, col))
            if piece:
                center = (col * 64 + 32, row * 64 + 32)
                cv2.circle(image, center, 20, (25, 25, 25), -1)
                color = (245, 245, 245) if piece.color else (40, 40, 40)
                cv2.circle(image, center, 17, color, -1)
                cv2.line(image, (center[0] - 8, center[1]),
                         (center[0] + 8, center[1]), (100, 100, 100), 2)
    return image


def render_with_shadows(board, top_color="black", strength=.45):
    image = render(board, top_color)
    mask = np.zeros((512, 512), np.float32)
    orientation = Orientation(top_color)
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.KNIGHT:
            continue
        row, col = orientation.cell(square)
        cv2.ellipse(mask, (col * 64 + 4, row * 64 + 34), (32, 20),
                    0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 4)
    return np.clip(image.astype(np.float32) * (1 - strength * mask[..., None]),
                   0, 255).astype(np.uint8)


def render_marked_white_pieces(board, top_color="black"):
    """White pieces blend into white squares except for small dark top marks."""
    orientation = Orientation(top_color)
    image = np.empty((512, 512, 3), np.uint8)
    for row in range(8):
        for col in range(8):
            image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = (
                (35, 35, 35) if (row + col) % 2 else (250, 250, 250))
            piece = board.piece_at(orientation.square(row, col))
            if piece:
                center = (col * 64 + 33, row * 64 + 32)
                cv2.circle(image, center, 16, (250, 250, 250) if piece.color else (12, 12, 12), -1)
                if piece.color:
                    cv2.circle(image, (center[0] + 3, center[1]), 4, (35, 35, 35), -1)
    return image


class MoveTests(unittest.TestCase):
    def feed(self, detector, image, start, frames=12, reliable=True, check=True):
        accepted = []
        with contextlib.redirect_stdout(io.StringIO()):
            if check and detector.baseline is not None and not detector.check_pending:
                detector.request_check(now=start)
            for i in range(frames):
                move = detector.update(image, reliable=reliable, now=start + i * .2)
                if move:
                    accepted.append(move.uci())
        return accepted

    def scenarios(self):
        return [
            (position(), "e2e4", {"e2", "e4"}),
            (position(["e2e4", "d7d5"]), "e4d5", {"e4", "d5"}),
            (position(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
             "e1g1", {"e1", "f1", "g1", "h1"}),
            (position(fen="r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
             "e8c8", {"a8", "c8", "d8", "e8"}),
            (position(["e2e4", "a7a6", "e4e5", "d7d5"]),
             "e5d6", {"e5", "d5", "d6"}),
        ]

    def test_orientation(self):
        for color in ("white", "black"):
            orientation = Orientation(color)
            self.assertEqual(len({orientation.square(r, c) for r in range(8) for c in range(8)}), 64)
            for square in chess.SQUARES:
                self.assertEqual(orientation.square(*orientation.cell(square)), square)
        self.assertEqual(Orientation("white").square(0, 0), chess.H1)
        self.assertEqual(Orientation("white").square(7, 7), chess.A8)
        self.assertEqual(Orientation("black").square(0, 0), chess.A8)
        self.assertEqual(Orientation("black").square(7, 7), chess.H1)

    def test_legal_footprints_and_ranking(self):
        for board, uci, names in self.scenarios():
            with self.subTest(move=uci):
                move = chess.Move.from_uci(uci)
                expected = {chess.parse_square(name) for name in names}
                self.assertEqual(changed_squares(board, move), expected)
                scores = np.full(64, .003, np.float32)
                scores[list(expected)] = .3
                result = match_legal_moves(board, scores)
                self.assertEqual(result.move, move, result.reason)
                self.assertGreater(result.confidence, .9)

    def test_visual_pipeline_both_orientations(self):
        for color in ("white", "black"):
            for board, uci, _ in self.scenarios():
                with self.subTest(color=color, move=uci):
                    detector = MoveDetector(board.fen(), color, stable_seconds=.6, confirm_seconds=.4)
                    self.feed(detector, render(board, color), 0)
                    after = board.copy()
                    after.push_uci(uci)
                    accepted = self.feed(detector, render(after, color), 3)
                    self.assertEqual(accepted, [uci], detector.status)
                    self.assertEqual(detector.board.fen(), after.fen())
                    self.assertEqual(len(detector.ranked_squares()), 64)
                    self.assertEqual(self.feed(detector, render(after, color), 6), [])

    def test_sideways_camera_rotation_keeps_square_orientation(self):
        board = position()
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)

        def from_side(image):
            sideways = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return rotate_board(sideways, "cw")

        self.feed(detector, from_side(render(board)), 0)
        board.push_uci("e2e4")
        self.assertEqual(self.feed(detector, from_side(render(board)), 3), ["e2e4"])
        self.assertEqual(detector.board.fen(), board.fen())

    def test_small_camera_and_lighting_changes(self):
        before = render(position())
        shifted = cv2.warpAffine(before, np.float32([[1, 0, 2], [0, 1, -2]]),
                                (512, 512), borderMode=cv2.BORDER_REFLECT_101)
        shifted = np.clip(shifted.astype(np.float32) * 0.94 + 8, 0, 255).astype(np.uint8)
        aligned, okay = align_image(before, shifted)
        self.assertTrue(okay)
        scores = square_scores(before, aligned, Orientation("black"))
        self.assertLess(float(scores.max()), .035)
        self.assertIsNone(match_legal_moves(position(), scores).move)

    def test_move_with_camera_and_lighting_change(self):
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        board = position()
        self.feed(detector, render(board), 0)
        board.push_uci("e2e4")
        after = cv2.warpAffine(render(board), np.float32([[1, 0, 2], [0, 1, 1]]),
                              (512, 512), borderMode=cv2.BORDER_REFLECT_101)
        after = np.clip(after.astype(np.float32) * .94 + 8, 0, 255).astype(np.uint8)
        self.assertEqual(self.feed(detector, after, 3), ["e2e4"], detector.status)

    def test_knight_moves_with_shadows_on_neighboring_squares(self):
        board = position(["e2e4", "e7e5", "g1f3"])
        for color in ("black", "white"):
            for uci in ("b8c6", "g8f6"):
                with self.subTest(color=color, move=uci):
                    detector = MoveDetector(board.fen(), color, stable_seconds=.6,
                                            confirm_seconds=.4)
                    self.feed(detector, render_with_shadows(board, color), 0)
                    after = board.copy()
                    after.push_uci(uci)
                    self.assertEqual(self.feed(detector, render_with_shadows(after, color), 3),
                                     [uci], detector.status)

    def test_changed_shadows_and_two_piece_moves_do_not_advance(self):
        board = position(["e2e4", "e7e5", "g1f3"])
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render_with_shadows(board), 0)
        self.assertEqual(self.feed(detector, render_with_shadows(board, strength=.65), 3), [])
        after = board.copy()
        after.push_uci("g8f6")
        after.set_piece_at(chess.A6, after.remove_piece_at(chess.A7))
        self.assertEqual(self.feed(detector, render_with_shadows(after), 6), [])
        self.assertEqual(detector.board.fen(), board.fen())

    def test_small_top_marks_on_white_pieces_are_not_diluted(self):
        for color in ("black", "white"):
            for uci in ("e2e4", "b1c3"):
                with self.subTest(color=color, move=uci):
                    before = position()
                    detector = MoveDetector(top_color=color, stable_seconds=.6, confirm_seconds=.4)
                    self.feed(detector, render_marked_white_pieces(before, color), 0)
                    after = before.copy()
                    after.push_uci(uci)
                    self.assertEqual(self.feed(detector, render_marked_white_pieces(after, color), 3), [uci], detector.status)
                    self.assertEqual(self.feed(detector, render_marked_white_pieces(after, color), 6), [])

    def test_marked_pieces_still_require_one_complete_legal_move(self):
        before = position()
        for target in (None, chess.E5):
            detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
            self.feed(detector, render_marked_white_pieces(before), 0)
            after = before.copy()
            pawn = after.remove_piece_at(chess.E2)
            if target is not None:
                after.set_piece_at(target, pawn)
            self.assertEqual(self.feed(detector, render_marked_white_pieces(after), 3), [])
            self.assertEqual(detector.board.fen(), before.fen())
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render_marked_white_pieces(before), 0)
        after = before.copy()
        after.push_uci("e2e4")
        after.set_piece_at(chess.D4, after.remove_piece_at(chess.D2))
        self.assertEqual(self.feed(detector, render_marked_white_pieces(after), 3), [])
        self.assertEqual(detector.board.fen(), before.fen())

    def real_black_pawn_fixture(self):
        folder = Path(__file__).resolve().parent / "test_fixtures"
        before = cv2.imread(str(folder / "black_pawn_before.png"))
        after = cv2.imread(str(folder / "black_pawn_after.png"))
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        return position(["e2e4"]), before, after

    def test_real_black_pawn_on_dark_destination_is_recorded_once(self):
        board, before, after = self.real_black_pawn_fixture()
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, before, 0)
        self.assertEqual(self.feed(detector, after, 3), ["e7e5"], detector.status)
        board.push_uci("e7e5")
        self.assertEqual(detector.board.fen(), board.fen())
        self.assertEqual(self.feed(detector, after, 6), [])

    def test_real_dark_pawn_fixture_rejects_lifted_piece_and_no_change(self):
        board, before, after = self.real_black_pawn_fixture()
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, before, 0)
        self.assertEqual(self.feed(detector, before, 3), [])
        # Keep the observed emptied e7, but restore e5's empty square: a pawn
        # lifted without a visible destination must never advance the game.
        lifted = after.copy()
        lifted[192:256, 256:320] = before[192:256, 256:320]
        self.assertEqual(self.feed(detector, lifted, 6), [])
        self.assertEqual(detector.board.fen(), board.fen())
        self.assertEqual(self.feed(detector, after, 9), ["e7e5"], detector.status)

    def test_weak_extra_and_incomplete_changes_rejected(self):
        for changes in ({"e2": .3}, {"e2": .3, "e4": .3, "a6": .3},
                        {"e2": .035, "e4": .035}, {"e2": .3, "e4": .3, "d2": .3, "d4": .3}):
            scores = np.zeros(64, np.float32)
            for name, score in changes.items():
                scores[chess.parse_square(name)] = score
            self.assertIsNone(match_legal_moves(position(), scores).move)

    def test_clear_move_with_background_camera_noise(self):
        # A real camera can show small changes across all 64 squares even
        # when only the source and destination contain a moved piece.
        scores = np.linspace(.01, .06, 64, dtype=np.float32)
        scores[[chess.D2, chess.D4]] = [.50, .47]
        result = match_legal_moves(position(), scores)
        self.assertEqual(result.move, chess.Move.from_uci("d2d4"), result.reason)
        self.assertGreater(result.confidence, .9)

    def test_clear_move_with_one_weak_stray_square(self):
        board = position(["e2e4"])
        scores = np.full(64, .025, np.float32)
        scores[[chess.E7, chess.E5]] = [.204, .202]
        scores[chess.E8] = .094
        result = match_legal_moves(board, scores)
        self.assertEqual(result.move, chess.Move.from_uci("e7e5"), result.reason)

        scores[chess.E8] = .15
        self.assertIsNone(match_legal_moves(board, scores).move)

    def test_out_of_turn_move_is_explained_and_rejected(self):
        scores = np.full(64, .03, np.float32)
        scores[[chess.E7, chess.E5]] = [.29, .28]
        result = match_legal_moves(position(), scores)
        self.assertIsNone(result.move)
        self.assertIn("Out of turn: e5 (e7e5)", result.reason)
        self.assertIn("White to move", result.reason)

        impossible = np.full(64, .03, np.float32)
        impossible[[chess.E2, chess.E5]] = .3
        result = match_legal_moves(position(), impossible)
        self.assertIsNone(result.move)
        self.assertNotIn("Out of turn", result.reason)

    def test_promotion_requires_explicit_choice(self):
        board = position(fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        after = board.copy()
        after.push_uci("a7a8q")
        self.assertEqual(self.feed(detector, render(after), 3), [])
        self.assertEqual(detector.board.fen(), board.fen())
        self.assertEqual(len(detector.confirmed_choices), 4)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(detector.confirm(ord("2")).uci(), "a7a8r")
        self.assertEqual(detector.board.piece_type_at(chess.A8), chess.ROOK)

    def test_partial_castling_not_committed_as_rook_move(self):
        board = position(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        scores = np.zeros(64, np.float32)
        scores[[chess.H1, chess.F1]] = .3
        result = match_legal_moves(board, scores)
        self.assertIsNone(result.move)
        self.assertEqual(result.choices, (chess.Move.from_uci("h1f1"),))

    def test_black_en_passant_and_capture_promotions(self):
        board = position(["a2a3", "e7e5", "a3a4", "e5e4", "d2d4"])
        move = chess.Move.from_uci("e4d3")
        self.assertEqual(changed_squares(board, move), {chess.E4, chess.D4, chess.D3})
        board = position(fen="4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1")
        scores = np.zeros(64, np.float32)
        scores[[chess.A2, chess.B1]] = .3
        result = match_legal_moves(board, scores)
        self.assertIsNone(result.move)
        self.assertEqual({move.uci() for move in result.choices},
                         {"a2b1q", "a2b1r", "a2b1b", "a2b1n"})

    def test_incomplete_move_keeps_original_baseline(self):
        board = position()
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        baseline = detector.baseline.copy()
        lifted = board.copy()
        lifted.remove_piece_at(chess.E2)
        self.assertEqual(self.feed(detector, render(lifted), 3), [])
        np.testing.assert_array_equal(detector.baseline, baseline)
        self.assertEqual(detector.board.fen(), board.fen())
        board.push_uci("e2e4")
        self.assertEqual(self.feed(detector, render(board), 6), ["e2e4"])

    def test_promotion_confirmation_cleared_on_motion_or_tracking_loss(self):
        board = position(fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        after = board.copy()
        after.push_uci("a7a8q")
        self.feed(detector, render(after), 3)
        self.assertEqual(len(detector.confirmed_choices), 4)
        detector.update(None, reliable=False, now=6)
        self.assertIsNone(detector.confirm(ord("1")))
        self.assertEqual(detector.board.fen(), board.fen())

    def test_stationary_large_occluder_does_not_advance_board(self):
        board = position()
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        covered = render(board)
        cv2.rectangle(covered, (100, 100), (350, 400), (70, 100, 130), -1)
        self.assertEqual(self.feed(detector, covered, 3), [])
        self.assertEqual(detector.board.fen(), board.fen())
        self.assertEqual(self.feed(detector, render(board), 6), [])

    def test_held_geometry_pauses_and_preserves_baseline(self):
        board = position()
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        baseline = detector.baseline.copy()
        board.push_uci("e2e4")
        self.assertEqual(self.feed(detector, render(board), 3, reliable=False), [])
        self.assertEqual(detector.board.fen(), chess.STARTING_FEN)
        np.testing.assert_array_equal(detector.baseline, baseline)
        detector.update(render(board), reliable=True, now=6)
        self.assertEqual(self.feed(detector, render(board), 6), ["e2e4"])

    def test_initial_stability_needs_time_and_frames(self):
        detector = MoveDetector(stable_seconds=1)
        image = render(position())
        self.feed(detector, image, 0, frames=3)
        self.assertIsNone(detector.baseline)
        self.feed(detector, image, .6, frames=4)
        self.assertIsNotNone(detector.baseline)

    def test_starting_baseline_tolerates_camera_noise(self):
        detector = MoveDetector(stable_seconds=.6)
        image = render(position()).astype(np.float32)
        rng = np.random.default_rng(12)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(12):
                noisy = np.clip(image + rng.normal(0, 15, image.shape), 0, 255).astype(np.uint8)
                detector.update(noisy, now=i * .2)
        self.assertIsNotNone(detector.baseline, detector.status)
        self.assertEqual(detector.board.fen(), chess.STARTING_FEN)

    def test_successive_moves_advance_turn_and_image_baseline(self):
        board = position()
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        for index, uci in enumerate(("e2e4", "d7d5", "e4d5", "d8d5", "b1c3")):
            board.push_uci(uci)
            self.assertEqual(self.feed(detector, render(board), 3 + index * 3), [uci], detector.status)
            self.assertEqual(detector.board.fen(), board.fen())

    def test_explicit_resync_and_undo_keep_position_in_step(self):
        detector = MoveDetector(stable_seconds=.6, confirm_seconds=.4)
        board = position()
        self.feed(detector, render(board), 0)
        original_fen = detector.board.fen()
        returned_piece = render(board).copy()
        cv2.circle(returned_piece, (5 * 64 + 40, 64 + 36), 8, (0, 0, 0), -1)
        with contextlib.redirect_stdout(io.StringIO()):
            detector.rebaseline(returned_piece)
        self.assertEqual(detector.board.fen(), original_fen)
        np.testing.assert_array_equal(detector.baseline, returned_piece)

        board.push_uci("e2e4")
        after = render(board)
        # The piece adjustment that was explicitly resynced remains unchanged
        # while the next pawn move is made.
        cv2.circle(after, (5 * 64 + 40, 64 + 36), 8, (0, 0, 0), -1)
        self.assertEqual(self.feed(detector, after, 3), ["e2e4"])
        with contextlib.redirect_stdout(io.StringIO()):
            detector.undo_and_rebaseline(returned_piece)
        self.assertEqual(detector.board.fen(), original_fen)
        np.testing.assert_array_equal(detector.baseline, returned_piece)
        with self.assertRaisesRegex(ValueError, "no recorded move"):
            detector.undo_and_rebaseline(returned_piece)

    def test_finish_castling_after_rook_first(self):
        board = position(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        self.feed(detector, render(board), 0)
        intermediate = board.copy()
        intermediate.push_uci("h1f1")
        self.assertEqual(self.feed(detector, render(intermediate), 3), [])
        self.assertEqual(detector.board.fen(), board.fen())
        board.push_uci("e1g1")
        self.assertEqual(self.feed(detector, render(board), 6), ["e1g1"])


class ButtonMoveTests(unittest.TestCase):
    feed = MoveTests.feed

    def detector(self, board=None):
        board = board or chess.Board()
        detector = MoveDetector(board.fen(), stable_seconds=.6, confirm_seconds=.4)
        with contextlib.redirect_stdout(io.StringIO()):
            detector.rebaseline(render(board))
        return detector

    def test_no_move_is_recorded_without_a_click_and_each_click_records_once(self):
        detector = self.detector()
        baseline = detector.baseline.copy()
        after = position(["e2e4"])
        self.assertEqual(self.feed(detector, render(after), 0, check=False), [])
        self.assertEqual(detector.board.fen(), chess.STARTING_FEN)
        np.testing.assert_array_equal(detector.baseline, baseline)
        detector.request_check(now=3)
        # Previously observed frames cannot complete a new request.
        self.assertEqual(self.feed(detector, render(after), 3, frames=3, check=False), [])
        self.assertEqual(self.feed(detector, render(after), 3.6, check=False), ["e2e4"])
        self.assertFalse(detector.check_pending)
        after.push_uci("e7e5")
        self.assertEqual(self.feed(detector, render(after), 6, check=False), [])
        self.assertEqual(len(detector.board.move_stack), 1)
        self.assertEqual(self.feed(detector, render(after), 9), ["e7e5"])
        self.assertIn("Black pawn from e7 to e5", detector.check_result)

    def test_no_change_ends_request_and_does_not_arm_a_later_move(self):
        detector = self.detector()
        self.assertEqual(self.feed(detector, render(position()), 0), [])
        self.assertFalse(detector.check_pending)
        self.assertIn("No move found", detector.check_result)
        self.assertEqual(self.feed(detector, render(position(["e2e4"])), 3, check=False), [])
        self.assertEqual(detector.board.fen(), chess.STARTING_FEN)

    def test_invalid_move_preserves_baseline_and_requires_another_click(self):
        detector = self.detector()
        baseline = detector.baseline.copy()
        lifted = position()
        lifted.remove_piece_at(chess.E2)
        self.assertEqual(self.feed(detector, render(lifted), 0), [])
        self.assertFalse(detector.check_pending)
        self.assertIn("Could not recognize", detector.check_result)
        np.testing.assert_array_equal(detector.baseline, baseline)
        after = position(["e2e4"])
        self.assertEqual(self.feed(detector, render(after), 3, check=False), [])
        self.assertEqual(self.feed(detector, render(after), 6), ["e2e4"])

    def test_duplicate_tracking_loss_and_timeout_leave_the_game_unchanged(self):
        detector = self.detector()
        detector.request_check(now=0)
        with self.assertRaisesRegex(ValueError, "already in progress"):
            detector.request_check(now=0)
        detector.update(None, reliable=False, now=.2)
        self.assertFalse(detector.check_pending)
        with self.assertRaisesRegex(ValueError, "camera view"):
            detector.request_check(now=1)
        after = position(["e2e4"])
        self.assertEqual(self.feed(detector, render(after), 1, check=False), [])
        detector.request_check(now=3)
        self.assertIsNone(detector.update(render(after), now=12))
        self.assertIn("timed out", detector.check_result)
        self.assertEqual(self.feed(detector, render(after), 13, check=False), [])
        self.assertEqual(detector.board.fen(), chess.STARTING_FEN)

    def test_promotion_choices_need_a_click_and_motion_invalidates_them(self):
        board = position(fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        detector = self.detector(board)
        after = board.copy()
        after.push_uci("a7a8q")
        self.feed(detector, render(after), 0, check=False)
        self.assertFalse(detector.confirmed_choices)
        self.feed(detector, render(after), 3)
        self.assertEqual(len(detector.confirmed_choices), 4)
        self.assertFalse(detector.check_pending)
        detector.update(render(board), now=6)
        self.assertIsNone(detector.confirm(ord("1")))
        self.feed(detector, render(after), 7, check=False)
        self.assertFalse(detector.confirmed_choices)
        self.feed(detector, render(after), 10)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(detector.confirm(ord("1")).uci(), "a7a8q")

    def test_resync_cancels_a_pending_check(self):
        detector = self.detector()
        detector.request_check(now=0)
        with contextlib.redirect_stdout(io.StringIO()):
            detector.rebaseline(render(position()))
        self.assertFalse(detector.check_pending)
        self.assertEqual(self.feed(detector, render(position(["e2e4"])), 1, check=False), [])
        with self.assertRaisesRegex(ValueError, "starting position"):
            MoveDetector().request_check()


if __name__ == "__main__":
    unittest.main()
