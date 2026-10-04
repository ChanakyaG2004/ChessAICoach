"""Camera-free tests for local engine coaching: python -m unittest test_coach."""

from pathlib import Path
import tempfile
import unittest

import chess
import chess.engine

from coach import (ChessCoach, CoachError, EngineUnavailable, find_stockfish)


def analysis(board, score, best):
    return {
        "score": chess.engine.PovScore(score, chess.WHITE),
        "pv": [chess.Move.from_uci(best)] if best else [],
    }


class FakeEngine:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.closed = False

    def analyse(self, board, limit):
        self.calls.append((board.fen(), limit.time))
        return self.answers[board.fen()]

    def quit(self):
        self.closed = True


class CoachTests(unittest.TestCase):
    def _review_with_reply(self, board, move):
        after = board.copy()
        after.push(move)
        reply = next(iter(after.legal_moves))
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Cp(30), move.uci()),
            after.fen(): analysis(after, chess.engine.Cp(10), reply.uci()),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.reply_move, reply)
        self.assertEqual(review.reply_san, after.san(reply))
        self.assertIn(f"Next reply: {review.reply_plain}", review.summary)
        self.assertEqual(len(engine.calls), 2)
        return review

    def test_find_explicit_executable_and_missing_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stockfish"
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
            self.assertEqual(find_stockfish(path), path.resolve())
            with self.assertRaisesRegex(EngineUnavailable, "not executable"):
                find_stockfish(path.with_name("missing"))

    def test_suggestion_is_legal_and_keeps_board_unchanged(self):
        board = chess.Board()
        before = board.fen()
        engine = FakeEngine({before: analysis(board, chess.engine.Cp(32), "e2e4")})
        with ChessCoach(analysis_time=0.05, engine=engine) as coach:
            suggestion = coach.suggest_move(board)
            self.assertEqual(suggestion.move.uci(), "e2e4")
            self.assertEqual(suggestion.san, "e4")
            self.assertIn("+0.32", suggestion.summary)
            self.assertEqual(board.fen(), before)
            self.assertEqual(engine.calls, [(before, 0.05)])
        self.assertTrue(engine.closed)

    def test_review_centipawn_loss_and_best_alternative(self):
        board = chess.Board()
        move = chess.Move.from_uci("d2d4")
        after = board.copy()
        after.push(move)
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Cp(50), "e2e4"),
            after.fen(): analysis(after, chess.engine.Cp(-80), "d7d5"),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.san, "d4")
        self.assertEqual(review.grade, "Mistake")
        self.assertEqual(review.centipawn_loss, 130)
        self.assertEqual(review.best_san, "e4")
        self.assertIn("Stockfish preferred pawn from e2 to e4", review.summary)
        self.assertEqual(board.fen(), chess.Board().fen())

    def test_black_move_scores_use_black_perspective(self):
        board = chess.Board()
        board.push_uci("e2e4")
        move = chess.Move.from_uci("e7e5")
        after = board.copy()
        after.push(move)
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Cp(60), "c7c5"),
            after.fen(): analysis(after, chess.engine.Cp(20), "g1f3"),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.before.centipawns, -60)
        self.assertEqual(review.after.centipawns, -20)
        self.assertEqual(review.centipawn_loss, 0)
        self.assertEqual(review.grade, "Good")

    def test_best_move_is_not_penalized_for_search_variation(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        after = board.copy()
        after.push(move)
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Cp(34), "e2e4"),
            after.fen(): analysis(after, chess.engine.Cp(10), "c7c5"),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.grade, "Best")
        self.assertEqual(review.centipawn_loss, 0)
        self.assertIn("top choice", review.summary)

    def test_factual_notes_and_short_post_move_reply(self):
        capture = chess.Board()
        capture.push_uci("e2e4")
        capture.push_uci("d7d5")
        bishop = chess.Board()
        bishop.push_uci("e2e4")
        bishop.push_uci("a7a6")
        en_passant = chess.Board()
        for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
            en_passant.push_uci(uci)
        cases = (
            (chess.Board(), "e2e4", "pawn to center"),
            (chess.Board(), "g1f3", "develops knight"),
            (bishop, "f1c4", "develops bishop"),
            (capture, "e4d5", "captures on d5"),
            (en_passant, "e5d6", "captures en passant"),
            (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
             "e1g1", "castles kingside"),
            (chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
             "a7a8q", "promotes to queen"),
            (chess.Board("4k3/8/8/8/8/8/8/K2Q4 w - - 0 1"),
             "d1e2", "gives check"),
        )
        for board, uci, note in cases:
            with self.subTest(move=uci, note=note):
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves)
                review = self._review_with_reply(board, move)
                self.assertIn(note, review.notes)
                self.assertIn(note, review.summary)
                self.assertLessEqual(len(review.summary), 160)

    def test_returned_minor_piece_is_not_marked_as_new_development(self):
        board = chess.Board()
        for uci in ("g1f3", "a7a6", "f3g1", "a6a5"):
            board.push_uci(uci)
        review = self._review_with_reply(board, chess.Move.from_uci("g1f3"))
        self.assertNotIn("develops knight", review.notes)

    def test_missing_post_move_pv_omits_reply_without_losing_feedback(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        after = board.copy()
        after.push(move)
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Cp(30), "e2e4"),
            after.fen(): analysis(after, chess.engine.Cp(10), None),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.grade, "Best")
        self.assertIsNone(review.reply_move)
        self.assertNotIn("Next reply", review.summary)

    def test_mate_is_reported_without_huge_centipawn_loss(self):
        board = chess.Board("k7/8/1QK5/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("b6b7")
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Mate(1), "b6b7"),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.grade, "Checkmate")
        self.assertEqual(review.after.text, "checkmate")
        self.assertEqual(len(engine.calls), 1)
        self.assertIn("checkmate", review.summary)

    def test_stalemate_that_throws_away_a_win_is_rated(self):
        board = chess.Board("k7/8/1QK5/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("b6c7")
        after = board.copy()
        after.push(move)
        self.assertTrue(after.is_stalemate())
        engine = FakeEngine({
            board.fen(): analysis(board, chess.engine.Cp(800), "b6b7"),
        })
        with ChessCoach(engine=engine) as coach:
            review = coach.review_move(board, move)
        self.assertEqual(review.grade, "Blunder")
        self.assertEqual(review.centipawn_loss, 800)
        self.assertEqual(review.after.text, "draw")

    def test_mate_scores_do_not_turn_into_centipawn_numbers(self):
        board = chess.Board()
        move = chess.Move.from_uci("d2d4")
        after = board.copy()
        after.push(move)
        cases = (
            (chess.engine.Mate(2), chess.engine.Cp(50), "Missed forced mate"),
            (chess.engine.Cp(0), chess.engine.Mate(-2), "Blunder"),
        )
        for before_score, after_score, grade in cases:
            with self.subTest(grade=grade):
                engine = FakeEngine({
                    board.fen(): analysis(board, before_score, "e2e4"),
                    after.fen(): analysis(after, after_score, "d7d5"),
                })
                with ChessCoach(engine=engine) as coach:
                    review = coach.review_move(board, move)
                self.assertEqual(review.grade, grade)
                self.assertIsNone(review.centipawn_loss)

    def test_finished_game_has_no_suggestion(self):
        board = chess.Board()
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push_uci(uci)
        engine = FakeEngine({})
        with ChessCoach(engine=engine) as coach:
            self.assertIsNone(coach.suggest_move(board))
        self.assertEqual(engine.calls, [])

    def test_invalid_move_and_engine_response(self):
        board = chess.Board()
        engine = FakeEngine({board.fen(): {"score": chess.engine.PovScore(
            chess.engine.Cp(0), chess.WHITE), "pv": []}})
        with ChessCoach(engine=engine) as coach:
            with self.assertRaisesRegex(ValueError, "not legal"):
                coach.review_move(board, chess.Move.from_uci("e7e5"))
            with self.assertRaisesRegex(CoachError, "no legal suggested move"):
                coach.suggest_move(board)

    def test_analysis_time_must_be_positive_and_finite(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ChessCoach(analysis_time=value, engine=FakeEngine({}))


if __name__ == "__main__":
    unittest.main()
