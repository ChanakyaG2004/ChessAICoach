"""Grounded text Q&A tests without camera, network, or engine subprocesses."""

import json
import unittest
from unittest.mock import patch

import chess
import chess.engine

from coach import ChessCoach
from conversation import ConversationCoach, _move_description, _position_facts, _purpose_text


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False

    def analyse(self, board, limit):
        self.calls.append((board.fen(), limit.time))
        last = board.move_stack[-1].uci() if board.move_stack else None
        if not last:
            score, uci = 30, "e2e4"
        elif last == "d2d4":
            score, uci = -100, "d7d5"
        elif last == "e2e4":
            score, uci = 10, "e7e5"
        elif last == "g1f3":
            score, uci = 0, "e7e5"
        elif last == "e7e5":
            score, uci = 15, "g1f3"
        else:
            score, uci = 0, next(iter(board.legal_moves)).uci()
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(score), chess.WHITE),
            "pv": [chess.Move.from_uci(uci)],
        }

    def quit(self):
        self.closed = True


def coached(enabled=True):
    conversation = ConversationCoach(use_ollama=False, engine_enabled=enabled)
    fake = FakeEngine()
    if enabled:
        conversation._coach = ChessCoach(engine=fake, analysis_time=0.01)
    return conversation, fake


class ConversationTests(unittest.TestCase):
    def test_simpler_center_follow_up_still_explains_the_term_when_model_is_offline(self):
        chat, _ = coached()
        board = chess.Board()
        chat.set_game_context(board, "Coach's move: White pawn from e2 to e4.", "")
        chat.answer("Why should I move that pawn?", board, "black")
        answer = chat.answer("I don't understand controlling the center. Explain it more simply.", board, "black")
        self.assertTrue(answer.startswith("The center is"))
        self.assertIn("attack those squares", answer)
        self.assertNotIn("short engine line", answer)
        short = chat.answer("Explain that move in one sentence", board, "black")
        self.assertNotIn("short engine line", short)
        self.assertEqual(short.count("."), 1)
        chat.close()
    def test_promotion_explanation_does_not_describe_the_new_piece_as_a_pawn(self):
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        text = _purpose_text(board, chess.Move.from_uci("a7a8q"))
        self.assertIn("becomes a queen", text)
        self.assertNotIn("pawn now guards", text)
        self.assertNotIn("cannot move back", text)
    def test_why_understands_displayed_pawn_without_a_previous_chat_question(self):
        chat, _ = coached(enabled=False)
        board = chess.Board()
        chat.set_game_context(board, "Coach's move: White pawn from e2 to e4. Play it.", "")
        answer = chat.answer("Why should I move a pawn?", board, "black")
        self.assertIn("claim space in the center", answer)
        self.assertIn("bishop on f1", answer)
        self.assertIn("d5, f5", answer)
        self.assertIn("pawn cannot move back", answer)
        self.assertNotIn("Which move", answer)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_flexible_follow_up_receives_history_and_current_move_facts(self):
        class Writer:
            calls = []
            def respond(self, question, history, facts, moves):
                self.calls.append((question, list(history), facts, dict(moves)))
                return "Think of it as giving your bishop a doorway out of its starting square."
        chat, _ = coached(enabled=False)
        chat._writer = Writer()
        board = chess.Board()
        chat.set_game_context(board, "Coach's move: White pawn from e2 to e4.", "")
        first = chat.answer("Why should I move a pawn?", board, "black")
        second = chat.answer("Can you give me an analogy for that?", board, "black")
        question, history, facts, moves = chat._writer.calls[-1]
        self.assertEqual(question, "Can you give me an analogy for that?")
        self.assertEqual(history[-1]["content"], first)
        self.assertIn("bishop", facts["move_purpose"])
        self.assertIn("e2e4", moves)
        self.assertEqual(first, second)

    def test_changed_position_cannot_reuse_displayed_old_recommendation(self):
        chat, _ = coached(enabled=False)
        board = chess.Board()
        chat.set_game_context(board, "Coach's move: White pawn from e2 to e4.", "")
        chat.answer("Why?", board, "black")
        board.push_uci("a2a3")
        answer = chat.answer("Why?", board, "black")
        self.assertIn("Which move", answer)
        self.assertIsNone(chat._context)
        self.assertIn(chess.STARTING_FEN, chat._history[0]["content"])

    def test_alternative_knight_moves_are_checked_for_a_flexible_comparison(self):
        class Writer:
            facts = None
            def respond(self, question, history, facts, moves):
                self.facts = facts
                return None
        chat, _ = coached()
        chat._writer = Writer()
        chat.answer("Why not move my knight first instead?", chess.Board(), "white")
        alternatives = chat._writer.facts["requested_alternatives"]
        self.assertEqual(len(alternatives), 2)
        self.assertTrue(all("comparison" in item for item in alternatives))
        self.assertTrue(all("knight" in item["move"] for item in alternatives))
        chat.close()

    def test_suggestion_and_why_are_physical_and_grounded(self):
        board = chess.Board()
        original = board.fen()
        chat, fake = coached()
        suggestion = chat.answer("What should I play now?", board, "white")
        self.assertIn("move your pawn from e2 to e4", suggestion)
        self.assertIn("for White", suggestion)
        why = chat.answer("Why?", board, "white")
        self.assertIn("places a pawn in the center", why)
        self.assertIn("opens a line for bishop on f1", why)
        self.assertIn("short engine line", why)
        self.assertEqual(board.fen(), original)
        self.assertTrue(fake.calls)
        chat.close()
        chat.close()
        self.assertTrue(fake.closed)

    def test_better_move_and_why_follow_up_use_last_human_move(self):
        board = chess.Board()
        board.push_uci("d2d4")
        board.push_uci("d7d5")
        chat, _ = coached()
        better = chat.answer("What should I have played instead?", board, "white")
        self.assertIn("last pawn from d2 to d4", better)
        self.assertIn("preferred move your pawn from e2 to e4", better)
        why = chat.answer("Why was that better?", board, "white")
        self.assertIn("evaluation changed", why)
        self.assertIn("After pawn from e2 to e4, one short engine line is Black pawn from e7 to e5", why)
        self.assertIn("opens a line", why)
        chat.close()

    def test_last_human_move_is_found_even_after_opponent_reply(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        chat, _ = coached()
        answer = chat.answer("Review my last move", board, "white")
        self.assertIn("last White move was pawn from e2 to e4", answer)
        self.assertIn("top choice", answer)
        self.assertEqual(len(board.move_stack), 2)
        chat.close()

    def test_candidate_square_notation_is_hypothetical(self):
        board = chess.Board()
        chat, _ = coached()
        answer = chat.answer("What if I move my knight from g1 to f3?", board, "white")
        self.assertIn("Hypothetical", answer)
        self.assertIn("develops knight", answer)
        self.assertIn("recorded board has not changed", answer)
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        invalid = chat.answer("What if e2 to e5?", board, "white")
        self.assertIn("not a legal move", invalid)
        chat.close()

    def test_opponent_reply_identifies_side_and_validated_line(self):
        board = chess.Board()
        board.push_uci("e2e4")
        chat, _ = coached()
        answer = chat.answer("What is my opponent's reply?", board, "white")
        self.assertIn("strongest Black reply is to move your pawn from e7 to e5", answer)
        self.assertIn("move your pawn from e7 to e5", answer)
        self.assertIn("Short line: Black pawn from e7 to e5", answer)
        self.assertEqual(board.peek().uci(), "e2e4")
        chat.close()

    def test_combined_recommendation_and_why_uses_current_position(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        chat, _ = coached()
        answer = chat.answer("What is the best move here, and why?", board, "white")
        self.assertIn("move your knight from g1 to f3", answer)
        self.assertIn("develops the knight", answer)
        answer = chat.answer("What is a better move to make?", board, "white")
        self.assertIn("knight from g1 to f3", answer)
        self.assertNotIn("For your last", answer)
        chat.close()

    def test_current_threat_does_not_replay_an_earlier_position(self):
        board = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
            board.push_uci(move)
        chat, _ = coached(enabled=False)
        answer = chat.answer("What is the main threat in this position?", board, "black")
        self.assertIn("pawn on e5 is attacked by knight on f3", answer)
        self.assertNotIn("your last", answer)
        chat.close()

    def test_definition_and_disabled_engine_are_offline(self):
        chat, fake = coached(enabled=False)
        board = chess.Board()
        self.assertIn("no legal move", chat.answer("What is stalemate?", board, "white"))
        candidate = chat.answer("What if Nf3?", board, "white")
        self.assertIn("Hypothetical knight from g1 to f3 is legal", candidate)
        self.assertIn("develops the knight", candidate)
        self.assertIn("Engine analysis is disabled", candidate)
        self.assertIn("disabled", chat.answer("What is the best move?", board, "white"))
        self.assertEqual(fake.calls, [])
        chat.close()

    def test_follow_up_context_is_invalidated_when_fen_changes(self):
        chat, _ = coached()
        board = chess.Board()
        chat.answer("What should I play now?", board, "white")
        changed = board.copy()
        changed.push_uci("a2a3")
        answer = chat.answer("Why?", changed, "black")
        self.assertIn("Which move should I explain", answer)
        chat.close()

    def test_move_description_and_position_facts_are_verified(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        facts = _position_facts(board, move)
        self.assertIn("pawn to center", facts)
        self.assertIn("opens a line for bishop on f1", " ".join(facts))
        self.assertEqual(_move_description(board, move), "move your pawn from e2 to e4")
        castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertIn("rook h1 to f1", _move_description(castle, chess.Move.from_uci("e1g1")))

    def test_ollama_json_is_only_parsed_then_move_is_legality_checked(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, _size):
                data = {"message": {"content": json.dumps({
                    "intent": "candidate", "move": "e2e5"})}}
                return json.dumps(data).encode()

        chat = ConversationCoach(use_ollama=True, engine_enabled=False)
        with patch("conversation.request.urlopen", return_value=FakeResponse()):
            answer = chat.answer("Can my little soldier leap there?", chess.Board(), "white")
        self.assertIn("not a legal move", answer)
        chat.close()


if __name__ == "__main__":
    unittest.main()
