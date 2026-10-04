"""Grounding, history, and failure handling for conversational replies."""

import json
import unittest
from unittest.mock import patch

import chess

from coach_language import LocalCoachWriter


class Response:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        pass
    def read(self, _size):
        return json.dumps({"message": {"content": json.dumps(self.value)}}).encode()


class WriterTests(unittest.TestCase):
    facts = {"move_purpose": "The pawn controls d5 and f5 and opens the bishop's path."}
    moves = {"e2e4": {"before_position": chess.STARTING_FEN,
                       "move": "White pawn from e2 to e4"}}

    def response(self, **extra):
        return {"answer": "The White pawn from e2 to e4 gives the bishop room to come out.",
                "used_facts": ["move_purpose"], "moves": ["e2e4"], **extra}

    def test_real_question_and_recent_conversation_are_sent_with_evidence(self):
        writer = LocalCoachWriter()
        history = [{"role": "user", "content": "What is development?"},
                   {"role": "assistant", "content": "Bringing pieces into play."}]
        with patch("coach_language.request.urlopen", return_value=Response(self.response())) as call:
            reply = writer.respond("Explain that more simply", history, self.facts, self.moves)
        self.assertIn("bishop room", reply)
        self.assertTrue(writer.last_used_model)
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload["messages"][1:3], history)
        self.assertIn("Explain that more simply", payload["messages"][-1]["content"])
        self.assertIn("move_purpose", payload["format"]["properties"]["used_facts"]["items"]["enum"])

    def test_invented_evidence_or_moves_are_rejected(self):
        for value in (self.response(used_facts=["invented tactic"]),
                      self.response(moves=["e2e5"]), self.response(answer=""),
                      self.response(used_facts="move_purpose"), []):
            with self.subTest(value=value), patch("coach_language.request.urlopen", return_value=Response(value)):
                self.assertIsNone(LocalCoachWriter().respond("Why?", [], self.facts, self.moves))

    def test_wrong_piece_or_move_in_prose_is_rejected_even_with_valid_metadata(self):
        for text in ("Move your bishop from e2 to e4.", "Move the pawn from e2 to e5.",
                     "Move the Black pawn from e2 to e4."):
            with self.subTest(text=text), patch("coach_language.request.urlopen",
                                               return_value=Response(self.response(answer=text))):
                self.assertIsNone(LocalCoachWriter().respond("Why?", [], self.facts, self.moves))

    def test_general_explanation_does_not_require_a_move_instruction(self):
        value = self.response(answer="Development means getting pieces into useful positions.", moves=[])
        with patch("coach_language.request.urlopen", return_value=Response(value)):
            self.assertIn("useful positions", LocalCoachWriter().respond("What is development?", [], self.facts, {}))

    def test_simplifying_center_control_cannot_change_piece_movement_rules(self):
        value = self.response(answer="Pieces in the center can move in any direction, to any square.", moves=[])
        with patch("coach_language.request.urlopen", return_value=Response(value)):
            self.assertIsNone(LocalCoachWriter().respond("Explain the center simply", [], self.facts, {}))

    def test_unavailable_model_falls_back_without_repeating_slow_requests(self):
        writer = LocalCoachWriter()
        with patch("coach_language.request.urlopen", side_effect=TimeoutError) as call:
            self.assertIsNone(writer.respond("Why?", [], self.facts, self.moves))
            self.assertIsNone(writer.respond("Explain it simply", [], self.facts, self.moves))
        self.assertEqual(call.call_count, 1)
        self.assertFalse(writer.last_used_model)


if __name__ == "__main__":
    unittest.main()
