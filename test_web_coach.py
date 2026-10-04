"""HTTP security, snapshot, action, and voice checks for the local coach UI."""

import json
import http.client
import re
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import chess

from web_coach import CoachWebServer


class FakeConversation:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.seen = []
        self.contexts = []
        self.closed = False

    def answer(self, question, board, human_color):
        self.seen.append((question, board.fen(), human_color))
        self.started.set()
        self.release.wait(5)
        return f"From {board.fen()}: develop a knight."

    def set_game_context(self, board, instruction, feedback):
        self.contexts.append((board.fen(), instruction, feedback))

    def close(self):
        self.closed = True


class FakeSpeech:
    def status(self):
        return {"available": True, "tts_available": True, "reason": "Ready"}

    def transcribe(self, wav_bytes):
        if wav_bytes != b"RIFF" + b"a" * 40:
            raise ValueError("Bad clip")
        return "What should I play?"

    def synthesize(self, text):
        return b"RIFF" + text.encode("utf-8")


class WebCoachTests(unittest.TestCase):
    def setUp(self):
        self.conversation = FakeConversation()
        self.web = CoachWebServer(self.conversation, port=0, speech=FakeSpeech()).start()
        self.addCleanup(self.web.close)
        html = self.get("/")[1].decode("utf-8")
        self.token = re.search(r'name="coach-csrf" content="([^"]+)"', html).group(1)

    def get(self, path, headers=None):
        request = Request(self.web.url + path, headers=headers or {})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read(), response.headers
        except HTTPError as exc:
            return exc.code, exc.read(), exc.headers

    def post(self, path, value, *, token=None, origin=None, content_type="application/json"):
        body = (json.dumps(value).encode("utf-8") if content_type == "application/json"
                else value)
        request = Request(self.web.url + path, data=body, method="POST", headers={
            "Content-Type": content_type,
            "Origin": origin or self.web.url,
            "X-Coach-CSRF": token or self.token,
        })
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read(), response.headers
        except HTTPError as exc:
            return exc.code, exc.read(), exc.headers

    def state(self):
        status, body, _ = self.get("/api/state")
        self.assertEqual(status, 200)
        return json.loads(body)

    def wait_reply(self, status):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            replies = [message for message in self.state()["messages"]
                       if message["role"] == "assistant"]
            if replies and replies[-1]["status"] == status:
                return replies[-1]
            time.sleep(.03)
        self.fail(f"No {status} assistant reply")

    def test_static_ui_state_and_image_lifecycle(self):
        self.assertIn(b"Coach chat", self.get("/")[1])
        self.assertEqual(self.get("/app.js")[0], 200)
        self.assertEqual(self.get("/style.css")[0], 200)
        self.assertEqual(self.get("/../../etc/passwd")[0], 404)
        board = chess.Board()
        board.push_san("e4")
        jpeg = b"\xff\xd8" + b"image" + b"\xff\xd9"
        self.web.publish(board, "black", "Make a black move", "White played e4",
                         "Board tracked", None, jpeg, "Stockfish ready")
        state = self.state()
        self.assertEqual(state["fen"], board.fen())
        self.assertEqual(state["human_color"], "black")
        self.assertEqual(state["pieces"]["e4"], "P")
        self.assertEqual(state["moves"], ["Move 1: White pawn from e2 to e4"])
        self.assertEqual(state["messages"][-1]["text"], "White played e4")
        self.web.publish(board, "black", "Make a black move", "White played e4",
                         "Board tracked", None, jpeg, "Stockfish ready")
        self.assertEqual(len(self.state()["messages"]), 1)
        self.assertEqual(self.get("/board.jpg")[0], 403)
        self.assertEqual(self.get(state["image_url"])[1], jpeg)
        self.web.publish(board, "black", "Waiting", "", "Lost", None)
        self.assertIsNone(self.state()["image_url"])
        self.assertEqual(self.get(state["image_url"])[0], 404)

    def test_chat_snapshot_and_stale_response(self):
        self.conversation.release.clear()
        board = chess.Board()
        self.web.publish(board, "white", "Play", "", "Tracked", None)
        self.assertEqual(self.post("/api/chat", {"question": "What now?"})[0], 202)
        self.assertTrue(self.conversation.started.wait(2))
        board.push_san("e4")
        self.web.publish(board, "white", "Play", "", "Tracked", None)
        self.conversation.release.set()
        reply = self.wait_reply("stale")
        self.assertIn("board changed", reply["text"].lower())
        self.assertNotIn("develop a knight", reply["text"])
        self.assertEqual(self.conversation.seen[0][1], chess.STARTING_FEN)

        self.assertEqual(self.post("/api/chat", {"question": "Plan?"})[0], 202)
        reply = self.wait_reply("done")
        self.assertIn(board.fen(), reply["text"])

    def test_chat_receives_the_displayed_move_from_the_question_snapshot(self):
        self.conversation.release.clear()
        board = chess.Board()
        self.web.publish(board, "black", "Coach's move: White pawn from e2 to e4.",
                         "Previous feedback", "Tracked", None)
        self.assertEqual(self.post("/api/chat", {"question": "Why should I move that pawn?"})[0], 202)
        self.assertTrue(self.conversation.started.wait(2))
        self.web.publish(board, "black", "Changed instruction", "Changed feedback", "Tracked", None)
        self.assertEqual(self.conversation.contexts, [(board.fen(),
                         "Coach's move: White pawn from e2 to e4.", "Previous feedback")])
        self.conversation.release.set()
        self.wait_reply("done")

    def test_action_queue_validates_fen_and_inputs(self):
        fen = self.state()["fen"]
        self.assertEqual(self.post("/api/action", {"action": ["undo"], "expected_fen": fen})[0], 400)
        self.assertEqual(self.post("/api/action", {"action": "confirm", "expected_fen": fen})[0], 409)
        self.assertEqual(self.post("/api/action", {"action": "save", "expected_fen": "wrong"})[0], 409)
        self.assertEqual(self.post("/api/action", {"action": "manual", "expected_fen": fen,
                                                      "move": "e7e5"})[0], 400)
        self.assertEqual(self.post("/api/action", {"action": "manual", "expected_fen": fen,
                                                      "move": "e2e4"})[0], 202)
        self.assertEqual(self.post("/api/action", {"action": "save", "expected_fen": fen})[0], 429)
        self.assertEqual(self.web.pop_action(), {"action": "manual", "expected_fen": fen,
                                                 "move": "e2e4"})
        self.assertIsNone(self.web.pop_action())
        self.assertEqual(self.post("/api/action", {"action": "manual", "expected_fen": fen,
                                                      "move": "pawn from e2 to e4"})[0], 202)
        self.assertEqual(self.web.pop_action(), {"action": "manual", "expected_fen": fen,
                                                 "move": "e2e4"})
        self.web.publish(chess.Board(), "white", "Check pieces", "", "Tracked",
                         "Press Y", recovery_ready=True)
        self.assertEqual(self.post("/api/action", {"action": "confirm", "expected_fen": fen})[0], 202)

    def test_target_board_differs_from_current_until_camera_confirms(self):
        current = chess.Board()
        current.push_san("e4")
        target = chess.Board()
        self.web.publish(current, "white", "Undo", "", "Tracked", "Restore pieces",
                         recovery_ready=True, expected_board=target)
        state = self.state()
        self.assertEqual(state["fen"], current.fen())
        self.assertEqual(state["expected_fen"], target.fen())
        self.assertIn("e4", state["pieces"])
        self.assertIn("e2", state["expected_pieces"])

    def test_move_check_requires_setup_and_camera_and_blocks_duplicate_requests(self):
        board = chess.Board()
        request = {"action": "check", "expected_fen": board.fen()}
        self.assertIn(b"Check my move", self.get("/")[1])
        self.assertEqual(self.post("/api/action", request)[0], 409)
        self.web.publish(board, "black", "Play", "", "Tracked", "Check setup",
                         recovery_ready=True, move_check_ready=True)
        self.assertFalse(self.state()["move_check_ready"])
        self.assertEqual(self.post("/api/action", request)[0], 409)
        self.web.publish(board, "black", "Play", "", "Tracked", None, move_check_ready=True)
        self.assertEqual(self.post("/api/action", {**request, "expected_fen": "stale"})[0], 409)
        self.assertEqual(self.post("/api/action", request)[0], 202)
        self.assertTrue(self.state()["move_check_pending"])
        self.assertFalse(self.state()["move_check_ready"])
        self.assertEqual(self.web.pop_action(), request)
        # Even after the camera takes the queued action, a second click cannot
        # request the same check before the next camera snapshot is published.
        self.assertEqual(self.post("/api/action", request)[0], 409)
        self.web.publish(board, "black", "Play", "", "Tracked", None,
                         move_check_pending=True, move_check_ready=True)
        self.assertFalse(self.state()["move_check_ready"])
        self.web.publish(board, "black", "Play", "", "Tracked", None,
                         move_check_ready=True, move_check_result="No move found.")
        self.assertEqual(self.state()["move_check_result"], "No move found.")
        self.assertEqual(self.post("/api/action", request)[0], 202)
        self.web.pop_action()
        finished = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.web.publish(finished, "black", "Game over", "", "Tracked", None, move_check_ready=True)
        self.assertFalse(self.state()["move_check_ready"])
        self.assertEqual(self.post("/api/action", {**request, "expected_fen": finished.fen()})[0], 409)

    def test_host_origin_csrf_and_body_limits(self):
        self.assertEqual(self.get("/api/state", {"Host": "evil.example"})[0], 403)
        self.assertEqual(self.get("/api/state", {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.post("/api/chat", {"question": "Hi"}, token="wrong")[0], 403)
        self.assertEqual(self.post("/api/chat", {"question": "Hi"}, origin="https://evil.example")[0], 403)
        self.assertEqual(self.post("/api/chat", {"question": "x" * 1501})[0], 400)

    def test_bounded_local_voice_routes(self):
        status, body, _ = self.post("/api/transcribe", b"RIFF" + b"a" * 40,
                                    content_type="audio/wav")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["transcript"], "What should I play?")
        connection = http.client.HTTPConnection("127.0.0.1", self.web.port, timeout=5)
        connection.putrequest("POST", "/api/transcribe")
        connection.putheader("Origin", self.web.url)
        connection.putheader("X-Coach-CSRF", self.token)
        connection.putheader("Content-Type", "audio/wav")
        connection.putheader("Content-Length", str(1024 * 1024 + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        connection.close()
        status, body, headers = self.post("/api/speak", {"text": "Play e4"})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "audio/wav")
        self.assertEqual(body, b"RIFFPlay e4")


if __name__ == "__main__":
    unittest.main()
