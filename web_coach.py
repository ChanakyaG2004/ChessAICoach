"""Loopback-only chat and control surface for the physical chess coach.

The HTTP server never changes the chess game. It snapshots published positions,
answers chat questions on one worker, and queues user actions for the camera
loop to validate and apply.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import secrets
import threading
from urllib.parse import parse_qs, urlsplit

import chess

from move_language import move_text, parse_move_text


WEB_FILES = Path(__file__).resolve().parent / "web"
MAX_CHAT_BYTES = 4096
MAX_CHAT_CHARS = 1500
MAX_SPEAK_CHARS = 3000
MAX_WAV_BYTES = 1024 * 1024
MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_MESSAGES = 100
ACTIONS = frozenset(("confirm", "resync", "undo", "new", "cancel",
                     "manual", "save", "promotion", "rook", "check"))


def _moves_and_pieces(board: chess.Board) -> tuple[list[str], dict[str, str]]:
    replay = board.root()
    moves = []
    for number, move in enumerate(board.move_stack, 1):
        if move not in replay.legal_moves:
            break
        moves.append(f"Move {number}: {move_text(replay, move, include_color=True)}")
        replay.push(move)
    pieces = {chess.square_name(square): piece.symbol()
              for square, piece in board.piece_map().items()}
    return moves[-24:], pieces


class CoachWebServer:
    """Serve the chat dashboard on 127.0.0.1 with a stale-answer guard."""

    def __init__(self, conversation, host="127.0.0.1", port=8765, speech=None):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("Coach web server must bind to loopback")
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer from 0 to 65535")
        if not callable(getattr(conversation, "answer", None)):
            raise ValueError("conversation must provide answer(question, board, human_color)")
        self.conversation = conversation
        self.speech = speech
        self.host = "127.0.0.1"
        self.port = port
        self.url = ""
        self._csrf = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._speech_lock = threading.Lock()
        self._jobs = queue.Queue(maxsize=4)
        self._actions = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._httpd = None
        self._http_thread = None
        self._worker = None
        self._board = chess.Board()
        self._expected_board = None
        self._human_color = "white"
        self._revision = 0
        self._image_revision = 0
        self._image_jpeg = None
        self._instruction = "Check the physical setup before making a move."
        self._feedback = ""
        self._detection_status = "Waiting for the camera"
        self._pending_action = None
        self._recovery_ready = False
        self._move_check_ready = False
        self._move_check_pending = False
        self._move_check_result = ""
        self._engine_status = ""
        self._messages = []
        self._next_message_id = 1

    def start(self):
        """Start HTTP and chat worker threads; returns self for convenience."""
        if self._httpd is not None:
            return self
        self._stop.clear()
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._http_thread = threading.Thread(target=self._httpd.serve_forever,
                                             name="coach-http", daemon=True)
        self._worker = threading.Thread(target=self._answer_loop,
                                        name="coach-chat", daemon=True)
        self._http_thread.start()
        self._worker.start()
        return self

    def close(self):
        """Stop accepting requests. A running engine answer may finish later."""
        self._stop.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
            self._http_thread = None
        if self._worker is not None:
            self._worker.join(timeout=20)
            self._worker = None

    def publish(self, board: chess.Board, human_color: str, instruction: str,
                feedback: str, detection_status: str, pending_action: str | None,
                image_jpeg: bytes | None = None, engine_status: str = "",
                recovery_ready: bool = False,
                expected_board: chess.Board | None = None,
                move_check_ready: bool = False, move_check_pending: bool = False,
                move_check_result: str = "") -> None:
        if not isinstance(board, chess.Board):
            raise TypeError("board must be a chess.Board")
        if expected_board is not None and not isinstance(expected_board, chess.Board):
            raise TypeError("expected_board must be a chess.Board or None")
        if human_color not in ("white", "black"):
            raise ValueError("human_color must be white or black")
        if image_jpeg is not None:
            if (not isinstance(image_jpeg, bytes) or len(image_jpeg) > MAX_JPEG_BYTES
                    or not image_jpeg.startswith(b"\xff\xd8")):
                raise ValueError("image_jpeg must be a JPEG of at most 2 MiB")
        snapshot = board.copy(stack=True)
        target = expected_board.copy(stack=True) if expected_board is not None else None
        with self._lock:
            position_changed = snapshot.fen() != self._board.fen()
            if (position_changed or
                    human_color != self._human_color):
                self._revision += 1
            if position_changed and feedback and feedback != self._feedback:
                self._append_message({"id": self._next_message_id, "role": "assistant",
                                      "text": str(feedback), "status": "done", "source": "game"})
                self._next_message_id += 1
            self._board = snapshot
            self._expected_board = target
            self._human_color = human_color
            self._instruction = str(instruction)
            self._feedback = str(feedback)
            self._detection_status = str(detection_status)
            self._pending_action = None if pending_action is None else str(pending_action)
            self._recovery_ready = bool(recovery_ready)
            self._move_check_pending = bool(move_check_pending)
            self._move_check_ready = (bool(move_check_ready) and not self._move_check_pending
                                      and pending_action is None and not board.is_game_over(claim_draw=False))
            self._move_check_result = str(move_check_result)
            self._engine_status = str(engine_status)
            if image_jpeg is not None:
                self._image_jpeg = image_jpeg
                self._image_revision += 1
            elif self._image_jpeg is not None:
                self._image_jpeg = None
                self._image_revision += 1

    def pop_action(self) -> dict | None:
        """Remove one requested action for the camera loop to revalidate."""
        try:
            return self._actions.get_nowait()
        except queue.Empty:
            return None

    def _speech_status(self) -> dict:
        if self.speech is None:
            return {"available": False, "tts_available": False,
                    "reason": "Local speech is not configured"}
        try:
            status = self.speech.status()
            return {"available": bool(status.get("available")),
                    "tts_available": bool(status.get("tts_available")),
                    "reason": str(status.get("reason", ""))}
        except Exception as exc:
            return {"available": False, "tts_available": False,
                    "reason": f"Speech service unavailable: {exc}"}

    def _state(self) -> dict:
        with self._lock:
            board = self._board.copy(stack=True)
            expected_board = (self._expected_board.copy(stack=True)
                              if self._expected_board is not None else board.copy(stack=True))
            human_color = self._human_color
            state = {
                "fen": board.fen(),
                "human_color": human_color,
                "turn": "white" if board.turn else "black",
                "in_check": board.is_check(),
                "game_over": board.is_game_over(claim_draw=False),
                "result": board.result(claim_draw=False),
                "instruction": self._instruction,
                "feedback": self._feedback,
                "detection_status": self._detection_status,
                "pending_action": self._pending_action,
                "recovery_ready": self._recovery_ready,
                "move_check_ready": self._move_check_ready,
                "move_check_pending": self._move_check_pending,
                "move_check_result": self._move_check_result,
                "engine_status": self._engine_status,
                "revision": self._revision,
                "image_revision": self._image_revision,
                "image_url": (f"/board.jpg?v={self._image_revision}&token={self._csrf}"
                              if self._image_jpeg is not None else None),
                "messages": list(self._messages),
                "busy": any(item.get("status") == "pending" for item in self._messages),
                "action_queued": not self._actions.empty(),
            }
        moves, pieces = _moves_and_pieces(board)
        state["moves"] = moves
        state["pieces"] = pieces
        state["expected_fen"] = expected_board.fen()
        state["expected_pieces"] = {
            chess.square_name(square): piece.symbol()
            for square, piece in expected_board.piece_map().items()
        }
        state["speech"] = self._speech_status()
        return state

    def _append_message(self, message: dict):
        self._messages.append(message)
        self._messages = self._messages[-MAX_MESSAGES:]

    def _answer_loop(self):
        try:
            while not self._stop.is_set():
                try:
                    message_id, question, board, human_color, revision, instruction, feedback = self._jobs.get(timeout=.25)
                except queue.Empty:
                    continue
                try:
                    context = getattr(self.conversation, "set_game_context", None)
                    if callable(context):
                        context(board, instruction, feedback)
                    answer = self.conversation.answer(question, board, human_color)
                    if not isinstance(answer, str):
                        answer = str(answer)
                    answer = answer.strip()[:6000] or "I could not form an answer for that position."
                    error = None
                except Exception as exc:
                    answer = "I could not answer right now. Please try again."
                    error = str(exc)[:300]
                with self._lock:
                    for message in self._messages:
                        if message["id"] != message_id:
                            continue
                        if revision != self._revision:
                            message["status"] = "stale"
                            message["text"] = ("The board changed while I was answering. "
                                               "Ask again for the current position.")
                        elif error is not None:
                            message["status"] = "error"
                            message["text"] = answer
                        else:
                            message["status"] = "done"
                            message["text"] = answer
                        break
                self._jobs.task_done()
        finally:
            # The worker owns the conversation so its engine cannot be closed
            # by the camera thread during an active answer.
            close = getattr(self.conversation, "close", None)
            if callable(close):
                close()

    def _handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "PhysicalChessCoach/1"
            sys_version = ""

            def log_message(self, _format, *_args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(15)

            def _headers(self, code: int, content_type: str, length: int):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; script-src 'self'; style-src 'self'; "
                                 "img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; "
                                 "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
                self.end_headers()

            def _send(self, code: int, body: bytes, content_type: str):
                self._headers(code, content_type, len(body))
                self.wfile.write(body)

            def _json(self, code: int, value):
                self._send(code, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            def _valid_request(self, post=False):
                if self.headers.get("Host") != f"127.0.0.1:{app.port}":
                    self._json(403, {"error": "Invalid host"})
                    return False
                origin = self.headers.get("Origin")
                if (origin is not None and origin != app.url) or (post and origin != app.url):
                    self._json(403, {"error": "Invalid origin"})
                    return False
                fetch_site = self.headers.get("Sec-Fetch-Site")
                if fetch_site not in (None, "none", "same-origin"):
                    self._json(403, {"error": "Cross-site request denied"})
                    return False
                if post and self.headers.get("X-Coach-CSRF") != app._csrf:
                    self._json(403, {"error": "Invalid CSRF token"})
                    return False
                return True

            def _body(self, limit: int) -> bytes | None:
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self._json(411, {"error": "Content-Length is required"})
                    return None
                if length < 1 or length > limit:
                    self._json(413, {"error": f"Body must be 1 to {limit} bytes"})
                    return None
                try:
                    body = self.rfile.read(length)
                except TimeoutError:
                    self._json(408, {"error": "Request body timed out"})
                    return None
                if len(body) != length:
                    self._json(400, {"error": "Incomplete request body"})
                    return None
                return body

            def _parse_json(self, limit: int):
                if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                    self._json(415, {"error": "Expected application/json"})
                    return None
                body = self._body(limit)
                if body is None:
                    return None
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    self._json(400, {"error": "Invalid JSON"})
                    return None
                if not isinstance(parsed, dict):
                    self._json(400, {"error": "Expected a JSON object"})
                    return None
                return parsed

            def do_GET(self):
                if not self._valid_request():
                    return
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    html = (WEB_FILES / "index.html").read_text(encoding="utf-8")
                    html = html.replace("{{CSRF_TOKEN}}", app._csrf)
                    self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/app.js":
                    self._send(200, (WEB_FILES / "app.js").read_bytes(),
                               "text/javascript; charset=utf-8")
                elif path == "/style.css":
                    self._send(200, (WEB_FILES / "style.css").read_bytes(),
                               "text/css; charset=utf-8")
                elif path == "/api/state":
                    self._json(200, app._state())
                elif path == "/board.jpg":
                    token = parse_qs(urlsplit(self.path).query).get("token", [None])[0]
                    if token != app._csrf:
                        self._json(403, {"error": "Invalid image token"})
                        return
                    with app._lock:
                        image = app._image_jpeg
                    if image is None:
                        self._json(404, {"error": "No board image yet"})
                    else:
                        self._send(200, image, "image/jpeg")
                else:
                    self._json(404, {"error": "Not found"})

            def do_POST(self):
                if not self._valid_request(post=True):
                    return
                path = urlsplit(self.path).path
                if path == "/api/chat":
                    value = self._parse_json(MAX_CHAT_BYTES)
                    if value is None:
                        return
                    question = value.get("question")
                    if (not isinstance(question, str) or
                            not 1 <= len(question.strip()) <= MAX_CHAT_CHARS):
                        self._json(400, {"error": "Question must be 1 to 1500 characters"})
                        return
                    with app._lock:
                        busy = app._jobs.full()
                        if not busy:
                            board = app._board.copy(stack=True)
                            human_color = app._human_color
                            revision = app._revision
                            user_id = app._next_message_id
                            reply_id = user_id + 1
                            app._next_message_id += 2
                            app._append_message({"id": user_id, "role": "user",
                                                 "text": question.strip(), "status": "done"})
                            app._append_message({"id": reply_id, "role": "assistant",
                                                 "text": "Thinking about this position...",
                                                 "status": "pending"})
                            app._jobs.put_nowait((reply_id, question.strip(), board,
                                                 human_color, revision, app._instruction, app._feedback))
                    if busy:
                        self._json(429, {"error": "Coach is busy; try again shortly"})
                        return
                    self._json(202, {"accepted": True, "message_id": reply_id})
                elif path == "/api/action":
                    value = self._parse_json(MAX_CHAT_BYTES)
                    if value is None:
                        return
                    action = value.get("action")
                    expected_fen = value.get("expected_fen")
                    if (not isinstance(action, str) or action not in ACTIONS or
                            not isinstance(expected_fen, str)):
                        self._json(400, {"error": "Invalid action or expected_fen"})
                        return
                    request = {"action": action, "expected_fen": expected_fen}
                    if action == "manual":
                        with app._lock:
                            board_for_move = app._board.copy(stack=True)
                        try:
                            move = parse_move_text(board_for_move, value.get("move"))
                        except ValueError as exc:
                            self._json(400, {"error": str(exc)})
                            return
                        request["move"] = move.uci()
                    if action == "promotion":
                        choice = value.get("choice")
                        if choice not in (1, 2, 3, 4):
                            self._json(400, {"error": "Promotion choice must be 1 to 4"})
                            return
                        request["choice"] = choice
                    with app._lock:
                        failure = None
                        if expected_fen != app._board.fen():
                            failure = (409, "Board changed; refresh and retry")
                        elif action == "confirm" and (
                                app._pending_action is None or not app._recovery_ready):
                            failure = (409, "Physical position is not ready to confirm")
                        elif action == "check" and not app._move_check_ready:
                            failure = (409, "Confirm the position and wait for a clear camera view or the current check to finish")
                        elif action == "manual" and move not in app._board.legal_moves:
                            failure = (400, "Manual move is not legal in this position")
                        elif action == "manual" and app._pending_action is not None:
                            failure = (409, "Finish the pending position check first")
                        elif app._actions.full():
                            failure = (429, "An action is already waiting for the camera")
                        else:
                            app._actions.put_nowait(request)
                            if action == "check":
                                app._move_check_ready = False
                                app._move_check_pending = True
                                app._move_check_result = "Checking your move… Keep your hand clear of the board."
                    if failure is not None:
                        self._json(failure[0], {"error": failure[1]})
                        return
                    self._json(202, {"accepted": True})
                elif path == "/api/transcribe":
                    if app.speech is None or not app._speech_status()["available"]:
                        self._json(503, {"error": "Local speech recognition is unavailable"})
                        return
                    if self.headers.get("Content-Type", "").split(";", 1)[0] != "audio/wav":
                        self._json(415, {"error": "Expected audio/wav"})
                        return
                    wav = self._body(MAX_WAV_BYTES)
                    if wav is None:
                        return
                    if not app._speech_lock.acquire(blocking=False):
                        self._json(429, {"error": "Speech service is busy"})
                        return
                    try:
                        transcript = app.speech.transcribe(wav)
                        self._json(200, {"transcript": str(transcript)[:MAX_CHAT_CHARS]})
                    except Exception as exc:
                        code = 400 if (isinstance(exc, ValueError) or
                                       exc.__class__.__name__ == "SpeechError") else 503
                        self._json(code, {"error": str(exc)[:300]})
                    finally:
                        app._speech_lock.release()
                elif path == "/api/speak":
                    if app.speech is None or not app._speech_status()["tts_available"]:
                        self._json(503, {"error": "Local speech output is unavailable"})
                        return
                    value = self._parse_json(MAX_CHAT_BYTES)
                    if value is None:
                        return
                    text = value.get("text")
                    if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_SPEAK_CHARS:
                        self._json(400, {"error": "Text must be 1 to 3000 characters"})
                        return
                    if not app._speech_lock.acquire(blocking=False):
                        self._json(429, {"error": "Speech service is busy"})
                        return
                    try:
                        wav = app.speech.synthesize(text.strip())
                        if not isinstance(wav, bytes) or len(wav) > 8 * 1024 * 1024:
                            raise ValueError("Speech output exceeded the size limit")
                        self._send(200, wav, "audio/wav")
                    except Exception as exc:
                        code = 400 if (isinstance(exc, ValueError) or
                                       exc.__class__.__name__ == "SpeechError") else 503
                        self._json(code, {"error": str(exc)[:300]})
                    finally:
                        app._speech_lock.release()
                else:
                    self._json(404, {"error": "Not found"})

        return Handler
