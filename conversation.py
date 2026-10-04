"""Grounded text questions about the recorded physical chess game.

Stockfish supplies moves and scores. Local Ollama writes conversational replies
from verified position facts, analysis, and recent discussion. Call ``answer`` from a background worker so
camera capture remains responsive. The engine is started lazily on that worker.
"""

from dataclasses import dataclass
import json
import re
import threading
from urllib import error, request

import chess

from coach import (ChessCoach, CoachError, EngineUnavailable, MoveReview,
                   _factual_notes)
from move_language import move_text, parse_move_text
from coach_language import LocalCoachWriter, MODEL as OLLAMA_MODEL


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TIMEOUT = 6.0
MAX_QUESTION = 1500

DEFINITIONS = {
    "center": "The center is the group of squares d4, e4, d5, and e5. Controlling them means your pieces or pawns attack those squares, which can make it harder for an opponent to use them safely. Pawns and knights can both help control the center.",
    "development": "Development means bringing pieces from their starting squares into useful positions, so they can help control squares, defend, or attack.",
    "tempo": "A tempo is one turn used to do something useful. Making your opponent spend a turn responding can give you time to improve another piece.",
    "space": "Space is room for your pieces to move and work together. Advanced pawns can gain room and restrict the opponent, but they also leave squares behind them and cannot move back.",
    "checkmate": "Checkmate means the king is in check and has no legal escape; the game ends.",
    "stalemate": "Stalemate means the player to move has no legal move but is not in check. The game is a draw.",
    "check": "Check means a king is attacked. The next move must remove that attack.",
    "castling": "Castling moves the king two squares toward a rook, then places that rook next to the king. It is legal only when the path and king safety conditions are met.",
    "en passant": "En passant is a pawn capture available immediately after an adjacent enemy pawn advances two squares. The capturing pawn moves diagonally behind it.",
    "promotion": "A pawn reaching the farthest rank becomes a queen, rook, bishop, or knight. Choose the piece when the camera asks.",
    "fork": "A fork is one piece attacking two or more enemy pieces at the same time.",
    "pin": "A pin is an attack on a piece that cannot move freely without exposing a more valuable piece or king behind it.",
    "centipawn": "A centipawn is one hundredth of a pawn in an engine evaluation. Short engine searches make these values approximate.",
    "evaluation": "An engine evaluation estimates a position for one side. Positive favors that side; negative favors the opponent. It is not a win probability.",
    "legal move": "A legal move follows the piece's movement rules and does not leave that side's king in check.",
}


@dataclass(frozen=True)
class _Intent:
    name: str
    move_token: str | None = None
    term: str | None = None


@dataclass
class _FollowUp:
    fen: str
    before: chess.Board
    move: chess.Move
    review: MoveReview | None
    kind: str


def _side(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def _last_move(board: chess.Board) -> tuple[chess.Board, chess.Move] | None:
    if not board.move_stack:
        return None
    before = board.copy(stack=True)
    move = before.pop()
    return before, move


def _last_human_move(board: chess.Board, human_color: str) -> tuple[chess.Board, chess.Move] | None:
    human = human_color == "white"
    replay = board.root()
    found = None
    for move in board.move_stack:
        if replay.turn == human:
            found = (replay.copy(stack=True), move)
        replay.push(move)
    return found


def _move_token(text: str) -> str | None:
    natural = re.search(r"\b([a-h][1-8])\s+(?:to|into|->|→)\s+([a-h][1-8])\b",
                        text, re.IGNORECASE)
    if natural:
        return (natural.group(1) + natural.group(2)).lower()
    named = re.search(r"\b(?:king|queen|rook|bishop|knight|pawn)\s+"
                      r"(?:from\s+[a-h][1-8]\s+)?(?:to|on|takes|captures)\s+[a-h][1-8]\b",
                      text, re.IGNORECASE)
    if named:
        return named.group()
    uci = re.search(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", text, re.IGNORECASE)
    if uci:
        return uci.group().lower()
    san = re.search(
        r"\b(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b",
        text)
    return san.group() if san else None


def _parse_move(board: chess.Board, token: str | None) -> chess.Move | None:
    if not token:
        return None
    token = _move_token(token) or token
    token = token.strip().rstrip(".,?!")
    try:
        return parse_move_text(board, token)
    except ValueError:
        return None


def _definition_term(question: str) -> str | None:
    lower = question.lower()
    for term in sorted(DEFINITIONS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term) + r"\b", lower):
            return term
    return None


def _offline_intent(question: str) -> _Intent:
    lower = question.lower().strip()
    token = _move_token(question)
    if any(phrase in lower for phrase in ("what if", "how about", "consider ",
                                           "instead of", "analyze ", "analyse ")):
        return _Intent("candidate", token)
    if any(phrase in lower for phrase in ("should have", "should i have", "best instead",
                                           "preferred move", "what was better",
                                           "better than my last", "better than the last")):
        return _Intent("better")
    if "better move" in lower:
        return _Intent("suggest")
    if ("last" in lower or "just played" in lower or "previous move" in lower):
        return _Intent("last")
    if any(phrase in lower for phrase in ("best move", "suggest", "should i play",
                                           "what now", "next move", "what to play")) and not token:
        return _Intent("suggest")
    term = _definition_term(question)
    if term and not token and (any(phrase in lower for phrase in (
            "what is", "what's", "what does", "what do you mean", "don't understand",
            "do not understand", "define", "definition of")) or
            re.search(r"\bexplain\s+(?:the\s+)?" + re.escape(term) + r"\b", lower)):
        return _Intent("definition", term=term)
    if any(phrase in lower for phrase in ("more simply", "simpler", "simplify",
                                           "plain english", "one sentence", "shorter")):
        return _Intent("simplify", token)
    if re.search(r"\bwhy\b|\bexplain (?:that|it|this|the move)\b", lower):
        return _Intent("why", token)
    if "threat" in lower:
        return _Intent("threat")
    if any(phrase in lower for phrase in ("reply", "respond", "retaliat",
                                           "what can they", "opponent play",
                                           "opponent do", "they do next")):
        return _Intent("reply")
    if any(phrase in lower for phrase in ("how can i improve", "what should i focus on",
                                           "how do i improve", "tips for me", "plan")):
        return _Intent("learning")
    if any(phrase in lower for phrase in ("best move", "suggest", "should i play",
                                           "what now", "next move", "what to play")):
        return _Intent("suggest")
    term = _definition_term(question)
    if term and any(word in lower for word in ("what", "mean", "define", "explain", "how does")):
        return _Intent("definition", term=term)
    if token and (len(lower.split()) <= 3 or "move" in lower or "play" in lower):
        return _Intent("candidate", token)
    return _Intent("unknown")


def _move_description(board: chess.Board, move: chess.Move) -> str:
    """Translate a legal SAN/UCI move into a physical instruction."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return move.uci()
    start = chess.square_name(move.from_square)
    end = chess.square_name(move.to_square)
    if board.is_kingside_castling(move):
        rook_from = "h1" if piece.color else "h8"
        rook_to = "f1" if piece.color else "f8"
        return f"castle kingside: king {start} to {end}, rook {rook_from} to {rook_to}"
    if board.is_queenside_castling(move):
        rook_from = "a1" if piece.color else "a8"
        rook_to = "d1" if piece.color else "d8"
        return f"castle queenside: king {start} to {end}, rook {rook_from} to {rook_to}"
    name = chess.piece_name(piece.piece_type)
    action = f"move your {name} from {start} to {end}"
    if board.is_en_passant(move):
        action += " and capture en passant"
    elif board.is_capture(move):
        action += " and capture the piece there"
    if move.promotion:
        action += f", promoting to {chess.piece_name(move.promotion)}"
    return action


def _position_facts(board: chess.Board, move: chess.Move) -> list[str]:
    """Concrete consequences of one legal move; no claim of causal engine logic."""
    after = board.copy(stack=True)
    after.push(move)
    facts = list(_factual_notes(board, move, after))
    mover = not after.turn
    piece = after.piece_at(move.to_square)
    if piece and piece.color == mover:
        center = [chess.square_name(square) for square in (chess.D4, chess.E4,
                                                          chess.D5, chess.E5)
                  if square in after.attacks(move.to_square)]
        if center:
            facts.append("attacks central " + ", ".join(center))
    opened = []
    for square in chess.SQUARES:
        slider = board.piece_at(square)
        if (slider is None or slider.color != mover or
                slider.piece_type not in (chess.BISHOP, chess.QUEEN) or
                after.piece_at(square) != slider):
            continue
        if set(after.attacks(square)) - set(board.attacks(square)):
            opened.append((slider.piece_type,
                           f"{chess.piece_name(slider.piece_type)} on {chess.square_name(square)}"))
    if opened:
        opened.sort(key=lambda item: 0 if item[0] == chess.BISHOP else 1)
        facts.append("opens a line for " + " and ".join(item[1] for item in opened[:2]))
    return facts[:4]


def _explain_facts(facts) -> str:
    """Turn compact engine-review labels into readable coaching phrases."""
    phrases = []
    for fact in facts:
        phrase = fact.replace("pawn to center", "places a pawn in the center")
        phrase = phrase.replace("develops knight", "develops the knight")
        phrase = phrase.replace("develops bishop", "develops the bishop")
        phrase = phrase.replace("attacks central ", "controls central squares ")
        phrases.append(phrase)
    return "; ".join(phrases)


def _purpose_text(board: chess.Board, move: chess.Move) -> str:
    """Explain the practical meaning of verified move consequences."""
    after = board.copy(stack=True)
    after.push(move)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return ""
    facts = _position_facts(board, move)
    plain = move_text(board, move)
    sentences = []
    if move.promotion:
        sentences.append(f"The pawn becomes a {chess.piece_name(move.promotion)}, gaining that piece's movement and attacking options.")
    elif "pawn to center" in facts:
        sentences.append(f"The idea of {plain} is to claim space in the center, where pieces can influence more of the board.")
    elif any(fact.startswith("develops") for fact in facts):
        sentences.append(f"The idea of {plain} is to bring a piece off its starting square and into play.")
    elif board.is_castling(move):
        sentences.append("Castling repositions your king and brings the rook toward the center. Whether the king is safer depends on the surrounding pieces.")
    elif board.is_capture(move):
        sentences.append(f"{plain.capitalize()} removes an opposing piece. The useful question is what your opponent can recapture or threaten afterward.")
    else:
        sentences.append(f"To understand {plain}, look at what the piece can influence from its new square.")
    if facts:
        sentences.append("It " + _explain_facts(facts) + ".")
    if any(fact.startswith("opens a line") for fact in facts):
        sentences.append("That gives those pieces room to join the game; having a route out is useful even before you move them.")
    if piece.piece_type == chess.PAWN and not move.promotion:
        guarded = ", ".join(chess.square_name(square) for square in sorted(after.attacks(move.to_square)))
        if guarded:
            sentences.append(f"The pawn now guards {guarded}, making those squares harder for an opponent to occupy safely.")
        sentences.append("The tradeoff is that a pawn cannot move back, so the space you gain is a lasting commitment.")
    captures = [candidate for candidate in after.legal_moves
                if candidate.to_square == move.to_square and after.is_capture(candidate)]
    if captures:
        sentences.append("Your opponent has a legal capture of that piece: " +
                         move_text(after, captures[0], include_color=True) +
                         ". A legal capture is not necessarily a good trade; check the response before deciding.")
    return " ".join(sentences)


class ConversationCoach:
    """Answer typed questions from a board snapshot, without changing the game."""

    def __init__(self, engine_path=None, analysis_time: float = 0.3,
                 use_ollama: bool = True, engine_enabled: bool = True):
        if analysis_time <= 0 or not 0 < analysis_time < float("inf"):
            raise ValueError("analysis_time must be positive and finite")
        self.engine_path = engine_path
        self.analysis_time = analysis_time
        self.use_ollama = use_ollama
        self.engine_enabled = engine_enabled
        self._coach: ChessCoach | None = None
        self._context: _FollowUp | None = None
        self._writer = LocalCoachWriter() if use_ollama else None
        self._history = []
        self._verified_moves = {}
        self._game_context_fen = None
        self._game_instruction = ""
        self._game_feedback = ""
        self._closed = False
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._coach is not None:
                self._coach.close()
                self._coach = None

    def _engine(self) -> ChessCoach:
        if not self.engine_enabled:
            raise EngineUnavailable("engine analysis is disabled")
        if self._coach is None:
            self._coach = ChessCoach(self.engine_path, self.analysis_time)
        return self._coach

    def _ollama_intent(self, question: str, board: chess.Board) -> _Intent:
        """Use a local model only as an untrusted parser, never as the answerer."""
        if not self.use_ollama:
            return _Intent("unknown")
        legal = ", ".join(f"{board.san(m)}:{m.uci()}" for m in board.legal_moves)
        prompt = ("Map the user's chess question to JSON only: "
                  '{"intent":"candidate|better|why|last|reply|suggest|definition|unknown",'
                  '"move":"SAN or UCI if stated","term":"basic chess term if asked"}. '
                  "Do not answer, analyze, or invent a move. The position is "
                  f"{board.fen()}; legal moves are {legal[:2500]}. Question: {question}")
        payload = {"model": OLLAMA_MODEL, "stream": False, "format": "json",
                   "options": {"temperature": 0, "num_predict": 100},
                   "messages": [{"role": "user", "content": prompt}]}
        data = json.dumps(payload).encode("utf-8")
        call = request.Request(OLLAMA_URL, data=data, headers={
            "Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(call, timeout=OLLAMA_TIMEOUT) as response:
                raw = response.read(8193)
            if len(raw) > 8192:
                return _Intent("unknown")
            content = json.loads(raw).get("message", {}).get("content", "")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return _Intent("unknown")
            intent = parsed.get("intent")
            if intent not in ("candidate", "better", "why", "last", "reply", "suggest",
                              "definition", "learning"):
                return _Intent("unknown")
            move = parsed.get("move")
            term = parsed.get("term")
            move = move[:24] if isinstance(move, str) else None
            term = term.lower() if isinstance(term, str) and term.lower() in DEFINITIONS else None
            return _Intent(intent, move, term)
        except (OSError, TimeoutError, ValueError, TypeError, KeyError,
                error.URLError, error.HTTPError):
            return _Intent("unknown")

    def set_game_context(self, board: chess.Board, instruction: str, feedback: str):
        """Supply the displayed action from the same snapshot as the question."""
        with self._lock:
            self._game_context_fen = board.fen()
            self._game_instruction = str(instruction)[:1200]
            self._game_feedback = str(feedback)[:1200]

    def _remember_move(self, board, move):
        if move is not None and move in board.legal_moves:
            after = board.copy(stack=True)
            after.push(move)
            self._verified_moves[move.uci()] = {
                "move": move_text(board, move, include_color=True),
                "before_position": board.fen(),
                "consequences": _position_facts(board, move),
                "central_control": [chess.square_name(square) for square in
                                    (chess.D4, chess.E4, chess.D5, chess.E5)
                                    if square in after.attacks(move.to_square)],
                "occupies_center": chess.square_name(move.to_square)
                                    if move.to_square in (chess.D4, chess.E4, chess.D5, chess.E5) else None,
            }

    def _writing_facts(self, board, human_color, grounded, question):
        facts = {
            "teaching_rules": {"center": DEFINITIONS["center"],
                               "development": DEFINITIONS["development"],
                               "movement": "Being in the center does not change a piece's movement rules. A knight still moves in an L shape, a bishop diagonally, a rook along ranks and files, and a pawn forward with diagonal captures. Controlling a square means attacking it, not being able to move anywhere."},
            "current_position": {"fen": board.fen(), "side_to_move": _side(board.turn),
                                 "human_side": human_color,
                                 "pieces": {chess.square_name(s): f"{_side(p.color)} {chess.piece_name(p.piece_type)}"
                                            for s, p in board.piece_map().items()}},
            "analysis": grounded,
        }
        if self._game_context_fen == board.fen():
            facts["displayed_action"] = self._game_instruction
            facts["last_feedback"] = self._game_feedback
        context = self._context
        if context is not None and context.fen == board.fen():
            self._remember_move(context.before, context.move)
            facts["focus_move"] = {**self._verified_moves[context.move.uci()], "discussion": context.kind}
            facts["move_purpose"] = _purpose_text(context.before, context.move)
            if context.review is not None:
                self._remember_move(context.before, context.review.best_move)
                after = context.before.copy(stack=True)
                after.push(context.move)
                self._remember_move(after, context.review.reply_move)
                if context.review.best_move is not None:
                    facts["alternative_purpose"] = _purpose_text(context.before, context.review.best_move)
        lower = question.lower()
        if any(word in lower for word in ("instead", "rather", "first", "why not", "what about")):
            names = [name for name in ("pawn", "knight", "bishop", "rook", "queen", "king")
                     if re.search(r"\b" + name + r"\b", lower)]
            alternatives = []
            for name in names:
                candidates = [move for move in board.legal_moves
                              if chess.piece_name(board.piece_at(move.from_square).piece_type) == name]
                candidates.sort(key=lambda move: abs(chess.square_file(move.to_square) - 3.5) +
                                                abs(chess.square_rank(move.to_square) - 3.5))
                for move in candidates[:2]:
                    self._remember_move(board, move)
                    item = {"move": move_text(board, move, include_color=True),
                            "purpose": _purpose_text(board, move),
                            "central_control": self._verified_moves[move.uci()]["central_control"]}
                    if self.engine_enabled:
                        try:
                            review = self._engine().review_move(board, move)
                            item["comparison"] = review.summary
                            self._remember_move(board, review.best_move)
                            after = board.copy(stack=True)
                            after.push(move)
                            self._remember_move(after, review.reply_move)
                        except (CoachError, EngineUnavailable, OSError, TimeoutError):
                            item["comparison"] = "Engine comparison unavailable; do not rank this option."
                    alternatives.append(item)
            facts["requested_alternatives"] = alternatives or "No legal move by that named piece in the current position."
            if alternatives and context is not None and context.before.fen() == board.fen():
                focused = self._verified_moves[context.move.uci()]
                facts["comparison_distinction"] = {
                    "focus": focused,
                    "alternatives": alternatives,
                    "principle": "Occupying a central square, controlling central squares from elsewhere, and opening lines for other pieces are distinct benefits. A move can control the center immediately without occupying it. Compare the listed effects and checked engine results; do not invent an immediate threat or assume the alternative fails to control the center.",
                }
        return facts

    def answer(self, question: str, board: chess.Board, human_color: str) -> str:
        """Compose a contextual answer; retain factual answers when Ollama is unavailable."""
        with self._lock:
            self._verified_moves = {}
            grounded = self._grounded_answer(question, board, human_color)
            question = question.strip()[:MAX_QUESTION]
            if not question:
                return grounded
            snapshot = board.copy(stack=True)
            facts = self._writing_facts(snapshot, human_color, grounded, question)
            reply = (self._writer.respond(question, self._history, facts, self._verified_moves)
                     if self._writer is not None else None)
            answer = reply or grounded
            self._history.extend([
                {"role": "user", "content": f"[Recorded position: {snapshot.fen()}] {question}"},
                {"role": "assistant", "content": answer},
            ])
            self._history = self._history[-12:]
            return answer

    def _grounded_answer(self, question: str, board: chess.Board, human_color: str) -> str:
        """Answer a text question about a copy of the current recorded position."""
        if human_color not in ("white", "black"):
            raise ValueError("human_color must be white or black")
        if not isinstance(board, chess.Board) or not board.is_valid():
            raise ValueError("board must be a valid chess.Board")
        if not isinstance(question, str):
            raise ValueError("question must be text")
        with self._lock:
            if self._closed:
                raise RuntimeError("ConversationCoach is closed")
            snapshot = board.copy(stack=True)
            if self._context is not None and self._context.fen != snapshot.fen():
                self._context = None
            question = question.strip()[:MAX_QUESTION]
            if not question:
                return "Ask about a move, a better option, the opponent's reply, or a chess term."
            intent = _offline_intent(question)
            if intent.name == "unknown":
                intent = self._ollama_intent(question, snapshot)
            if intent.name == "definition":
                return DEFINITIONS.get(intent.term or "", "Which chess term would you like explained?")
            try:
                if intent.name == "candidate":
                    return self._candidate(intent.move_token, snapshot)
                if intent.name == "last":
                    return self._last(snapshot, human_color, question)
                if intent.name == "better":
                    return self._better(snapshot, human_color)
                if intent.name == "suggest":
                    return self._suggest(snapshot)
                if intent.name == "why":
                    return self._why(snapshot, human_color, intent.move_token)
                if intent.name == "simplify":
                    explanation = self._why(snapshot, human_color, intent.move_token)
                    count = 1 if "one sentence" in question.lower() else 2
                    return ". ".join(explanation.split(". ")[:count]).rstrip(".") + "."
                if intent.name == "reply":
                    return self._reply(snapshot, human_color)
                if intent.name == "threat":
                    return self._threats(snapshot, human_color)
                if intent.name == "learning":
                    return self._learning(snapshot, human_color)
            except (CoachError, EngineUnavailable, OSError, TimeoutError) as exc:
                return (f"Stockfish analysis is unavailable ({exc}). I can still explain "
                        "legal moves and basic rules; the recorded game is unchanged.")
            return ("I can explain your last move, compare a better move, check a legal "
                    "candidate such as 'knight to f3', show the opponent's likely reply, "
                    "or define a chess term.")

    def _candidate(self, token: str | None, board: chess.Board) -> str:
        if not token:
            return "Name the move you want to explore, such as 'knight from g1 to f3'."
        move = _parse_move(board, token)
        if move is None or move not in board.legal_moves:
            return (f"{token} is not a legal move for {_side(board.turn)} in this position. "
                    "The recorded board has not changed.")
        plain = move_text(board, move)
        after = board.copy(stack=True)
        after.push(move)
        notes = _factual_notes(board, move, after)
        if not self.engine_enabled:
            self._context = _FollowUp(board.fen(), board.copy(stack=True), move, None, "candidate")
            detail = f" It {_explain_facts(notes)}." if notes else ""
            return (f"Hypothetical {plain} is legal for {_side(board.turn)}.{detail} "
                    "Engine analysis is disabled; the recorded board has not changed.")
        review = self._engine().review_move(board, move)
        self._context = _FollowUp(board.fen(), board.copy(stack=True), move, review, "candidate")
        return f"Hypothetical only: {review.summary} The recorded board has not changed."

    def _last(self, board: chess.Board, human_color: str, question: str) -> str:
        wanted = (_last_human_move(board, human_color) if
                  any(part in question.lower() for part in ("my ", "i ", "human"))
                  else _last_move(board))
        if wanted is None:
            return "No recorded move is available yet."
        before, move = wanted
        plain = move_text(before, move)
        after = before.copy(stack=True)
        after.push(move)
        notes = _factual_notes(before, move, after)
        factual = f"The last {_side(before.turn)} move was {plain}."
        self._context = _FollowUp(board.fen(), before, move, None, "last")
        if notes:
            factual += f" It {_explain_facts(notes)}."
        if not self.engine_enabled or before.turn != (human_color == "white"):
            return factual
        review = self._engine().review_move(before, move)
        self._context = _FollowUp(board.fen(), before, move, review, "last")
        return factual + " " + review.summary

    def _better(self, board: chess.Board, human_color: str) -> str:
        wanted = _last_human_move(board, human_color)
        if wanted is None:
            return self._suggest(board)
        before, move = wanted
        if not self.engine_enabled:
            return "Engine comparisons are disabled. Ask about a legal candidate or chess rule."
        review = self._engine().review_move(before, move)
        self._context = _FollowUp(board.fen(), before, move, review, "review")
        if review.best_move == move:
            return (f"For your last move, {move_text(before, move)}, Stockfish found that move best "
                    "in its short search. " + review.summary)
        if review.best_move is None:
            return (f"Your last {move_text(before, move)} was recorded. Stockfish returned a score "
                    "but no legal alternative move in this search.")
        loss = (f" The estimated difference is about {review.centipawn_loss} centipawns."
                if review.centipawn_loss is not None else "")
        return (f"For your last {move_text(before, move)}, Stockfish preferred "
                f"{_move_description(before, review.best_move)}.{loss} "
                "Ask why to compare the continuations.")

    def _suggest(self, board: chess.Board) -> str:
        if not self.engine_enabled:
            return "Engine suggestions are disabled. You can still ask about legal moves or rules."
        suggestion = self._engine().suggest_move(board)
        if suggestion is None:
            return "The recorded game is over; there is no next legal move."
        self._context = _FollowUp(board.fen(), board.copy(stack=True), suggestion.move,
                                  None, "suggestion")
        facts = _position_facts(board, suggestion.move)
        fact_text = " It " + _explain_facts(facts[:3]) + "." if facts else ""
        line = ""
        after = board.copy(stack=True)
        after.push(suggestion.move)
        if not after.is_game_over():
            line = self._pv_san(after, 3)
        continuation = f" One engine continuation is {line}." if line else ""
        return (f"For {_side(board.turn)} to move: "
                f"{_move_description(board, suggestion.move)}. "
                f"Stockfish's short search evaluates it at {suggestion.evaluation.text} "
                f"for {_side(board.turn)}.{fact_text}{continuation} "
                "This is analysis only; the physical board has not changed.")

    def _why(self, board: chess.Board, human_color: str, token: str | None) -> str:
        if token:
            candidate = _parse_move(board, token)
            if candidate is not None and candidate in board.legal_moves:
                self._candidate(token, board)
            else:
                return f"{token} is not legal in the current position; name another move."
        context = self._context
        if context is None:
            displayed = (_parse_move(board, _move_token(self._game_instruction))
                         if self._game_context_fen == board.fen() and
                         self._game_instruction.startswith("Coach's move:") else None)
            if displayed is not None:
                context = _FollowUp(board.fen(), board.copy(stack=True), displayed, None, "suggestion")
                self._context = context
        if context is None:
            wanted = _last_human_move(board, human_color)
            if wanted is None:
                return "Which move should I explain? Say something like 'knight to f3'."
            before, move = wanted
            review = self._engine().review_move(before, move) if self.engine_enabled else None
            context = _FollowUp(board.fen(), before, move, review, "review")
            self._context = context
        before, move, review = context.before, context.move, context.review
        plain = move_text(before, move)
        if context.kind == "suggestion":
            after = before.copy(stack=True)
            after.push(move)
            facts = _position_facts(before, move)
            line = self._pv_san(after, 3) if self.engine_enabled else ""
            text = _purpose_text(before, move)
            if line:
                text += f" After it, one short engine line is {line}."
            return text
        if review is None:
            after = before.copy(stack=True)
            after.push(move)
            return (_purpose_text(before, move) + " " +
                    "Stockfish is disabled, so I cannot compare its alternatives.")
        mover = _side(before.turn)
        if review.best_move == move:
            text = (f"For {mover}, {plain} was Stockfish's top choice in this short search. "
                    f"The evaluation was {review.before.text} before the move "
                    f"and {review.after.text} after it.")
        elif review.best_move is not None:
            best_plain = move_text(before, review.best_move)
            text = (f"For {mover}, Stockfish preferred {best_plain} to {plain}. "
                    f"The evaluation changed from {review.before.text} before that move "
                    f"to {review.after.text} after it.")
            if review.centipawn_loss is not None:
                text += f" That is about {review.centipawn_loss} centipawns lower."
        else:
            text = f"The engine returned no alternative for {plain}."
        text = _purpose_text(before, move) + " " + text
        if review.notes:
            text += f" That move {'; '.join(review.notes)}."
        if review.best_move is not None and review.best_move != move:
            facts = _position_facts(before, review.best_move)
            if facts:
                text += f" The suggested move {'; '.join(facts)}."
        if review.reply_move:
            after_played = before.copy(stack=True)
            after_played.push(move)
            text += (f" After {plain}, the engine's next reply is "
                     f"{move_text(after_played, review.reply_move)}.")
        if review.best_move is not None and review.best_move != move and self.engine_enabled:
            after_best = before.copy(stack=True)
            after_best.push(review.best_move)
            line = self._pv_san(after_best, 3)
            if line:
                text += f" After {best_plain}, one short engine line is {line}."
        if not review.notes and review.best_move is None:
            text += " This short search does not identify a specific tactical cause."
        return text

    def _pv_san(self, board: chess.Board, max_plies: int = 3) -> str:
        """Render legal moves in plain language from a bounded engine line."""
        if board.is_game_over(claim_draw=False):
            return ""
        info = self._engine()._analyze(board)
        replay = board.copy(stack=True)
        names = []
        for move in (info.get("pv") or [])[:max_plies]:
            if move not in replay.legal_moves:
                break
            self._remember_move(replay, move)
            names.append(move_text(replay, move, include_color=True))
            replay.push(move)
        return "; ".join(names)

    def _reply(self, board: chess.Board, human_color: str) -> str:
        human = human_color == "white"
        if self._context is not None and self._context.kind in ("candidate", "suggestion"):
            source = self._context.before.copy(stack=True)
            move = self._context.move
            original = f"hypothetical {move_text(source, move)}"
            source.push(move)
        elif board.turn != human:
            source = board.copy(stack=True)
            original = "the current position"
        else:
            return ("It is your turn. Ask about a candidate move or the best move first, "
                    "then ask about the opponent's reply after it.")
        if source.is_game_over(claim_draw=False):
            return f"After {original}, the game is over; there is no opponent reply."
        if not self.engine_enabled:
            return "Engine reply analysis is disabled. The recorded board has not changed."
        suggestion = self._engine().suggest_move(source)
        if suggestion is None:
            return f"After {original}, the game is over; there is no opponent reply."
        after = source.copy(stack=True)
        self._remember_move(source, suggestion.move)
        after.push(suggestion.move)
        facts = _factual_notes(source, suggestion.move, after)
        forcing = [fact for fact in facts if ("captures" in fact or "check" in fact
                                            or "promotes" in fact)]
        result = (f"After {original}, Stockfish's strongest {_side(source.turn)} reply "
                  f"is to {_move_description(source, suggestion.move)}.")
        if after.is_checkmate():
            result += " That is checkmate."
        elif forcing:
            result += f" It {'; '.join(forcing)}."
        line = self._pv_san(source, 3)
        if line:
            result += f" Short line: {line}."
        return result

    def _threats(self, board: chess.Board, human_color: str) -> str:
        human = human_color == "white"
        if board.is_game_over():
            return "The game is over; there is no next move to defend against."
        if board.is_check():
            king = board.king(board.turn)
            attackers = ", ".join(chess.square_name(s) for s in board.checkers())
            return (f"{_side(board.turn)}'s king on {chess.square_name(king)} is in check "
                    f"from {attackers}. The next move must remove that check.")
        if board.turn != human and self.engine_enabled:
            return self._reply(board, human_color) if self._context is None else self._current_opponent_reply(board)
        targets = []
        for square, piece in board.piece_map().items():
            if piece.color != human or piece.piece_type == chess.KING:
                continue
            attackers = board.attackers(not human, square)
            if attackers:
                named = ", ".join(f"{chess.piece_name(board.piece_at(s).piece_type)} on {chess.square_name(s)}"
                                  for s in sorted(attackers))
                targets.append((piece.piece_type, f"Your {chess.piece_name(piece.piece_type)} on "
                                f"{chess.square_name(square)} is attacked by {named}."))
        if targets:
            targets.sort(reverse=True)
            return (" ".join(text for _, text in targets[:3]) +
                    " Check the defenders and king safety before deciding whether to move or capture. "
                    "Ask about a candidate move to test the opponent's reply.")
        return ("No opponent piece currently attacks one of your pieces. "
                "Check their possible checks and captures after your intended move. "
                "Ask for the best move, then ask about the reply.")

    def _current_opponent_reply(self, board: chess.Board) -> str:
        suggestion = self._engine().suggest_move(board)
        if suggestion is None:
            return "The game is over."
        after = board.copy(stack=True)
        self._remember_move(board, suggestion.move)
        after.push(suggestion.move)
        facts = _position_facts(board, suggestion.move)
        detail = f" It {_explain_facts(facts)}." if facts else ""
        return (f"In the current position, {_side(board.turn)} can "
                f"{_move_description(board, suggestion.move)}.{detail}")

    def _learning(self, board: chess.Board, human_color: str) -> str:
        if board.turn == (human_color == "white") and board.is_check():
            return ("Your king is in check. Start by finding a legal way to remove the "
                    "check, then examine the opponent's strongest reply.")
        suggestion = self._suggest(board) + "\n\n" if self.engine_enabled and not board.is_game_over() else ""
        return suggestion + ("Before moving, look for checks, captures, and direct threats; "
                "check whether your king stays safe; develop your minor pieces; "
                "then consider the opponent's strongest reply. Ask about a specific "
                "legal move for position-based feedback.")
