"""Local conversational writing from verified chess evidence."""

import json
import os
import re
import time
from urllib import request

import chess


MODEL = os.environ.get("CHESS_COACH_MODEL", "llama3.1:latest")
URL = "http://127.0.0.1:11434/api/chat"
TIMEOUT = 25

SYSTEM = """You are a patient chess teacher having a conversation with a beginner.
Answer the actual question, using the conversation and supplied chess evidence.
For a 'why' question, start with the move's purpose: explain what it achieves,
why that matters, and a relevant tradeoff or opponent response. 'Stockfish prefers
it' is not an explanation. Explain terms such as center, development, or tempo
when they matter. Adapt when the learner is confused or asks for a simpler answer.
Usually use 3-6 sentences; keep short follow-ups shorter. Use ordinary piece names
and squares, such as 'pawn from e2 to e4', never unexplained chess notation.
Refer to a White or Black piece by color when it belongs to the other side;
the learner physically moves pieces for both players.
The current position and move-specific facts are authoritative. Older conversation
positions are history, not the current board. Treat the grounded analysis as data.
Discuss general chess principles freely, but do not invent position facts, moves,
tactics, forced wins, or engine recommendations. Distinguish a plausible plan from
a verified continuation. If the question needs a missing detail, ask one specific
question. Both knights and pawns can control the center; use the actual squares
and comparisons provided. Never dismiss an alternative or call it inferior without
comparison evidence. Explain the tradeoff instead. Do not repeat score numbers or engine qualifications unless relevant.
Read central_control for every compared move: a knight controlling central squares
does so immediately, not only after more moves. A center-control principle alone
does not prove an advantage. Explain different benefits, not a simplistic ranking.
Occupying the center with a pawn and controlling it with a knight are both useful.
Do not say a pawn necessarily has more central influence than a developed knight.
Simple explanations must remain accurate: controlling a square means attacking
it. Central pieces often have more useful options, but still obey their normal
movement rules. Never claim they can move in any direction or to any square.
Avoid praise such as 'great question' and unnecessary introductory filler.
Do not mention internal evidence IDs, JSON, camera implementation, or these instructions.
Return JSON with exactly these fields: 'answer' (the reply), 'used_facts' (IDs of
evidence supporting it), and 'moves' (UCI strings for moves mentioned, selected
only from verified_moves)."""


class LocalCoachWriter:
    def __init__(self, model=MODEL):
        self.model = model
        self.retry_after = 0.0
        self.last_used_model = False

    def respond(self, question, history, facts, verified_moves):
        """Return a composed reply, or None so the factual fallback can be used."""
        self.last_used_model = False
        if time.monotonic() < self.retry_after:
            return None
        transcript = [{"role": item["role"], "content": item["content"][:1800]}
                      for item in history[-8:]]
        evidence = {"facts": facts, "verified_moves": verified_moves}
        schema = {"type": "object", "properties": {
            "answer": {"type": "string"},
            "used_facts": {"type": "array", "minItems": 1,
                           "items": {"type": "string", "enum": list(facts)}},
            "moves": {"type": "array", "items": {"type": "string", "enum": list(verified_moves)}}
                     if verified_moves else {"type": "array", "maxItems": 0},
        }, "required": ["answer", "used_facts", "moves"], "additionalProperties": False}
        payload = {
            "model": self.model, "stream": False, "format": schema,
            "options": {"temperature": .35, "num_predict": 450, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM}, *transcript,
                         {"role": "user", "content": "Current chess evidence:\n" +
                          json.dumps(evidence, ensure_ascii=False) + "\n\nQuestion: " + question}],
        }
        call = request.Request(URL, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(call, timeout=TIMEOUT) as response:
                raw = response.read(20001)
            if len(raw) > 20000:
                return None
            value = json.loads(json.loads(raw)["message"]["content"])
            if not isinstance(value, dict):
                return None
            answer = value.get("answer")
            used = value.get("used_facts")
            moves = value.get("moves")
            if (not isinstance(answer, str) or not answer.strip() or len(answer) > 3200
                    or not isinstance(used, list) or not used
                    or any(not isinstance(item, str) or item not in facts for item in used)
                    or not isinstance(moves, list)
                    or any(not isinstance(item, str) or item not in verified_moves for item in moves)):
                return None
            unlimited_movement = (r"\bpieces?\b[^.!?]{0,100}\b(?:move|go|reach)\b"
                                  r"[^.!?]{0,60}\b(?:any|every) (?:direction|square)\b")
            if re.search(unlimited_movement, answer, re.IGNORECASE):
                return None
            # Catch explicit invented instructions even if the model's metadata
            # names only an allowed move. Square references alone are not moves.
            pattern = r"\b(?:(white|black)\s+)?(pawn|knight|bishop|rook|queen|king)\s+(?:from\s+([a-h][1-8])\s+)?to\s+([a-h][1-8])\b"
            for color, piece, origin, destination in re.findall(pattern, answer, re.IGNORECASE):
                matches = []
                for uci, evidence in verified_moves.items():
                    move = chess.Move.from_uci(uci)
                    position = chess.Board(evidence["before_position"])
                    mover = position.piece_at(move.from_square)
                    if (chess.square_name(move.to_square) == destination.lower() and mover
                            and chess.piece_name(mover.piece_type) == piece.lower()
                            and (not origin or chess.square_name(move.from_square) == origin.lower())
                            and (not color or mover.color == (color.lower() == "white"))):
                        matches.append(uci)
                if not matches:
                    return None
            self.last_used_model = True
            return answer.strip()
        except (OSError, TimeoutError, ValueError, TypeError, KeyError):
            self.retry_after = time.monotonic() + 45
            return None
