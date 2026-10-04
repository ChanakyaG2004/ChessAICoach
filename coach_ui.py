"""Compact operator view for the physical chess coach's OpenCV window."""

import re

import chess
import cv2
import numpy as np

from move_language import move_text


WIDTH, HEIGHT = 1280, 760
BOARD_X, BOARD_Y, BOARD_SIZE = 48, 70, 620
PANEL_X, PANEL_WIDTH = 700, 540
FONT = cv2.FONT_HERSHEY_SIMPLEX

BACKGROUND = (27, 29, 25)
PANEL = (39, 42, 38)
CARD = (52, 55, 49)
WHITE = (237, 238, 231)
MUTED = (170, 183, 175)
ACCENT = (105, 211, 139)
AMBER = (98, 206, 245)


def _plain(value: str) -> str:
    """OpenCV's Hershey font is ASCII-only; normalize before measuring text."""
    value = str(value).replace("→", "->").replace("×", "x").replace("—", "-")
    return re.sub(r"\s+", " ", value.encode("ascii", "replace").decode("ascii")).strip()


def _width(value: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(value, FONT, scale, thickness)[0][0]


def _wrapped(value: str, max_width: int, scale: float, max_lines: int,
             thickness: int = 1) -> list[str]:
    """Pixel-aware wrapping with an ellipsis if the allotted lines fill up."""
    remaining = _plain(value)
    if not remaining:
        return []
    lines = []
    for _ in range(max_lines):
        if not remaining:
            break
        if _width(remaining, scale, thickness) <= max_width:
            lines.append(remaining)
            remaining = ""
            break
        words = remaining.split(" ")
        line = ""
        consumed = 0
        for word in words:
            candidate = f"{line} {word}" if line else word
            if _width(candidate, scale, thickness) <= max_width:
                line = candidate
                consumed += len(word) + (1 if consumed else 0)
            else:
                break
        if not line:
            # An unbroken path or FEN field can be wider than the panel.
            consumed = 1
            while (consumed < len(remaining) and
                   _width(remaining[:consumed + 1], scale, thickness) <= max_width):
                consumed += 1
            line = remaining[:consumed]
        lines.append(line)
        remaining = remaining[consumed:].lstrip()
    if remaining and lines:
        last = lines[-1]
        while last and _width(last + "...", scale, thickness) > max_width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    return lines


def _draw_lines(canvas: np.ndarray, text: str, x: int, y: int, max_width: int,
                max_lines: int, scale: float = 0.57, line_height: int = 25,
                color=WHITE, thickness: int = 1) -> None:
    for index, line in enumerate(_wrapped(text, max_width, scale, max_lines, thickness)):
        cv2.putText(canvas, line, (x, y + index * line_height), FONT, scale,
                    color, thickness, cv2.LINE_AA)


def _card(canvas: np.ndarray, top: int, height: int, heading: str,
          body: str, *, emphasis: bool = False) -> None:
    left, right = PANEL_X + 17, PANEL_X + PANEL_WIDTH - 17
    cv2.rectangle(canvas, (left, top), (right, top + height), CARD, -1)
    cv2.rectangle(canvas, (left, top), (left + 4, top + height),
                  ACCENT if emphasis else MUTED, -1)
    text_x = left + 18
    cv2.putText(canvas, heading.upper(), (text_x, top + 23), FONT, 0.46,
                ACCENT if emphasis else MUTED, 1, cv2.LINE_AA)
    scale = 0.69 if emphasis else 0.57
    line_height = 29 if emphasis else 24
    max_lines = max(1, (height - 43) // line_height)
    _draw_lines(canvas, body, text_x, top + 55, right - text_x - 16,
                max_lines, scale, line_height, WHITE, 1)


def _move_history(board: chess.Board) -> str:
    replay = board.root()
    labels = []
    for number, move in enumerate(board.move_stack, 1):
        if move not in replay.legal_moves:
            return "Move history unavailable"
        labels.append(f"Move {number}: {move_text(replay, move, include_color=True)}")
        replay.push(move)
    return " ".join(labels[-10:]) if labels else "No moves yet"


def render_coach_view(warped: np.ndarray | None, board: chess.Board,
                      human_color: str, detection_status: str, instruction: str,
                      feedback: str, engine_status: str, save_path: str,
                      pending_action: str | None = None) -> np.ndarray:
    """Render a readable BGR dashboard without mutating the game or source image."""
    if human_color not in ("white", "black"):
        raise ValueError("human_color must be 'white' or 'black'")
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    cv2.rectangle(canvas, (PANEL_X, 19),
                  (PANEL_X + PANEL_WIDTH, HEIGHT - 19), PANEL, -1)
    cv2.putText(canvas, "PHYSICAL CHESS COACH", (BOARD_X, 36), FONT, 0.83,
                WHITE, 2, cv2.LINE_AA)

    if warped is None:
        cv2.rectangle(canvas, (BOARD_X, BOARD_Y),
                      (BOARD_X + BOARD_SIZE, BOARD_Y + BOARD_SIZE), CARD, -1)
        _draw_lines(canvas, "Waiting for a reliable board image", BOARD_X + 65,
                    BOARD_Y + BOARD_SIZE // 2, BOARD_SIZE - 100, 2, 0.76, 35)
    else:
        image = cv2.resize(warped, (BOARD_SIZE, BOARD_SIZE), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        canvas[BOARD_Y:BOARD_Y + BOARD_SIZE, BOARD_X:BOARD_X + BOARD_SIZE] = image
    cv2.rectangle(canvas, (BOARD_X, BOARD_Y),
                  (BOARD_X + BOARD_SIZE, BOARD_Y + BOARD_SIZE), ACCENT, 2)

    step = BOARD_SIZE / 8
    ranks = "87654321" if human_color == "black" else "12345678"
    files = "abcdefgh" if human_color == "black" else "hgfedcba"
    for index in range(8):
        cv2.putText(canvas, ranks[index],
                    (BOARD_X - 25, int(BOARD_Y + (index + 0.55) * step)),
                    FONT, 0.59, WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, files[index],
                    (int(BOARD_X + (index + 0.44) * step), BOARD_Y + BOARD_SIZE + 26),
                    FONT, 0.59, WHITE, 1, cv2.LINE_AA)
    _draw_lines(canvas, f"Top side: {human_color}  |  Position: {board.fullmove_number}",
                BOARD_X, HEIGHT - 22, BOARD_SIZE, 1, 0.55, 24, MUTED)

    outcome = board.outcome(claim_draw=False)
    if outcome:
        headline = f"GAME OVER  {outcome.result()}"
        action = f"{outcome.termination.name.replace('_', ' ').title()}. Start a new game when ready."
    else:
        headline = f"{'WHITE' if board.turn else 'BLACK'} TO MOVE"
        if board.is_check():
            headline += " - CHECK"
        action = pending_action or instruction or "Make one move, clear your hand, then press Space to check."
    cv2.putText(canvas, headline, (PANEL_X + 18, 62), FONT, 0.77,
                AMBER if outcome else ACCENT, 2, cv2.LINE_AA)
    _card(canvas, 84, 112, "Next physical action", action, emphasis=True)
    _card(canvas, 210, 106, "Coach feedback", feedback or "Waiting for a move.")
    _card(canvas, 330, 74, "Engine", engine_status or "Engine unavailable")
    _card(canvas, 418, 88, "Board detection", detection_status or "Waiting for camera")
    _card(canvas, 520, 75, "Recent moves", _move_history(board))
    _card(canvas, 609, 72, "Current FEN", board.fen())
    _draw_lines(canvas, f"Saved: {save_path or 'not saved'}", PANEL_X + 18, 704,
                PANEL_WIDTH - 36, 1, 0.43, 20, MUTED)
    _draw_lines(canvas, "Space Check move  q Exit  c Corners  r Sync  u Undo  n New",
                PANEL_X + 18, 732, PANEL_WIDTH - 36, 1, 0.45, 20, WHITE)
    return canvas
