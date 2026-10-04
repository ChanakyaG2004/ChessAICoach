"""Stable image comparison and legal move matching; no visual piece recognition."""

from dataclasses import dataclass
import textwrap
import time

import chess
import cv2
import numpy as np

from move_language import move_text


ANALYSIS_SIZE = 512


@dataclass(frozen=True)
class Orientation:
    top_color: str

    def __post_init__(self):
        if self.top_color not in ("white", "black"):
            raise ValueError("top_color must be white or black")

    @property
    def human_color(self):
        return self.top_color

    @property
    def coach_color(self):
        return "black" if self.top_color == "white" else "white"

    def square(self, row, col):
        # White at bottom: top-left=a8. White at top: top-left=h1.
        return chess.square(7 - col, row) if self.top_color == "white" else chess.square(col, 7 - row)

    def cell(self, square):
        file, rank = chess.square_file(square), chess.square_rank(square)
        return (rank, 7 - file) if self.top_color == "white" else (7 - rank, file)


@dataclass
class Candidate:
    move: chess.Move
    san: str
    changed: frozenset[int]
    fit: float


@dataclass
class Match:
    candidates: list[Candidate]
    confidence: float
    reason: str
    move: chess.Move | None = None
    choices: tuple[chess.Move, ...] = ()


def changed_squares(board: chess.Board, move: chess.Move) -> frozenset[int]:
    """Compare the known piece maps; includes the captured pawn and castling rook."""
    after = board.copy(stack=False)
    after.push(move)
    return frozenset(square for square in chess.SQUARES
                     if board.piece_at(square) != after.piece_at(square))


def _rank_moves(board: chess.Board, evidence: np.ndarray) -> list[Candidate]:
    candidates = []
    for move in board.legal_moves:
        changed = changed_squares(board, move)
        expected = np.zeros(64, np.float32)
        expected[list(changed)] = 1
        fit = float(np.exp(-np.sum((evidence - expected) ** 2) / 2))
        candidates.append(Candidate(move, board.san(move), changed, fit))
    candidates.sort(key=lambda item: item.fit, reverse=True)
    return candidates


def match_legal_moves(board: chess.Board, scores: np.ndarray, threshold=0.07) -> Match:
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Change threshold must be positive and finite")
    scores = np.asarray(scores, np.float32)
    if scores.shape != (64,) or not np.isfinite(scores).all():
        raise ValueError("Expected 64 finite square scores indexed by chess square")
    median = float(np.median(scores))
    noise = float(np.median(np.abs(scores - median)))
    threshold = max(threshold, median + 6 * noise)
    # Camera noise can affect every square a little. Only changes above the
    # measured noise gate should contribute to the legal-move footprint.
    evidence = np.clip((scores - threshold) / (0.8 * threshold), 0, 1)
    candidates = _rank_moves(board, evidence)
    if not candidates:
        return Match([], 0.0, "No legal moves in current position")
    best = candidates[0]
    if float(scores.max()) < threshold:
        return Match(candidates, 0.0, "No significant change")
    # Explain a clear move by the wrong side without accepting it. This is
    # diagnostic only: the real board and its turn stay unchanged.
    other_board = board.copy(stack=False)
    other_board.turn = not board.turn
    other_board.ep_square = None
    other_candidates = _rank_moves(other_board, evidence)
    if other_candidates:
        other_best = other_candidates[0]
        alternatives = [item for item in other_candidates
                        if item.changed != other_best.changed]
        margin = other_best.fit - (alternatives[0].fit if alternatives else 0)
        current_fit = best.fit
        if (current_fit < 0.75 and other_best.fit >= 0.75 and margin >= 0.15
                and all(scores[square] >= threshold for square in other_best.changed)
                and not any(scores[square] >= threshold for square in chess.SQUARES
                            if square not in other_best.changed)):
            to_move = "White" if board.turn == chess.WHITE else "Black"
            return Match(candidates, other_best.fit,
                         f"Out of turn: {other_best.san} ({other_best.move.uci()}); "
                         f"{to_move} to move. Restore the last accepted position")
    if any(scores[square] < threshold for square in best.changed):
        return Match(candidates, best.fit, "Incomplete or weak move evidence")
    unexplained = [square for square in chess.SQUARES
                   if square not in best.changed and scores[square] >= threshold]
    if unexplained:
        # A small shadow or piece wobble can cross the absolute noise gate on
        # one square. Keep rejecting a second real move, but allow weak stray
        # evidence when both squares of one legal move changed much more.
        weakest_expected = min(float(scores[square]) for square in best.changed)
        weak_limit = min(threshold * 1.5, weakest_expected * .55)
        stray_excess = sum(float(scores[square]) - threshold for square in unexplained)
        expected_excess = weakest_expected - threshold
        if (best.fit < .8 or any(scores[square] > weak_limit for square in unexplained)
                or stray_excess > expected_excess * .5):
            names = ", ".join(chess.square_name(s) for s in unexplained[:5])
            return Match(candidates, best.fit, "Unexplained changes: " + names)
    alternatives = [item for item in candidates if item.changed != best.changed]
    margin = best.fit - (alternatives[0].fit if alternatives else 0)
    if best.fit < 0.75 or margin < 0.15:
        return Match(candidates, best.fit, "Ambiguous legal moves or weak visual fit")
    identical = tuple(item.move for item in candidates if item.changed == best.changed)
    if len(identical) > 1:
        return Match(candidates, best.fit, "Promotion: 1=queen 2=rook 3=bishop 4=knight",
                     choices=identical)
    # Moving the rook first can be an unfinished castle. Do not commit that
    # intermediate state as an ordinary rook move without confirmation.
    if any(best.changed < item.changed and board.is_castling(item.move) for item in alternatives):
        return Match(candidates, best.fit,
                     "Possible unfinished castle: finish it, or Enter confirms rook move",
                     choices=(best.move,))
    return Match(candidates, best.fit, "Unique legal match", move=best.move)


def prepare_image(image):
    return cv2.resize(image, (ANALYSIS_SIZE, ANALYSIS_SIZE), interpolation=cv2.INTER_AREA)


def align_image(reference, current):
    """Small residual affine alignment plus robust exposure correction."""
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    cur_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    matrix = np.eye(2, 3, dtype=np.float32)
    try:
        _, matrix = cv2.findTransformECC(
            ref_gray, cur_gray, matrix, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, 1e-4), None, 5)
    except cv2.error:
        return current, False
    # Larger changes require a fresh reliable board mapping, not stretching
    # the images to force a legal move match.
    if (np.max(np.abs(matrix[:, :2] - np.eye(2))) > 0.025 or
            np.max(np.abs(matrix[:, 2])) > 10):
        return current, False
    aligned = cv2.warpAffine(current, matrix, (ANALYSIS_SIZE, ANALYSIS_SIZE),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT_101)
    # Fit exposure on all squares, trimming changed pixels rather than allowing
    # a hand or moved piece to determine the brightness correction.
    x = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)[::4, ::4].astype(np.float32).ravel()
    y = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[::4, ::4].astype(np.float32).ravel()
    design = np.column_stack((x, np.ones_like(x)))
    keep = np.ones(len(x), bool)
    gain, offset = 1.0, 0.0
    for _ in range(3):
        gain, offset = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        residual = np.abs(y - (gain * x + offset))
        keep = residual <= max(3.0, float(np.percentile(residual, 75)))
    if not (0.65 <= gain <= 1.5 and abs(offset) <= 65):
        return current, False
    return np.clip(aligned.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8), True


def _score_squares(difference, orientation):
    scores = np.zeros(64, np.float32)
    step = ANALYSIS_SIZE // 8
    margin = 6  # Exclude the grid boundary, where small warp jitter is strongest.
    for row in range(8):
        for col in range(8):
            patch = difference[row * step + margin:(row + 1) * step - margin,
                               col * step + margin:(col + 1) * step - margin].ravel()
            count = max(1, int(len(patch) * 0.25))
            scores[orientation.square(row, col)] = np.mean(np.partition(patch, -count)[-count:])
    return scores


def square_scores(reference, current, orientation):
    """Raw appearance changes for detecting motion and occlusion."""
    first = cv2.GaussianBlur(reference, (3, 3), 0).astype(np.float32)
    second = cv2.GaussianBlur(current, (3, 3), 0).astype(np.float32)
    appearance = np.mean(np.abs(first - second), axis=2) / 255
    gray_a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    edge_a = cv2.Laplacian(gray_a, cv2.CV_32F, ksize=3)
    edge_b = cv2.Laplacian(gray_b, cv2.CV_32F, ksize=3)
    difference = 0.8 * appearance + 0.2 * np.minimum(np.abs(edge_a - edge_b) / 255, 1)
    return _score_squares(difference, orientation)


def move_scores(reference, current, orientation):
    """Compare changed contours relative to the detail visible on each square.

    Denoising and local averaging retain piece contours while suppressing sensor
    speckle on dark squares and broad cast shadows. Normalizing
    contour energy makes a small dark marker on a pale piece count when it
    appears or disappears, instead of diluting it across a quarter of a square.
    The energy floor prevents textureless squares from amplifying camera noise.
    Raw appearance still gates motion before this comparison is used.
    """
    def detail(image):
        gray = cv2.cvtColor(cv2.GaussianBlur(image, (7, 7), 0), cv2.COLOR_BGR2GRAY)
        brightness = np.log(gray.astype(np.float32) + 16)
        return brightness - cv2.GaussianBlur(brightness, (0, 0), 3)

    first, second = detail(reference), detail(current)
    scores = np.zeros(64, np.float32)
    step, margin = ANALYSIS_SIZE // 8, 6
    for row in range(8):
        for col in range(8):
            rows = slice(row * step + margin, (row + 1) * step - margin)
            cols = slice(col * step + margin, (col + 1) * step - margin)
            a, b = first[rows, cols], second[rows, cols]
            changed_energy = float(np.mean((a - b) ** 2))
            visible_energy = float(np.mean(a ** 2 + b ** 2))
            scores[orientation.square(row, col)] = .5 * changed_energy / max(visible_energy, .003)
    return scores


class MoveDetector:
    def __init__(self, fen=chess.STARTING_FEN, top_color="black", stable_seconds=0.9,
                 change_threshold=0.07, confirm_seconds=0.5):
        self.board = chess.Board(fen)
        if not self.board.is_valid():
            raise ValueError("The starting FEN must describe a valid chess position")
        if (not np.isfinite([stable_seconds, change_threshold, confirm_seconds]).all()
                or stable_seconds <= 0 or change_threshold <= 0 or confirm_seconds < 0):
            raise ValueError("Stability time and change threshold must be positive")
        self.orientation = Orientation(top_color)
        self.stable_seconds = stable_seconds
        self.change_threshold = change_threshold
        # Live high-contrast boards can have ~0.06 square-score jitter from
        # camera noise and subpixel warp changes, even after alignment. Motion
        # needs a slightly wider gate than the final legal-move evidence.
        self.motion_threshold = min(0.10, max(0.03, change_threshold * 1.15))
        self.confirm_seconds = confirm_seconds
        self.baseline = None
        self.anchor = None
        self.samples = []
        self.stable_since = 0.0
        self.stable = False
        self.status = "Waiting for reliable board and stable starting position"
        self.scores = np.zeros(64, np.float32)
        self.match = None
        self.pending = None
        self.pending_since = 0.0
        self.pending_image = None
        self.confirmed_choices = ()
        self.reliable = False
        self.check_pending = False
        self.check_started = 0.0
        self.check_result = ""
        self.last_move = ""
        self.last_confidence = 0.0
        self._last_log = None
        self._last_log_time = 0.0

    def _clear_pending(self):
        self.pending = None
        self.pending_image = None
        self.confirmed_choices = ()

    def request_check(self, now=None):
        """Check one completed physical move using fresh frames after the click."""
        if self.check_pending:
            raise ValueError("A move check is already in progress")
        if self.baseline is None:
            raise ValueError("Confirm the physical starting position first")
        if not self.reliable:
            raise ValueError("Clear the camera view before checking your move")
        if self.board.is_game_over(claim_draw=False):
            raise ValueError("The game is over; start a new game")
        self.cancel_check()
        self.check_pending = True
        self.check_started = time.monotonic() if now is None else now
        self.check_result = "Checking your move… Keep your hand clear of the board."
        self.status = self.check_result

    def cancel_check(self, reason=""):
        """Discard a request or stale special-move choice without changing the game."""
        self.check_pending = False
        self.anchor = None
        self.samples = []
        self.stable = False
        self.match = None
        self._clear_pending()
        self.check_result = reason
        if reason:
            self.status = reason

    def _failed_check(self):
        reason = self.match.reason
        if reason == "No significant change":
            message = "No move found. Finish one move, clear your hand, then press Check my move."
        elif reason.startswith("Out of turn"):
            side = "White" if self.board.turn else "Black"
            message = f"No move recorded: it is {side}'s turn. Restore the last recorded position and try again."
        else:
            message = ("Could not recognize one complete legal move. Check the pieces and camera view, "
                       "then press Check my move again.")
        self.check_pending = False
        self._clear_pending()
        self.check_result = message
        self.status = message + " (" + reason + ")"

    def update(self, warped, reliable=True, now=None):
        """Return a committed chess.Move once, only after stable, unique evidence."""
        now = time.monotonic() if now is None else now
        if self.check_pending and now - self.check_started >= max(8, self.stable_seconds * 3 + self.confirm_seconds):
            self.cancel_check("Move check timed out. Clear your hand, then press Check my move again.")
        if warped is None or not reliable:
            self.reliable = False
            if self.check_pending or self.confirmed_choices:
                self.cancel_check("Board view lost. Clear your hand, then press Check my move again.")
            self.anchor = None
            self.samples = []
            self.stable = False
            self.match = None
            self._clear_pending()
            self.status = "Paused: board geometry is held or lost"
            return None
        self.reliable = True
        # The live camera stays visible, but an idle detector never matches or
        # records a move. Special choices still monitor motion to stay valid.
        if self.baseline is not None and not self.check_pending and not self.confirmed_choices:
            self.status = self.check_result or "Waiting for Check my move"
            return None
        image = prepare_image(warped)
        reference = self.baseline if self.baseline is not None else self.anchor
        if reference is not None:
            image, aligned = align_image(reference, image)
            if not aligned:
                if self.check_pending or self.confirmed_choices:
                    self.cancel_check("Camera view changed. Clear the board view and press Check my move again; resync if the camera moved.")
                self.anchor = None
                self.samples = []
                self.stable = False
                self.match = None
                self._clear_pending()
                self.status = "Paused: alignment/exposure unreliable; clear the board view"
                return None
        if self.baseline is not None:
            self.scores = move_scores(self.baseline, image, self.orientation)
        motion = (float(square_scores(self.anchor, image, self.orientation).max())
                  if self.anchor is not None else float("inf"))
        if motion > self.motion_threshold:
            if self.confirmed_choices:
                self.cancel_check("Pieces changed before confirmation. Finish the move and press Check my move again.")
                return None
            self.anchor = image.copy()
            self.samples = [image]
            self.stable_since = now
            self.stable = False
            self.match = None
            self._clear_pending()
            self.status = "Moving / occluded: waiting for board to settle"
            return None
        if self.confirmed_choices and not self.check_pending:
            return None
        self.samples.append(image)
        self.samples = self.samples[-7:]
        self.stable = len(self.samples) >= 4 and now - self.stable_since >= self.stable_seconds
        if not self.stable:
            self.status = "Settling: collecting consistent frames"
            return None
        stable_image = np.median(np.stack(self.samples), axis=0).astype(np.uint8)
        if self.baseline is None:
            self.baseline = stable_image
            self.anchor = stable_image
            self.status = "Starting position recorded. Press Check my move after each physical move."
            print(self.status, flush=True)
            return None
        self.scores = move_scores(self.baseline, stable_image, self.orientation)
        self.match = match_legal_moves(self.board, self.scores, self.change_threshold)
        self.status = "Stable: " + self.match.reason
        self._log_match()
        choices = self.match.choices or ((self.match.move,) if self.match.move else ())
        signature = tuple(move.uci() for move in choices)
        if not choices:
            self._failed_check()
            return None
        if signature != self.pending:
            self.pending = signature
            self.pending_since = now
            self.confirmed_choices = ()
        self.pending_image = stable_image
        if now - self.pending_since < self.confirm_seconds:
            self.status = "Stable: verifying " + ", ".join(signature)
            return None
        if self.match.choices:
            self.confirmed_choices = self.match.choices
            self.check_pending = False
            self.check_result = ("Choose the piece your pawn promoted to below." if any(
                move.promotion for move in self.confirmed_choices) else
                "Finish castling and press Check my move again, or confirm this as a rook move below.")
            return None
        return self._commit(self.match.move, stable_image, self.match.confidence)

    def confirm(self, key):
        """Resolve only an explicitly displayed, stable ambiguity."""
        if not self.stable or not self.confirmed_choices or self.pending_image is None:
            return None
        promotion = {ord("1"): chess.QUEEN, ord("2"): chess.ROOK,
                     ord("3"): chess.BISHOP, ord("4"): chess.KNIGHT}.get(key)
        options = [move for move in self.confirmed_choices
                   if (promotion is not None and move.promotion == promotion)
                   or (key in (10, 13) and len(self.confirmed_choices) == 1 and not move.promotion)]
        if len(options) != 1:
            return None
        return self._commit(options[0], self.pending_image, self.match.confidence)

    def _commit(self, move, image, confidence):
        if move not in self.board.legal_moves:
            raise RuntimeError("Position changed before move confirmation")
        san = self.board.san(move)
        plain = move_text(self.board, move, include_color=True)
        self.board.push(move)
        self.baseline = image.copy()
        self.anchor = None
        self.samples = []
        self.stable = False
        self._clear_pending()
        self.check_pending = False
        self.check_result = f"Recorded: {plain}."
        self.last_move = f"{san} ({move.uci()})"
        self.last_confidence = confidence
        self.status = f"Accepted {self.last_move}; visual fit {confidence:.2f}"
        self.match = None
        print(self.status + "\nFEN: " + self.board.fen(), flush=True)
        return move

    def rebaseline(self, warped, fen=None):
        """Trust a user-confirmed physical position and replace its image baseline.

        This is an explicit recovery operation. It does not infer piece identity
        from pixels, so the caller must confirm that the board matches the FEN.
        """
        if warped is None:
            raise ValueError("A visible board is required to resync")
        if fen is not None:
            board = chess.Board(fen)
            if not board.is_valid():
                raise ValueError("The resync FEN must be a valid chess position")
            self.board = board
        image = prepare_image(warped)
        self.cancel_check()
        self.reliable = True
        self.baseline = image.copy()
        self.anchor = image.copy()
        self.samples = []
        self.stable_since = time.monotonic()
        self.stable = False
        self.scores.fill(0)
        self.match = None
        self._clear_pending()
        self.last_move = ""
        self.last_confidence = 0.0
        self._last_log = None
        self._last_log_time = 0.0
        self.status = "Position confirmed. Press Check my move after each physical move."
        print(self.status + "\nFEN: " + self.board.fen(), flush=True)

    def undo_and_rebaseline(self, warped):
        """Undo one recorded move after the user restores the physical board."""
        if not self.board.move_stack:
            raise ValueError("There is no recorded move to undo")
        previous = self.board.copy(stack=True)
        previous.pop()
        self.rebaseline(warped, previous.fen())
        self.board = previous

    def ranked_squares(self):
        return [(chess.square_name(int(square)), float(self.scores[square]))
                for square in np.argsort(-self.scores, kind="stable")]

    def _log_match(self):
        # Camera noise can reorder poor candidates on every frame. The live
        # dashboard stays current; throttle routine terminal diagnostics.
        important = (self.match.move is not None or bool(self.match.choices)
                     or self.match.reason.startswith("Out of turn"))
        signature = (self.match.reason if important else
                     self.match.reason.split(":", 1)[0])
        if signature == self._last_log:
            return
        now = time.monotonic()
        if not important and now - self._last_log_time < 5.0:
            return
        self._last_log = signature
        self._last_log_time = now
        print(self.status, flush=True)
        print("  Changed: " + ", ".join(f"{name}={score:.3f}" for name, score in self.ranked_squares()[:10]), flush=True)
        print("  Legal candidates: " + ", ".join(
            f"{item.san}/{item.move.uci()} fit={item.fit:.2f} squares="
            + "/".join(chess.square_name(s) for s in sorted(item.changed))
            for item in self.match.candidates[:5]), flush=True)

    def debug_view(self, warped):
        size, side = 640, 470
        board_image = (cv2.resize(warped, (size, size)) if warped is not None
                       else np.zeros((size, size, 3), np.uint8))
        canvas = np.zeros((size, size + side, 3), np.uint8)
        canvas[:, :size] = board_image
        cell = size // 8
        for row in range(8):
            for col in range(8):
                square = self.orientation.square(row, col)
                score = float(self.scores[square])
                x, y = col * cell, row * cell
                color = (0, 50, 255) if score >= self.change_threshold else (80, 200, 80)
                cv2.rectangle(canvas, (x, y), (x + cell - 1, y + cell - 1), color, 2)
                label = f"{chess.square_name(square)} {score:.2f}"
                cv2.putText(canvas, label, (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 0, 0), 3)
                cv2.putText(canvas, label, (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (255, 255, 255), 1)
        lines = [f"Human/top: {self.orientation.human_color}",
                 f"Coach/bottom: {self.orientation.coach_color}",
                 "To move: " + ("White" if self.board.turn else "Black"),
                 f"STABLE: {'yes' if self.stable else 'no'}", ""]
        # Wrap status so diagnostics are never silently clipped.
        lines.extend(textwrap.wrap(self.status, 49))
        lines.extend(["", "Top changed squares (all 64 ranked):"])
        ranked = self.ranked_squares()
        lines.extend("  ".join(f"{name}: {score:.3f}" for name, score in ranked[i:i + 2])
                     for i in range(0, 10, 2))
        lines.extend(["", "Legal candidates / visual fit:"])
        if self.match:
            lines.extend(f"{item.san} ({item.move.uci()}): {item.fit:.2f}"
                         for item in self.match.candidates[:5])
        lines.extend(["", "Last: " + (self.last_move or "none"),
                      f"Last confidence: {self.last_confidence:.2f} (heuristic)",
                      "Promotion: 1=Q 2=R 3=B 4=N", "q: quit"])
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (size + 12, 22 + index * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas
