"""Physical chess coach: detect legal moves, review them, and guide engine replies."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

# Some macOS OpenCV builds crash while loading a cached OpenCL kernel for SB.
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import chess
import cv2
import numpy as np

from board_detector import BoardDetector, BoardTracker
from coach import ChessCoach, EngineUnavailable
from coach_ui import render_coach_view
from conversation import ConversationCoach
from game_controller import GameController
from move_detector import MoveDetector
from move_language import move_text
from perspective import BOARD_SIZE, manual_corners, rotate_board, warp_board
from recovery import StableFrameGate
from session_store import load_game
from speech import LocalSpeech
from web_coach import CoachWebServer


CAMERA_WINDOW = "Camera - board outline"
COACH_WINDOW = "Physical Chess Coach"


def camera_point(value: str) -> tuple[float, float]:
    try:
        x, y = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use X,Y for each corner") from exc
    if not np.isfinite([x, y]).all():
        raise argparse.ArgumentTypeError("Corner coordinates must be finite")
    return x, y


def new_session_path(folder: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return folder / f"game_{stamp}.pgn"


def action_prompt(action: str, board: chess.Board) -> str:
    if action == "setup":
        return ("SETUP: Match every physical piece to Expected pieces. "
                "Clear your hand, then confirm the physical position or press Y.")
    if action == "resync":
        return ("RESYNC: Match every physical piece to Expected pieces. "
                "Clear your hand, then confirm the physical position or press Y.")
    if action == "undo":
        last = board.peek().uci() if board.move_stack else "the last move"
        return (f"UNDO {last}: Restore the position shown in Expected pieces. "
                "Clear your hand, then confirm the physical position or press Y.")
    if action == "new":
        return ("NEW GAME: Arrange the standard starting position. "
                "Clear your hand, then confirm the physical position or press Y.")
    raise ValueError(f"Unknown action: {action}")


def _camera_diagnostics(camera: cv2.VideoCapture) -> np.ndarray:
    ok, sample = camera.read()
    if not ok:
        raise RuntimeError("Webcam opened but did not return a frame.")
    started = time.perf_counter()
    for _ in range(30):
        ok, sample = camera.read()
        if not ok:
            raise RuntimeError("Webcam opened but did not return a frame.")
    measured_fps = 30 / (time.perf_counter() - started)
    sample_gray = cv2.cvtColor(cv2.resize(sample, (320, 180)), cv2.COLOR_BGR2GRAY)
    median_light = float(np.median(sample_gray))
    bright_end = float(np.percentile(sample_gray, 90))
    reported_fps = camera.get(cv2.CAP_PROP_FPS)
    if (median_light < 35 and bright_end < 70 and
            (reported_fps >= 45 or (reported_fps == 0 and measured_fps >= 45))):
        print("Very dark frames at 60 FPS; requesting 30 FPS for more exposure time.", flush=True)
        camera.set(cv2.CAP_PROP_FPS, 30)
        started = time.perf_counter()
        for _ in range(30):
            ok, sample = camera.read()
            if not ok:
                raise RuntimeError("Webcam opened but did not return a frame.")
        measured_fps = 30 / (time.perf_counter() - started)
    height, width = sample.shape[:2]
    reported_fps = camera.get(cv2.CAP_PROP_FPS)
    active_gray = cv2.cvtColor(cv2.resize(sample, (320, 180)), cv2.COLOR_BGR2GRAY)
    print(f"Camera resolution: {width}x{height}", flush=True)
    print(f"Camera FPS: {reported_fps:.1f} reported; {measured_fps:.1f} measured", flush=True)
    print(f"Camera brightness: median {np.median(active_gray):.0f}/255, "
          f"90th percentile {np.percentile(active_gray, 90):.0f}/255", flush=True)
    return sample


def save_debug_capture(project, controller, moves, frame, result, panel, warped, corners):
    """Keep the accepted reference with a failed view for reproducible diagnosis."""
    controller.save_now()
    folder = project / "debug_captures"
    folder.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    captures = {"camera": frame, "edges": result.edges,
                "candidates": result.candidates_view, "coach": panel,
                "board": warped, "baseline": moves.baseline,
                "comparison": moves.samples[-1] if moves.samples else None}
    for name, image in captures.items():
        if image is not None:
            cv2.imwrite(str(folder / f"{stamp}_{name}.png"), image)
    diagnostics = {"fen": moves.board.fen(), "status": moves.status,
                   "orientation": moves.orientation.top_color,
                   "scores": dict(moves.ranked_squares()),
                   "corners": corners.tolist() if corners is not None else None,
                   "reason": moves.match.reason if moves.match else moves.check_result}
    (folder / f"{stamp}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Saved game to {controller.save_path} and camera diagnostics to {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-color", choices=("white", "black"),
                        help="Color at the top of the rotated board (the human side)")
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--fen", help="Known physical position as a full FEN")
    position.add_argument("--resume", type=Path, help="Resume a saved PGN; arrange its final position")
    parser.add_argument("--save-pgn", type=Path, help="Save this game to a chosen PGN path")
    parser.add_argument("--engine-path", help="Stockfish executable; auto-detected by default")
    parser.add_argument("--analysis-time", type=float, default=0.25,
                        help="Seconds Stockfish spends on each position; default: 0.25")
    parser.add_argument("--no-engine", action="store_true", help="Run move detection without coaching")
    parser.add_argument("--chat-port", type=int, default=8765,
                        help="Local text and optional voice chat port; default: 8765")
    parser.add_argument("--no-chat", action="store_true", help="Disable the local browser chat")
    parser.add_argument("--no-language-model", action="store_true",
                        help="Use factual explanations and the built-in question parser without local Ollama")
    parser.add_argument("--assume-setup", action="store_true",
                        help="Skip physical setup confirmation when the board was already checked")
    parser.add_argument("--debug-windows", action="store_true",
                        help="Show edge, candidate, lighting, and square-score diagnostics")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index; default: 0")
    parser.add_argument("--stable-seconds", type=float, default=0.9)
    parser.add_argument("--change-threshold", type=float, default=0.07)
    parser.add_argument("--board-rotation", choices=("none", "cw", "ccw", "half"), default="none")
    parser.add_argument("--corners", nargs=4, type=camera_point, metavar="X,Y",
                        help="Optional four outer board corners in camera pixels")
    args = parser.parse_args()

    saved = None
    if args.resume:
        try:
            saved = load_game(args.resume)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.top_color is not None and args.top_color != saved.human_color:
            parser.error("--top-color differs from the resumed game's HumanColor")
        human_color = saved.human_color
        initial_fen = saved.initial_fen
    else:
        human_color = args.top_color
        if human_color is None:
            try:
                while human_color not in ("white", "black"):
                    human_color = input("Human color at the TOP [white/black]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                parser.error("Provide --top-color white or --top-color black")
        initial_fen = args.fen or chess.STARTING_FEN

    try:
        moves = MoveDetector(initial_fen, human_color, args.stable_seconds, args.change_threshold)
    except ValueError as exc:
        parser.error(str(exc))
    if saved:
        moves.board = saved.board
    if not np.isfinite(args.analysis_time) or args.analysis_time <= 0:
        parser.error("--analysis-time must be positive and finite")
    if not 0 <= args.chat_port <= 65535:
        parser.error("--chat-port must be between 0 and 65535 (0 chooses a free port)")

    engine = None
    if not args.no_engine:
        try:
            engine = ChessCoach(args.engine_path, args.analysis_time)
            print(f"Stockfish: {engine.engine_path or 'provided engine'}", flush=True)
        except EngineUnavailable as exc:
            print(f"Coaching unavailable: {exc}", flush=True)
    project = Path(__file__).resolve().parent
    save_path = args.save_pgn or args.resume or new_session_path(project / "games")
    controller = GameController(moves, initial_fen, human_color, save_path, engine)
    print(f"Human/top: {human_color}; coach/bottom: {moves.orientation.coach_color}", flush=True)
    print(f"Board rotation: {args.board_rotation}", flush=True)
    print("Expected physical FEN: " + moves.board.fen(), flush=True)
    print("Arrange that position before the image baseline is recorded.", flush=True)
    print(f"Game PGN: {controller.save_path}", flush=True)
    print("Keys: Space check move, q quit, c corners, r resync, u undo, n new, m manual move, s save/debug, "
          "1-4 promotion, Enter rook confirmation.", flush=True)

    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        controller.close()
        raise RuntimeError("Could not open webcam. Check macOS camera permission and other camera apps.")
    web = None
    conversation = None
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.set(cv2.CAP_PROP_FPS, 60):
            camera.set(cv2.CAP_PROP_FPS, 30)
        _camera_diagnostics(camera)

        if not args.no_chat:
            conversation = ConversationCoach(
                args.engine_path, max(.3, args.analysis_time),
                use_ollama=not args.no_language_model, engine_enabled=not args.no_engine)
            try:
                web = CoachWebServer(conversation, port=args.chat_port, speech=LocalSpeech())
                web.start()
            except OSError as exc:
                print(f"Could not open chat port {args.chat_port}: {exc}. "
                      "Retry with --chat-port 0 to choose a free port.", flush=True)
                raise
            print(f"Text coach: {web.url} (open in Codex's browser). Voice is optional and off.", flush=True)

        detector = BoardDetector()
        tracker = BoardTracker()
        calibrating = False
        manual_mode = False
        clicked_points = []
        pending_corners = args.corners
        pending_action = None if args.assume_setup else "setup"
        manual_buffer = ""
        manual_move = None
        calibration_status = ("" if args.assume_setup else
                              "Check physical setup against FEN, then press Y")
        recovery_gate = StableFrameGate()

        def camera_click(event, x, y, _flags, _param):
            nonlocal calibrating, pending_corners, calibration_status
            if event != cv2.EVENT_LBUTTONDOWN or not calibrating:
                return
            clicked_points.append((x, y))
            if len(clicked_points) == 4:
                pending_corners = clicked_points.copy()
                clicked_points.clear()
                calibrating = False
                calibration_status = "Checking clicked board corners"

        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW, 1280, 720)
        cv2.setMouseCallback(CAMERA_WINDOW, camera_click)
        cv2.namedWindow(COACH_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(COACH_WINDOW, 1280, 760)

        frame_number = 0
        last_web_publish = 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam opened but did not return a frame.")
            full_search = not manual_mode and (tracker.corners is None or frame_number % 10 == 0)
            result = detector.detect(frame, full_search=full_search)
            detection = None if manual_mode else result.corners
            if pending_corners is not None:
                try:
                    selected = manual_corners(pending_corners, frame.shape)
                except ValueError as exc:
                    calibration_status = f"Invalid corners: {exc}; press c to retry"
                    print(calibration_status, flush=True)
                else:
                    tracker = BoardTracker()
                    manual_mode = True
                    detection = selected
                    result.method = "manual corners"
                    result.reason = "OK"
                    calibration_status = "Board outline set; confirm the physical position"
                    pending_action = "setup" if moves.baseline is None else "resync"
                    recovery_gate.reset()
                    print(calibration_status, flush=True)
                pending_corners = None
            corners, state = tracker.update(frame, detection)
            if state == "lost" and not full_search and not manual_mode:
                result = detector.detect(frame)
                corners, state = tracker.update(frame, result.corners)
            frame_number += 1

            view = frame.copy()
            if calibrating:
                cv2.putText(view, f"Click 4 OUTER board corners: {len(clicked_points)}/4",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2, cv2.LINE_AA)
                for index, point in enumerate(clicked_points, 1):
                    cv2.circle(view, point, 7, (0, 255, 255), -1)
                    cv2.putText(view, str(index), (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
            elif corners is None:
                cv2.putText(view, "Press c to click the 4 outer board corners", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(view, "c: recalibrate corners", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2, cv2.LINE_AA)
            if calibration_status:
                cv2.putText(view, calibration_status[:65], (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2, cv2.LINE_AA)

            warped = None
            source = "No board detected"
            if corners is not None:
                color = ((0, 255, 0) if state == "live" else
                         (255, 220, 0) if state == "tracked" else (0, 180, 255))
                cv2.polylines(view, [corners.astype(np.int32)], True, color, 3)
                for label, point in zip(("TL", "TR", "BR", "BL"), corners):
                    x, y = point.astype(int)
                    cv2.circle(view, (x, y), 5, color, -1)
                    cv2.putText(view, label, (x + 7, y - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
                warped = rotate_board(warp_board(frame, corners), args.board_rotation)
                source = (result.method if state == "live" else
                          f"tracked: {tracker.feature_count} points" if state == "tracked" else
                          "held: last verified outline")
                cv2.putText(view, source, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(view, "NO BOARD: see camera setup", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)

            reliable = not calibrating and state in ("live", "tracked")
            if pending_action is None:
                accepted = moves.update(warped, reliable=reliable)
                if accepted is not None:
                    controller.on_accepted_move(accepted)
            else:
                moves.update(None, reliable=False)
                ready = recovery_gate.update(warped, reliable)
                moves.status = ("Paused: stable board ready for confirmation" if ready else
                                "Paused: waiting for a stable board")

            if pending_action == "manual":
                action = (f"Manual move: {move_text(moves.board, manual_move)} is legal. Press Y after the "
                          "board settles; Esc cancels." if manual_move else
                          f"Type a UCI move such as e2e4, then Enter. Input: {manual_buffer or '(empty)'}. "
                          "Esc cancels.")
            else:
                action = action_prompt(pending_action, moves.board) if pending_action else None
            engine_status = controller.engine_status
            if controller.storage_status.startswith("SAVE FAILED"):
                engine_status += "; " + controller.storage_status
            panel = render_coach_view(
                warped, moves.board, human_color, f"{source}: {moves.status}",
                controller.instruction, controller.feedback, engine_status,
                str(controller.save_path), action)
            if web is not None and time.monotonic() - last_web_publish >= .25:
                expected_board = moves.board.copy(stack=True)
                if pending_action == "undo" and expected_board.move_stack:
                    expected_board.pop()
                elif pending_action == "new":
                    expected_board = chess.Board()
                elif pending_action == "manual" and manual_move is not None:
                    expected_board.push(manual_move)
                jpeg = None
                if warped is not None:
                    encoded, payload = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if encoded:
                        jpeg = payload.tobytes()
                web.publish(moves.board, human_color, controller.instruction,
                            controller.feedback, f"{source}: {moves.status}" +
                            (f" {calibration_status}" if calibration_status else ""),
                            action, image_jpeg=jpeg, engine_status=engine_status,
                            recovery_ready=recovery_gate.ready(), expected_board=expected_board,
                            move_check_pending=moves.check_pending,
                            move_check_ready=(pending_action is None and reliable and
                                              moves.baseline is not None and not moves.check_pending and
                                              not moves.board.is_game_over(claim_draw=False)),
                            move_check_result=moves.check_result)
                last_web_publish = time.monotonic()
            cv2.imshow(CAMERA_WINDOW, view)
            cv2.imshow(COACH_WINDOW, panel)
            if args.debug_windows:
                cv2.imshow("Edges", result.edges)
                cv2.imshow("Hough lines", result.lines_view)
                cv2.imshow("Candidates and internal corners", result.candidates_view)
                lighting_view = result.lighting_view
                if lighting_view is None:
                    lighting_view = np.zeros_like(result.edges)
                    cv2.putText(lighting_view, "Lighting pass idle", (12, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, .62, 255, 2, cv2.LINE_AA)
                cv2.imshow("Lighting normalized", lighting_view)
                debug_board = warped if warped is not None else np.zeros(
                    (BOARD_SIZE, BOARD_SIZE, 3), np.uint8)
                cv2.imshow(f"Warped board {BOARD_SIZE}x{BOARD_SIZE}", debug_board)
                cv2.imshow("Square changes and legal moves", moves.debug_view(warped))

            key = cv2.waitKey(1) & 0xFF
            request = web.pop_action() if web is not None else None
            if request is not None:
                # Recheck against the live position: a move can land after enqueue.
                if request.get("expected_fen") != moves.board.fen():
                    calibration_status = "Position changed; repeat that action for the current board"
                    continue
                requested_action = request["action"]
                if requested_action == "save":
                    save_debug_capture(project, controller, moves, frame, result, panel, warped, corners)
                    calibration_status = controller.storage_status
                    continue
                if requested_action == "check":
                    key = 32
                elif requested_action == "promotion":
                    key = ord(str(request["choice"]))
                if pending_action == "manual" and requested_action in ("resync", "undo", "new"):
                    pending_action = None
                    manual_buffer = ""
                    manual_move = None
                    recovery_gate.reset()
                if requested_action == "manual":
                    try:
                        candidate = chess.Move.from_uci(request["move"])
                        if candidate not in moves.board.legal_moves:
                            raise ValueError("move is not legal for the side to move")
                    except ValueError as exc:
                        calibration_status = f"Invalid manual move: {exc}"
                    else:
                        pending_action = "manual"
                        manual_buffer = candidate.uci()
                        manual_move = candidate
                        recovery_gate.reset()
                        calibration_status = "Manual move ready; verify physical position, then confirm"
                    continue
                if requested_action not in ("check", "promotion"):
                    key = {"confirm": ord("y"), "resync": ord("r"), "undo": ord("u"),
                           "new": ord("n"), "cancel": 27, "save": ord("s"),
                           "rook": 13}[requested_action]
            if key == 32 and pending_action != "manual":
                if pending_action is not None or not reliable:
                    moves.check_result = "Confirm the physical position and clear the camera view first."
                else:
                    try:
                        moves.request_check()
                    except ValueError as exc:
                        moves.check_result = str(exc)
                continue
            if pending_action == "manual":
                if key == 27:
                    pending_action = ("setup" if moves.baseline is None and not args.assume_setup else None)
                    manual_buffer = ""
                    manual_move = None
                    calibration_status = "Manual move cancelled; check setup" if pending_action else "Manual move cancelled"
                    recovery_gate.reset()
                elif key in (8, 127):
                    manual_buffer = manual_buffer[:-1]
                    manual_move = None
                elif key in (10, 13):
                    try:
                        candidate = chess.Move.from_uci(manual_buffer)
                        if candidate not in moves.board.legal_moves:
                            raise ValueError("move is not legal for the side to move")
                    except ValueError as exc:
                        manual_move = None
                        calibration_status = f"Invalid manual move: {exc}"
                    else:
                        manual_move = candidate
                        calibration_status = (f"Legal move: {move_text(moves.board, candidate)}; "
                                              "press Y after the board settles")
                elif key == ord("y"):
                    if manual_move is None:
                        calibration_status = "Type a legal UCI move and press Enter first"
                    elif warped is None or state not in ("live", "tracked") or calibrating:
                        calibration_status = "Cannot confirm: board outline is not reliable"
                    else:
                        try:
                            controller.record_manual_move(manual_move, recovery_gate.snapshot())
                        except ValueError as exc:
                            calibration_status = f"Manual move failed: {exc}"
                        else:
                            pending_action = None
                            manual_buffer = ""
                            manual_move = None
                            calibration_status = "Manual move recorded"
                            recovery_gate.reset()
                elif 32 <= key <= 126 and chr(key).lower() in "abcdefgh12345678qrbn":
                    if len(manual_buffer) < 5:
                        manual_buffer += chr(key).lower()
                        manual_move = None
                continue
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrating = True
                clicked_points.clear()
                pending_corners = None
                pending_action = None
                calibration_status = "Click four outer corners in any order"
                recovery_gate.reset()
                continue
            if key == 27:
                pending_action = ("setup" if moves.baseline is None and not args.assume_setup else None)
                calibrating = False
                calibration_status = "Setup confirmation is required" if pending_action else "Action cancelled"
                recovery_gate.reset()
                continue
            if key in (ord("r"), ord("u"), ord("n")):
                desired = {ord("r"): "resync", ord("u"): "undo", ord("n"): "new"}[key]
                if desired == "undo" and not moves.board.move_stack:
                    calibration_status = "No recorded move to undo"
                else:
                    pending_action = desired
                    calibration_status = "Confirm only after the physical board is correct"
                    recovery_gate.reset()
                continue
            if key == ord("m"):
                pending_action = "manual"
                manual_buffer = ""
                manual_move = None
                calibration_status = "Type the completed legal move in UCI notation"
                recovery_gate.reset()
                continue
            if key == ord("y") and pending_action is not None:
                if warped is None or state not in ("live", "tracked") or calibrating:
                    calibration_status = "Cannot confirm: board outline is not reliable"
                    continue
                try:
                    baseline_image = recovery_gate.snapshot()
                    if pending_action in ("setup", "resync"):
                        controller.resync(baseline_image)
                    elif pending_action == "undo":
                        controller.undo(baseline_image)
                    else:
                        folder = controller.save_path.parent
                        controller.new_game(baseline_image, new_session_path(folder))
                except ValueError as exc:
                    calibration_status = f"Action failed: {exc}"
                    continue
                pending_action = None
                calibration_status = "Position confirmed"
                recovery_gate.reset()
                continue
            if key == ord("s"):
                save_debug_capture(project, controller, moves, frame, result, panel, warped, corners)
            if pending_action is None:
                accepted = moves.confirm(key)
                if accepted is not None:
                    controller.on_accepted_move(accepted)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if web is not None:
            web.close()
        elif conversation is not None:
            conversation.close()
        controller.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
